"""Reproduce every number and figure in the manuscript from the three CSV extracts.

Usage:  python analysis.py --data data --out results

Inputs (one row per record):
  claims.csv     claim_id, claim_type (O/I), claim_status, net_amount,
                 services_rendered, service_date (YYYY-MM-DD)
  providers.csv  provider_id, licensing_status (A/H/C), city, region,
                 facility_type, volume_of_claims
  members.csv    member_id, sex, city, region, policy_type, age_band,
                 total_healthcare_visits, chronic, diabetes, hypertension,
                 cad, dyslipidemia, thyroid

Outputs: results/results.json, results/numbers.tex (LaTeX macros used by
main.tex) and results/fig_*.pdf.

The rule thresholds, severities, score mapping, fusion weights and risk bands
below are copied from the Fraudspect rule engine and scorer. The ten CHI
predicates and the graph agent are re-implemented from the same code and run on
the repository's ten test cases.
"""
import argparse, json, math, os
from collections import Counter, OrderedDict
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- Fraudspect constants
SEV_SCORE = {"HIGH": 100, "MEDIUM": 60, "LOW": 30}
W_LLM, W_RULE, W_GRAPH = 0.40, 0.35, 0.25
HIGH_CUT, MED_CUT = 70, 40
CLAIM_HIGH_AMT, CLAIM_MED_AMT, CLAIM_SERVICES = 5000, 2500, 10
PROV_EXTREME, PROV_ELEVATED = 100000, 50000
MEM_VERY_HIGH, MEM_ELEVATED = 100, 50


def fraud_score(llm_score, viols, graph_score):
    rule = max([SEV_SCORE[s] for _, s in viols], default=0)
    final = round(min(100.0, llm_score * W_LLM + rule * W_RULE + graph_score * W_GRAPH), 2)
    cat = "HIGH" if final >= HIGH_CUT else "MEDIUM" if final >= MED_CUT else "LOW"
    if any(s == "HIGH" for _, s in viols) and cat == "LOW":
        cat = "MEDIUM"
    return final, cat, cat == "HIGH"


def claim_rules(amt, sv, status):
    v = []
    if amt >= CLAIM_HIGH_AMT: v.append(("CSV-CLAIM-001", "HIGH"))
    elif amt >= CLAIM_MED_AMT: v.append(("CSV-CLAIM-002", "MEDIUM"))
    if sv >= CLAIM_SERVICES: v.append(("CSV-CLAIM-003", "MEDIUM"))
    if status and status.upper() not in ("N", "A", "APPROVED"): v.append(("CSV-CLAIM-004", "LOW"))
    return v


def provider_rules(lic, vol):
    v = []
    if lic and lic.upper() != "A": v.append(("CSV-PROVIDER-001", "HIGH"))
    if vol >= PROV_EXTREME: v.append(("CSV-PROVIDER-002", "HIGH"))
    elif vol >= PROV_ELEVATED: v.append(("CSV-PROVIDER-003", "MEDIUM"))
    return v


def member_rules(visits):
    if visits >= MEM_VERY_HIGH: return [("CSV-MEMBER-001", "HIGH")]
    if visits >= MEM_ELEVATED: return [("CSV-MEMBER-002", "MEDIUM")]
    return []


# ---------------------------------------------------------------- CHI rule engine + graph agent (test cases)
PROHIBITED = {("27446", "27447"), ("44950", "44960"), ("99213", "99215"), ("93000", "93010")}
PEDS = {"99381", "99382", "99383", "99391", "99392", "90460", "90461"}

TEST_CASES = [
    ("Diabetes upcoding", dict(patient_id="P001", provider_id="DR001", diagnosis_codes=["E11.9"], procedure_codes=["99215"], medications=["insulin"], amount=450, date="2024-01-15")),
    ("Phantom appendectomy", dict(patient_id="P002", provider_id="DR002", diagnosis_codes=["K37"], procedure_codes=["44950"], medications=[], amount=8500, date="2024-01-20")),
    ("Duplicate claim", dict(patient_id="P003", provider_id="DR003", diagnosis_codes=["I10"], procedure_codes=["99213"], medications=[], amount=150, date="2024-01-22")),
    ("Doctor shopping", dict(patient_id="P004", provider_id="DR004", diagnosis_codes=["M54.5"], procedure_codes=["99213"], medications=["morphine"], amount=200, date="2024-01-25")),
    ("Asthma bronchodilator burst", dict(patient_id="P005", provider_id="DR005", diagnosis_codes=["J45.909"], procedure_codes=["94010", "94640", "94060"], medications=["salbutamol"], amount=750, date="2024-02-01")),
    ("Influenza extended antiviral", dict(patient_id="P006", provider_id="DR006", diagnosis_codes=["J11.1"], procedure_codes=["99213"], medications=["oseltamivir"], amount=180, date="2024-02-05")),
    ("RA contraindicated combination", dict(patient_id="P007", provider_id="DR007", diagnosis_codes=["M06.9"], procedure_codes=["99215"], medications=["methotrexate", "ibuprofen"], amount=380, date="2024-02-10")),
    ("IBS repeat colonoscopy", dict(patient_id="P008", provider_id="DR008", diagnosis_codes=["K58.9"], procedure_codes=["45378"], medications=[], amount=2200, date="2024-02-15")),
    ("Opioid over-prescribing", dict(patient_id="P009", provider_id="DR009", diagnosis_codes=["M54.5"], procedure_codes=["99215"], medications=["morphine"], amount=600, date="2024-02-20")),
    ("Provider referral loop", dict(patient_id="P010", provider_id="DR010", diagnosis_codes=["Z00.00"], procedure_codes=["99213"], medications=[], amount=300, date="2024-02-25", referral_provider_id="DR011")),
]


def d(s):
    return date.fromisoformat(s)


def rule_engine(p, history):
    v = []
    procs = p.get("procedure_codes", []); dx = p.get("diagnosis_codes", [])
    meds = [m.lower() for m in p.get("medications", [])]; t = d(p["date"])
    for a, b in PROHIBITED:
        if a in procs and b in procs: v.append(("CHI-001", "HIGH"))
    if p.get("patient_age") is not None:
        for c in procs:
            if c in PEDS and p["patient_age"] >= 18: v.append(("CHI-002", "MEDIUM"))
    win = t - timedelta(days=30)
    if any(h["patient_id"] == p["patient_id"] and h["provider_id"] == p["provider_id"] and d(h["date"]) >= win
           and set(h.get("procedure_codes", [])) & set(procs) for h in history):
        v.append(("CHI-003", "HIGH"))
    if any(c in ("E11.9", "E10.9") for c in dx) and "insulin" in meds and "83036" not in procs:
        v.append(("CHI-005", "LOW"))
    if "J45.909" in dx and sum(c in procs for c in ("94010", "94640", "94060")) >= 3:
        v.append(("CHI-006", "MEDIUM"))
    if "M06.9" in dx and "methotrexate" in meds and ("ibuprofen" in meds or "naproxen" in meds) and "85027" not in procs:
        v.append(("CHI-007", "MEDIUM"))
    if "K58.9" in dx and "45378" in procs:
        y = t - timedelta(days=365)
        if any(h["patient_id"] == p["patient_id"] and d(h["date"]) >= y and "45378" in h.get("procedure_codes", []) for h in history):
            v.append(("CHI-008", "MEDIUM"))
    provs = {h["provider_id"] for h in history if h["patient_id"] == p["patient_id"] and d(h["date"]) >= win}
    provs.add(p["provider_id"])
    if len(provs) >= 3: v.append(("CHI-010", "HIGH"))
    return v


def graph_agent(p, edges, history):
    pr, pt = "provider:" + p["provider_id"], "patient:" + p["patient_id"]
    edges.append((pr, pt))
    if p.get("referral_provider_id"): edges.append((pt, "provider:" + p["referral_provider_id"]))
    adj = {}
    for a, b in edges: adj.setdefault(a, []).append(b)

    def reach(f, t, seen):
        if f == t: return True
        seen.add(f)
        return any(n not in seen and reach(n, t, seen) for n in adj.get(f, []))
    loop = any(reach(n, pr, set()) for n in adj.get(pr, []))
    burst = sum(1 for h in history if h["provider_id"] == p["provider_id"] and h["date"] == p["date"]) + 1 >= 10
    nodes = {x for e in edges for x in e}
    deg = len(adj.get(pr, []))
    cent = 0 if len(nodes) < 3 else min(1, deg / max(1, len(nodes) - 1)) * 0.3
    g = min(100, cent * 50 + (30 if loop else 0) + (20 if burst else 0))
    return round(cent, 4), loop, burst, round(g, 2)


# ---------------------------------------------------------------- helpers
def pct(a, b): return 100.0 * a / b


def fmt_int(x): return f"{int(x):,}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    C = pd.read_csv(os.path.join(a.data, "claims.csv"), dtype={"claim_id": str, "claim_status": str})
    P = pd.read_csv(os.path.join(a.data, "providers.csv"), dtype={"provider_id": str, "licensing_status": str})
    M = pd.read_csv(os.path.join(a.data, "members.csv"), dtype={"member_id": str})
    R = OrderedDict()
    N = OrderedDict()  # LaTeX macros

    # ---------------- claims descriptives
    amt = C.net_amount.values; sv = C.services_rendered.values
    R["claims"] = dict(n=len(C), total=float(amt.sum()), median=float(np.median(amt)), mean=float(amt.mean()),
                       p90=float(np.percentile(amt, 90)), p99=float(np.percentile(amt, 99)), max=float(amt.max()), min=float(amt.min()),
                       sv_mean=float(sv.mean()), sv_var=float(sv.var(ddof=1)), sv_median=float(np.median(sv)), sv_max=int(sv.max()),
                       inpatient=int((C.claim_type == "I").sum()), status=C.claim_status.value_counts().to_dict(),
                       date_min=C.service_date.min(), date_max=C.service_date.max(), days=int(C.service_date.nunique()))
    ip = C[C.claim_type == "I"].net_amount; op = C[C.claim_type == "O"].net_amount
    R["claims"].update(ip_median=float(ip.median()), op_median=float(op.median()), ip_share_value=float(ip.sum() / amt.sum()))
    top10 = np.sort(amt)[::-1][: int(len(amt) * 0.10)].sum() / amt.sum()
    R["claims"]["top10_value_share"] = float(top10)
    rho_c = stats.spearmanr(amt, sv)
    R["claims"]["spearman_amt_sv"] = float(rho_c.statistic)
    daily = C.groupby("service_date").size()
    R["claims"]["daily_max"] = int(daily.max()); R["claims"]["daily_max_date"] = daily.idxmax(); R["claims"]["daily_median"] = float(daily.median())

    # ---------------- providers
    vol = P.volume_of_claims.values
    R["providers"] = dict(n=len(P), lic=P.licensing_status.value_counts().to_dict(), vol_median=float(np.median(vol)),
                          vol_mean=float(vol.mean()), vol_q1=float(np.percentile(vol, 25)), vol_q3=float(np.percentile(vol, 75)),
                          vol_max=int(vol.max()), vol_min=int(vol.min()), types=P.facility_type.value_counts().to_dict(),
                          regions=P.region.value_counts().to_dict())
    sv_ = np.sort(vol)[::-1]
    R["providers"]["top1pct_vol_share"] = float(sv_[: max(1, int(round(len(vol) * 0.01)))].sum() / vol.sum())
    R["providers"]["top10pct_vol_share"] = float(sv_[: int(round(len(vol) * 0.10))].sum() / vol.sum())
    R["providers"]["vol_by_type_median"] = P.groupby("facility_type").volume_of_claims.median().to_dict()

    # ---------------- members
    vis = M.total_healthcare_visits.values
    conds = ["diabetes", "hypertension", "dyslipidemia", "thyroid", "cad"]
    R["members"] = dict(rows=len(M), distinct=int(M.member_id.nunique()), dup_ids=int(len(M) - M.member_id.nunique()),
                        visits_median=float(np.median(vis)), visits_mean=float(vis.mean()), visits_max=int(vis.max()),
                        zero_visits=int((vis == 0).sum()), chronic=int(M.chronic.sum()),
                        conds={c: int(M[c].sum()) for c in conds}, age=M.age_band.value_counts().to_dict(),
                        sex=M.sex.value_counts().to_dict(), policy=M.policy_type.value_counts().to_dict(),
                        region=M.region.value_counts().to_dict())
    R["members"]["age46plus"] = int(M.age_band.isin(["46-60", "61+"]).sum())
    ch = M[M.chronic == 1].total_healthcare_visits; nc = M[M.chronic == 0].total_healthcare_visits
    mw = stats.mannwhitneyu(ch, nc, alternative="two-sided")
    R["members"].update(visits_median_chronic=float(ch.median()), visits_median_nonchronic=float(nc.median()), mw_p=float(mw.pvalue),
                        mw_U=float(mw.statistic))
    M["ncond"] = M[conds].sum(axis=1).clip(upper=4)  # 4 = four or more
    R["members"]["visits_median_by_ncond"] = {int(k): float(v) for k, v in M.groupby("ncond").total_healthcare_visits.median().items()}
    R["members"]["n_by_ncond"] = {int(k): int(v) for k, v in M.ncond.value_counts().sort_index().items()}
    rho_m = stats.spearmanr(M.ncond, M.total_healthcare_visits)
    R["members"]["spearman_ncond_visits"] = float(rho_m.statistic); R["members"]["spearman_ncond_visits_p"] = float(rho_m.pvalue)
    R["members"]["visits_median_by_age"] = M.groupby("age_band").total_healthcare_visits.median().to_dict()
    vh = M[M.total_healthcare_visits >= MEM_VERY_HIGH]
    R["members"]["veryhigh_chronic"] = int(vh.chronic.sum()); R["members"]["veryhigh_n"] = len(vh)
    el = M[M.total_healthcare_visits >= MEM_ELEVATED]
    R["members"]["elev_or_more_chronic"] = int(el.chronic.sum()); R["members"]["elev_or_more_n"] = len(el)

    # ---------------- bulk-path rule engine and scorer
    hits = Counter(); flagged = {"claims": 0, "providers": 0, "members": 0}; cats = {k: Counter() for k in flagged}; scores = {k: Counter() for k in flagged}
    escal = 0; flagged_value = 0.0; claim_multi = 0
    for r in C.itertuples():
        v = claim_rules(r.net_amount, r.services_rendered, r.claim_status)
        for x in v: hits[x[0]] += 1
        s, cat, e = fraud_score(0, v, 0); cats["claims"][cat] += 1; scores["claims"][s] += 1; escal += e
        if v: flagged["claims"] += 1; flagged_value += r.net_amount
        if len(v) > 1: claim_multi += 1
    for r in P.itertuples():
        v = provider_rules(r.licensing_status, r.volume_of_claims)
        for x in v: hits[x[0]] += 1
        s, cat, e = fraud_score(0, v, 0); cats["providers"][cat] += 1; scores["providers"][s] += 1; escal += e
        if v: flagged["providers"] += 1
    for r in M.itertuples():
        v = member_rules(r.total_healthcare_visits)
        for x in v: hits[x[0]] += 1
        s, cat, e = fraud_score(0, v, 0); cats["members"][cat] += 1; scores["members"][s] += 1; escal += e
        if v: flagged["members"] += 1
    R["bulk"] = dict(hits=dict(sorted(hits.items())), flagged=flagged, categories={k: dict(v) for k, v in cats.items()},
                     scores={k: {str(s): n for s, n in sorted(v.items())} for k, v in scores.items()}, escalated=escal,
                     flagged_claim_value=flagged_value, claims_multi_rule=claim_multi)
    # amount + service overlap
    big = C.net_amount >= CLAIM_MED_AMT; many = C.services_rendered >= CLAIM_SERVICES
    R["bulk"]["amount_and_services"] = int((big & many).sum())
    # threshold percentiles in the data
    R["thresholds"] = dict(
        amt5000_pct=float(stats.percentileofscore(amt, CLAIM_HIGH_AMT, kind="strict")),
        amt2500_pct=float(stats.percentileofscore(amt, CLAIM_MED_AMT, kind="strict")),
        sv10_pct=float(stats.percentileofscore(sv, CLAIM_SERVICES, kind="strict")),
        vis100_pct=float(stats.percentileofscore(vis, MEM_VERY_HIGH, kind="strict")),
        vis50_pct=float(stats.percentileofscore(vis, MEM_ELEVATED, kind="strict")),
        vol100k_pct=float(stats.percentileofscore(vol, PROV_EXTREME, kind="strict")),
        vol50k_pct=float(stats.percentileofscore(vol, PROV_ELEVATED, kind="strict")))
    # provider flags by type
    P["flag"] = [bool(provider_rules(l, v)) for l, v in zip(P.licensing_status, P.volume_of_claims)]
    P["nonactive"] = P.licensing_status != "A"; P["extreme"] = P.volume_of_claims >= PROV_EXTREME
    bt = P.groupby("facility_type").agg(n=("flag", "size"), flagged=("flag", "sum"), nonactive=("nonactive", "sum"), extreme=("extreme", "sum"))
    R["providers"]["by_type"] = {k: {kk: int(vv) for kk, vv in row.items()} for k, row in bt.iterrows()}
    hosp_ext = int(P[(P.facility_type == "Hospital") & P.extreme].shape[0])
    R["providers"]["extreme_hospitals"] = hosp_ext
    # chi-square: extreme volume vs hospital
    tab = pd.crosstab(P.facility_type == "Hospital", P.extreme)
    chi = stats.chi2_contingency(tab)
    R["providers"]["chi2_hosp_extreme_p"] = float(chi.pvalue)
    # member flags vs chronic
    M["flag"] = M.total_healthcare_visits >= MEM_ELEVATED
    R["members"]["flag_rate_chronic"] = float(M[M.chronic == 1].flag.mean()); R["members"]["flag_rate_nonchronic"] = float(M[M.chronic == 0].flag.mean())
    tabm = pd.crosstab(M.chronic, M.flag); chim = stats.chi2_contingency(tabm)
    R["members"]["chi2_chronic_flag_p"] = float(chim.pvalue)
    R["members"]["flag_rate_by_policy"] = M.groupby("policy_type").flag.mean().to_dict()

    # ---------------- score ceilings (analytic, from scorer)
    ceil = OrderedDict()
    for name, (l, r_, g) in {"rules": (0, 100, 0), "rules+graph": (0, 100, 100), "llm": (100, 0, 0), "llm+graph": (100, 0, 100), "llm+rules": (100, 100, 0), "all": (100, 100, 100)}.items():
        ceil[name] = round(min(100, l * W_LLM + r_ * W_RULE + g * W_GRAPH), 2)
    R["ceilings"] = ceil
    # minimum LLM score for escalation when rules and graph are maxed
    R["min_llm_for_high_all_max"] = (HIGH_CUT - 100 * W_RULE - 100 * W_GRAPH) / W_LLM
    R["min_llm_for_high_rules_only"] = (HIGH_CUT - 100 * W_RULE) / W_LLM

    # ---------------- test cases: rule engine + graph agent (LLM not run)
    history, edges = [], [("provider:DR011", "patient:P010"), ("patient:P010", "provider:DR010")]
    tc = []
    for name, p in TEST_CASES:
        v = rule_engine(p, history); cent, loop, burst, g = graph_agent(p, edges, history)
        s, cat, e = fraud_score(0, v, g)
        tc.append(dict(name=name, rules=[x[0] for x in v], centrality=cent, loop=loop, burst=burst, graph=g, score_no_llm=s, cat_no_llm=cat,
                       llm_needed_for_high=round(max(0, (HIGH_CUT - (max([SEV_SCORE[x[1]] for x in v], default=0) * W_RULE + g * W_GRAPH)) / W_LLM), 1)))
        history.append(p)
    # second pass of the duplicate claim, as the repository instructs
    p = dict(TEST_CASES[2][1]); v = rule_engine(p, history); cent, loop, burst, g = graph_agent(p, edges, history); s, cat, e = fraud_score(0, v, g)
    tc.append(dict(name="Duplicate claim (second submission)", rules=[x[0] for x in v], centrality=cent, loop=loop, burst=burst, graph=g, score_no_llm=s, cat_no_llm=cat,
                   llm_needed_for_high=round(max(0, (HIGH_CUT - (max([SEV_SCORE[x[1]] for x in v], default=0) * W_RULE + g * W_GRAPH)) / W_LLM), 1)))
    R["test_cases"] = tc

    # ---------------- readiness matrix
    R["readiness"] = [
        ("CSV-CLAIM-001 to 004", "amount, services, status", True),
        ("CSV-PROVIDER-001 to 003", "licence, volume", True),
        ("CSV-MEMBER-001/002", "visit count", True),
        ("CHI-001 prohibited CPT pairs", "procedure codes", False),
        ("CHI-002 paediatric CPT", "procedure codes, age", False),
        ("CHI-003 duplicate billing", "member id, provider id, codes, dates", False),
        ("CHI-005 to CHI-008", "diagnosis, procedure, medication codes", False),
        ("CHI-010 doctor shopping", "member id, provider id, dates", False),
        ("Graph: loops, bursts, centrality", "provider id, member id, referrals", False),
        ("LLM clinical review", "diagnoses, procedures, medications", False)]

    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)

    # ---------------- LaTeX macros
    c, pv, m, b, t = R["claims"], R["providers"], R["members"], R["bulk"], R["thresholds"]
    def mac(k, v): N[k] = v
    mac("nClaims", fmt_int(c["n"])); mac("claimTotal", f"{c['total']:,.0f}"); mac("claimMedian", f"{c['median']:.2f}"); mac("claimMean", f"{c['mean']:,.2f}")
    mac("claimPninety", f"{c['p90']:,.0f}"); mac("claimPnn", f"{c['p99']:,.0f}"); mac("claimMax", f"{c['max']:,.2f}"); mac("claimMin", f"{c['min']:.2f}")
    mac("svMean", f"{c['sv_mean']:.2f}"); mac("svVar", f"{c['sv_var']:.1f}"); mac("svMedian", f"{c['sv_median']:.0f}"); mac("svMax", str(c["sv_max"]))
    mac("nInpatient", str(c["inpatient"])); mac("nStatusJ", str(c["status"].get("J", 0))); mac("nStatusN", fmt_int(c["status"].get("N", 0)))
    mac("ipMedian", f"{c['ip_median']:,.0f}"); mac("opMedian", f"{c['op_median']:.0f}"); mac("ipShareValue", f"{100*c['ip_share_value']:.1f}")
    mac("topTenValueShare", f"{100*c['top10_value_share']:.1f}"); mac("rhoAmtSv", f"{c['spearman_amt_sv']:.2f}")
    mac("dailyMax", str(c["daily_max"])); mac("dailyMedian", f"{c['daily_median']:.0f}"); mac("nDays", str(c["days"]))
    mac("nProv", str(pv["n"])); mac("nProvActive", str(pv["lic"].get("A", 0))); mac("nProvHold", str(pv["lic"].get("H", 0))); mac("nProvCancel", str(pv["lic"].get("C", 0)))
    mac("provActivePct", f"{pct(pv['lic'].get('A',0), pv['n']):.1f}")
    mac("volMedian", f"{pv['vol_median']:,.0f}"); mac("volMean", f"{pv['vol_mean']:,.0f}"); mac("volQone", f"{pv['vol_q1']:,.0f}"); mac("volQthree", f"{pv['vol_q3']:,.0f}"); mac("volMax", f"{pv['vol_max']:,}")
    mac("volTopOne", f"{100*pv['top1pct_vol_share']:.1f}"); mac("volTopTen", f"{100*pv['top10pct_vol_share']:.1f}")
    mac("nPoly", str(pv["types"].get("Polyclinic", 0))); mac("nHosp", str(pv["types"].get("Hospital", 0))); mac("nDental", str(pv["types"].get("Dental Center", 0))); mac("nTypes", str(len(pv["types"])))
    mac("hospVolMedian", f"{pv['vol_by_type_median'].get('Hospital',0):,.0f}"); mac("polyVolMedian", f"{pv['vol_by_type_median'].get('Polyclinic',0):,.0f}")
    mac("extremeHosp", str(pv["extreme_hospitals"])); mac("chiHospP", "<0.001" if pv["chi2_hosp_extreme_p"] < 0.001 else f"{pv['chi2_hosp_extreme_p']:.3f}")
    mac("nMemRows", fmt_int(m["rows"])); mac("nMemDistinct", fmt_int(m["distinct"])); mac("nMemDup", str(m["dup_ids"]))
    mac("visMedian", f"{m['visits_median']:.1f}"); mac("visMean", f"{m['visits_mean']:.1f}"); mac("visMax", str(m["visits_max"])); mac("visZero", str(m["zero_visits"]))
    mac("nChronic", str(m["chronic"])); mac("chronicPct", f"{pct(m['chronic'], m['rows']):.1f}")
    for k, nm in [("diabetes", "Dm"), ("hypertension", "Htn"), ("dyslipidemia", "Dys"), ("thyroid", "Thy"), ("cad", "Cad")]: mac("n" + nm, str(m["conds"][k]))
    mac("ageFortySixPct", f"{pct(m['age46plus'], m['rows']):.1f}"); mac("nMale", str(m["sex"].get("M", 0))); mac("nFemale", str(m["sex"].get("F", 0)))
    mac("visMedChronic", f"{m['visits_median_chronic']:.0f}"); mac("visMedNonChronic", f"{m['visits_median_nonchronic']:.0f}")
    mac("mwP", "<0.001" if m["mw_p"] < 0.001 else f"{m['mw_p']:.3f}")
    mac("rhoCondVis", f"{m['spearman_ncond_visits']:.2f}")
    mac("vhN", str(m["veryhigh_n"])); mac("vhChronic", str(m["veryhigh_chronic"])); mac("vhChronicPct", f"{pct(m['veryhigh_chronic'], m['veryhigh_n']):.1f}")
    mac("elN", str(m["elev_or_more_n"])); mac("elChronic", str(m["elev_or_more_chronic"])); mac("elChronicPct", f"{pct(m['elev_or_more_chronic'], m['elev_or_more_n']):.1f}")
    mac("flagRateChronic", f"{100*m['flag_rate_chronic']:.1f}"); mac("flagRateNonChronic", f"{100*m['flag_rate_nonchronic']:.1f}")
    for k, v in b["hits"].items(): mac("hit" + k.replace("CSV-", "").replace("-", "").replace("001", "A").replace("002", "B").replace("003", "C").replace("004", "D"), str(v))
    mac("flagClaims", str(b["flagged"]["claims"])); mac("flagProv", str(b["flagged"]["providers"])); mac("flagMem", str(b["flagged"]["members"]))
    mac("flagClaimsPct", f"{pct(b['flagged']['claims'], c['n']):.1f}"); mac("flagProvPct", f"{pct(b['flagged']['providers'], pv['n']):.1f}"); mac("flagMemPct", f"{pct(b['flagged']['members'], m['rows']):.1f}")
    mac("flagClaimValue", f"{b['flagged_claim_value']:,.0f}"); mac("flagClaimValuePct", f"{100*b['flagged_claim_value']/c['total']:.1f}")
    mac("claimMulti", str(b["claims_multi_rule"])); mac("amtAndSv", str(b["amount_and_services"])); mac("nEscalated", str(b["escalated"]))
    nmed = sum(v.get("MEDIUM", 0) for v in b["categories"].values()); nlow = sum(v.get("LOW", 0) for v in b["categories"].values()); nhigh = sum(v.get("HIGH", 0) for v in b["categories"].values())
    mac("nMedium", fmt_int(nmed)); mac("nLow", fmt_int(nlow)); mac("nHigh", str(nhigh)); mac("nRecords", fmt_int(c["n"] + pv["n"] + m["rows"]))
    mac("pctAmtFiveK", f"{100-t['amt5000_pct']:.1f}"); mac("pctAmtTwoFiveK", f"{100-t['amt2500_pct']:.1f}"); mac("pctSvTen", f"{100-t['sv10_pct']:.1f}")
    mac("pctVisHundred", f"{100-t['vis100_pct']:.1f}"); mac("pctVisFifty", f"{100-t['vis50_pct']:.1f}"); mac("pctVolHundredK", f"{100-t['vol100k_pct']:.1f}"); mac("pctVolFiftyK", f"{100-t['vol50k_pct']:.1f}")
    mac("ceilRules", f"{ceil['rules']:.0f}"); mac("ceilRulesGraph", f"{ceil['rules+graph']:.0f}"); mac("ceilLlmGraph", f"{ceil['llm+graph']:.0f}"); mac("ceilLlmRules", f"{ceil['llm+rules']:.0f}")
    mac("nTestReachable", str(sum(1 for x in tc if x["llm_needed_for_high"] <= 100))); mac("nTestRuns", str(len(tc)))
    mac("nTestRuleHit", str(sum(1 for x in tc if x["rules"])))
    chron_none = int(((M.chronic == 1) & (M[conds].sum(axis=1) == 0)).sum()); mac("chronicNoListed", str(chron_none))
    mac("chiChronicP", "<0.001" if m["chi2_chronic_flag_p"] < 0.001 else f"{m['chi2_chronic_flag_p']:.3f}")
    hv = C[C.net_amount >= CLAIM_HIGH_AMT]; mac("hvInpatient", str(int((hv.claim_type == "I").sum()))); mac("hvValuePct", f"{100*hv.net_amount.sum()/c['total']:.1f}")
    mac("ipHighValue", str(int(((C.claim_type == "I") & (C.net_amount >= CLAIM_HIGH_AMT)).sum())))
    mac("nZeroAmount", str(int((C.net_amount == 0).sum())))
    mac("svFlagValuePct", f"{100*C[C.services_rendered >= CLAIM_SERVICES].net_amount.sum()/c['total']:.1f}")
    mac("dupRowsPolicy", str(m["dup_ids"]))
    pt = P[P.extreme]; mac("extremeVolShare", f"{100*pt.volume_of_claims.sum()/P.volume_of_claims.sum():.1f}")
    mac("nonActiveVolMedian", f"{P[P.nonactive].volume_of_claims.median():,.0f}")
    bt = pv["by_type"]; mac("extremePoly", str(bt.get("Polyclinic", {}).get("extreme", 0))); mac("extremePharm", str(bt.get("Pharmacy", {}).get("extreme", 0)))
    mac("hospShareExtreme", f"{100*bt['Hospital']['extreme']/b['hits'].get('CSV-PROVIDER-002',1):.1f}"); mac("hospShareAll", f"{100*pv['types']['Hospital']/pv['n']:.1f}")
    mac("hospExtremeRate", f"{100*bt['Hospital']['extreme']/bt['Hospital']['n']:.1f}"); mac("polyExtremeRate", f"{100*bt['Polyclinic']['extreme']/bt['Polyclinic']['n']:.1f}")
    ftot = sum(b["flagged"].values()); mac("flaggedTotal", str(ftot)); mac("flaggedLow", str(ftot - nmed))
    mac("minLlmAll", f"{R['min_llm_for_high_all_max']:.0f}"); mac("minLlmRules", f"{R['min_llm_for_high_rules_only']:.1f}")
    with open(os.path.join(a.out, "testcases.tex"), "w") as f:
        f.write("% Generated by analysis.py. Do not edit by hand.\n")
        for x in tc:
            need = f"{x['llm_needed_for_high']:.1f}" if x["llm_needed_for_high"] <= 100 else "not reachable"
            f.write(f"{x['name']} & {', '.join(x['rules']) or '--'} & {'yes' if x['loop'] else 'no'} & {x['graph']:.2f} & {x['score_no_llm']:.2f} & {x['cat_no_llm']} & {need}\\\\\n")
    with open(os.path.join(a.out, "numbers.tex"), "w") as f:
        f.write("% Generated by analysis.py from the CSV extracts. Do not edit by hand.\n")
        for k, v in N.items():
            f.write("\\newcommand{\\" + k + "}{" + v.replace("<", "$<$") + "}\n")

    # ---------------- figures
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
                         "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555", "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
    BLUE, ORANGE, GREY = "#257de0", "#f47230", "#9c9c9c"

    # Fig A: claim amount and services, with rule thresholds
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    bins = np.logspace(np.log10(max(1, amt.min())), np.log10(amt.max()), 40)
    ax[0].hist(amt, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6); ax[0].set_xscale("log")
    for x, lab, ha, dx in [(CLAIM_MED_AMT, "2,500", "right", 0.92), (CLAIM_HIGH_AMT, "5,000", "left", 1.08)]:
        ax[0].axvline(x, color="#333333", lw=0.9, ls="--"); ax[0].text(x * dx, ax[0].get_ylim()[1] * 0.92, lab, fontsize=7.5, color="#333333", ha=ha)
    ax[0].set_xlabel("net claim amount (SAR, log scale)"); ax[0].set_ylabel("claims"); ax[0].set_title("(a) claim amount", loc="left", fontsize=9)
    sb = np.arange(0.5, min(60, sv.max()) + 1.5, 1)
    ax[1].hist(np.clip(sv, 0, 60), bins=sb, color=BLUE, edgecolor="white", linewidth=0.4)
    ax[1].axvline(CLAIM_SERVICES - 0.5, color="#333333", lw=0.9, ls="--"); ax[1].text(CLAIM_SERVICES + 0.5, ax[1].get_ylim()[1] * 0.92, "10 services", fontsize=7.5)
    ax[1].set_xlabel("services per claim (capped at 60)"); ax[1].set_ylabel("claims"); ax[1].set_title("(b) services per claim", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_claims.pdf")); plt.close(fig)

    # Fig B: member visits by chronic status
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    vb = np.arange(0, 210, 10)
    ax[0].hist([np.clip(nc, 0, 200), np.clip(ch, 0, 200)], bins=vb, color=[GREY, BLUE], label=["no chronic condition", "one or more chronic conditions"], edgecolor="white", linewidth=0.5)
    for x in (MEM_ELEVATED, MEM_VERY_HIGH): ax[0].axvline(x, color="#333333", lw=0.9, ls="--")
    ax[0].set_xlabel("total healthcare visits (capped at 200)"); ax[0].set_ylabel("members"); ax[0].legend(frameon=False, fontsize=7.5)
    ax[0].set_title("(a) visits by chronic status", loc="left", fontsize=9)
    ks = sorted(m["visits_median_by_ncond"].keys())
    ax[1].bar([str(k) for k in ks], [m["visits_median_by_ncond"][k] for k in ks], color=BLUE, width=0.6)
    for i, k in enumerate(ks): ax[1].text(i, m["visits_median_by_ncond"][k] + 1, f"n={m['n_by_ncond'][k]}", ha="center", fontsize=7)
    ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels([str(k) if k < 4 else "4+" for k in ks]); ax[1].set_xlabel("number of recorded chronic conditions"); ax[1].set_ylabel("median visits"); ax[1].set_title("(b) median visits by condition count", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_members.pdf")); plt.close(fig)

    # Fig C: provider volume by facility type (top types)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.7))
    top = [k for k, _ in sorted(pv["types"].items(), key=lambda x: -x[1])[:6]]
    data = [P[P.facility_type == k].volume_of_claims.clip(lower=1).values for k in top]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, medianprops=dict(color="#111111"), flierprops=dict(marker="o", markersize=2.5, markerfacecolor=BLUE, markeredgecolor="none", alpha=0.6))
    for bx in bp["boxes"]: bx.set(facecolor="#cce3fd", edgecolor=BLUE)
    ax.set_yticks(range(1, len(top) + 1)); ax.set_yticklabels([f"{k} (n={pv['types'][k]})" for k in top]); ax.set_xscale("log")
    for x in (PROV_ELEVATED, PROV_EXTREME): ax.axvline(x, color="#333333", lw=0.9, ls="--")
    ax.set_xlabel("annual claim volume (log scale); dashed lines at 50,000 and 100,000")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_providers.pdf")); plt.close(fig)

    # Fig D: threshold sensitivity curves
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    def curve(axx, vals, grid, mark, xl, title):
        frac = [100 * np.mean(vals >= g) for g in grid]
        axx.plot(grid, frac, color=BLUE, lw=1.8); axx.set_xscale("log")
        for mk in mark: axx.plot([mk], [100 * np.mean(vals >= mk)], "o", color=ORANGE, ms=4.5); axx.annotate(f"{100*np.mean(vals>=mk):.1f}%", (mk, 100 * np.mean(vals >= mk)), textcoords="offset points", xytext=(4, 4), fontsize=7)
        axx.set_xlabel(xl); axx.set_title(title, loc="left", fontsize=9)
    curve(ax[0], amt, np.logspace(2, 4.5, 80), [CLAIM_MED_AMT, CLAIM_HIGH_AMT], "amount threshold (SAR)", "(a) claims flagged, %")
    curve(ax[1], vis.astype(float), np.logspace(1, 2.6, 80), [MEM_ELEVATED, MEM_VERY_HIGH], "visit threshold", "(b) members flagged, %")
    curve(ax[2], vol.astype(float), np.logspace(3.5, 6, 80), [PROV_ELEVATED, PROV_EXTREME], "volume threshold", "(c) providers flagged, %")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_thresholds.pdf")); plt.close(fig)

    print(json.dumps({k: R[k] for k in ("bulk", "ceilings")}, indent=1, default=str))
    print(len(N), "macros")


if __name__ == "__main__":
    main()
