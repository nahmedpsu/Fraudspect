"""Build dashboard/data/data.js from the three CSV extracts.

Usage:  python scripts/csv_to_dashboard.py --data data --out dashboard/data/data.js

Expects the same files and column names as analysis/analysis.py:
  claims.csv     claim_id, claim_type, claim_status, net_amount, services_rendered, service_date
  providers.csv  provider_id, licensing_status, city, region, facility_type, volume_of_claims
  members.csv    member_id, sex, city, region, policy_type, age_band, total_healthcare_visits,
                 chronic, diabetes, hypertension, cad, dyslipidemia, thyroid

The output file holds the data rows, so it is listed in .gitignore.
"""
import argparse, json, os

import pandas as pd

CLAIM_COLS = ["claim_id", "claim_type", "claim_status", "net_amount", "services_rendered", "service_date"]
PROV_COLS = ["provider_id", "licensing_status", "city", "region", "facility_type", "volume_of_claims"]
MEM_COLS = ["member_id", "sex", "city", "region", "policy_type", "age_band", "total_healthcare_visits",
            "chronic", "diabetes", "hypertension", "cad", "dyslipidemia", "thyroid"]


def rows(df, cols, num):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    out = []
    for rec in df[cols].itertuples(index=False):
        r = []
        for c, v in zip(cols, rec):
            if c in num:
                v = 0 if pd.isna(v) else float(v)
                r.append(int(v) if v.is_integer() and c != "net_amount" else v)
            else:
                r.append("" if pd.isna(v) else str(v))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dashboard/data/data.js")
    a = ap.parse_args()
    C = pd.read_csv(os.path.join(a.data, "claims.csv"), dtype={"claim_id": str, "claim_status": str})
    P = pd.read_csv(os.path.join(a.data, "providers.csv"), dtype={"provider_id": str, "licensing_status": str})
    M = pd.read_csv(os.path.join(a.data, "members.csv"), dtype={"member_id": str})
    C["service_date"] = pd.to_datetime(C["service_date"]).dt.strftime("%Y-%m-%d")
    data = {
        "claims": rows(C, CLAIM_COLS, {"net_amount", "services_rendered"}),
        "providers": rows(P, PROV_COLS, {"volume_of_claims"}),
        "members": rows(M, MEM_COLS, set(MEM_COLS[6:])),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("window.FRAUDSPECT_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {a.out}: {len(data['claims'])} claims, {len(data['providers'])} providers, {len(data['members'])} members")


if __name__ == "__main__":
    main()
