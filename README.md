# Fraudspect

Healthcare claims fraud screening for payer data. Fraudspect scores each claim with
three signals and combines them:

| Signal | Weight | What it does |
|---|---|---|
| LLM clinical reasoner | 40% | Reads diagnosis, procedure and medication codes and judges clinical fit (Groq, `gpt-oss-20b`) |
| Rule engine | 35% | CHI rules (prohibited code pairs, paediatric codes on adults, duplicates, bursts, doctor shopping) and CSV rules on amount, services, licence status, provider volume and member visits. HIGH = 100, MEDIUM = 60, LOW = 30 |
| Provider graph | 25% | Centrality, referral loops and billing bursts across the provider-patient network |

A final score of 70 or more is HIGH and escalates to a case; 40 to 69 is MEDIUM. A HIGH rule
hit is never left in the LOW band.

## Repository layout

```
dashboard/            Investigator dashboard (single HTML page, ECharts + Lucide from CDN)
  index.html
  data/               data.js goes here (built from the CSVs, git-ignored)
api/ai.js             Serverless endpoint for the dashboard's AI features (Groq)
analysis/
  analysis.py         Reproduces every number, table and figure in the manuscript
  results/            Output of analysis.py: results.json, numbers.tex, testcases.tex, fig_*.pdf
scripts/
  csv_to_dashboard.py Converts the three CSVs into dashboard/data/data.js
data/                 Put the CSV extracts here (git-ignored)
```

## Data

The three extracts are not in this repository. Place them in `data/`:

- `claims.csv`: claim_id, claim_type (O/I), claim_status, net_amount, services_rendered, service_date (YYYY-MM-DD)
- `providers.csv`: provider_id, licensing_status (A/H/C), city, region, facility_type, volume_of_claims
- `members.csv`: member_id, sex, city, region, policy_type, age_band, total_healthcare_visits,
  chronic, diabetes, hypertension, cad, dyslipidemia, thyroid

The results in `analysis/results/` were produced from extracts with 1,000 claims, 989 providers
and 1,016 member rows (claims dated 17 Dec 2024 to 20 Jan 2025).

## Reproduce the analysis

```bash
pip install -r requirements.txt
python analysis/analysis.py --data data --out analysis/results
```

## Run the dashboard

```bash
python scripts/csv_to_dashboard.py --data data --out dashboard/data/data.js
cd dashboard && python -m http.server 8000
# open http://localhost:8000
```

Without `data.js` the dashboard opens with empty tables. The rule engine, scorer and the
"Analyze a Claim" test cases run in the browser.

### AI features

"Ask AI", "AI briefing" and "Run AI review" call `POST /api/ai`. To enable them, deploy the
repository to Vercel (the root redirects to `/dashboard/`) and set these environment variables
in the Vercel project settings:

```
GROQ_API_KEY=<your key>
GROQ_MODEL=openai/gpt-oss-20b   # optional
```

See `.env.example`. Never commit a real key. If an old key was ever shared in code or chat,
revoke it in the Groq console and create a new one.

## Citation

If you use this code, please cite the Fraudspect manuscript.
