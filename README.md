# Heart Disease Risk Explorer — Streamlit Dashboard

An interactive dashboard for exploring which patient factors track
with heart disease in the UCI Cleveland dataset (297 patients after
cleaning). Built as the stakeholder-facing companion to the
heart-disease ML pipeline: a clinic team can filter the patient
population and see prevalence patterns without touching code.

**This is a pattern-exploration tool, not a diagnostic tool.**

## What it shows

- Headline metrics for the filtered population: patient count, heart
  disease prevalence, average age, average max heart rate
- Prevalence by age group and by sex
- Resting blood pressure vs cholesterol, colored by disease status
- Random forest feature importance — which measurements a model leans
  on (chest pain type, number of major vessels, thalassemia result)
- Sidebar filters for sex and age range; every chart and metric
  updates with the filter

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually localhost:8501).

## Deploy it (free)

1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, pick the repo and
   `app.py`.
3. Streamlit Cloud installs `requirements.txt` and gives you a public
   URL — that URL goes on the resume and LinkedIn.

## How it works

- `@st.cache_data` loads and cleans the CSV once instead of on every
  interaction; Streamlit reruns the script top-to-bottom on each
  filter change, so the cache is what keeps it fast.
- The model section trains a small random forest on the full dataset
  at load — 297 rows, so it trains instantly.

## Interview notes — be able to explain

- **Why a dashboard at all?** The ML team's output only matters if
  stakeholders can use it. This is the "visualizations and dashboards
  for internal stakeholders" part of real data work.
- **Why the disclaimer?** In healthcare, an exploration tool must
  never present itself as a diagnostic device — different standard of
  evidence, and real patient-safety consequences.
- **Caching:** what `@st.cache_data` does and why the app would feel
  slow without it.
