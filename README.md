# 📊 AI Data Scientist

A system where you upload any CSV and get **senior data scientist–style** analysis: profiling, EDA, statistical structure, anomalies, modeling recommendations, and an executive summary — in under a few seconds.

**Vibe:** *"I just hired a data scientist for 10 seconds."*

---

## What it does

1. **Profiles the dataset** — dtypes, missing %, skewness, kurtosis, cardinality, class imbalance, leakage indicators, and a **data health score**.
2. **Runs intelligent EDA** — correlation matrix, mutual information, PCA variance, IQR outliers, distribution fitting and transform suggestions (e.g. “Feature ‘income’ is highly right-skewed. Log transform recommended.”).
3. **Detects statistical structure** — high correlations, multicollinearity hints, feature clustering (PCA).
4. **Identifies anomalies** — Isolation Forest, Z-score, DBSCAN; summarizes e.g. “X% of data points exhibit high-leverage anomaly patterns.”
5. **Suggests modeling strategies** — classification (Logistic, RF, XGBoost) or regression (Ridge, RF, LightGBM) or clustering (KMeans); cross-validation, feature importance, SHAP, overfitting detection.
6. **Explains in natural language** — executive summary, business implications, risks, next steps (template or LLM if `OPENAI_API_KEY` is set).
7. **Cognitive flags** — data leakage risk, Simpson’s paradox possibility, multicollinearity, high cardinality, small sample bias, feature dominance, overfitting risk.

---

## Quick start

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app (recommended)

```bash
streamlit run frontend/app.py
```

- Upload a CSV in the sidebar.
- Optionally set a **target column** for supervised modeling.
- Click **Run full analysis**.
- Switch between **Executive (plain English)** and **Technical (full stats)**.
- Use the **insight cards** (expandable) for math and recommendations.

### 3. Optional: Run the FastAPI backend

```bash
cd backend && uvicorn app.main:app --reload
```

- `POST /api/analyze` with form data: `file` (CSV), optional `target_column`.
- Returns full JSON: `profile`, `statistical`, `modeling`, `anomaly`, `cognitive_flags`, `executive_summary`.

---

## Optional: LLM executive summary

For an LLM-generated executive summary instead of the template:

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY=your_key` (and optionally `OPENAI_API_BASE`, `OPENAI_MODEL`).

If the key is not set, the app still runs and uses a **template-based** summary.

---

## Project layout

```
├── backend/
│   └── app/
│       ├── main.py           # FastAPI app
│       ├── api/routes.py     # POST /api/analyze
│       ├── agents/
│       │   ├── profiler.py           # Data profiler + health score
│       │   ├── statistical.py       # Correlation, PCA, MI, outliers, distributions
│       │   ├── modeling.py           # Classification / regression / clustering + SHAP
│       │   ├── anomaly.py            # Isolation Forest, Z-score, DBSCAN
│       │   ├── cognitive_flags.py   # Leakage, Simpson, multicollinearity, etc.
│       │   └── insight_generator.py # Executive summary (template or LLM)
│       ├── core/config.py
│       └── schemas/
├── frontend/
│   └── app.py                # Streamlit UI (Data Story + Technical mode, insight cards)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech stack

- **Backend:** FastAPI, Pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, SHAP.
- **Frontend:** Streamlit, Plotly.
- **Optional:** OpenAI (or compatible) API for natural-language executive summary.

---

## Killer features

- **Cognitive flags** — leakage, Simpson’s paradox, multicollinearity, high cardinality, small sample bias, feature dominance, overfitting.
- **Data Story Mode** — toggle: Technical (full stats) vs Executive (plain English).
- **Interactive insight cards** — each flag expandable with recommendation and math explanation.

You can extend with: Bayesian inference, drift detection, fairness metrics, or LangGraph for multi-agent reasoning.
