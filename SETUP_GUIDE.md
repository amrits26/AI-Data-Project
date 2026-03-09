# 🚀 Quick Start & Setup Guide

This guide walks you through setting up and running the AI Data Scientist project for development or deployment.

---

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **pip** (comes with Python)
- A **CSV file** to analyze

### Optional (for AI-powered insights)
- **Anthropic API key** ([get one here](https://console.anthropic.com/)) — recommended
- **OpenAI API key** ([get one here](https://platform.openai.com/account/api-keys)) — fallback

---

## Installation (5 minutes)

### 1. Clone/Download the project
```bash
cd path/to/AI_Data_Science_Project
```

### 2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables (optional, for LLM features)
```bash
# Copy the example
cp .env.example .env

# Edit .env and add your API keys:
#   ANTHROPIC_API_KEY=your_key_here
#   OPENAI_API_KEY=your_key_here (fallback)
```

---

## Running the App

### Option A: Streamlit Web UI (Recommended)
```bash
streamlit run frontend/app.py
```
- Opens automatically at `http://localhost:8501`
- Upload CSV, set target column, click **Run full analysis**
- Switch between **Executive** (plain English) and **Technical** (full stats) views

**Features:**
- Interactive data preview with statistics
- Drag-and-drop CSV upload
- Smart target column detection
- Real-time analysis with visualizations
- Exportable reports

### Option B: FastAPI Backend (For integration)
In a separate terminal:
```bash
cd backend
uvicorn app.main:app --reload
```

Then POST to `http://localhost:8000/api/analyze`:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@sample.csv" \
  -F "target_column=target_name"
```

Returns JSON with:
- `profile` — data profiling (dtype, missing %, skewness, health score)
- `statistical` — correlations, PCA, distributions, outliers
- `modeling` — model recommendations, feature importance, SHAP
- `anomaly` — anomaly detection (Isolation Forest, Z-score, DBSCAN)
- `cognitive_flags` — data leakage, multicollinearity, Simpson's paradox
- `executive_summary` — AI-powered business insights

---

## Testing

### Quick verification
```bash
python backend/app/agents/modeling.py
```

### With sample data
1. Download `sample_data/sample.csv`
2. Open Streamlit: `streamlit run frontend/app.py`
3. Upload file → Set target → **Run full analysis**

---

## Project Structure
```
AI Data Science Project/
├── frontend/
│   └── app.py                 # Streamlit UI
├── backend/
│   └── app/
│       ├── agents/            # Analysis engines
│       │   ├── profiler.py
│       │   ├── statistical.py
│       │   ├── modeling.py
│       │   ├── anomaly.py
│       │   ├── cognitive_flags.py
│       │   └── insight_generator.py
│       ├── core/              # Shared utilities
│       │   ├── config.py
│       │   ├── data_health.py
│       │   ├── problem_inference.py
│       │   └── multicollinearity.py
│       ├── api/               # FastAPI routes
│       │   └── routes.py
│       └── main.py            # API entry point
├── sample_data/
│   └── sample.csv             # Example dataset
├── requirements.txt           # Python packages
└── .env.example               # Config template
```

---

## Configuration

### Environment Variables (`.env`)
```bash
# LLM Configuration
ANTHROPIC_API_KEY=              # Preferred LLM
ANTHROPIC_MODEL=claude-haiku-4-5-20251001  # Or claude-sonnet-4-6
OPENAI_API_KEY=                 # Fallback LLM
OPENAI_API_BASE=                # Custom OpenAI endpoint (optional)

# Upload Settings
UPLOAD_DIR=uploads              # Where to save uploaded CSVs
MAX_UPLOAD_MB=50                # Max file size in MB
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named..."
```bash
pip install -r requirements.txt
```

### "Streamlit not found"
```bash
pip install streamlit
```

### "Port 8501 already in use"
```bash
streamlit run frontend/app.py --server.port=8502
```

### "ANTHROPIC_API_KEY not found"
- Optional — app works without it (uses template or OpenAI)
- To enable: Add key to `.env` (copy from `.env.example`)

### API timeouts on large files
- Increase `MAX_UPLOAD_MB` in `.env`
- Streamlit: file is processed in memory (may need 2-4GB RAM for 100MB+ CSV)

---

## Performance Tips

1. **For large CSVs (>50MB):**
   - Use FastAPI backend instead of Streamlit (lower memory)
   - Set `MAX_UPLOAD_MB` appropriately

2. **For faster SHAP analysis:**
   - Modeling automatically samples 200 rows (configurable in `modeling.py`)
   - Skip SHAP if slow: catch exception returns empty dict

3. **For better insights:**
   - Set `ANTHROPIC_API_KEY` (uses Claude, faster & better than GPT)
   - Or set `OPENAI_API_KEY` (uses GPT-4o-mini)

---

## API Reference

### POST /api/analyze
Upload a CSV and get full analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@data.csv" \
  -F "target_column=column_name"
```

**Response (JSON):**
```json
{
  "profile": {
    "shape": [1000, 10],
    "data_health_score": 85,
    "missing_data": {...},
    "numeric_stats": {...}
  },
  "statistical": {
    "correlation_matrix": [...],
    "pca_variance": [...],
    "anomalies": {...}
  },
  "modeling": {
    "problem_type": "classification",
    "model_used": "RandomForestClassifier",
    "cross_val_score": 0.92,
    "feature_importance": {...}
  },
  "cognitive_flags": {
    "data_leakage_risk": false,
    "multicollinearity": "low"
  },
  "executive_summary": "Your data appears clean..."
}
```

---

## Development

### Running Tests
```bash
# Syntax check
python -m py_compile backend/app/agents/*.py

# Import check
python -c "from backend.app.agents import profiler; print('✓ Imports OK')"
```

### Code Quality
- Uses **type hints** for clarity
- Error handling with try/except
- No circular imports (imports are inside functions)

---

## Deployment

### Streamlit Cloud
```bash
git push  # Your repo
# Go to https://share.streamlit.io → Deploy
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku / Cloud Run
```bash
heroku create your-app-name
git push heroku main
```

---

## Support & Bugs

- **Sample data:** `sample_data/sample.csv`
- **Issues:** Check error logs in terminal
- **API docs:** Run backend, visit `http://localhost:8000/docs`

---

## What's Analyzed?

✅ **Data Profiling** — dtypes, nullness, cardinality, distributions  
✅ **Correlation & PCA** — feature relationships, variance explained  
✅ **Statistical Tests** — distribution fitting, skewness, kurtosis  
✅ **Anomaly Detection** — Isolation Forest, Z-score, DBSCAN  
✅ **ML Modeling** — RFC/RFR, cross-validation, feature importance  
✅ **SHAP Values** — per-feature prediction impact  
✅ **Cognitive Flags** — data leakage, multicollinearity, imbalance  
✅ **Executive Summary** — AI-powered business insights  

---

**Ready to analyze? Run:**
```bash
streamlit run frontend/app.py
```

Enjoy! 🚀
