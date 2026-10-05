# NEXUS AI

Autonomous Data Analysis & Predictive Intelligence Platform.

## Architecture

```
User Question → LLM (intent) → Structured Request → Validation →
Python/Polars/DuckDB/ML Engine (actual calculation) →
Structured Result → LLM (explanation) → User
```

The LLM understands intent and explains results — it **never** performs
factual numerical calculations itself. Code is the source of truth.

## UI Design & Aesthetics

- **Executive Dark Theme**: High-contrast Black (`#0A0B0E`), Crimson Red (`#FF334B`), and Crisp White (`#FFFFFF`).
- **Dedicated Chatting Box**: Built as a dedicated conversational card featuring live status indicators, quick prompt pills (`📊 Overview`, `🔮 Best Model`), reset controls, and verified calculation badges.
- **Optimized Layout**: Balanced split screen with a 65% analytics workbench and a 35% interactive chatting box.
- **Data sources**: CSV, Excel (sheet picker), multiple files (stack / join / load-one — never a silent bad merge), PostgreSQL or SQLite (read-only SELECT), Google Sheets (service account).
- **Visualization Studio**: Column/bar aggregates, line trends, box & violin plots, category counts, donut composition, correlation heatmaps, and a visible chart gallery.
- **Data Quality Tab**: Missing-value charts, duplicate counts, and per-column type/null inventory.

## Autonomous ML Tournament Engine

- **Automated Problem Detection**: Intelligently identifies continuous regression, binary classification, and multiclass targets with optional manual override.
- **Multi-Model Tournament**: Evaluates candidate models across:
  - *Regression*: RandomForest, HistGradientBoosting, XGBoost, ExtraTrees, Ridge, Baseline
  - *Classification*: RandomForest, HistGradientBoosting, XGBoost, ExtraTrees, LogisticRegression, Baseline
- **Rigorous Evaluation**: Combines K-Fold Cross-Validation (mean ± std) and unseen holdout test metrics to crown the true champion model.
- **Diagnostics**: Automated Feature Importance ranking bar charts, Actual vs. Predicted scatter plots, Confusion Matrices, and an interactive **What-If Scenario Simulator**.

## LLM Backend — Ollama Cloud

NEXUS uses **Ollama Cloud** for LLM inference. No local GPU or local model
download is required.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_URL` | `https://ollama.com` | Ollama Cloud base URL |
| `OLLAMA_MODEL` | `gemma4:31b` | Model to use (change without code edits) |
| `OLLAMA_API_KEY` | *(required)* | Your API key from [ollama.com/settings/keys](https://ollama.com/settings/keys) |

The model can be swapped by changing the `OLLAMA_MODEL` environment variable.
No code changes are needed — the LLM service layer is fully modular.

## Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in:
   - `OLLAMA_API_KEY` — get one from [ollama.com/settings/keys](https://ollama.com/settings/keys)
   - Optionally change `OLLAMA_MODEL` to any model available on Ollama Cloud
5. (Optional) SQL: set `DB_DIALECT` (`postgresql` or `sqlite`) and connection fields. Queries are read-only.
6. (Optional) Google Sheets: set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON that has access to the spreadsheet.

## Run

```bash
python run.py
```
Or directly with Streamlit:
```bash
python -m streamlit run app/main.py
```

## Deployment

### Option 1: Streamlit Community Cloud (Recommended & Free)
1. Push your repository to GitHub (ensure `.env` is ignored by `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set **Main file path** to `app/main.py`.
4. In **Advanced Settings > Secrets**, paste the contents of your `.env` (such as `OLLAMA_API_KEY`).
5. Click **Deploy**.

### Option 2: Docker Container (Render, Railway, AWS ECS, GCP Cloud Run)
```bash
# Build the container
docker build -t nexus-ai .

# Run locally or in production
docker run -p 8501:8501 --env-file .env nexus-ai
```

### Option 3: Render or Railway Web Service
- Use the included `Procfile` and `Dockerfile`.
- Set environment variables (`OLLAMA_API_KEY`, etc.) in the dashboard settings.

---

## 🔒 GitHub Security & Privacy Checklist

Before publishing to GitHub:
- [x] **`.env` is ignored**: Verified in `.gitignore`. Secrets, tokens, and database passwords will never be published.
- [x] **`.env.example` provided**: Only placeholder names exist in `.env.example`.
- [x] **Analytical storage protected**: Generated Parquet tables, SQLite databases, and temporary run files are excluded in `.gitignore`.
- [x] **Sample data preserved**: `data/test_sales.csv` remains tracked so anyone cloning can explore instantly with the bundled sample.

---

## Tech Stack

Python 3.11 · Polars · DuckDB · Scikit-Learn · XGBoost · Parquet · Plotly · Ollama Cloud · Streamlit · ReportLab
