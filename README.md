# CineProfit: Pre-Release Financial Forecasting for Film

Forecasts a film's net profit and ROI-based outcome class (**Loss-Making** < 1.0× · **Value-Generating** 1.0–3.0× ·
**Breakout** > 3.0×) using only information available **before release**. The model is a heterogeneous ensemble
(Random Forest for bagging, Gradient Boosting and XGBoost for boosting, combined by a Ridge meta-learner), and each
forecast is explained with SHAP attribution.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retraining (optional)
Put `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` (Kaggle: "TMDB 5000 Movie Dataset") in a `data/` folder, then:
```bash
python train_model.py      # rebuilds model.joblib and movies_table.csv
```

## Deployment (free public link)
1. Create a GitHub repo and upload everything in this folder (`app.py`, `core.py`, `train_model.py`,
   `model.joblib`, `movies_table.csv`, `requirements.txt`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, choose the repo, main file `app.py`, **Deploy**.

## Files
| File | Purpose |
|---|---|
| `app.py` | The website |
| `core.py` | Leakage-free feature engineering shared by training and the app |
| `train_model.py` | Chronological evaluation and final model training |
| `model.joblib` | Trained models and lookup tables |
| `movies_table.csv` | Historical releases used for the "Comparable releases" table |
