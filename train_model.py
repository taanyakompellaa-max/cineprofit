"""Trains the ensemble models and saves everything the website needs.
Run:  python train_model.py  (expects tmdb_5000_movies.csv + tmdb_5000_credits.csv in ./data or ./_data_tmp)"""
import os, json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, StackingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, roc_auc_score
from xgboost import XGBRegressor
from core import *

folder = next(p for p in ["data", "_data_tmp", "."] if os.path.exists(f"{p}/tmdb_5000_movies.csv"))
df = load_data(f"{folder}/tmdb_5000_movies.csv", f"{folder}/tmdb_5000_credits.csv")
df = add_basic_features(df)
df, actor_hist, dir_hist, g_rev, g_roi = add_history_features(df)
df, genres = add_genre_flags(df)
FEATURES = BASE_FEATURES + genre_cols(genres)

def make_reg():
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42)
    xg = XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
    return StackingRegressor([("rf", rf), ("gb", gb), ("xgb", xg)], final_estimator=RidgeCV())
def make_clf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features=0.5,
                                  class_weight="balanced", random_state=42, n_jobs=-1)

# ---- honest evaluation: train on the oldest 80% of films, test on the newest 20% ----
cut = df.release_date.quantile(0.8)
tr, te = df[df.release_date <= cut], df[df.release_date > cut]
reg = make_reg().fit(tr[FEATURES], tr.log_roi)
p = reg.predict(te[FEATURES]); prof = te.budget * (np.exp(p) - 1)
res = te.log_roi - p
clf = make_clf().fit(tr[FEATURES], tr.tier)
pc = clf.predict_proba(te[FEATURES]); pred_tier = pc.argmax(1)
metrics = {
    "n_films": int(len(df)), "n_train": int(len(tr)), "n_test": int(len(te)),
    "test_from": str(te.release_date.min().date()),
    "profit_r2": float(r2_score(te.profit, prof)),
    "profit_mae_musd": float(mean_absolute_error(te.profit, prof) / 1e6),
    "naive_mae_musd": float(mean_absolute_error(te.profit, np.full(len(te), tr.profit.median())) / 1e6),
    "tier_macro_f1": float(f1_score(te.tier, pred_tier, average="macro")),
    "profitable_auc": float(roc_auc_score((te.roi >= 1).astype(int), 1 - pc[:, 0])),
}
resid_q = [float(np.quantile(res, 0.1)), float(np.quantile(res, 0.9))]
print(json.dumps(metrics, indent=2), "\nresidual 10/90%:", resid_q)

# ---- final models trained on ALL films ----
reg = make_reg().fit(df[FEATURES], df.log_roi)
clf = make_clf().fit(df[FEATURES], df.tier)
actor_stats = {a: float(np.mean(v)) for a, v in actor_hist.items()}
director_stats = {d: (float(np.mean(v)), len(v)) for d, v in dir_hist.items()}
art = dict(reg=reg, clf=clf, xgb=reg.named_estimators_["xgb"], features=FEATURES, genres=genres,
           actor_stats=actor_stats, director_stats=director_stats, g_rev=float(g_rev), g_roi=float(g_roi),
           max_year=int(df.year.max()), year_median_logbudget=df.groupby("year").log_budget.median().to_dict(),
           fallback_logbudget=float(df.log_budget.median()), resid_q=resid_q, metrics=metrics,
           actors=sorted(a for a, v in actor_hist.items() if len(v) >= 1),
           directors=sorted(d for d in dir_hist))
joblib.dump(art, "model.joblib", compress=3)
df[["title", "year", "genre_list", "budget", "revenue", "profit", "roi", "tier"]].assign(
    genres=df.genre_list.apply(lambda g: "|".join(g))).drop(columns="genre_list").to_csv("movies_table.csv", index=False)
print("saved model.joblib", os.path.getsize("model.joblib") // 1024, "KB;", len(art["actors"]), "actors,", len(art["directors"]), "directors")
