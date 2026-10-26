"""Shared feature engineering for the Box-Office Profit Predictor.
Used by train_model.py (training) and app.py (website)."""
import ast, numpy as np, pandas as pd
from collections import defaultdict

TIERS = ["Loss-Making", "Value-Generating", "Breakout"]   # ROI <1x | 1-3x | >3x
TIER_RULES = ["ROI below 1.0×", "ROI between 1.0× and 3.0×", "ROI above 3.0×"]
SUMMER, HOLIDAY = [5, 6, 7], [11, 12]
FRANCHISE_KW = {"sequel", "franchise", "superhero", "based on comic book"}

def tier(roi):
    return 0 if roi < 1 else (1 if roi <= 3 else 2)

def _names(s, key="name", n=None, job=None):
    try:
        items = ast.literal_eval(s)
    except Exception:
        return []
    if job:
        items = [i for i in items if i.get("job") == job]
    out = [i[key] for i in items]
    return out[:n] if n else out

def load_data(movies_csv, credits_csv):
    m = pd.read_csv(movies_csv)
    c = pd.read_csv(credits_csv).rename(columns={"movie_id": "id"})
    df = m.merge(c[["id", "cast", "crew"]], on="id")
    # budget / revenue of 0 are missing values, not real zeros
    df = df[(df.budget > 1000) & (df.revenue > 1000)].copy()
    df["release_date"] = pd.to_datetime(df.release_date)
    df = df.dropna(subset=["release_date", "runtime"]).sort_values("release_date").reset_index(drop=True)
    df["profit"] = df.revenue - df.budget
    df["roi"] = df.revenue / df.budget
    df["log_roi"] = np.log(df.roi)
    df["tier"] = df.roi.apply(tier)
    return df

def add_basic_features(df):
    df["genre_list"] = df.genres.apply(_names)
    df["top_cast"] = df.cast.apply(lambda s: _names(s, n=3))
    df["director"] = df.crew.apply(lambda s: (_names(s, job="Director") or [None])[0])
    kw = df.keywords.apply(lambda s: {k.lower() for k in _names(s)})
    df["is_franchise"] = kw.apply(lambda k: int(bool(k & FRANCHISE_KW)))
    df["month"] = df.release_date.dt.month
    df["year"] = df.release_date.dt.year
    df["log_budget"] = np.log(df.budget)
    df["is_english"] = (df.original_language == "en").astype(int)
    df["is_summer"] = df.month.isin(SUMMER).astype(int)
    df["is_holiday"] = df.month.isin(HOLIDAY).astype(int)
    df["budget_vs_year"] = df.log_budget - df.groupby("year").log_budget.transform("median")
    return df

def add_history_features(df):
    """Star power / director record using ONLY earlier films (prevents data leakage).
    Returns df plus the final histories (used for brand-new movies in the app)."""
    actor_hist, dir_hist = defaultdict(list), defaultdict(list)
    g_rev, g_roi = np.log(df.revenue).mean(), df.log_roi.mean()
    sp, dr, dn = [], [], []
    for r in df.itertuples():                       # df is sorted by release date
        vals = [np.mean(actor_hist[a]) for a in r.top_cast if actor_hist[a]]
        sp.append(np.mean(vals) if vals else g_rev)
        has_dir = isinstance(r.director, str)
        h = dir_hist[r.director] if has_dir else []
        dr.append(np.mean(h) if h else g_roi)
        dn.append(len(h))
        for a in r.top_cast:                        # update AFTER predicting this film
            actor_hist[a].append(np.log(r.revenue))
        if has_dir:
            dir_hist[r.director].append(r.log_roi)
    df["star_power"], df["director_roi"], df["director_films"] = sp, dr, dn
    return df, actor_hist, dir_hist, g_rev, g_roi

def add_genre_flags(df):
    genres = sorted({g for gl in df.genre_list for g in gl})
    for g in genres:
        df["g_" + g.replace(" ", "_")] = df.genre_list.apply(lambda gl: int(g in gl))
    return df, genres

BASE_FEATURES = ["log_budget", "runtime", "month", "year", "is_english", "is_summer", "is_holiday",
                 "is_franchise", "star_power", "director_roi", "director_films", "budget_vs_year"]

def genre_cols(genres):
    return ["g_" + g.replace(" ", "_") for g in genres]

def make_row(inp, art):
    """Turn website inputs into one model-ready row using saved artifacts."""
    yr = min(int(inp["year"]), art["max_year"])
    ah, dh = art["actor_stats"], art["director_stats"]
    vals = [ah[a] for a in inp["actors"] if a in ah]
    star = float(np.mean(vals)) if vals else art["g_rev"]
    d = dh.get(inp["director"])
    row = {
        "log_budget": np.log(inp["budget"]), "runtime": inp["runtime"], "month": inp["month"], "year": yr,
        "is_english": int(inp["english"]), "is_summer": int(inp["month"] in SUMMER),
        "is_holiday": int(inp["month"] in HOLIDAY), "is_franchise": int(inp["franchise"]),
        "star_power": star, "director_roi": d[0] if d else art["g_roi"], "director_films": d[1] if d else 0,
        "budget_vs_year": np.log(inp["budget"]) - art["year_median_logbudget"].get(yr, art["fallback_logbudget"]),
    }
    for g in art["genres"]:
        row["g_" + g.replace(" ", "_")] = int(g in inp["genres"])
    return pd.DataFrame([row])[art["features"]]
