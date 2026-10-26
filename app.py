import numpy as np, pandas as pd, joblib, shap, streamlit as st
import matplotlib.pyplot as plt
from core import make_row, TIERS, TIER_RULES

st.set_page_config(page_title="CineProfit | Pre-Release Financial Forecasting for Film", page_icon="🎬", layout="wide")

@st.cache_resource
def load():
    return joblib.load("model.joblib"), pd.read_csv("movies_table.csv").fillna({"genres": ""})

art, movies = load()
explainer = shap.TreeExplainer(art["xgb"])

NICE = {"log_budget": "Budget size", "runtime": "Runtime", "month": "Release month", "year": "Release era",
        "is_english": "English-language", "is_summer": "Summer release", "is_holiday": "Holiday-season release",
        "is_franchise": "Sequel / franchise", "star_power": "Cast star power", "director_roi": "Director's past ROI",
        "director_films": "Director's past film count", "budget_vs_year": "Budget vs. era norm"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.markdown("# 🎬 CineProfit")
st.markdown("**Pre-release financial forecasting for feature films.** A bagging, boosting and stacking ensemble "
            "trained on 3,200+ theatrical releases, with every forecast explained through SHAP attribution.")

with st.sidebar:
    st.header("Production profile")
    title = st.text_input("Working title (optional)", "My Movie")
    budget_m = st.number_input("Budget (USD millions)", 0.5, 500.0, 60.0, 5.0)
    runtime = st.slider("Runtime (minutes)", 60, 210, 118)
    month = st.selectbox("Release month", range(1, 13), index=5, format_func=lambda m: MONTHS[m - 1])
    genres = st.multiselect("Genres", art["genres"], default=["Action", "Adventure"])
    director = st.selectbox("Director", ["(unknown / new director)"] + art["directors"])
    actors = st.multiselect("Top-billed cast (up to 3)", art["actors"], max_selections=3)
    english = st.checkbox("English-language film", True)
    franchise = st.checkbox("Sequel / franchise / comic-book IP", False)
    go = st.button("Generate forecast", type="primary", width="stretch")

if not genres:
    st.info("Select at least one genre in the sidebar to generate a forecast.")
    st.stop()

inp = dict(budget=budget_m * 1e6, runtime=runtime, month=month, year=2016, genres=genres,
           director=None if director.startswith("(") else director, actors=actors,
           english=english, franchise=franchise)
X = make_row(inp, art)

log_roi = float(art["reg"].predict(X)[0])
q10, q90 = art["resid_q"]
b = inp["budget"]
profit = b * (np.exp(log_roi) - 1)
lo, hi = b * (np.exp(log_roi + q10) - 1), b * (np.exp(log_roi + q90) - 1)
probs = art["clf"].predict_proba(X)[0]
p_profit = 1 - probs[0]
verdict = TIERS[int(np.argmax(probs))]

fmt = lambda v: f"{'-' if v < 0 else ''}${abs(v) / 1e6:,.0f}M"
c1, c2, c3 = st.columns(3)
c1.metric("Projected net profit", fmt(profit), help="Projected worldwide revenue minus production budget, produced by the stacked ensemble.")
c2.metric("80% confidence interval", f"{fmt(lo)} to {fmt(hi)}",
          help="Derived from the model's out-of-sample forecast errors on the most recent 20% of releases.")
c3.metric("Probability of positive return", f"{p_profit:.0%}")

st.subheader(f"Projected outcome class for “{title}”: {verdict}")
fig, ax = plt.subplots(figsize=(7, 1.6))
cols = ["#d9534f", "#f0ad4e", "#3a9d5d"]
ax.barh(TIERS, probs, color=cols)
for i, v in enumerate(probs):
    ax.text(v + 0.01, i, f"{v:.0%}", va="center")
ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Outcome-class probabilities  (" + "  |  ".join(f"{t}: {r}" for t, r in zip(TIERS, TIER_RULES)) + ")", fontsize=9, loc="left")
st.pyplot(fig, width="content")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Key drivers of this forecast")
    sv = explainer.shap_values(X)[0]
    imp = pd.Series(sv, index=X.columns)
    imp.index = [NICE.get(i, i.replace("g_", "Genre: ").replace("_", " ")) for i in imp.index]
    top = imp.reindex(imp.abs().sort_values(ascending=False).index)[:8][::-1]
    fig2, ax2 = plt.subplots(figsize=(6, 3.4))
    ax2.barh(top.index, top.values, color=["#3a9d5d" if v > 0 else "#d9534f" for v in top.values])
    ax2.axvline(0, color="grey", lw=.8)
    ax2.set_xlabel("Contribution to predicted ROI   ← lowers  |  raises →")
    for s in ["top", "right"]: ax2.spines[s].set_visible(False)
    st.pyplot(fig2, width="content")
    st.caption("SHAP attribution from the XGBoost (boosting) component of the ensemble. Green bars raise predicted ROI; red bars lower it.")
with right:
    st.subheader("Comparable releases")
    sim = movies.copy()
    sim["overlap"] = sim.genres.apply(lambda g: len(set(g.split("|")) & set(genres)))
    sim = sim[(sim.overlap > 0) & sim.budget.between(b / 2.5, b * 2.5)]
    sim = sim.sort_values(["overlap"], ascending=False).assign(d=lambda d: (np.log(d.budget) - np.log(b)).abs())
    sim = sim.sort_values(["overlap", "d"], ascending=[False, True]).head(8)
    if len(sim):
        st.dataframe(sim.assign(Budget=sim.budget.map(fmt), Revenue=sim.revenue.map(fmt), Profit=sim.profit.map(fmt),
                                ROI=sim.roi.map("{:.1f}×".format))[["title", "year", "Budget", "Revenue", "Profit", "ROI"]]
                     .rename(columns={"title": "Film", "year": "Year"}), hide_index=True, width="stretch")
        st.caption(f"Median net profit of comparable releases (shared genres, similar budget): {fmt(movies.loc[sim.index, 'profit'].median())}")
    else:
        st.write("No comparable releases found for this budget and genre combination.")

with st.expander("Model reliability and limitations (model card)"):
    m = art["metrics"]
    st.markdown(f"""
- **Training data:** {m['n_films']:,} releases with verified budget and revenue (TMDB 5000). Evaluated out-of-sample on the **most recent {m['n_test']} films
  (released after {m['test_from']})**, using a strict chronological split.
- **Mean absolute error (profit):** ${m['profit_mae_musd']:.0f}M, compared with ${m['naive_mae_musd']:.0f}M for a naive median-profit benchmark.
- **Outcome-class accuracy (macro-F1):** {m['tier_macro_f1']:.2f}, above the ~0.33 chance level but well short of precise.
- **Profitability discrimination (ROC-AUC):** {m['profitable_auc']:.2f}.
- **Leakage-free by design:** only pre-release information is used. Ratings, vote counts and popularity are excluded because they are only observable after release.
- **Inherent uncertainty:** marketing spend, reception and competition are unobservable pre-release. Treat output as a **decision-support estimate**, not a guarantee.
- **Scope:** release years beyond 2016 are mapped to the 2016 era. Cast and director effects derive solely from their prior films in the dataset.
""")
