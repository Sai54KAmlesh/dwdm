import sqlite3, pandas as pd, plotly.express as px, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config("Fleet Management Dashboard", layout="wide")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COLORS
FEATS = ["age", "odometer_km", "avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "days_since_last_service"]
CL_FEATS = ["avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "age", "corrective_rate"]
LABELS = ["Low", "Medium", "High"]
TYPES = ["Bus", "Car", "Truck", "Van"]

@st.cache_data
def load():
    con = sqlite3.connect("fleet_dw.db")
    df = pd.read_sql("""SELECT f.*, t.year, t.quarter, t.full_date, v.vehicle_id, v.vehicle_type, v.make, v.purchase_year,
        l.region, l.city, s.service_type FROM fact_service f JOIN dim_time t USING(time_key)
        JOIN dim_vehicle v USING(vehicle_key) JOIN dim_location l USING(location_key)
        JOIN dim_service_type s USING(service_type_key)""", con, parse_dates=["full_date"])
    df["age"] = df.year - df.purchase_year
    df["Year"] = df.year.astype(str)
    df["Quarter"] = df.year.astype(str) + " Q" + df.quarter.astype(str)
    df["Month"] = df.full_date.dt.strftime("%Y-%m")
    parts = pd.read_sql("SELECT b.service_key, p.part_name FROM bridge_service_part b JOIN dim_part p USING(part_key)", con)
    return df, parts

def design(d):
    X = d[FEATS].copy()
    for t in TYPES: X["type_" + t] = (d.vehicle_type == t).astype(int)
    return X

def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2, title=None),
                      title=dict(font=dict(size=15, color="#0b0b0b")), font=dict(color="#52514e"))
    return fig

@st.cache_data
def run_clf(d, model, depth, trees, test):
    X, y = design(d), d.maintenance_need
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test, random_state=1, stratify=y)
    m = (DecisionTreeClassifier(max_depth=depth or None, random_state=1) if model == "Decision Tree"
         else RandomForestClassifier(trees, max_depth=depth or None, random_state=1))
    p = m.fit(Xtr, ytr).predict(Xte)
    return dict(model=m, acc=accuracy_score(yte, p), f1=f1_score(yte, p, average="macro"),
                cm=confusion_matrix(yte, p, labels=LABELS), imp=pd.Series(m.feature_importances_, X.columns).sort_values())

@st.cache_data
def vehicle_table(d):
    return d.groupby("vehicle_id").agg(avg_daily_km=("avg_daily_km", "mean"), idle_hours=("idle_hours", "mean"),
        harsh_events=("harsh_events", "mean"), engine_temp_c=("engine_temp_c", "mean"), age=("age", "mean"),
        corrective_rate=("service_type", lambda s: (s == "Corrective").mean()),
        total_cost_lakh=("cost_inr", lambda s: s.sum() / 1e5)).reset_index()

@st.cache_data
def run_km(v, feats, k):
    Z = StandardScaler().fit_transform(v[list(feats)])
    fit = lambda n: KMeans(n, n_init=10, random_state=1).fit_predict(Z)
    out = v.copy(); out["cl"] = fit(k)
    order = out.groupby("cl").corrective_rate.mean().sort_values().index
    names = ["Low wear", "Moderate wear", "High wear", "Extreme wear"] if k == 4 else [f"Group {i + 1}" for i in range(k)]
    out["Group"] = out.cl.map(dict(zip(order, names)))
    return out, {n: silhouette_score(Z, fit(n)) for n in range(3, 7)}

@st.cache_data
def run_rules(d, parts, sup, conf, lift):
    b = parts[parts.service_key.isin(d.service_key)]
    basket = pd.crosstab(b.service_key, b.part_name).astype(bool)
    fi = apriori(basket, min_support=sup, use_colnames=True)
    try:
        r = association_rules(fi, num_itemsets=len(basket), metric="confidence", min_threshold=conf).query("lift >= @lift")
    except ValueError:
        r = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    r = r.assign(Rule=r.antecedents.map(lambda s: ", ".join(sorted(s))) + "  →  " + r.consequents.map(lambda s: ", ".join(sorted(s))))
    return r[["Rule", "support", "confidence", "lift"]].sort_values("lift", ascending=False)

@st.cache_data
def run_reg(d, model, trees, test):
    X, y = design(d), d.days_to_next_service
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test, random_state=1)
    m = LinearRegression() if model == "Linear Regression" else RandomForestRegressor(trees, random_state=1)
    p = m.fit(Xtr, ytr).predict(Xte)
    return dict(model=m, mae=mean_absolute_error(yte, p), r2=r2_score(yte, p), actual=yte.values, pred=p)

df, parts = load()

sb = st.sidebar
sb.header("Filters")
year = sb.multiselect("Year", sorted(df.Year.unique()), sorted(df.Year.unique()))
region = sb.multiselect("Region", sorted(df.region.unique()), sorted(df.region.unique()))
vtype = sb.multiselect("Vehicle type", TYPES, TYPES)
d = df[df.Year.isin(year) & df.region.isin(region) & df.vehicle_type.isin(vtype)].reset_index(drop=True)
sb.caption(f"{len(d):,} of {len(df):,} records selected")

st.title("Fleet Management Dashboard")
if len(d) < 150 or d.vehicle_id.nunique() < 10:
    st.warning("Too few records for modelling. Widen the filters in the sidebar.")
    st.stop()

tabs = st.tabs(["Overview", "Classification", "Clustering", "Association rules", "Prediction", "Watchlist"])

with tabs[0]:
    k = st.columns(4)
    k[0].metric("Services", f"{len(d):,}")
    k[1].metric("Total cost (INR lakh)", f"{d.cost_inr.sum() / 1e5:,.1f}")
    k[2].metric("Avg downtime (hrs)", f"{d.downtime_hours.mean():.1f}")
    k[3].metric("Repair share", f"{(d.service_type == 'Corrective').mean():.0%}")
    c = st.columns(3)
    meas = c[0].selectbox("Measure", ["Total cost (INR lakh)", "Services", "Avg downtime (hrs)"])
    level = c[1].selectbox("Time level", ["Year", "Quarter", "Month"], index=1)
    dim = c[2].selectbox("Break down by", ["region", "city", "vehicle_type", "make", "service_type"])
    value = lambda g: {"Total cost (INR lakh)": g.cost_inr.sum() / 1e5, "Services": len(g), "Avg downtime (hrs)": g.downtime_hours.mean()}[meas]
    agg = lambda key: d.groupby(key).apply(value, include_groups=False).reset_index(name="value")
    a, b = st.columns(2)
    a.plotly_chart(style(px.line(agg(level), x=level, y="value", markers=True, title=f"{meas} by {level.lower()}", labels={"value": meas})), width="stretch")
    b.plotly_chart(style(px.bar(agg(dim).sort_values("value", ascending=False), x=dim, y="value", title=f"{meas} by {dim.replace('_', ' ')}",
                                labels={"value": meas})), width="stretch")
    pc = parts[parts.service_key.isin(d.service_key)].part_name.value_counts().reset_index().sort_values("count")
    st.plotly_chart(style(px.bar(pc, x="count", y="part_name", orientation="h", title="Parts replaced", labels={"part_name": ""}), 380), width="stretch")

with tabs[1]:
    c = st.columns(4)
    model = c[0].radio("Model", ["Decision Tree", "Random Forest"], index=1, horizontal=True)
    depth = c[1].slider("Max depth (0 = unlimited)", 0, 20, 6 if model == "Decision Tree" else 0, key=f"cd_{model}")
    trees = c[2].slider("Trees", 50, 500, 200, 50, disabled=model == "Decision Tree")
    test = c[3].slider("Test share", 0.1, 0.5, 0.25, 0.05, key="ct")
    clf = run_clf(d, model, depth, trees, test)
    m = st.columns(3)
    m[0].metric("Accuracy", f"{clf['acc']:.1%}")
    m[1].metric("Macro F1", f"{clf['f1']:.3f}")
    m[2].metric("High-need recall", f"{clf['cm'][2, 2] / max(clf['cm'][2].sum(), 1):.0%}")
    a, b = st.columns(2)
    a.plotly_chart(style(px.imshow(clf["cm"], x=LABELS, y=LABELS, text_auto=True, color_continuous_scale=SEQ,
                                   labels=dict(x="Predicted", y="Actual", color="Records"), title="Confusion matrix")), width="stretch")
    b.plotly_chart(style(px.bar(clf["imp"], orientation="h", title="Which inputs matter most", labels={"value": "Importance", "index": ""})
                         .update_layout(showlegend=False)), width="stretch")

with tabs[2]:
    v = vehicle_table(d)
    c = st.columns([1, 2])
    kk = c[0].slider("Number of clusters (k)", 3, 6, 4)
    cf = tuple(c[1].multiselect("Features used to cluster", CL_FEATS, CL_FEATS))
    if len(cf) < 2:
        st.error("Pick at least 2 features.")
    else:
        cv, sil = run_km(v, cf, kk)
        order = cv.groupby("Group").corrective_rate.mean().sort_values().index.tolist()
        st.metric(f"Silhouette score at k = {kk}", f"{sil[kk]:.3f}")
        a, b = st.columns(2)
        a.plotly_chart(style(px.scatter(cv, x="harsh_events", y="engine_temp_c", color="Group", hover_name="vehicle_id",
                                        category_orders={"Group": order}, title="Vehicle clusters")), width="stretch")
        s = pd.DataFrame({"k": list(sil), "silhouette": list(sil.values())})
        b.plotly_chart(style(px.line(s, x="k", y="silhouette", markers=True, title="Silhouette score by k (higher is better)")
                             .update_xaxes(dtick=1).add_scatter(x=[kk], y=[sil[kk]], mode="markers", marker=dict(size=14, color="#eb6834"), name="chosen k")), width="stretch")
        prof = cv.groupby("Group").agg(Vehicles=("vehicle_id", "size"), Daily_km=("avg_daily_km", "mean"), Harsh_events=("harsh_events", "mean"),
                                       Temp_C=("engine_temp_c", "mean"), Repair_share=("corrective_rate", "mean"), Cost_lakh_per_vehicle=("total_cost_lakh", "mean")).loc[order]
        st.dataframe(prof.round(2), width="stretch")

with tabs[3]:
    c = st.columns(3)
    sup = c[0].slider("Min support", 0.01, 0.20, 0.04, 0.005)
    conf = c[1].slider("Min confidence", 0.1, 1.0, 0.6, 0.05)
    lift = c[2].slider("Min lift", 1.0, 12.0, 1.5, 0.5)
    rules = run_rules(d, parts, sup, conf, lift)
    st.metric("Rules found", len(rules))
    if rules.empty:
        st.info("No rules pass these thresholds. Lower the minimum support, confidence or lift.")
    else:
        st.dataframe(rules.head(20).rename(columns=str.title), hide_index=True, width="stretch", height=420,
                     column_config={"Lift": st.column_config.NumberColumn(format="%.2f"), "Support": st.column_config.NumberColumn(format="%.3f"),
                                    "Confidence": st.column_config.NumberColumn(format="%.2f")})

with tabs[4]:
    c = st.columns(3)
    rm = c[0].radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)
    rtrees = c[1].slider("Trees", 50, 500, 200, 50, key="rt", disabled=rm == "Linear Regression")
    rtest = c[2].slider("Test share", 0.1, 0.5, 0.25, 0.05, key="rts")
    reg = run_reg(d, rm, rtrees, rtest)
    m = st.columns(2)
    m[0].metric("Average error (days)", f"{reg['mae']:.2f}")
    m[1].metric("R²", f"{reg['r2']:.3f}")
    a, b = st.columns([3, 2])
    ap = pd.DataFrame({"Actual": reg["actual"], "Predicted": reg["pred"]})
    a.plotly_chart(style(px.scatter(ap, x="Actual", y="Predicted", opacity=0.5, title="Predicted vs actual days to next service")
                         .add_shape(type="line", x0=ap.Actual.min(), y0=ap.Actual.min(), x1=ap.Actual.max(), y1=ap.Actual.max(),
                                    line=dict(color="#e34948", width=2))), width="stretch")
    with b.container(border=True):
        st.markdown("**Try a vehicle**")
        row = d[FEATS].median().to_dict()
        row["age"] = st.slider("Vehicle age (years)", int(d.age.min()), int(d.age.max()), int(d.age.median()))
        row["harsh_events"] = st.slider("Harsh events", int(d.harsh_events.min()), int(d.harsh_events.max()), int(d.harsh_events.median()))
        row["engine_temp_c"] = st.slider("Engine temp (C)", float(d.engine_temp_c.min()), float(d.engine_temp_c.max()), float(d.engine_temp_c.median()))
        vt = st.selectbox("Vehicle type", TYPES, index=2)
        X1 = pd.DataFrame([row])
        for t in TYPES: X1["type_" + t] = int(vt == t)
        st.metric("Predicted days to next service", f"{float(reg['model'].predict(X1)[0]):.0f}")

with tabs[5]:
    st.caption("Uses the models set in the Classification and Prediction tabs on each vehicle's latest service record.")
    last = d.sort_values("full_date").groupby("vehicle_id").tail(1).reset_index(drop=True)
    last["Predicted need"] = clf["model"].predict(design(last))
    last["Days to next service"] = pd.Series(reg["model"].predict(design(last))).round().clip(lower=0)
    last["Next service due"] = (last.full_date + pd.to_timedelta(last["Days to next service"], unit="D")).dt.date
    c = st.columns(2)
    within = c[0].slider("Due within (days of the latest record)", 10, 120, 60)
    need = c[1].multiselect("Predicted need", LABELS, LABELS)
    asof = d.full_date.max().date()
    w = last[last["Predicted need"].isin(need)].sort_values("Days to next service")
    w = w[(w["Next service due"] - asof).map(lambda x: x.days) <= within]
    st.metric("Vehicles shown", len(w))
    st.dataframe(w[["vehicle_id", "vehicle_type", "city", "full_date", "Predicted need", "Days to next service", "Next service due"]]
                 .rename(columns={"full_date": "Last service"}), hide_index=True, width="stretch", height=420,
                 column_config={"Last service": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                "Days to next service": st.column_config.ProgressColumn(min_value=0, max_value=120, format="%d")})
