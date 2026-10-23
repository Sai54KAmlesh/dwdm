import os, subprocess, sys

# ── Auto-setup for Streamlit Cloud (runs once if DB/CSV are missing) ──────────
def _bootstrap():
    if not os.path.exists("fleet_maintenance.csv"):
        subprocess.run([sys.executable, "generate_data.py"], check=True)
    if not os.path.exists("fleet_dw.db"):
        subprocess.run([sys.executable, "build_dw.py"], check=True)

_bootstrap()
# ─────────────────────────────────────────────────────────────────────────────

import re, sqlite3, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config("Fleet Intelligence", page_icon="🚚", layout="wide", initial_sidebar_state="auto")

# ───────────────────────── design tokens ─────────────────────────
ORANGE, AMBER, TEAL, VIOLET, PINK, SKY, LIME, RED = "#ff7a3d", "#ffb547", "#2dd4bf", "#8b7bff", "#ff6b9d", "#4cc9f0", "#b6f36a", "#ff4d5e"
COLORS = [ORANGE, TEAL, VIOLET, AMBER, PINK, SKY, LIME, RED]
WARM = [[0, "rgba(255,122,61,0.28)"], [0.5, "rgba(255,122,61,0.65)"], [1, AMBER]]
COOL = [[0, "rgba(139,123,255,0.55)"], [0.5, "rgba(76,201,240,0.7)"], [1, TEAL]]
TXT, MUTED, GRID = "#f4efe9", "#a49c93", "rgba(255,255,255,0.07)"
FEATS = ["age", "odometer_km", "avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "days_since_last_service"]
CL_FEATS = ["avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "age", "corrective_rate"]
LABELS = ["Low", "Medium", "High"]
TYPES = ["Bus", "Car", "Truck", "Van"]
NEED_COL = {"Low": TEAL, "Medium": AMBER, "High": ORANGE}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--glass:rgba(255,255,255,.055);--glass-2:rgba(255,255,255,.09);--stroke:rgba(255,255,255,.12);--txt:#f4efe9;--muted:#a49c93;--orange:#ff7a3d;--amber:#ffb547;--teal:#2dd4bf;--violet:#8b7bff;--r:24px}
html,body,.stApp,[class*="css"]{font-family:'Inter',system-ui,sans-serif!important;color:var(--txt)}
.stApp{background:#09090b;background-image:
 radial-gradient(60vw 50vh at 8% -5%,rgba(255,122,61,.30),transparent 60%),
 radial-gradient(50vw 45vh at 100% 5%,rgba(139,123,255,.22),transparent 60%),
 radial-gradient(55vw 55vh at 85% 105%,rgba(45,212,191,.18),transparent 60%),
 radial-gradient(45vw 40vh at 0% 100%,rgba(255,181,71,.14),transparent 60%);background-attachment:fixed}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{padding:1.4rem clamp(.8rem,3vw,2.6rem) 4rem;max-width:1500px}
h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif!important;letter-spacing:-.02em;color:var(--txt)}
p,label,span,.stMarkdown{color:var(--txt)}
[data-testid="stCaptionContainer"],small{color:var(--muted)!important}

/* glass surfaces */
.glass,[class*="st-key-glass"],[data-testid="stExpander"],[data-testid="stDataFrame"]{background:linear-gradient(145deg,rgba(255,255,255,.085),rgba(255,255,255,.03));border:1px solid var(--stroke);border-radius:var(--r);backdrop-filter:blur(22px) saturate(140%);-webkit-backdrop-filter:blur(22px) saturate(140%);box-shadow:0 10px 40px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.10)}
[class*="st-key-glass"]{padding:1.1rem 1.2rem .6rem;transition:transform .25s ease,border-color .25s ease}
[class*="st-key-glass"]:hover{border-color:rgba(255,181,71,.35)}
[data-testid="stDataFrame"]{padding:.4rem;overflow:hidden}

/* hero */
.hero{display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-end;justify-content:space-between;padding:1.5rem 1.7rem;margin-bottom:1.1rem;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;inset:auto -10% -60% 40%;height:120%;background:radial-gradient(closest-side,rgba(255,122,61,.35),transparent);filter:blur(30px);pointer-events:none}
.brand{display:flex;align-items:center;gap:.8rem;margin-bottom:.9rem}
.logo{width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,var(--orange),var(--amber));display:grid;place-items:center;font-size:20px;box-shadow:0 6px 24px rgba(255,122,61,.5)}
.brand b{font-family:'Space Grotesk','Inter',system-ui,sans-serif;font-size:1rem}.brand span{display:block;font-size:.72rem;color:var(--muted)}
.hero h1{font-size:clamp(1.7rem,4.2vw,3rem);line-height:1.05;margin:0;font-weight:600}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--orange),var(--amber),var(--teal));-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:var(--muted);margin:.6rem 0 0;font-size:clamp(.85rem,1.6vw,1rem)}
.chips{display:flex;flex-wrap:wrap;gap:.45rem;position:relative}
.chip{padding:.38rem .8rem;border-radius:999px;font-size:.76rem;background:var(--glass-2);border:1px solid var(--stroke);backdrop-filter:blur(10px);color:var(--txt)}
.chip i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--teal);margin-right:.45rem;box-shadow:0 0 10px var(--teal)}

/* KPI cards */
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin-bottom:1.1rem}
.kpi{padding:1.1rem 1.2rem;border-radius:var(--r);position:relative;overflow:hidden;min-height:128px}
.kpi .ic{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;font-size:16px;background:var(--c);box-shadow:0 6px 20px -4px var(--c)}
.kpi .lb{font-size:.76rem;color:var(--muted);margin-top:.7rem;letter-spacing:.02em}
.kpi .v{font-family:'Space Grotesk','Inter',system-ui,sans-serif;font-size:clamp(1.5rem,2.6vw,2.1rem);font-weight:600;line-height:1.1}
.kpi .v small{font-size:.8rem;color:var(--muted)!important;font-weight:400;margin-left:.25rem}
.kpi svg{position:absolute;right:0;bottom:0;width:42%;height:40px;opacity:.85;pointer-events:none}.kpi .v,.kpi .lb{position:relative;z-index:1}
.kpi .tg{position:absolute;top:1.1rem;right:1.2rem;font-size:.7rem;padding:.2rem .55rem;border-radius:999px;background:var(--glass-2);border:1px solid var(--stroke);color:var(--muted)}

/* Streamlit metric (other tabs) */
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(255,255,255,.09),rgba(255,255,255,.03));border:1px solid var(--stroke);border-radius:20px;padding:.9rem 1.1rem;backdrop-filter:blur(18px);box-shadow:inset 0 1px 0 rgba(255,255,255,.1)}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:.78rem}
[data-testid="stMetricValue"]{font-family:'Space Grotesk','Inter',system-ui,sans-serif;font-size:1.9rem;background:linear-gradient(90deg,#fff,var(--amber));-webkit-background-clip:text;background-clip:text;color:transparent}

/* tabs → pill nav */
[data-baseweb="tab-list"],[role="tablist"]{gap:.4rem;background:var(--glass);border:1px solid var(--stroke);padding:.35rem;border-radius:999px;backdrop-filter:blur(18px);overflow-x:auto;scrollbar-width:none;margin-bottom:1rem}
[data-baseweb="tab-list"]::-webkit,[role="tablist"]::-webkit-scrollbar{display:none}
button[data-baseweb="tab"],[role="tab"]{border-radius:999px!important;padding:.5rem 1.05rem!important;height:auto!important;background:transparent!important;white-space:nowrap}
button[data-baseweb="tab"] p,[role="tab"] p{color:var(--muted)!important;font-size:.88rem;font-weight:500}
button[data-baseweb="tab"][aria-selected="true"],[role="tab"][aria-selected="true"]{background:linear-gradient(135deg,var(--orange),var(--amber))!important;box-shadow:0 6px 22px rgba(255,122,61,.45)}
button[data-baseweb="tab"][aria-selected="true"] p,[role="tab"][aria-selected="true"] p{color:#1a0f08!important;font-weight:600}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"],[role="tablist"] [class*="ighlight"],[data-testid="stTabsHighlight"],[role="tablist"]>div:not([role="tab"]){display:none!important}

/* inputs */
[data-baseweb="select"]>div,[data-baseweb="input"],.stNumberInput input{background:var(--glass)!important;border:1px solid var(--stroke)!important;border-radius:14px!important;color:var(--txt)!important}
[data-baseweb="tag"]{background:rgba(255,122,61,.22)!important;border-radius:10px!important}
[data-testid="stSlider"] [role="slider"]{background:var(--amber)!important;box-shadow:0 0 0 5px rgba(255,181,71,.2)}
[data-testid="stSlider"] [data-testid="stTickBar"]{display:none}
.stRadio [role="radiogroup"]{gap:.4rem}
.stRadio label{background:var(--glass);border:1px solid var(--stroke);padding:.3rem .8rem;border-radius:999px}
[data-testid="stSidebar"]{background:rgba(14,14,18,.6)!important;backdrop-filter:blur(26px);border-right:1px solid var(--stroke)}
[data-testid="stAlert"]{border-radius:18px;background:var(--glass);border:1px solid var(--stroke)}

.sec{display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .2rem}
.sec h3{font-size:1.02rem;margin:0;font-weight:600}.sec span{font-size:.76rem;color:var(--muted)}
.sec:before{content:"";width:6px;height:18px;border-radius:4px;background:linear-gradient(var(--orange),var(--amber))}

/* ───── responsive ───── */
[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:1rem}
[data-testid="stColumn"]{min-width:min(100%,280px)}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){
 .kpis{gap:.7rem}.kpi{min-height:112px;padding:.9rem}.kpi svg{height:34px}
 .hero{padding:1.1rem}.block-container{padding-top:.8rem}
 [data-testid="stColumn"]{min-width:100%!important}
 [class*="st-key-glass"]{padding:.8rem .7rem .3rem;border-radius:20px}
 [data-testid="stMetricValue"]{font-size:1.5rem}
}
@media (max-width:380px){.kpis{grid-template-columns:1fr}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────── plotly theme ─────────────────────────
def style(fig, h=340, legend=True):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=10, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=MUTED, size=12), colorway=COLORS, showlegend=legend,
                      legend=dict(orientation="h", y=-0.18, title=None, bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)),
                      hoverlabel=dict(bgcolor="rgba(20,20,24,.92)", bordercolor="rgba(255,255,255,.15)", font=dict(color=TXT, family="Inter")),
                      xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", title=None),
                      yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", title=None))
    return fig

def rounded(fig, r=8):
    try:
        fig.update_traces(marker_cornerradius=r, selector=dict(type="bar"))
    except Exception:
        pass
    return fig

def plot(fig, h=340, legend=True):
    st.plotly_chart(style(fig, h, legend), width="stretch", config=dict(displayModeBar=False))

def card(title, sub=""):
    slug = re.sub(r"[^a-z0-9]+", "_", (title + "_" + sub).lower()).strip("_")[:60]
    c = st.container(key=f"glass_{slug}")
    c.markdown(f'<div class="sec"><h3>{title}</h3><span>{sub}</span></div>', unsafe_allow_html=True)
    return c

def rgba(hex_, a):
    h = hex_.lstrip("#"); return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

def spark(vals, color):
    v = np.asarray(vals, float)
    if len(v) < 2 or np.ptp(v) == 0: return ""
    x = np.linspace(0, 100, len(v)); y = 34 - (v - v.min()) / np.ptp(v) * 28
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    gid = color.strip("#")
    return (f'<svg viewBox="0 0 100 40" preserveAspectRatio="none"><defs><linearGradient id="g{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".45"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,40 {pts} 100,40" fill="url(#g{gid})"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>')

def kpi_row(items):
    html = '<div class="kpis">' + "".join(
        f'<div class="kpi glass" style="--c:{c}"><div class="ic">{ic}</div><div class="tg">{tag}</div><div class="lb">{lb}</div>'
        f'<div class="v">{val}<small>{unit}</small></div>{spark(sp, c)}</div>' for lb, val, unit, ic, c, tag, sp in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def gauge(value, title, color=ORANGE, suffix="%", rng=(0, 100), h=260):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number=dict(suffix=suffix, font=dict(size=44, color=TXT, family="Space Grotesk, Inter, sans-serif")),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(shape="angular", axis=dict(range=list(rng), tickcolor="rgba(255,255,255,.25)", tickfont=dict(size=10, color=MUTED), nticks=6),
                   bar=dict(color=color, thickness=.28), bgcolor="rgba(255,255,255,.05)", borderwidth=0,
                   steps=[dict(range=[rng[0], rng[0] + (rng[1] - rng[0]) * .5], color="rgba(255,255,255,.03)"),
                          dict(range=[rng[0] + (rng[1] - rng[0]) * .5, rng[1]], color="rgba(255,255,255,.07)")])))
    return fig

# ───────────────────────── data + models (logic unchanged) ─────────────────────────
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

# ───────────────────────── sidebar ─────────────────────────
sb = st.sidebar
sb.markdown('<div class="brand"><div class="logo">🚚</div><div><b>Fleet Intelligence</b><span>Filters</span></div></div>', unsafe_allow_html=True)
year = sb.multiselect("Year", sorted(df.Year.unique()), sorted(df.Year.unique()))
region = sb.multiselect("Region", sorted(df.region.unique()), sorted(df.region.unique()))
vtype = sb.multiselect("Vehicle type", TYPES, TYPES)
d = df[df.Year.isin(year) & df.region.isin(region) & df.vehicle_type.isin(vtype)].reset_index(drop=True)
sb.caption(f"{len(d):,} of {len(df):,} records selected")

# ───────────────────────── hero ─────────────────────────
span = f"{df.full_date.min():%b %Y} – {df.full_date.max():%b %Y}"
st.markdown(f"""<div class="hero glass"><div style="position:relative"><div class="brand"><div class="logo">🚚</div>
<div><b>Fleet Management</b><span>Data warehouse &amp; mining console</span></div></div>
<h1>Keep every vehicle <em>ahead of failure.</em></h1>
<p>Explore cost, wear patterns and predicted service needs across {df.vehicle_id.nunique()} vehicles.</p></div>
<div class="chips"><span class="chip"><i></i>Live recompute</span><span class="chip">📅 {span}</span><span class="chip">{len(d):,} records</span></div></div>""", unsafe_allow_html=True)

if len(d) < 150 or d.vehicle_id.nunique() < 10:
    st.warning("Too few records for modelling. Widen the filters in the sidebar.")
    st.stop()

tabs = st.tabs(["◉ Overview", "◈ Classification", "◎ Clustering", "⛓ Association rules", "↗ Prediction", "⚑ Watchlist"])

# ───────────────────────── Overview ─────────────────────────
with tabs[0]:
    mon = d.groupby("Month").agg(cost=("cost_inr", "sum"), n=("cost_inr", "size"), dt=("downtime_hours", "mean"),
                                 rep=("service_type", lambda s: (s == "Corrective").mean()))
    kpi_row([("Services", f"{len(d):,}", "", "🛠", ORANGE, "all time", mon.n),
             ("Total cost", f"{d.cost_inr.sum() / 1e5:,.1f}", "INR lakh", "₹", AMBER, "monthly trend", mon.cost),
             ("Avg downtime", f"{d.downtime_hours.mean():.1f}", "hrs", "⏱", TEAL, "per service", mon.dt),
             ("Repair share", f"{(d.service_type == 'Corrective').mean():.0%}", "", "⚠", VIOLET, "corrective", mon.rep)])

    with st.container(key="glass_ctrl"):
        c = st.columns(3)
        meas = c[0].selectbox("Measure", ["Total cost (INR lakh)", "Services", "Avg downtime (hrs)"])
        level = c[1].selectbox("Time level", ["Year", "Quarter", "Month"], index=1)
        dim = c[2].selectbox("Break down by", ["region", "city", "vehicle_type", "make", "service_type"])
    value = lambda g: {"Total cost (INR lakh)": g.cost_inr.sum() / 1e5, "Services": len(g), "Avg downtime (hrs)": g.downtime_hours.mean()}[meas]
    agg = lambda key: d.groupby(key).apply(value, include_groups=False).reset_index(name="value")

    a, b = st.columns([3, 2])
    with a:
        with card(f"{meas}", f"by {level.lower()}"):
            t = agg(level)
            fig = go.Figure()
            fig.add_scatter(x=t[level], y=t.value, mode="lines+markers", line=dict(color=ORANGE, width=3, shape="spline", smoothing=.8),
                            marker=dict(size=8, color="#fff", line=dict(color=ORANGE, width=2)), fill="tozeroy",
                            fillgradient=dict(type="vertical", colorscale=[[0, "rgba(255,122,61,0)"], [1, "rgba(255,122,61,.45)"]]),
                            hovertemplate="%{x}<br><b>%{y:,.1f}</b><extra></extra>")
            fig.update_yaxes(rangemode="tozero")
            plot(fig, 340, False)
    with b:
        with card(f"{meas}", f"by {dim.replace('_', ' ')}"):
            t = agg(dim).sort_values("value", ascending=False)
            fig = go.Figure(go.Bar(x=t[dim], y=t.value, marker=dict(color=t.value, colorscale=WARM, line=dict(width=0)),
                                   hovertemplate="%{x}<br><b>%{y:,.1f}</b><extra></extra>"))
            rounded(fig, 10); plot(fig, 340, False)

    a, b, c3 = st.columns([2, 2, 3])
    with a:
        with card("Service mix", "preventive vs corrective"):
            sm = d.service_type.value_counts().reset_index()
            fig = go.Figure(go.Pie(labels=sm.service_type, values=sm["count"], hole=.7, sort=False, textinfo="percent",
                                   marker=dict(colors=[TEAL, ORANGE, VIOLET, AMBER], line=dict(color="rgba(10,10,12,.9)", width=3)), textfont=dict(color="#111")))
            fig.add_annotation(text=f"<b>{len(d):,}</b><br><span style='font-size:11px;color:{MUTED}'>services</span>", showarrow=False, font=dict(size=22, color=TXT))
            plot(fig, 300)
    with b:
        with card("Fleet health index", "share of low / medium need"):
            hi = ((d.maintenance_need == "Low").mean() + .5 * (d.maintenance_need == "Medium").mean()) * 100
            fig = gauge(hi, "Balanced condition score", TEAL if hi >= 60 else AMBER if hi >= 40 else ORANGE)
            plot(fig, 300, False)
    with c3:
        with card("Cost heatmap", "region × quarter · INR lakh"):
            hm = d.pivot_table(index="region", columns="Quarter", values="cost_inr", aggfunc="sum").fillna(0) / 1e5
            fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=WARM, xgap=4, ygap=4, showscale=False,
                                       hovertemplate="%{y} · %{x}<br><b>%{z:,.1f}</b> lakh<extra></extra>"))
            plot(fig, 300, False)

    with card("Parts replaced", "most frequent components"):
        pc = parts[parts.service_key.isin(d.service_key)].part_name.value_counts().reset_index().sort_values("count")
        fig = go.Figure(go.Bar(y=pc.part_name, x=pc["count"], orientation="h", marker=dict(color=pc["count"], colorscale=COOL),
                               hovertemplate="%{y}<br><b>%{x:,}</b><extra></extra>"))
        rounded(fig, 8); plot(fig, max(300, 26 * len(pc)), False)

# ───────────────────────── Classification ─────────────────────────
with tabs[1]:
    with st.container(key="glass_clfctrl"):
        c = st.columns(4)
        model = c[0].radio("Model", ["Decision Tree", "Random Forest"], index=1, horizontal=True)
        depth = c[1].slider("Max depth (0 = unlimited)", 0, 20, 6 if model == "Decision Tree" else 0, key=f"cd_{model}")
        trees = c[2].slider("Trees", 50, 500, 200, 50, disabled=model == "Decision Tree")
        test = c[3].slider("Test share", 0.1, 0.5, 0.25, 0.05, key="ct")
    clf = run_clf(d, model, depth, trees, test)
    hr = clf["cm"][2, 2] / max(clf["cm"][2].sum(), 1)
    g = st.columns(3)
    for col, (v, t, clr) in zip(g, [(clf["acc"] * 100, "Accuracy", ORANGE), (clf["f1"] * 100, "Macro F1 (×100)", VIOLET), (hr * 100, "High-need recall", TEAL)]):
        with col:
            with card(t):
                plot(gauge(v, "", clr, "%" if "F1" not in t else ""), 220, False)
    a, b = st.columns(2)
    with a:
        with card("Confusion matrix", "actual vs predicted"):
            fig = go.Figure(go.Heatmap(z=clf["cm"], x=LABELS, y=LABELS, colorscale=WARM, xgap=5, ygap=5, showscale=False,
                                       text=clf["cm"], texttemplate="<b>%{text}</b>", textfont=dict(size=18, color=TXT),
                                       hovertemplate="Actual %{y} → Predicted %{x}<br><b>%{z}</b><extra></extra>"))
            fig.update_yaxes(autorange="reversed", title="Actual"); fig.update_xaxes(title="Predicted")
            plot(fig, 360, False)
    with b:
        with card("Which inputs matter most", "feature importance"):
            imp = clf["imp"]
            fig = go.Figure()
            fig.add_bar(y=imp.index, x=imp.values, orientation="h", width=.07, marker_color=rgba(AMBER, .5), hoverinfo="skip")
            fig.add_scatter(y=imp.index, x=imp.values, mode="markers", marker=dict(size=15, color=imp.values, colorscale=[[0, VIOLET], [.5, ORANGE], [1, AMBER]],
                            line=dict(color="#fff", width=2)), hovertemplate="%{y}<br><b>%{x:.3f}</b><extra></extra>")
            plot(fig, 360, False)

# ───────────────────────── Clustering ─────────────────────────
with tabs[2]:
    v = vehicle_table(d)
    with st.container(key="glass_kctrl"):
        c = st.columns([1, 2])
        kk = c[0].slider("Number of clusters (k)", 3, 6, 4)
        cf = tuple(c[1].multiselect("Features used to cluster", CL_FEATS, CL_FEATS))
    if len(cf) < 2:
        st.error("Pick at least 2 features.")
    else:
        cv, sil = run_km(v, cf, kk)
        order = cv.groupby("Group").corrective_rate.mean().sort_values().index.tolist()
        pal = dict(zip(order, [TEAL, SKY, AMBER, ORANGE, PINK, VIOLET]))
        a, b = st.columns([3, 2])
        with a:
            with card("Vehicle clusters", "bubble size = total cost"):
                fig = px.scatter(cv, x="harsh_events", y="engine_temp_c", color="Group", size="total_cost_lakh", size_max=22, hover_name="vehicle_id",
                                 category_orders={"Group": order}, color_discrete_map=pal)
                fig.update_traces(marker=dict(opacity=.8, line=dict(color="rgba(255,255,255,.55)", width=1)))
                fig.update_xaxes(title="Harsh events"); fig.update_yaxes(title="Engine temp (°C)")
                plot(fig, 400)
        with b:
            with card("Silhouette score by k", f"k = {kk}: {sil[kk]:.3f}"):
                s = pd.DataFrame({"k": list(sil), "silhouette": list(sil.values())})
                fig = go.Figure()
                fig.add_scatter(x=s.k, y=s.silhouette, mode="lines+markers", line=dict(color=VIOLET, width=3, shape="spline"), marker=dict(size=9, color=VIOLET),
                                fill="tozeroy", fillgradient=dict(type="vertical", colorscale=[[0, "rgba(139,123,255,0)"], [1, "rgba(139,123,255,.4)"]]), name="score")
                fig.add_scatter(x=[kk], y=[sil[kk]], mode="markers", marker=dict(size=18, color=ORANGE, line=dict(color="#fff", width=3)), name="chosen k")
                fig.update_xaxes(dtick=1); fig.update_yaxes(range=[min(s.silhouette) * .8, max(s.silhouette) * 1.1])
                plot(fig, 400)
        prof = cv.groupby("Group").agg(Vehicles=("vehicle_id", "size"), Daily_km=("avg_daily_km", "mean"), Harsh_events=("harsh_events", "mean"),
                                       Temp_C=("engine_temp_c", "mean"), Repair_share=("corrective_rate", "mean"), Cost_lakh_per_vehicle=("total_cost_lakh", "mean")).loc[order]
        a, b = st.columns([2, 3])
        with a:
            with card("Cluster fingerprints", "normalised profile"):
                cols = ["Daily_km", "Harsh_events", "Temp_C", "Repair_share", "Cost_lakh_per_vehicle"]
                nz = (prof[cols] - prof[cols].min()) / (prof[cols].max() - prof[cols].min()).replace(0, 1)
                fig = go.Figure()
                for gname in nz.index:
                    r = nz.loc[gname].tolist()
                    fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=[c_.replace("_", " ") for c_ in cols] + [cols[0].replace("_", " ")], name=gname, fill="toself",
                                                  line=dict(color=pal[gname], width=2), fillcolor=rgba(pal[gname], .14)))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False, range=[0, 1.05]),
                                             angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED, size=10))))
                plot(fig, 380)
        with b:
            with card("Cluster profile", "group averages"):
                st.dataframe(prof.round(2), width="stretch")

# ───────────────────────── Association rules ─────────────────────────
with tabs[3]:
    with st.container(key="glass_rctrl"):
        c = st.columns(3)
        sup = c[0].slider("Min support", 0.01, 0.20, 0.04, 0.005)
        conf = c[1].slider("Min confidence", 0.1, 1.0, 0.6, 0.05)
        lift = c[2].slider("Min lift", 1.0, 12.0, 1.5, 0.5)
    rules = run_rules(d, parts, sup, conf, lift)
    st.metric("Rules found", len(rules))
    if rules.empty:
        st.info("No rules pass these thresholds. Lower the minimum support, confidence or lift.")
    else:
        a, b = st.columns([2, 3])
        with a:
            with card("Rule landscape", "size & colour = lift"):
                fig = go.Figure(go.Scatter(x=rules.support, y=rules.confidence, mode="markers", text=rules.Rule,
                                           marker=dict(size=np.clip(rules.lift * 5, 8, 40), color=rules.lift, colorscale=[[0, VIOLET], [.5, ORANGE], [1, AMBER]],
                                                       opacity=.8, line=dict(color="#fff", width=1), colorbar=dict(thickness=8, len=.7, tickfont=dict(color=MUTED))),
                                           hovertemplate="%{text}<br>support %{x:.3f} · conf %{y:.2f}<extra></extra>"))
                fig.update_xaxes(title="Support"); fig.update_yaxes(title="Confidence")
                plot(fig, 420, False)
        with b:
            with card("Top rules", "ranked by lift"):
                st.dataframe(rules.head(20).rename(columns=str.title), hide_index=True, width="stretch", height=380,
                             column_config={"Lift": st.column_config.ProgressColumn(min_value=1, max_value=float(max(rules.lift.max(), 2)), format="%.2f"),
                                            "Support": st.column_config.NumberColumn(format="%.3f"), "Confidence": st.column_config.NumberColumn(format="%.2f")})

# ───────────────────────── Prediction ─────────────────────────
with tabs[4]:
    with st.container(key="glass_pctrl"):
        c = st.columns(3)
        rm = c[0].radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)
        rtrees = c[1].slider("Trees", 50, 500, 200, 50, key="rt", disabled=rm == "Linear Regression")
        rtest = c[2].slider("Test share", 0.1, 0.5, 0.25, 0.05, key="rts")
    reg = run_reg(d, rm, rtrees, rtest)
    m = st.columns(2)
    m[0].metric("Average error (days)", f"{reg['mae']:.2f}")
    m[1].metric("R²", f"{reg['r2']:.3f}")
    a, b = st.columns([3, 2])
    with a:
        with card("Predicted vs actual", "days to next service"):
            ap = pd.DataFrame({"Actual": reg["actual"], "Predicted": reg["pred"]}); ap["err"] = (ap.Predicted - ap.Actual).abs()
            fig = go.Figure()
            fig.add_scatter(x=ap.Actual, y=ap.Predicted, mode="markers", marker=dict(size=7, color=ap.err, colorscale=[[0, TEAL], [.5, AMBER], [1, ORANGE]],
                            opacity=.7, colorbar=dict(title="|error|", thickness=8, len=.7, tickfont=dict(color=MUTED))), hovertemplate="actual %{x}<br>pred %{y:.0f}<extra></extra>")
            lo, hi_ = ap.Actual.min(), ap.Actual.max()
            fig.add_scatter(x=[lo, hi_], y=[lo, hi_], mode="lines", line=dict(color="rgba(255,255,255,.55)", dash="dash", width=1.5), hoverinfo="skip")
            fig.update_xaxes(title="Actual"); fig.update_yaxes(title="Predicted")
            plot(fig, 400, False)
    with b:
        with card("Try a vehicle", "what-if simulator"):
            row = d[FEATS].median().to_dict()
            row["age"] = st.slider("Vehicle age (years)", int(d.age.min()), int(d.age.max()), int(d.age.median()))
            row["harsh_events"] = st.slider("Harsh events", int(d.harsh_events.min()), int(d.harsh_events.max()), int(d.harsh_events.median()))
            row["engine_temp_c"] = st.slider("Engine temp (C)", float(d.engine_temp_c.min()), float(d.engine_temp_c.max()), float(d.engine_temp_c.median()))
            vt = st.selectbox("Vehicle type", TYPES, index=2)
            X1 = pd.DataFrame([row])
            for t in TYPES: X1["type_" + t] = int(vt == t)
            pr = float(reg["model"].predict(X1)[0])
            plot(gauge(pr, "Predicted days to next service", AMBER, "", (0, max(120, pr * 1.3)), 230), 230, False)

# ───────────────────────── Watchlist ─────────────────────────
with tabs[5]:
    st.caption("Uses the models set in the Classification and Prediction tabs on each vehicle's latest service record.")
    last = d.sort_values("full_date").groupby("vehicle_id").tail(1).reset_index(drop=True)
    last["Predicted need"] = clf["model"].predict(design(last))
    last["Days to next service"] = pd.Series(reg["model"].predict(design(last))).round().clip(lower=0)
    last["Next service due"] = (last.full_date + pd.to_timedelta(last["Days to next service"], unit="D")).dt.date
    with st.container(key="glass_wctrl"):
        c = st.columns(2)
        within = c[0].slider("Due within (days of the latest record)", 10, 120, 60)
        need = c[1].multiselect("Predicted need", LABELS, LABELS)
    asof = d.full_date.max().date()
    w = last[last["Predicted need"].isin(need)].sort_values("Days to next service")
    w = w[(w["Next service due"] - asof).map(lambda x: x.days) <= within]
    a, b = st.columns([1, 3])
    with a:
        st.metric("Vehicles shown", len(w))
        with card("Need split"):
            nc = w["Predicted need"].value_counts().reindex(LABELS).fillna(0)
            fig = go.Figure(go.Pie(labels=nc.index, values=nc.values, hole=.72, sort=False, textinfo="none",
                                   marker=dict(colors=[NEED_COL[x] for x in nc.index], line=dict(color="rgba(10,10,12,.9)", width=3))))
            plot(fig, 260)
    with b:
        with card("Service watchlist", "most urgent first"):
            st.dataframe(w[["vehicle_id", "vehicle_type", "city", "full_date", "Predicted need", "Days to next service", "Next service due"]]
                         .rename(columns={"full_date": "Last service"}), hide_index=True, width="stretch", height=420,
                         column_config={"Last service": st.column_config.DateColumn(format="YYYY-MM-DD"),
                                        "Days to next service": st.column_config.ProgressColumn(min_value=0, max_value=120, format="%d")})
