import re, sqlite3, base64, os, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config("Fleet Management Dashboard", page_icon="🚘", layout="wide", initial_sidebar_state="auto")

# ───────────────────────── palette & design tokens ─────────────────────────
CRIMSON = "#8b0e1e"        # primary deep crimson
CRIMSON_BRIGHT = "#c4152e" # vibrant crimson highlight
CRIMSON_GLOW = "rgba(196, 21, 46, 0.45)"
GREY = "#a8a8a8"           # neutral grey
GREY_LIGHT = "#e2dacc"     # warm platinum / light grey
GREY_DARK = "#2c2522"      # deep warm dark grey
TXT = "#ffffff"
MUTED = "#c4c0ba"
STROKE = "rgba(255, 255, 255, 0.22)"
GRID = "rgba(255, 255, 255, 0.09)"

# Color scales for plots
COLORS = [CRIMSON_BRIGHT, GREY, "#e04f5f", "#948e8a", "#e2dacc", "#5c504a", "#ff6b6b", "#403a37"]
CRIMSON_SCALE = [[0, "rgba(139,14,30,0.2)"], [0.5, "rgba(196,21,46,0.65)"], [1, "#ff6b7e"]]
GREY_SCALE = [[0, "rgba(44,37,34,0.4)"], [0.5, "rgba(168,168,168,0.7)"], [1, "#f5f3f0"]]
NEED_COL = {"Low": "#48c774", "Medium": "#a8a8a8", "High": CRIMSON_BRIGHT}

FEATS = ["age", "odometer_km", "avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "days_since_last_service"]
CL_FEATS = ["avg_daily_km", "idle_hours", "harsh_events", "engine_temp_c", "age", "corrective_rate"]
LABELS = ["Low", "Medium", "High"]
TYPES = ["Bus", "Car", "Truck", "Van"]

# Load car background image as base64
current_dir = os.path.dirname(os.path.abspath(__file__))
bg_path = os.path.join(current_dir, "car_bg.jpg")
b64_bg = ""
if os.path.exists(bg_path):
    with open(bg_path, "rb") as f:
        b64_bg = base64.b64encode(f.read()).decode("utf-8")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
    --crimson: {CRIMSON};
    --crimson-bright: {CRIMSON_BRIGHT};
    --crimson-glow: {CRIMSON_GLOW};
    --grey: {GREY};
    --grey-light: {GREY_LIGHT};
    --grey-dark: {GREY_DARK};
    --txt: {TXT};
    --muted: {MUTED};
    --stroke: {STROKE};
    /* Translucent glass layer from reference: highly frosted, deep glass with clear backdrop */
    --glass-card: linear-gradient(135deg, rgba(28, 26, 32, 0.46) 0%, rgba(14, 13, 18, 0.38) 100%);
    --glass-card-hover: linear-gradient(135deg, rgba(38, 32, 40, 0.54) 0%, rgba(20, 16, 24, 0.44) 100%);
    --r: 22px;
}}

html, body, .stApp, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--txt);
}}

/* Crisp Car Background pinned to the application viewport */
[data-testid="stAppViewContainer"], .stApp {{
    background-color: #0c0b0f !important;
    background-image: 
        radial-gradient(ellipse 90% 50% at 50% 0%, rgba(196, 21, 46, 0.22), transparent 75%),
        linear-gradient(180deg, rgba(10, 10, 14, 0.22) 0%, rgba(10, 10, 14, 0.40) 50%, rgba(8, 8, 11, 0.75) 100%),
        url("data:image/jpeg;base64,{b64_bg}") !important;
    background-size: 100% auto, 100% 100%, cover !important;
    background-position: top center, center, center 28% !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}

header[data-testid="stHeader"] {{
    background: transparent !important;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

.block-container {{
    padding: 1.5rem clamp(0.8rem, 2.5vw, 2.4rem) 4rem;
    max-width: 1540px;
}}

h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--txt);
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
}}

p, label, span, .stMarkdown {{
    color: var(--txt);
}}

[data-testid="stCaptionContainer"], small {{
    color: var(--muted) !important;
}}

/* True Optical Frosted Glass (Reference Style) */
.glass, [class*="st-key-glass"], [data-testid="stExpander"], [data-testid="stDataFrame"] {{
    background: var(--glass-card) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: var(--r) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55),
                inset 0 1px 1px rgba(255, 255, 255, 0.35),
                inset 0 -1px 1px rgba(0, 0, 0, 0.3) !important;
}}

[class*="st-key-glass"] {{
    padding: 1.2rem 1.3rem 0.8rem;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
}}

[class*="st-key-glass"]:hover {{
    background: var(--glass-card-hover) !important;
    border-color: rgba(196, 21, 46, 0.6) !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), 0 0 28px rgba(196, 21, 46, 0.3) !important;
}}

[data-testid="stDataFrame"] {{
    padding: 0.5rem;
    overflow: hidden;
}}

/* Window Controls (Mac OS style traffic dots from reference) */
.win-controls {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.7rem;
}}
.win-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.5);
}}
.win-dot-red {{ background: #ff5f56; border: 1px solid #e0443e; }}
.win-dot-yellow {{ background: #ffbd2e; border: 1px solid #dea123; }}
.win-dot-green {{ background: #27c93f; border: 1px solid #1aab29; }}
.win-url {{
    font-size: 0.74rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-left: 0.4rem;
    text-transform: uppercase;
    font-weight: 500;
}}

/* Hero Dashboard Header */
.hero {{
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.3rem;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1.2rem;
}}

.hero:before {{
    content: "";
    position: absolute;
    inset: -30% -10% auto 50%;
    height: 150%;
    background: radial-gradient(closest-side, rgba(196, 21, 46, 0.28), transparent);
    filter: blur(45px);
    pointer-events: none;
}}

.hero h1 {{
    font-size: clamp(2.2rem, 4.2vw, 3.3rem);
    line-height: 1.1;
    margin: 0;
    font-weight: 700;
    color: #ffffff;
}}

.hero h1 em {{
    font-style: normal;
    background: linear-gradient(90deg, #ff7a8d, #ffffff, var(--grey));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}}

.hero-subtitle {{
    color: var(--muted);
    margin: 0.6rem 0 0.8rem;
    font-size: 0.96rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    text-shadow: 0 1px 4px rgba(0,0,0,0.8);
}}

/* Live Fleet Vehicle Telemetry Highlights */
.vehicle-telemetry-bar {{
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    align-items: center;
    margin-top: 0.6rem;
}}

.telemetry-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid var(--stroke);
    padding: 0.38rem 0.8rem;
    border-radius: 12px;
    color: #ffffff;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}

.chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    position: relative;
    z-index: 2;
}}

.chip {{
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    background: rgba(22, 20, 26, 0.55);
    border: 1px solid var(--stroke);
    backdrop-filter: blur(14px);
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 0.45rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.35);
}}

.chip-status {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ff3b53;
    box-shadow: 0 0 10px #ff3b53;
}}

/* KPI Fleet Cards */
.kpis {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1rem;
    margin-bottom: 1.2rem;
}}

.kpi {{
    padding: 1.2rem 1.3rem;
    border-radius: var(--r);
    position: relative;
    overflow: hidden;
    min-height: 136px;
}}

.kpi .ic {{
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-size: 18px;
    background: var(--c);
    box-shadow: 0 8px 22px -4px var(--c);
}}

.kpi .lb {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}}

.kpi .v {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.65rem, 2.8vw, 2.25rem);
    font-weight: 700;
    line-height: 1.1;
    color: #ffffff;
    text-shadow: 0 2px 12px rgba(0,0,0,0.6);
}}

.kpi .v small {{
    font-size: 0.82rem;
    color: var(--muted) !important;
    font-weight: 400;
    margin-left: 0.3rem;
}}

.kpi svg {{
    position: absolute;
    right: 0;
    bottom: 0;
    width: 44%;
    height: 44px;
    opacity: 0.85;
    pointer-events: none;
}}

.kpi .v, .kpi .lb {{
    position: relative;
    z-index: 1;
}}

.kpi .tg {{
    position: absolute;
    top: 1.2rem;
    right: 1.3rem;
    font-size: 0.72rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid var(--stroke);
    color: var(--grey-light);
}}

/* Streamlit Metric Container (Other tabs) */
[data-testid="stMetric"] {{
    background: var(--glass-card) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 18px !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45) !important;
}}

[data-testid="stMetricLabel"] p {{
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

[data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2rem !important;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}}

/* Navigation Tabs */
[data-baseweb="tab-list"], [role="tablist"] {{
    gap: 0.5rem;
    background: rgba(18, 16, 22, 0.55);
    border: 1px solid var(--stroke);
    padding: 0.4rem;
    border-radius: 999px;
    backdrop-filter: blur(20px);
    overflow-x: auto;
    scrollbar-width: none;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 35px rgba(0,0,0,0.45);
}}

[data-baseweb="tab-list"]::-webkit-scrollbar, [role="tablist"]::-webkit-scrollbar {{
    display: none;
}}

button[data-baseweb="tab"], [role="tab"] {{
    border-radius: 999px !important;
    padding: 0.55rem 1.2rem !important;
    height: auto !important;
    background: transparent !important;
    white-space: nowrap;
    border: none !important;
}}

button[data-baseweb="tab"] p, [role="tab"] p {{
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}}

button[data-baseweb="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, var(--crimson-bright), var(--crimson)) !important;
    box-shadow: 0 6px 20px rgba(196, 21, 46, 0.6) !important;
}}

button[data-baseweb="tab"][aria-selected="true"] p, [role="tab"][aria-selected="true"] p {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}

[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], [role="tablist"] [class*="ighlight"], [data-testid="stTabsHighlight"] {{
    display: none !important;
}}

/* Form Controls & Sidebar */
[data-baseweb="select"] > div, [data-baseweb="input"], .stNumberInput input {{
    background: rgba(22, 20, 26, 0.65) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 12px !important;
    color: var(--txt) !important;
    backdrop-filter: blur(12px) !important;
}}

[data-baseweb="tag"] {{
    background: rgba(196, 21, 46, 0.4) !important;
    border: 1px solid rgba(196, 21, 46, 0.6) !important;
    border-radius: 8px !important;
}}

[data-testid="stSlider"] [role="slider"] {{
    background: var(--crimson-bright) !important;
    box-shadow: 0 0 0 5px rgba(196, 21, 46, 0.35) !important;
}}

[data-testid="stSlider"] [data-testid="stTickBar"] {{
    display: none;
}}

[data-testid="stSidebar"] {{
    background: rgba(12, 11, 15, 0.72) !important;
    backdrop-filter: blur(28px) !important;
    border-right: 1px solid var(--stroke) !important;
}}

[data-testid="stAlert"] {{
    border-radius: 16px;
    background: var(--glass-card) !important;
    border: 1px solid var(--stroke);
    backdrop-filter: blur(18px) !important;
}}

/* Card Title Section */
.sec {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 0.2rem 0 0.3rem;
}}
.sec h3 {{
    font-size: 1.05rem;
    margin: 0;
    font-weight: 600;
    color: #ffffff;
}}
.sec span {{
    font-size: 0.78rem;
    color: var(--muted);
}}
.sec:before {{
    content: "";
    width: 5px;
    height: 18px;
    border-radius: 4px;
    background: linear-gradient(var(--crimson-bright), var(--grey));
}}

/* Responsive Breakpoints */
[data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap;
    gap: 1rem;
}}
[data-testid="stColumn"] {{
    min-width: min(100%, 280px);
}}
@media (max-width: 1100px) {{
    .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 640px) {{
    .kpis {{ gap: 0.7rem; }}
    .kpi {{ min-height: 110px; padding: 0.9rem; }}
    .kpi svg {{ height: 34px; }}
    .hero {{ padding: 1.2rem; }}
    .block-container {{ padding-top: 0.8rem; }}
    [data-testid="stColumn"] {{ min-width: 100% !important; }}
}}
@media (max-width: 380px) {{
    .kpis {{ grid-template-columns: 1fr; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────── plotly theme ─────────────────────────
def style(fig, h=340, legend=True):
    fig.update_layout(
        height=h,
        margin=dict(l=8, r=8, t=14, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=MUTED, size=12),
        colorway=COLORS,
        showlegend=legend,
        legend=dict(orientation="h", y=-0.18, title=None, bgcolor="rgba(0,0,0,0)", font=dict(color=TXT)),
        hoverlabel=dict(bgcolor="rgba(20, 18, 24, 0.95)", bordercolor="rgba(196, 21, 46, 0.6)", font=dict(color="#ffffff", family="Plus Jakarta Sans")),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", title=None),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", title=None)
    )
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
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

def spark(vals, color):
    v = np.asarray(vals, float)
    if len(v) < 2 or np.ptp(v) == 0:
        return ""
    x = np.linspace(0, 100, len(v))
    y = 34 - (v - v.min()) / np.ptp(v) * 28
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    gid = color.replace("#", "").replace("(", "").replace(")", "").replace(",", "").replace(".", "").replace(" ", "")
    return (f'<svg viewBox="0 0 100 40" preserveAspectRatio="none"><defs><linearGradient id="g{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".5"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,40 {pts} 100,40" fill="url(#g{gid})"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>')

def kpi_row(items):
    html = '<div class="kpis">' + "".join(
        f'<div class="kpi glass" style="--c:{c}"><div class="ic">{ic}</div><div class="tg">{tag}</div><div class="lb">{lb}</div>'
        f'<div class="v">{val}<small>{unit}</small></div>{spark(sp, c)}</div>' for lb, val, unit, ic, c, tag, sp in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def gauge(value, title, color=CRIMSON_BRIGHT, suffix="%", rng=(0, 100), h=260):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=suffix, font=dict(size=42, color="#ffffff", family="Space Grotesk, sans-serif")),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        gauge=dict(
            shape="angular",
            axis=dict(range=list(rng), tickcolor="rgba(255,255,255,.25)", tickfont=dict(size=10, color=MUTED), nticks=6),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(255,255,255,.05)",
            borderwidth=0,
            steps=[
                dict(range=[rng[0], rng[0] + (rng[1] - rng[0]) * 0.5], color="rgba(255,255,255,.04)"),
                dict(range=[rng[0] + (rng[1] - rng[0]) * 0.5, rng[1]], color="rgba(255,255,255,.08)")
            ]
        )
    ))
    return fig

# ───────────────────────── data + models ─────────────────────────
@st.cache_data
def load():
    con = sqlite3.connect(os.path.join(current_dir, "fleet_dw.db"))
    df = pd.read_sql("""SELECT f.*, t.year, t.quarter, t.full_date, v.vehicle_id, v.vehicle_type, v.make, v.purchase_year,
        l.region, l.city, s.service_type FROM fact_service f JOIN dim_time t USING(time_key)
        JOIN dim_vehicle v USING(vehicle_key) JOIN dim_location l USING(location_key)
        JOIN dim_service_type s USING(service_type_key)""", con, parse_dates=["full_date"])
    df["age"] = df.year - df.purchase_year
    df["Year"] = df.year.astype(str)
    df["Quarter"] = df.year.astype(str) + " Q" + df.quarter.astype(str)
    df["Month"] = df.full_date.dt.strftime("%Y-%m")
    parts = pd.read_sql("SELECT b.service_key, p.part_name FROM bridge_service_part b JOIN dim_part p USING(part_key)", con)
    con.close()
    return df, parts

def design(d):
    X = d[FEATS].copy()
    for t in TYPES:
        X["type_" + t] = (d.vehicle_type == t).astype(int)
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
    return d.groupby("vehicle_id").agg(
        avg_daily_km=("avg_daily_km", "mean"),
        idle_hours=("idle_hours", "mean"),
        harsh_events=("harsh_events", "mean"),
        engine_temp_c=("engine_temp_c", "mean"),
        age=("age", "mean"),
        corrective_rate=("service_type", lambda s: (s == "Corrective").mean()),
        total_cost_lakh=("cost_inr", lambda s: s.sum() / 1e5)
    ).reset_index()

@st.cache_data
def run_km(v, feats, k):
    Z = StandardScaler().fit_transform(v[list(feats)])
    fit = lambda n: KMeans(n, n_init=10, random_state=1).fit_predict(Z)
    out = v.copy()
    out["cl"] = fit(k)
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
sb.markdown("""
<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:1rem;">
    <div style="font-size: 26px;">🚘</div>
    <div><b>Fleet Command</b><span style="display:block;font-size:0.75rem;color:var(--muted)">Telemetry & Filters</span></div>
</div>
""", unsafe_allow_html=True)

year = sb.multiselect("Year", sorted(df.Year.unique()), sorted(df.Year.unique()))
region = sb.multiselect("Region", sorted(df.region.unique()), sorted(df.region.unique()))
vtype = sb.multiselect("Vehicle Type", TYPES, TYPES)
d = df[df.Year.isin(year) & df.region.isin(region) & df.vehicle_type.isin(vtype)].reset_index(drop=True)
sb.caption(f"{len(d):,} of {len(df):,} records selected")

# Sidebar vehicle quick telemetry
sb.markdown("---")
sb.markdown("""
<div style="font-size:0.78rem;color:var(--muted);margin-bottom:0.4rem;font-weight:600;letter-spacing:0.04em;">FLEET TELEMETRY STATUS</div>
<div style="display:flex;flex-direction:column;gap:0.4rem;font-size:0.8rem;">
    <div>🏎 <b>Active Vehicles:</b> 100 units</div>
    <div>⛽ <b>Fuel/Service Health:</b> 94.2%</div>
    <div>🌡 <b>Avg Engine Temp:</b> 88.5°C</div>
    <div>🛰 <b>GPS Tracking:</b> Connected</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────── Hero Header (Title first) ─────────────────────────
span = f"{df.full_date.min():%b %Y} – {df.full_date.max():%b %Y}"
st.markdown(f"""
<div class="hero glass">
    <div style="position:relative; z-index: 2;">
        <div class="win-controls">
            <span class="win-dot win-dot-red"></span>
            <span class="win-dot win-dot-yellow"></span>
            <span class="win-dot win-dot-green"></span>
            <span class="win-url">fleet.telemetry.console // live</span>
        </div>
        <h1>Fleet Management Dashboard</h1>
        <div class="hero-subtitle">
            <span>Enterprise Telemetry & Predictive Maintenance</span>
            <span>•</span>
            <span>{df.vehicle_id.nunique()} active fleet vehicles</span>
        </div>
        <div class="vehicle-telemetry-bar">
            <div class="telemetry-item">🚗 <b>Cars:</b> {len(d[d.vehicle_type=='Car']):,} logs</div>
            <div class="telemetry-item">🚚 <b>Trucks:</b> {len(d[d.vehicle_type=='Truck']):,} logs</div>
            <div class="telemetry-item">🚌 <b>Buses:</b> {len(d[d.vehicle_type=='Bus']):,} logs</div>
            <div class="telemetry-item">🚐 <b>Vans:</b> {len(d[d.vehicle_type=='Van']):,} logs</div>
        </div>
    </div>
    <div class="chips">
        <span class="chip"><span class="chip-status"></span>Telemetry Active</span>
        <span class="chip">📅 {span}</span>
        <span class="chip">📊 {len(d):,} Records</span>
    </div>
</div>
""", unsafe_allow_html=True)

if len(d) < 150 or d.vehicle_id.nunique() < 10:
    st.warning("Too few records for modelling. Please broaden the filters in the sidebar.")
    st.stop()

tabs = st.tabs(["◉ Overview", "◈ Classification", "◎ Clustering", "⛓ Association Rules", "↗ Prediction", "⚑ Watchlist"])

# ───────────────────────── 1. Overview ─────────────────────────
with tabs[0]:
    mon = d.groupby("Month").agg(
        cost=("cost_inr", "sum"),
        n=("cost_inr", "size"),
        dt=("downtime_hours", "mean"),
        rep=("service_type", lambda s: (s == "Corrective").mean())
    )
    kpi_row([
        ("Total Services", f"{len(d):,}", "", "🔧", CRIMSON_BRIGHT, "all time", mon.n),
        ("Fleet Maintenance Cost", f"{d.cost_inr.sum() / 1e5:,.1f}", "Lakh", "₹", GREY, "monthly trend", mon.cost),
        ("Avg Downtime", f"{d.downtime_hours.mean():.1f}", "hrs", "⏱", "#e04f5f", "per service", mon.dt),
        ("Repair Ratio", f"{(d.service_type == 'Corrective').mean():.0%}", "", "🚨", "#948e8a", "corrective", mon.rep)
    ])

    with st.container(key="glass_ctrl"):
        c = st.columns(3)
        meas = c[0].selectbox("Metric to Analyze", ["Total cost (INR lakh)", "Services", "Avg downtime (hrs)"])
        level = c[1].selectbox("Time Level", ["Year", "Quarter", "Month"], index=1)
        dim = c[2].selectbox("Group By Dimension", ["region", "city", "vehicle_type", "make", "service_type"])
    
    value = lambda g: {"Total cost (INR lakh)": g.cost_inr.sum() / 1e5, "Services": len(g), "Avg downtime (hrs)": g.downtime_hours.mean()}[meas]
    agg = lambda key: d.groupby(key).apply(value, include_groups=False).reset_index(name="value")

    a, b = st.columns([3, 2])
    with a:
        with card(f"{meas}", f"Trend by {level.lower()}"):
            t = agg(level)
            fig = go.Figure()
            fig.add_scatter(
                x=t[level], y=t.value, mode="lines+markers",
                line=dict(color=CRIMSON_BRIGHT, width=3, shape="spline", smoothing=0.8),
                marker=dict(size=8, color="#ffffff", line=dict(color=CRIMSON_BRIGHT, width=2)),
                fill="tozeroy",
                fillgradient=dict(type="vertical", colorscale=[[0, "rgba(139,14,30,0)"], [1, "rgba(196,21,46,0.45)"]]),
                hovertemplate="%{x}<br><b>%{y:,.1f}</b><extra></extra>"
            )
            fig.update_yaxes(rangemode="tozero")
            plot(fig, 340, False)
    with b:
        with card(f"{meas}", f"Breakdown by {dim.replace('_', ' ')}"):
            t = agg(dim).sort_values("value", ascending=False)
            fig = go.Figure(go.Bar(
                x=t[dim], y=t.value,
                marker=dict(color=t.value, colorscale=CRIMSON_SCALE, line=dict(width=0)),
                hovertemplate="%{x}<br><b>%{y:,.1f}</b><extra></extra>"
            ))
            rounded(fig, 10)
            plot(fig, 340, False)

    a, b, c3 = st.columns([2, 2, 3])
    with a:
        with card("Service Distribution", "Preventive vs Corrective"):
            sm = d.service_type.value_counts().reset_index()
            fig = go.Figure(go.Pie(
                labels=sm.service_type, values=sm["count"], hole=0.7, sort=False, textinfo="percent",
                marker=dict(colors=[GREY, CRIMSON_BRIGHT, "#e04f5f", "#948e8a"], line=dict(color="rgba(10,10,12,.9)", width=2)),
                textfont=dict(color="#ffffff")
            ))
            fig.add_annotation(text=f"<b>{len(d):,}</b><br><span style='font-size:11px;color:{MUTED}'>services</span>", showarrow=False, font=dict(size=20, color="#ffffff"))
            plot(fig, 300)
    with b:
        with card("Fleet Health Index", "Low & moderate risk score"):
            hi = ((d.maintenance_need == "Low").mean() + 0.5 * (d.maintenance_need == "Medium").mean()) * 100
            gauge_clr = "#48c774" if hi >= 60 else GREY if hi >= 40 else CRIMSON_BRIGHT
            fig = gauge(hi, "Fleet Reliability Condition", gauge_clr)
            plot(fig, 300, False)
    with c3:
        with card("Regional Maintenance Cost Heatmap", "Region × Quarter (Lakh ₹)"):
            hm = d.pivot_table(index="region", columns="Quarter", values="cost_inr", aggfunc="sum").fillna(0) / 1e5
            fig = go.Figure(go.Heatmap(
                z=hm.values, x=hm.columns, y=hm.index, colorscale=CRIMSON_SCALE, xgap=4, ygap=4, showscale=False,
                hovertemplate="%{y} · %{x}<br><b>%{z:,.1f}</b> lakh<extra></extra>"
            ))
            plot(fig, 300, False)

    with card("High-Frequency Replaced Parts", "Most replaced vehicle components"):
        pc = parts[parts.service_key.isin(d.service_key)].part_name.value_counts().reset_index().sort_values("count")
        fig = go.Figure(go.Bar(
            y=pc.part_name, x=pc["count"], orientation="h",
            marker=dict(color=pc["count"], colorscale=GREY_SCALE),
            hovertemplate="%{y}<br><b>%{x:,}</b><extra></extra>"
        ))
        rounded(fig, 8)
        plot(fig, max(300, 26 * len(pc)), False)

# ───────────────────────── 2. Classification ─────────────────────────
with tabs[1]:
    with st.container(key="glass_clfctrl"):
        c = st.columns(4)
        model = c[0].radio("Model", ["Decision Tree", "Random Forest"], index=1, horizontal=True)
        depth = c[1].slider("Max Depth (0 = unlimited)", 0, 20, 6 if model == "Decision Tree" else 0, key=f"cd_{model}")
        trees = c[2].slider("Estimator Trees", 50, 500, 200, 50, disabled=model == "Decision Tree")
        test = c[3].slider("Test Split Share", 0.1, 0.5, 0.25, 0.05, key="ct")
    
    clf = run_clf(d, model, depth, trees, test)
    hr = clf["cm"][2, 2] / max(clf["cm"][2].sum(), 1)
    
    g = st.columns(3)
    for col, (v, t, clr) in zip(g, [(clf["acc"] * 100, "Accuracy", CRIMSON_BRIGHT), (clf["f1"] * 100, "Macro F1 Score", GREY), (hr * 100, "High-Need Recall", "#e04f5f")]):
        with col:
            with card(t):
                plot(gauge(v, "", clr, "%" if "F1" not in t else ""), 220, False)
    
    a, b = st.columns(2)
    with a:
        with card("Confusion Matrix", "Actual vs Predicted classes"):
            fig = go.Figure(go.Heatmap(
                z=clf["cm"], x=LABELS, y=LABELS, colorscale=CRIMSON_SCALE, xgap=5, ygap=5, showscale=False,
                text=clf["cm"], texttemplate="<b>%{text}</b>", textfont=dict(size=18, color="#ffffff"),
                hovertemplate="Actual %{y} → Predicted %{x}<br><b>%{z}</b><extra></extra>"
            ))
            fig.update_yaxes(autorange="reversed", title="Actual")
            fig.update_xaxes(title="Predicted")
            plot(fig, 360, False)
    with b:
        with card("Feature Importance in Wear Prediction", "Vehicle sensor weights"):
            imp = clf["imp"]
            fig = go.Figure()
            fig.add_bar(y=imp.index, x=imp.values, orientation="h", width=0.08, marker_color="rgba(168,168,168,0.3)", hoverinfo="skip")
            fig.add_scatter(
                y=imp.index, x=imp.values, mode="markers",
                marker=dict(size=15, color=imp.values, colorscale=[[0, GREY], [0.5, CRIMSON], [1, CRIMSON_BRIGHT]], line=dict(color="#ffffff", width=2)),
                hovertemplate="%{y}<br><b>%{x:.3f}</b><extra></extra>"
            )
            plot(fig, 360, False)

# ───────────────────────── 3. Clustering ─────────────────────────
with tabs[2]:
    v = vehicle_table(d)
    with st.container(key="glass_kctrl"):
        c = st.columns([1, 2])
        kk = c[0].slider("Number of Clusters (k)", 3, 6, 4)
        cf = tuple(c[1].multiselect("Features for Clustering", CL_FEATS, CL_FEATS))
    
    if len(cf) < 2:
        st.error("Please select at least 2 vehicle features.")
    else:
        cv, sil = run_km(v, cf, kk)
        order = cv.groupby("Group").corrective_rate.mean().sort_values().index.tolist()
        cluster_palette = [GREY, "#948e8a", "#e04f5f", CRIMSON_BRIGHT, "#ff6b6b", "#403a37"]
        pal = dict(zip(order, cluster_palette[:len(order)]))
        
        a, b = st.columns([3, 2])
        with a:
            with card("Vehicle Wear Segments", "Bubble size = Total maintenance cost"):
                fig = px.scatter(
                    cv, x="harsh_events", y="engine_temp_c", color="Group", size="total_cost_lakh", size_max=22,
                    hover_name="vehicle_id", category_orders={"Group": order}, color_discrete_map=pal
                )
                fig.update_traces(marker=dict(opacity=0.85, line=dict(color="rgba(255,255,255,.6)", width=1)))
                fig.update_xaxes(title="Harsh Braking & Acceleration Events")
                fig.update_yaxes(title="Avg Engine Temperature (°C)")
                plot(fig, 400)
        with b:
            with card("Silhouette Quality Score", f"k = {kk} Silhouette: {sil[kk]:.3f}"):
                s = pd.DataFrame({"k": list(sil), "silhouette": list(sil.values())})
                fig = go.Figure()
                fig.add_scatter(
                    x=s.k, y=s.silhouette, mode="lines+markers",
                    line=dict(color=GREY, width=3, shape="spline"),
                    marker=dict(size=9, color=GREY),
                    fill="tozeroy",
                    fillgradient=dict(type="vertical", colorscale=[[0, "rgba(168,168,168,0)"], [1, "rgba(168,168,168,0.3)"]]),
                    name="Score"
                )
                fig.add_scatter(x=[kk], y=[sil[kk]], mode="markers", marker=dict(size=18, color=CRIMSON_BRIGHT, line=dict(color="#ffffff", width=3)), name="Selected k")
                fig.update_xaxes(dtick=1)
                fig.update_yaxes(range=[min(s.silhouette) * 0.8, max(s.silhouette) * 1.1])
                plot(fig, 400)
        
        prof = cv.groupby("Group").agg(
            Vehicles=("vehicle_id", "size"),
            Daily_km=("avg_daily_km", "mean"),
            Harsh_events=("harsh_events", "mean"),
            Temp_C=("engine_temp_c", "mean"),
            Repair_share=("corrective_rate", "mean"),
            Cost_lakh_per_vehicle=("total_cost_lakh", "mean")
        ).loc[order]
        
        a, b = st.columns([2, 3])
        with a:
            with card("Cluster Radar Profiles", "Normalized attributes"):
                cols = ["Daily_km", "Harsh_events", "Temp_C", "Repair_share", "Cost_lakh_per_vehicle"]
                nz = (prof[cols] - prof[cols].min()) / (prof[cols].max() - prof[cols].min()).replace(0, 1)
                fig = go.Figure()
                for gname in nz.index:
                    r = nz.loc[gname].tolist()
                    fig.add_trace(go.Scatterpolar(
                        r=r + r[:1], theta=[c_.replace("_", " ") for c_ in cols] + [cols[0].replace("_", " ")],
                        name=gname, fill="toself",
                        line=dict(color=pal[gname], width=2), fillcolor=rgba(pal[gname], 0.16)
                    ))
                fig.update_layout(polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=False, range=[0, 1.05]),
                    angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED, size=10))
                ))
                plot(fig, 380)
        with b:
            with card("Cluster Fleet Averages", "Summary metrics"):
                st.dataframe(prof.round(2), width="stretch")

# ───────────────────────── 4. Association Rules ─────────────────────────
with tabs[3]:
    with st.container(key="glass_rctrl"):
        c = st.columns(3)
        sup = c[0].slider("Minimum Support", 0.01, 0.20, 0.04, 0.005)
        conf = c[1].slider("Minimum Confidence", 0.1, 1.0, 0.6, 0.05)
        lift = c[2].slider("Minimum Lift", 1.0, 12.0, 1.5, 0.5)
    
    rules = run_rules(d, parts, sup, conf, lift)
    st.metric("Rules Discovered", len(rules))
    
    if rules.empty:
        st.info("No association rules found. Adjust the support or lift sliders to widen threshold.")
    else:
        a, b = st.columns([2, 3])
        with a:
            with card("Part Replacement Rules", "Point size & color = Lift"):
                fig = go.Figure(go.Scatter(
                    x=rules.support, y=rules.confidence, mode="markers", text=rules.Rule,
                    marker=dict(
                        size=np.clip(rules.lift * 5, 8, 40),
                        color=rules.lift,
                        colorscale=CRIMSON_SCALE,
                        opacity=0.85,
                        line=dict(color="#ffffff", width=1),
                        colorbar=dict(thickness=8, len=0.7, tickfont=dict(color=MUTED))
                    ),
                    hovertemplate="%{text}<br>Support: %{x:.3f} · Conf: %{y:.2f}<extra></extra>"
                ))
                fig.update_xaxes(title="Support")
                fig.update_yaxes(title="Confidence")
                plot(fig, 420, False)
        with b:
            with card("Top Co-Occurring Maintenance Rules", "Ranked by lift factor"):
                st.dataframe(
                    rules.head(20).rename(columns=str.title),
                    hide_index=True, width="stretch", height=380,
                    column_config={
                        "Lift": st.column_config.ProgressColumn(min_value=1, max_value=float(max(rules.lift.max(), 2)), format="%.2f"),
                        "Support": st.column_config.NumberColumn(format="%.3f"),
                        "Confidence": st.column_config.NumberColumn(format="%.2f")
                    }
                )

# ───────────────────────── 5. Prediction ─────────────────────────
with tabs[4]:
    with st.container(key="glass_pctrl"):
        c = st.columns(3)
        rm = c[0].radio("Regression Algorithm", ["Linear Regression", "Random Forest"], horizontal=True)
        rtrees = c[1].slider("Trees Count", 50, 500, 200, 50, key="rt", disabled=rm == "Linear Regression")
        rtest = c[2].slider("Test Share Split", 0.1, 0.5, 0.25, 0.05, key="rts")
    
    reg = run_reg(d, rm, rtrees, rtest)
    m = st.columns(2)
    m[0].metric("Mean Absolute Error (Days)", f"{reg['mae']:.2f}")
    m[1].metric("R² Variance Explained", f"{reg['r2']:.3f}")
    
    a, b = st.columns([3, 2])
    with a:
        with card("Predicted vs Actual Interval", "Days until next required service"):
            ap = pd.DataFrame({"Actual": reg["actual"], "Predicted": reg["pred"]})
            ap["err"] = (ap.Predicted - ap.Actual).abs()
            fig = go.Figure()
            fig.add_scatter(
                x=ap.Actual, y=ap.Predicted, mode="markers",
                marker=dict(size=7, color=ap.err, colorscale=CRIMSON_SCALE, opacity=0.75, colorbar=dict(title="|error|", thickness=8, len=0.7, tickfont=dict(color=MUTED))),
                hovertemplate="Actual: %{x}<br>Predicted: %{y:.0f}<extra></extra>"
            )
            lo, hi_ = ap.Actual.min(), ap.Actual.max()
            fig.add_scatter(x=[lo, hi_], y=[lo, hi_], mode="lines", line=dict(color="rgba(255,255,255,.5)", dash="dash", width=1.5), hoverinfo="skip")
            fig.update_xaxes(title="Actual Days")
            fig.update_yaxes(title="Predicted Days")
            plot(fig, 400, False)
    with b:
        with card("Vehicle What-If Simulator", "Test specific vehicle parameters"):
            row = d[FEATS].median().to_dict()
            row["age"] = st.slider("Vehicle Age (years)", int(d.age.min()), int(d.age.max()), int(d.age.median()))
            row["harsh_events"] = st.slider("Harsh Driving Events", int(d.harsh_events.min()), int(d.harsh_events.max()), int(d.harsh_events.median()))
            row["engine_temp_c"] = st.slider("Engine Temperature (°C)", float(d.engine_temp_c.min()), float(d.engine_temp_c.max()), float(d.engine_temp_c.median()))
            vt = st.selectbox("Vehicle Type", TYPES, index=1)
            X1 = pd.DataFrame([row])
            for t in TYPES:
                X1["type_" + t] = int(vt == t)
            pr = float(reg["model"].predict(X1)[0])
            plot(gauge(pr, "Days Until Next Service", CRIMSON_BRIGHT, "", (0, max(120, pr * 1.3)), 230), 230, False)

# ───────────────────────── 6. Watchlist ─────────────────────────
with tabs[5]:
    st.caption("Predicts maintenance urgency using latest telemetry for each fleet vehicle.")
    last = d.sort_values("full_date").groupby("vehicle_id").tail(1).reset_index(drop=True)
    last["Predicted need"] = clf["model"].predict(design(last))
    last["Days to next service"] = pd.Series(reg["model"].predict(design(last))).round().clip(lower=0)
    last["Next service due"] = (last.full_date + pd.to_timedelta(last["Days to next service"], unit="D")).dt.date
    
    with st.container(key="glass_wctrl"):
        c = st.columns(2)
        within = c[0].slider("Service Due Within (Days)", 10, 120, 60)
        need = c[1].multiselect("Predicted Urgency Level", LABELS, LABELS)
    
    asof = d.full_date.max().date()
    w = last[last["Predicted need"].isin(need)].sort_values("Days to next service")
    w = w[(w["Next service due"] - asof).map(lambda x: x.days) <= within]
    
    a, b = st.columns([1, 3])
    with a:
        st.metric("Critical Vehicles in Queue", len(w))
        with card("Urgency Breakdown"):
            nc = w["Predicted need"].value_counts().reindex(LABELS).fillna(0)
            fig = go.Figure(go.Pie(
                labels=nc.index, values=nc.values, hole=0.72, sort=False, textinfo="none",
                marker=dict(colors=[NEED_COL[x] for x in nc.index], line=dict(color="rgba(10,10,12,.9)", width=2))
            ))
            plot(fig, 260)
    with b:
        with card("Fleet Service Watchlist", "Priority order"):
            st.dataframe(
                w[["vehicle_id", "vehicle_type", "city", "full_date", "Predicted need", "Days to next service", "Next service due"]]
                .rename(columns={"full_date": "Last service"}),
                hide_index=True, width="stretch", height=420,
                column_config={
                    "Last service": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Days to next service": st.column_config.ProgressColumn(min_value=0, max_value=120, format="%d")
                }
            )
