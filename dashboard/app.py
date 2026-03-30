import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 1. PAGE CONFIG  (must be the VERY first call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cisco AI Terminal",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* ── Root tokens ── */
:root {
    --bg:          #05060a;
    --surface:     #0c0e15;
    --surface2:    #111420;
    --border:      #1a1e2e;
    --border2:     #242840;
    --accent:      #4f6ef7;
    --accent2:     #7c5cfc;
    --teal:        #1de9b6;
    --warn:        #ffb300;
    --crit:        #f44336;
    --txt:         #e8eaf6;
    --txt-muted:   #5c6380;
    --txt-dim:     #333750;
}

/* ── Base ── */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--txt);
}

/* Remove Streamlit chrome */
header, footer, #MainMenu { display: none !important; }
[data-testid="block-container"] {
    padding: 1.2rem 2rem 0.5rem 2rem !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    width: 220px !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] .stSlider > div { padding-top: 0.5rem; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: var(--txt-muted) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    padding: 0.5rem !important;
    margin-top: 0.5rem !important;
}
[data-testid="stSidebar"] button:hover { opacity: 0.85 !important; }

/* ── Metric dataframe ── */
[data-testid="stDataFrame"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Plotly charts transparent ── */
.js-plotly-plot .plotly, .plot-container { background: transparent !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.4rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. DATA LOADING  (with safe fallback)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/processed/results.csv")
        # Make sure required columns exist
        required = ["device_id", "temp", "vibration", "latency",
                    "packet_loss", "health_score", "rul", "score", "network_score"]
        if not all(c in df.columns for c in required):
            raise ValueError("Missing columns")
        return df
    except Exception:
        rng = np.random.default_rng(42)
        n = 1000
        health  = np.clip(np.linspace(96, 18, n) + rng.normal(0, 3, n), 0, 100)
        rul_arr = np.clip(np.linspace(520, 5, n) + rng.normal(0, 8, n), 0, 600)
        score   = np.linspace(0.06, -0.07, n) + rng.normal(0, 0.008, n)
        return pd.DataFrame({
            "device_id":    [0] * n,
            "temp":         rng.normal(44, 3, n),
            "vibration":    rng.normal(0.48, 0.12, n),
            "latency":      rng.normal(19, 6, n),
            "packet_loss":  np.clip(rng.normal(0.6, 0.4, n), 0, 5),
            "health_score": health,
            "rul":          rul_arr,
            "score":        score,
            "network_score": np.linspace(99, 55, n),
        })


df = load_data()

# ─────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="
            font-family:'Orbitron',sans-serif;
            font-size:14px; font-weight:700;
            color:#fff; letter-spacing:2px;
            padding:10px 0 12px 0;
            border-bottom:1px solid #1a1e2e;
            margin-bottom:16px;">
            ⚙ CMD MENU
        </div>""", unsafe_allow_html=True)

    # Define your hardware mapping
    DEVICE_MAPPING = {
        0: "Cisco Catalyst 9300",
        1: "Cisco Nexus 9000",
        2: "Cisco ISR 4000",
        3: "Cisco UCS C-Series",
        4: "APC Smart-UPS"
    }

    device_ids = sorted(df["device_id"].unique())
    selected_device = st.selectbox(
        "TARGET DEVICE ID", 
        options=device_ids,
        format_func=lambda x: DEVICE_MAPPING.get(int(x), f"Unknown Device {x}")
    )

    max_pts = int(df[df["device_id"] == selected_device].shape[0])
    max_pts = max(max_pts, 200)   # guard against tiny datasets
    time_filter = st.slider(
        "DATA WINDOW",
        min_value=100,
        max_value=max_pts,
        value=min(300, max_pts),
        step=50,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 REFRESH FEED"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
        <div style="
            margin-top:24px;
            padding-top:16px;
            border-top:1px solid #1a1e2e;
            font-family:'JetBrains Mono',monospace;
            font-size:9px; color:#333750;
            letter-spacing:1px; line-height:1.8;">
            CISCO AI TERMINAL v2.1<br>
            ISOLATION FOREST ENGINE<br>
            © 2026 PREDICTIVE OPS
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. FILTER DATA
# ─────────────────────────────────────────────
device_data = (
    df[df["device_id"] == selected_device]
    .tail(time_filter)
    .reset_index(drop=True)
)

if device_data.empty:
    st.error("No data available for the selected device.")
    st.stop()

latest   = device_data.iloc[-1]
previous = device_data.iloc[-2] if len(device_data) > 1 else latest

# ─────────────────────────────────────────────
# 6. STATUS LOGIC
# ─────────────────────────────────────────────
def get_status(health: float, score: float):
    if health < 30 or score < -0.05:
        return "CRITICAL", "#f44336", "🔴"
    if health < 60 or score < -0.01:
        return "DEGRADING", "#ffb300", "🟠"
    return "NOMINAL", "#1de9b6", "🟢"

status_label, status_color, status_dot = get_status(
    latest["health_score"], latest["score"]
)

# ─────────────────────────────────────────────
# 7. HEADER BANNER
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 7. HEADER BANNER
# ─────────────────────────────────────────────
# Fetch the mapped name, defaulting to ID if not found, and make it uppercase
device_name = DEVICE_MAPPING.get(int(selected_device), f"DEVICE {selected_device}").upper()

st.markdown(f"""
<div style="display:flex; align-items:flex-end; justify-content:space-between;
            margin-bottom:6px;">
    <div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:10px; letter-spacing:4px;
            color:var(--accent); margin-bottom:2px;">
            ● QUANTUM NODE ALPHA · {device_name}
        </div>
        <div style="
            font-family:'Orbitron',sans-serif;
            font-size:34px; font-weight:900;
            font-style:italic; color:#fff;
            line-height:1; letter-spacing:2px;">
            CISCO&nbsp;<span style="color:#7c5cfc;">TERMINAL</span>
        </div>
        <div style="
            font-family:'JetBrains Mono',monospace;
            font-size:10px; color:#5c6380;
            margin-top:3px; letter-spacing:2px;">
            // PREDICTIVE MAINTENANCE &amp; REAL-TIME TELEMETRY
        </div>
    </div>
    <div style="
        background:var(--surface);
        border:1px solid var(--border2);
        border-radius:12px;
        padding:10px 20px;
        text-align:right;">
        <div style="font-family:'JetBrains Mono',monospace;
                    font-size:10px; color:#5c6380;
                    letter-spacing:2px; margin-bottom:4px;">
            SYSTEM STATUS
        </div>
        <div style="
            font-family:'Orbitron',sans-serif;
            font-size:18px; font-weight:700;
            color:{status_color}; letter-spacing:2px;">
            {status_dot}&nbsp;{status_label}
        </div>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# 8. KPI CARDS  (4 cards)
# ─────────────────────────────────────────────
def delta_arrow(val: float) -> str:
    if val > 0:  return f"<span style='color:#1de9b6;'>▲ +{val:.2f}</span>"
    if val < 0:  return f"<span style='color:#f44336;'>▼ {val:.2f}</span>"
    return f"<span style='color:#5c6380;'>— {val:.2f}</span>"

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex color to rgba() string for use inside HTML attributes."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def kpi_card(title: str, value: str, delta_html: str,
             subtitle: str, bar_pct: float = None,
             bar_color: str = "#4f6ef7", value_color: str = "#fff") -> str:
    glow = hex_to_rgba(bar_color, 0.13)
    bar_html = ""
    if bar_pct is not None:
        bar_pct = max(0.0, min(1.0, bar_pct))
        bar_html = (
            f'<div style="margin-top:10px;background:#1a1e2e;border-radius:4px;'
            f'height:3px;overflow:hidden;">'
            f'<div style="width:{bar_pct*100:.1f}%;height:100%;'
            f'background:{bar_color};border-radius:4px;'
            f'transition:width 0.6s ease;"></div></div>'
        )
    return (
        f'<div style="background:#0c0e15;border:1px solid #1a1e2e;'
        f'border-radius:12px;padding:16px 18px;height:100%;'
        f'position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;right:0;width:60px;height:60px;'
        f'background:radial-gradient(circle at top right,{glow},transparent 70%);'
        f'border-radius:0 12px 0 0;"></div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        f'font-weight:600;letter-spacing:2px;color:#5c6380;'
        f'text-transform:uppercase;margin-bottom:10px;">{title}</div>'
        f'<div style="font-family:\'Orbitron\',sans-serif;font-size:28px;'
        f'font-weight:900;font-style:italic;color:{value_color};'
        f'line-height:1.1;">{value}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        f'margin-top:6px;">{delta_html}</div>'
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:10px;'
        f'color:#5c6380;margin-top:4px;text-transform:uppercase;'
        f'letter-spacing:1px;">{subtitle}</div>'
        f'{bar_html}</div>'
    )

health_delta   = latest["health_score"] - previous["health_score"]
rul_delta      = latest["rul"]          - previous["rul"]
score_delta    = latest["score"]        - previous["score"]
net_delta      = latest["network_score"]- previous["network_score"]

rul_color   = "#1de9b6" if latest["rul"] > 100 else "#f44336"
score_val   = latest["score"]
ai_state    = "CRITICAL" if score_val < -0.05 else ("WARNING" if score_val < -0.01 else "STABLE")
ai_color    = "#f44336" if score_val < -0.05 else ("#ffb300" if score_val < -0.01 else "#1de9b6")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi_card(
        "System Health Score",
        f"{latest['health_score']:.1f}",
        delta_arrow(health_delta),
        "Live Telemetry Feed",
        bar_pct=latest["health_score"] / 100,
        bar_color="#4f6ef7",
    ), unsafe_allow_html=True)

with c2:
    st.markdown(kpi_card(
        "Remaining Useful Life",
        f"{latest['rul']:.0f} cyc",
        delta_arrow(rul_delta),
        "Predictive Degradation",
        bar_pct=latest["rul"] / 520,
        bar_color=rul_color,
        value_color=rul_color,
    ), unsafe_allow_html=True)

with c3:
    st.markdown(kpi_card(
        "AI Signal Matrix",
        ai_state,
        f"<span style='color:{ai_color};font-family:JetBrains Mono,monospace;font-size:11px;'>"
        f"SCORE: {score_val:.4f}</span>",
        "Isolation Forest Output",
        bar_pct=max(0, (score_val + 0.07) / 0.13),
        bar_color=ai_color,
        value_color=ai_color,
    ), unsafe_allow_html=True)

with c4:
    st.markdown(kpi_card(
        "Network Degradation",
        f"{latest['network_score']:.1f}",
        delta_arrow(net_delta),
        "AI Network Score",
        bar_pct=latest["network_score"] / 100,
        bar_color="#7c5cfc",
    ), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 9. SHARED PLOTLY THEME
# ─────────────────────────────────────────────
CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", size=11, color="#5c6380"),
    margin=dict(l=10, r=10, t=8, b=8),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="#1a1e2e", gridwidth=1,
               zeroline=False, tickfont=dict(size=10)),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0c0e15",
        bordercolor="#242840",
        font=dict(family="JetBrains Mono", size=11, color="#e8eaf6"),
    ),
)

def section(label: str, desc: str = "") -> None:
    desc_html = (f"<div style='font-family:DM Sans,sans-serif;font-size:10px;"
                 f"color:#333750;margin-bottom:6px;'>{desc}</div>") if desc else ""
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:10px;
                letter-spacing:2px;text-transform:uppercase;
                color:#5c6380;margin-bottom:2px;margin-top:10px;'>
        {label}
    </div>{desc_html}""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 10. ROW 1 — Telemetry + Network Panel
# ─────────────────────────────────────────────
col_hw, col_net = st.columns([3.5, 1], gap="medium")

# with col_hw:
#     section("📈 Intraday Trajectory — Temp & Vibration",
#             "Real-time thermal output (°C) and kinetic vibration stress (g) overlaid on dual axes.")

#     fig_hw = make_subplots(specs=[[{"secondary_y": True}]])
#     fig_hw.add_trace(
#         go.Scatter(
#             x=device_data.index, y=device_data["temp"],
#             name="Temp (°C)", mode="lines",
#             line=dict(color="#4f6ef7", width=2),
#             fill="tozeroy", fillcolor="rgba(79,110,247,0.07)",
#         ), secondary_y=False,
#     )
#     fig_hw.add_trace(
#         go.Scatter(
#             x=device_data.index, y=device_data["vibration"],
#             name="Vibration (g)", mode="lines",
#             line=dict(color="#b084cc", width=1.5, dash="dot"),
#         ), secondary_y=True,
#     )
#     fig_hw.update_layout(**CHART, height=230, showlegend=True,
#                          legend=dict(orientation="h", x=0, y=1.12,
#                                      font=dict(size=10, color="#5c6380")))
#     fig_hw.update_yaxes(title_text="Temp (°C)", title_font=dict(size=10), secondary_y=False)
#     fig_hw.update_yaxes(title_text="Vibration (g)", title_font=dict(size=10), secondary_y=True)
#     st.plotly_chart(fig_hw, width="stretch", config={"displayModeBar": False})

with col_hw:
    section("📈 Intraday Trajectory & Degradation Trends",
            "Real-time telemetry overlaid with Exponential Moving Averages (EMA) to expose long-term hardware stress.")

    # 1. Calculate Exponential Moving Averages (EMA)
    # We use a dynamic span based on the data window to ensure smooth curves
    trend_span = max(10, len(device_data) // 5) 
    device_data["temp_ema"] = device_data["temp"].ewm(span=trend_span, adjust=False).mean()
    device_data["vib_ema"] = device_data["vibration"].ewm(span=trend_span, adjust=False).mean()

    fig_hw = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 2. Raw Data (Dimmed so it doesn't distract from the trend)
    fig_hw.add_trace(
        go.Scatter(
            x=device_data.index, y=device_data["temp"],
            name="Temp Raw", mode="lines",
            line=dict(color="rgba(79,110,247,0.3)", width=1), 
            fill="tozeroy", fillcolor="rgba(79,110,247,0.05)",
        ), secondary_y=False,
    )
    fig_hw.add_trace(
        go.Scatter(
            x=device_data.index, y=device_data["vibration"],
            name="Vib Raw", mode="lines",
            line=dict(color="rgba(176,132,204,0.3)", width=1, dash="dot"), 
        ), secondary_y=True,
    )

    # 3. EMA Trendlines (Bold and highly visible)
    fig_hw.add_trace(
        go.Scatter(
            x=device_data.index, y=device_data["temp_ema"],
            name="Temp Trend (EMA)", mode="lines",
            line=dict(color="#4f6ef7", width=2.5), # Solid Blue
        ), secondary_y=False,
    )
    fig_hw.add_trace(
        go.Scatter(
            x=device_data.index, y=device_data["vib_ema"],
            name="Vib Trend (EMA)", mode="lines",
            line=dict(color="#1de9b6", width=2.5), # Vibrant Teal
        ), secondary_y=True,
    )

    fig_hw.update_layout(**CHART, height=230, showlegend=True,
                         legend=dict(orientation="h", x=0, y=1.12,
                                     font=dict(size=10, color="#5c6380")))
    fig_hw.update_yaxes(title_text="Temp (°C)", title_font=dict(size=10), secondary_y=False)
    fig_hw.update_yaxes(title_text="Vibration (g)", title_font=dict(size=10), secondary_y=True)
    st.plotly_chart(fig_hw, use_container_width=True, config={"displayModeBar": False})
with col_net:
    section("🌐 Sector Radar", "Live network integrity snapshot.")

    # Pre-compute everything — no nested f-strings inside st.markdown
    pkt_val   = f"{latest['packet_loss']:.2f}%"
    lat_val   = f"{latest['latency']:.1f}"
    net_val   = f"{latest['network_score']:.1f}"
    pkt_color = "#f44336" if latest["packet_loss"]   > 1  else "#1de9b6"
    lat_color = "#ffb300" if latest["latency"]       > 30 else "#ffffff"
    net_color = "#f44336" if latest["network_score"] < 70 else "#ffffff"

    def _radar_row(label, sub, val, val_color, border=True):
        bd = "border-bottom:1px solid #1a1e2e;padding-bottom:10px;margin-bottom:10px;" if border else ""
        return (
            '<div style="' + bd + '">'
            '<div style="display:flex;justify-content:space-between;align-items:center;">'
            '<div>'
            '<div style="font-family:Orbitron,sans-serif;font-size:12px;'
            'font-weight:700;color:#ffffff;">' + label + '</div>'
            '<div style="font-family:JetBrains Mono,monospace;font-size:9px;'
            'color:#5c6380;letter-spacing:1px;">' + sub + '</div>'
            '</div>'
            '<div style="font-family:JetBrains Mono,monospace;font-size:16px;'
            'font-weight:700;color:' + val_color + ';">' + val + '</div>'
            '</div></div>'
        )

    radar_html = (
        '<div style="background:#0c0e15;padding:14px;border-radius:12px;'
        'border:1px solid #1a1e2e;height:230px;'
        'display:flex;flex-direction:column;justify-content:space-around;">'
        + _radar_row("LATENCY",     "NETWORK MS",   lat_val, lat_color, border=True)
        + _radar_row("PKT LOSS",    "DROPPED %",    pkt_val, pkt_color, border=True)
        + _radar_row("DEGRADATION", "AI NET SCORE", net_val, net_color, border=False)
        + '</div>'
    )
    st.markdown(radar_html, unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 11. ROW 2 — AI Score + System Logs
# ─────────────────────────────────────────────
col_ai, col_logs = st.columns([2, 1], gap="medium")

with col_ai:
    section("🤖 AI Isolation Forest Score",
            "Unsupervised ML analysing multi-dimensional variance. Scores < -0.01 indicate anomalous behaviour.")

    fig_ai = go.Figure()
    fig_ai.add_trace(go.Scatter(
        x=device_data.index, y=device_data["score"],
        mode="lines", name="Anomaly Score",
        line=dict(color="#1de9b6", width=2),
        fill="tozeroy", fillcolor="rgba(29,233,182,0.05)",
    ))
    fig_ai.add_hline(
        y=-0.05, line_dash="dash", line_color="#f44336", line_width=1,
        annotation_text="CRITICAL", annotation_font_color="#f44336",
        annotation_position="bottom right",
    )
    fig_ai.add_hline(
        y=-0.01, line_dash="dot", line_color="#ffb300", line_width=1,
        annotation_text="WARNING", annotation_font_color="#ffb300",
        annotation_position="bottom right",
    )
    fig_ai.update_layout(
        **CHART, height=230,
        yaxis_title="Anomaly Score",
        showlegend=False,
    )
    st.plotly_chart(fig_ai, width="stretch", config={"displayModeBar": False})

with col_logs:
    section("⚠️ System Logs",
            "Recent threshold breaches, newest first.")

    anomalies = device_data[device_data["score"] < -0.01].copy()

    if not anomalies.empty:
        anomalies["SEVERITY"] = anomalies["score"].apply(
            lambda x: "CRITICAL" if x < -0.05 else "WARNING"
        )
        display_logs = (
            anomalies[["SEVERITY", "health_score", "temp", "latency"]]
            .tail(10)
            .iloc[::-1]
            .round(2)
        )
        display_logs.columns = ["SEVERITY", "HEALTH", "TEMP °C", "LAT ms"]

        def colour_severity(val):
            if val == "CRITICAL":
                return "background-color:rgba(244,67,54,0.15);color:#f44336;font-weight:700;"
            return "background-color:rgba(255,179,0,0.12);color:#ffb300;font-weight:700;"

        styled = display_logs.style.map(colour_severity, subset=["SEVERITY"])
        st.dataframe(styled, width="stretch", height=230, hide_index=True)
    else:
        st.markdown("""
        <div style="background:#0c0e15;padding:20px;border-radius:12px;
                    border:1px solid #1a1e2e;height:230px;
                    display:flex;align-items:center;justify-content:center;">
            <div style="color:#1de9b6;font-family:'JetBrains Mono',monospace;
                        font-size:13px;text-align:center;line-height:2;">
                ✓<br>SYSTEM STABLE<br>
                <span style='color:#333750;font-size:10px;'>
                    NO ANOMALIES DETECTED
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 12. ROW 3 — Health Score Timeline
# ─────────────────────────────────────────────
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
section("📉 Health Score Timeline",
        "Composite health index over the selected window. Colour-coded zones for quick assessment.")

fig_h = go.Figure()

# Coloured zone fills
fig_h.add_hrect(y0=0,  y1=30,  fillcolor="rgba(244,67,54,0.07)",  layer="below", line_width=0)
fig_h.add_hrect(y0=30, y1=60,  fillcolor="rgba(255,179,0,0.06)",  layer="below", line_width=0)
fig_h.add_hrect(y0=60, y1=100, fillcolor="rgba(29,233,182,0.04)", layer="below", line_width=0)

fig_h.add_trace(go.Scatter(
    x=device_data.index, y=device_data["health_score"],
    mode="lines", name="Health",
    line=dict(color="#4f6ef7", width=2.5),
    fill="tozeroy", fillcolor="rgba(79,110,247,0.08)",
))
fig_h.update_layout(**CHART, height=160, showlegend=False)
fig_h.update_yaxes(range=[0, 100], showgrid=True,
                   gridcolor="#1a1e2e", tickfont=dict(size=10))
st.plotly_chart(fig_h, width="stretch", config={"displayModeBar": False})

# ─────────────────────────────────────────────
# 13. FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:12px;
    padding:10px 0;
    border-top:1px solid #1a1e2e;
    display:flex;
    justify-content:space-between;
    align-items:center;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#333750;letter-spacing:2px;">
        CISCO AI TERMINAL · ISOLATION FOREST ENGINE · v2.1
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                color:#333750;letter-spacing:1px;">
        © 2026 PREDICTIVE OPERATIONS DIVISION
    </div>
</div>""", unsafe_allow_html=True)
