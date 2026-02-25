"""
app_streamlit.styles — Dark Purple/Orange Theme CSS
=====================================================
Inject via st.markdown(WORKBENCH_CSS, unsafe_allow_html=True)
"""

PALETTE = {
    "bg":          "#0a0a12",
    "bg2":         "#12121e",
    "bg3":         "#1a1a2e",
    "card":        "#16162a",
    "card_border": "#2d2d50",
    "purple":      "#a855f7",
    "purple_dark": "#7c3aed",
    "purple_glow": "rgba(168,85,247,0.15)",
    "orange":      "#f97316",
    "orange_dark": "#ea580c",
    "orange_glow": "rgba(249,115,22,0.15)",
    "text":        "#e2e8f0",
    "text_muted":  "#94a3b8",
    "text_faint":  "#475569",
    "success":     "#22c55e",
    "warning":     "#eab308",
    "error":       "#ef4444",
    "info":        "#06b6d4",
    "divider":     "#2d2d50",
    "green":       "#22c55e",
    "red":         "#ef4444",
}

WORKBENCH_CSS = f"""
<style>
/* ══════════════════════════════════════════════════════════
   Global reset — enforce dark mode
══════════════════════════════════════════════════════════ */
:root {{
    --bg:          {PALETTE['bg']};
    --bg2:         {PALETTE['bg2']};
    --bg3:         {PALETTE['bg3']};
    --card:        {PALETTE['card']};
    --card-border: {PALETTE['card_border']};
    --purple:      {PALETTE['purple']};
    --orange:      {PALETTE['orange']};
    --text:        {PALETTE['text']};
    --text-muted:  {PALETTE['text_muted']};
    --success:     {PALETTE['success']};
    --warning:     {PALETTE['warning']};
    --error:       {PALETTE['error']};
}}

html, body, [data-testid="stApp"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
}}

/* ══════════════════════════════════════════════════════════
   Main container
══════════════════════════════════════════════════════════ */
[data-testid="stAppViewContainer"] {{
    background-color: var(--bg) !important;
}}
[data-testid="block-container"] {{
    padding: 1.5rem 2rem !important;
    max-width: 1400px !important;
}}
section[data-testid="stSidebar"] {{
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--card-border) !important;
}}

/* ══════════════════════════════════════════════════════════
   Typography
══════════════════════════════════════════════════════════ */
h1 {{
    color: var(--purple) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    text-shadow: 0 0 30px {PALETTE['purple_glow']} !important;
}}
h2 {{
    color: var(--text) !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    border-bottom: 1px solid var(--card-border) !important;
    padding-bottom: 0.5rem !important;
}}
h3 {{
    color: var(--purple) !important;
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}

/* ══════════════════════════════════════════════════════════
   Step header badges
══════════════════════════════════════════════════════════ */
.step-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: linear-gradient(135deg, {PALETTE['bg3']}, {PALETTE['card']});
    border: 1px solid {PALETTE['card_border']};
    border-left: 4px solid {PALETTE['purple']};
    border-radius: 8px;
    margin: 1rem 0 0.75rem 0;
}}
.step-badge {{
    background: {PALETTE['purple']};
    color: white;
    font-weight: 800;
    font-size: 0.7rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    letter-spacing: 0.05em;
}}
.step-title {{
    color: {PALETTE['text']};
    font-size: 1.0rem;
    font-weight: 700;
    margin: 0;
}}
.step-subtitle {{
    color: {PALETTE['text_muted']};
    font-size: 0.78rem;
    margin: 0;
}}

/* ══════════════════════════════════════════════════════════
   Cards
══════════════════════════════════════════════════════════ */
.wb-card {{
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    margin: 0.5rem 0 !important;
}}
.wb-card-purple {{
    background: {PALETTE['bg3']} !important;
    border: 1px solid {PALETTE['purple_dark']} !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 0 20px {PALETTE['purple_glow']} !important;
}}
.wb-card-orange {{
    background: {PALETTE['bg3']} !important;
    border: 1px solid {PALETTE['orange_dark']} !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 0 20px {PALETTE['orange_glow']} !important;
}}

/* ══════════════════════════════════════════════════════════
   Metric tiles
══════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {{
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: {PALETTE['text_muted']} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
[data-testid="stMetricValue"] {{
    color: var(--text) !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] svg {{
    display: none;
}}

/* ══════════════════════════════════════════════════════════
   Buttons
══════════════════════════════════════════════════════════ */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {PALETTE['purple']}, {PALETTE['purple_dark']}) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 0 15px {PALETTE['purple_glow']} !important;
    transition: all 0.2s !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 25px {PALETTE['purple_glow']}, 0 4px 15px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:not([kind="primary"]) {{
    background: transparent !important;
    border: 1px solid {PALETTE['card_border']} !important;
    color: {PALETTE['text_muted']} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    border-color: {PALETTE['purple']} !important;
    color: {PALETTE['purple']} !important;
}}

/* ══════════════════════════════════════════════════════════
   Sliders & inputs
══════════════════════════════════════════════════════════ */
[data-testid="stSlider"] > div > div > div > div {{
    background: {PALETTE['purple']} !important;
}}
input, textarea {{
    background-color: var(--bg2) !important;
    color: var(--text) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 6px !important;
}}
input:focus, textarea:focus {{
    border-color: var(--purple) !important;
    box-shadow: 0 0 8px {PALETTE['purple_glow']} !important;
}}
.stSelectbox > div, .stMultiSelect > div {{
    background-color: var(--bg2) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 6px !important;
}}

/* ══════════════════════════════════════════════════════════
   Tabs
══════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] {{
    color: var(--text) !important;
}}
button[data-baseweb="tab"] {{
    color: {PALETTE['text_muted']} !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PALETTE['purple']} !important;
    border-bottom: 2px solid {PALETTE['purple']} !important;
}}

/* ══════════════════════════════════════════════════════════
   Expanders
══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {{
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{
    color: var(--text) !important;
    font-weight: 600 !important;
}}
[data-testid="stExpander"] summary:hover {{
    color: var(--purple) !important;
}}

/* ══════════════════════════════════════════════════════════
   Dataframes / Tables
══════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}

/* ══════════════════════════════════════════════════════════
   Status / progress
══════════════════════════════════════════════════════════ */
[data-testid="stStatusWidget"] {{
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px !important;
}}

/* ══════════════════════════════════════════════════════════
   Progress bar
══════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {PALETTE['purple']}, {PALETTE['orange']}) !important;
    border-radius: 999px !important;
}}

/* ══════════════════════════════════════════════════════════
   Alert boxes
══════════════════════════════════════════════════════════ */
[data-testid="stAlert"][data-kind="success"] {{
    background: rgba(34,197,94,0.1) !important;
    border-left: 4px solid {PALETTE['success']} !important;
    border-radius: 6px !important;
}}
[data-testid="stAlert"][data-kind="warning"] {{
    background: rgba(234,179,8,0.1) !important;
    border-left: 4px solid {PALETTE['warning']} !important;
    border-radius: 6px !important;
}}
[data-testid="stAlert"][data-kind="error"] {{
    background: rgba(239,68,68,0.1) !important;
    border-left: 4px solid {PALETTE['error']} !important;
    border-radius: 6px !important;
}}
[data-testid="stAlert"][data-kind="info"] {{
    background: rgba(6,182,212,0.08) !important;
    border-left: 4px solid {PALETTE['info']} !important;
    border-radius: 6px !important;
}}

/* ══════════════════════════════════════════════════════════
   Toggle / checkbox
══════════════════════════════════════════════════════════ */
[data-testid="stCheckbox"] label {{
    color: var(--text) !important;
}}
[data-testid="stToggle"] {{
    color: var(--text) !important;
}}

/* ══════════════════════════════════════════════════════════
   Divider
══════════════════════════════════════════════════════════ */
hr {{
    border-color: {PALETTE['divider']} !important;
    margin: 1.5rem 0 !important;
}}

/* ══════════════════════════════════════════════════════════
   Scrollbar
══════════════════════════════════════════════════════════ */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg2); }}
::-webkit-scrollbar-thumb {{ background: {PALETTE['card_border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {PALETTE['purple']}; }}

/* ══════════════════════════════════════════════════════════
   Result ribbon
══════════════════════════════════════════════════════════ */
.result-ribbon {{
    display: flex;
    gap: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, {PALETTE['bg3']}, {PALETTE['card']});
    border: 1px solid {PALETTE['purple_dark']};
    border-radius: 10px;
    margin: 1rem 0;
    flex-wrap: wrap;
}}
.result-tile {{
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 0.5rem;
}}
.result-tile .label {{
    color: {PALETTE['text_muted']};
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.result-tile .value {{
    color: {PALETTE['text']};
    font-size: 1.3rem;
    font-weight: 700;
}}
.result-tile .value.positive {{ color: {PALETTE['success']}; }}
.result-tile .value.negative {{ color: {PALETTE['error']}; }}
.result-tile .value.highlight {{ color: {PALETTE['orange']}; }}

/* ══════════════════════════════════════════════════════════
   Indicator card
══════════════════════════════════════════════════════════ */
.ind-card {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    transition: border-color 0.2s;
}}
.ind-card.active {{
    border-color: {PALETTE['purple']};
    box-shadow: 0 0 12px {PALETTE['purple_glow']};
}}
.ind-card.auto-tuned {{
    border-color: {PALETTE['orange']};
    box-shadow: 0 0 12px {PALETTE['orange_glow']};
}}
.ind-type-badge {{
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    background: {PALETTE['bg3']};
    color: {PALETTE['purple']};
    border: 1px solid {PALETTE['purple']};
    margin-left: 0.5rem;
    letter-spacing: 0.06em;
    vertical-align: middle;
}}

/* ══════════════════════════════════════════════════════════
   Live log console
══════════════════════════════════════════════════════════ */
.log-console {{
    background: #050510;
    border: 1px solid {PALETTE['card_border']};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: monospace;
    font-size: 0.78rem;
    color: {PALETTE['success']};
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.6;
}}
.log-line {{ margin: 0; padding: 0; }}
.log-line.info  {{ color: {PALETTE['info']}; }}
.log-line.warn  {{ color: {PALETTE['warning']}; }}
.log-line.error {{ color: {PALETTE['error']}; }}

/* ══════════════════════════════════════════════════════════
   Summary banner at top
══════════════════════════════════════════════════════════ */
.wb-banner {{
    background: linear-gradient(135deg, {PALETTE['bg3']} 0%, {PALETTE['bg2']} 100%);
    border: 1px solid {PALETTE['card_border']};
    border-top: 3px solid {PALETTE['purple']};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.wb-banner h1 {{
    margin: 0 0 0.25rem 0 !important;
}}
.wb-banner p {{
    color: {PALETTE['text_muted']};
    margin: 0;
    font-size: 0.85rem;
}}
</style>
"""


def step_header(number: int, title: str, subtitle: str = "") -> str:
    """Return HTML for a step header block."""
    return f"""
<div class="step-header">
  <span class="step-badge">STEP {number}</span>
  <div>
    <div class="step-title">{title}</div>
    {"<div class='step-subtitle'>" + subtitle + "</div>" if subtitle else ""}
  </div>
</div>
"""


def result_ribbon_html(
    start_cap: float,
    end_cap: float,
    net_pnl: float,
    net_pnl_pct: float,
    primary_label: str,
    primary_value: str,
) -> str:
    """Return HTML for the results summary ribbon."""
    pnl_class = "positive" if net_pnl >= 0 else "negative"
    pnl_sign  = "+" if net_pnl >= 0 else ""
    return f"""
<div class="result-ribbon">
  <div class="result-tile">
    <div class="label">Start Capital</div>
    <div class="value">${start_cap:,.0f}</div>
  </div>
  <div class="result-tile">
    <div class="label">End Capital</div>
    <div class="value">${end_cap:,.0f}</div>
  </div>
  <div class="result-tile">
    <div class="label">Net P&L ($)</div>
    <div class="value {pnl_class}">{pnl_sign}${net_pnl:,.0f}</div>
  </div>
  <div class="result-tile">
    <div class="label">Net P&L (%)</div>
    <div class="value {pnl_class}">{pnl_sign}{net_pnl_pct:.2%}</div>
  </div>
  <div class="result-tile">
    <div class="label">{primary_label}</div>
    <div class="value highlight">{primary_value}</div>
  </div>
</div>
"""
