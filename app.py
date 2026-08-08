import os
import re
import json
import pandas as pd
import pandas_ta as ta
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="Nifty Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CACHE_DIR = os.getenv("CACHE_DIR", "data/ohlcv_cache")
SCREENER_JSON = os.getenv("SCREENER_JSON_PATH", "data/latest_screener_results.json")
EQUITY_CSV = "EQUITY_L.csv"

# Load Env Vars for dynamic UI text interpolation
WATCHLIST_MIN_SCORE = int(os.getenv("SCREENER_WATCHLIST_MIN_SCORE", "3"))
TRADE_READY_MIN_SCORE = int(os.getenv("SCREENER_TRADE_READY_MIN_SCORE", "5"))
HOLD_DAYS = int(os.getenv("BACKTEST_HOLD_DAYS", "20"))
ATR_MULTIPLIER = float(os.getenv("BACKTEST_ATR_MULTIPLIER", "2.0"))

# Dictionary to map technical triggers to plain-English explanations for the UI
TRIGGER_HELP = {
    "ABOVE_EMA_200": "Long-term trend is positive (Price is trading above its 200-day average).",
    "EMA_200_BREAKOUT": "Fresh Breakout: Price just crossed above the 200-day average today. Strong reversal signal.",
    "ABOVE_EMA_50": "Medium-term trend is positive (Price is trading above its 50-day average).",
    "VOLUME_SPIKE_2X": "Unusual activity: today's volume is more than double its 20-day average.",
    "RSI_60_TO_70": "Goldilocks Momentum: RSI is between 60-70 (Strong momentum, but not dangerously overbought).",
    "MACD_BULLISH_CROSS": "MACD line crossed above the signal line today, indicating an immediate shift to upward momentum.",
}

st.markdown(
    """
<style>
:root {
    --bg:#080d12;
    --surface:#0e151d;
    --surface-2:#121b24;
    --surface-3:#17222d;
    --line:#263542;
    --line-soft:#1d2a35;
    --muted:#8796a5;
    --text:#edf2f6;
    --green:#35d99a;
    --amber:#f3bd55;
    --red:#ff6672;
    --blue:#73aaff;
}

.stApp {
    background:
        radial-gradient(circle at 88% -12%, rgba(53,217,154,.075), transparent 30rem),
        radial-gradient(circle at -8% 30%, rgba(115,170,255,.035), transparent 28rem),
        var(--bg);
    color:var(--text);
}

.block-container {
    max-width:1540px;
    padding:1.35rem 2.2rem 4rem;
}

#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

h1,h2,h3 {
    letter-spacing:-.035em;
}

h2 {
    margin-top:.15rem;
}

.hero {
    display:flex;
    justify-content:space-between;
    align-items:flex-end;
    gap:2rem;
    margin:.1rem 0 1.45rem;
    padding-bottom:1.15rem;
    border-bottom:1px solid var(--line-soft);
}

.eyebrow {
    color:var(--green);
    font-size:.68rem;
    font-weight:800;
    letter-spacing:.16em;
    text-transform:uppercase;
}

.hero h1 {
    margin:.3rem 0 .35rem;
    font-size:2.25rem;
    line-height:1.02;
}

.hero p {
    color:var(--muted);
    margin:0;
    font-size:.88rem;
}

.scan {
    color:var(--muted);
    font-size:.66rem;
    line-height:1.55;
    text-align:right;
    letter-spacing:.12em;
}

.scan b {
    color:#dfe6ec;
    letter-spacing:0;
    font-size:.8rem;
}

.card {
    background:linear-gradient(180deg,var(--surface-2),var(--surface));
    border:1px solid var(--line);
    border-radius:15px;
    padding:1rem 1.05rem;
    min-height:112px;
    box-shadow:0 10px 28px rgba(0,0,0,.12);
    transition:border-color .15s ease, transform .15s ease;
}

.card:hover {
    border-color:#3b4c5b;
    transform:translateY(-1px);
}

.card-label {
    color:var(--muted);
    font-size:.66rem;
    font-weight:750;
    text-transform:uppercase;
    letter-spacing:.11em;
}

.card-value {
    font-size:1.7rem;
    font-weight:800;
    margin:.42rem 0 .16rem;
    letter-spacing:-.04em;
}

.card-note {
    color:var(--muted);
    font-size:.72rem;
    line-height:1.4;
}

.dot {
    display:inline-block;
    width:7px;
    height:7px;
    border-radius:50%;
    margin-right:7px;
    background:currentColor;
    vertical-align:middle;
}

.good { color:var(--green); }
.warn { color:var(--amber); }
.bad { color:var(--red); }

.section-kicker {
    color:var(--muted);
    font-size:.65rem;
    font-weight:800;
    letter-spacing:.14em;
    text-transform:uppercase;
    margin-top:1.35rem;
    margin-bottom:.25rem;
}

.stock-head {
    background:linear-gradient(180deg,var(--surface-2),var(--surface));
    border:1px solid var(--line);
    border-radius:16px;
    padding:1.05rem 1.15rem;
    margin:.55rem 0 1rem;
    box-shadow:0 10px 28px rgba(0,0,0,.10);
}

.stock-title {
    font-size:1.42rem;
    font-weight:800;
    letter-spacing:-.03em;
}

.stock-meta {
    color:var(--muted);
    font-size:.78rem;
    margin-top:.3rem;
}

.pill {
    display:inline-flex;
    align-items:center;
    border:1px solid #30404e;
    background:#151e27;
    border-radius:999px;
    padding:.3rem .6rem;
    margin:.15rem .25rem .15rem 0;
    color:#d1dbe4;
    font-size:.7rem;
    font-weight:650;
    transition:all .15s ease;
}

.pill:hover {
    border-color:var(--green);
    background:rgba(53,217,154,.08);
    color:#a7efd2;
}

div[data-testid="stDataFrame"] {
    border:1px solid var(--line);
    border-radius:13px;
    overflow:auto !important;
    resize:vertical;
    max-height:800px;
    min-height:200px;
    background:var(--surface);
}

.stTabs [data-baseweb="tab-list"] {
    gap:2rem;
    border-bottom:1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    padding:.72rem 0;
    background:transparent;
    font-weight:650;
}

.stTabs [aria-selected="true"] {
    color:var(--green)!important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background:var(--surface-2);
    border-color:var(--line);
    border-radius:10px;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color:#3b4c5b;
}

[data-testid="stAlert"] {
    border-radius:12px;
}

[data-testid="stFileUploader"] {
    background:var(--surface);
    border:1px dashed #344552;
    border-radius:12px;
    padding:.25rem;
}

[data-testid="stMetric"] {
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:12px;
    padding:.8rem;
}

.glossary-term {
    color:var(--green);
    font-weight:750;
    font-size:1rem;
    margin-bottom:.25rem;
}

.glossary-def {
    color:#c3cdd6;
    font-size:.86rem;
    line-height:1.6;
    margin-bottom:1.35rem;
}

.glossary-def ul {
    margin-top:.35rem;
    margin-bottom:.35rem;
}

.glossary-def li {
    margin-bottom:.3rem;
}

@media (max-width: 900px) {
    .block-container { padding:1rem; }
    .hero { display:block; }
    .scan { text-align:left; margin-top:.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_screener_data():
    if os.path.exists(SCREENER_JSON):
        with open(SCREENER_JSON, "r") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_stock_ohlcv(ticker: str):
    file_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    else:
        try:
            yf_ticker = ticker if ticker.endswith(".NS") or "." in ticker else f"{ticker}.NS"
            df = yf.download(yf_ticker, period="1y", progress=False)
            if df.empty:
                df = yf.download(ticker, period="1y", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                os.makedirs(CACHE_DIR, exist_ok=True)
                df.to_parquet(file_path)
                return df
            return None
        except Exception:
            return None


@st.cache_data
def load_equity_master(filepath=EQUITY_CSV):
    """
    Loads the NSE equity master list safely with encoding fallback 
    and validates required columns and row count.
    """
    if not os.path.exists(filepath):
        st.warning(f"Warning: {filepath} not found. Name matching may fall back to ticker guessing.")
        return None

    df = None
    # 1. Attempt loading with UTF-8, fall back to latin-1 on decode errors
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except Exception as e:
            st.error(f"Error loading {filepath} with latin-1: {e}")
            return None
    except Exception as e:
        st.error(f"Error reading {filepath}: {e}")
        return None

    if df is None or df.empty:
        return None

    # 2. Clean and uppercase column headers to prevent matching issues due to spacing/case
    df.columns = [str(col).strip().upper() for col in df.columns]

    # 3. Validation checks
    required_cols = {"SYMBOL", "NAME OF COMPANY"}
    if not required_cols.issubset(df.columns):
        st.error(f"Error: Missing required columns in {filepath}. Found: {list(df.columns)}")
        return None

    # Optional sanity check assertion for row count (Full NSE equity list is typically 2000+)
    if len(df) < 1000:
        st.warning(f"Warning: Equity master row count ({len(df)}) is lower than expected for a full NSE list.")

    return df


def find_cached_ticker(company_name, cache_dir):
    """Maps broker company names to exact NSE symbols using EQUITY_L.csv."""
    original_name = str(company_name).strip().upper()
    
    # Comprehensive broker mapping overrides
    KNOWN_ALIASES = {
        # --- Popular ETFs & Indices ---
        "LIQUID BEES": "LIQUIDBEES.NS",
        "LIQUIDBEES": "LIQUIDBEES.NS",
        "NIP IND ETF LIQUID BEES": "LIQUIDBEES.NS",
        "NIPPON INDIA ETF LIQUID BEES": "LIQUIDBEES.NS",
        "GOLD BEES": "GOLDBEES.NS",
        "NIPPON INDIA ETF GOLDBEES": "GOLDBEES.NS",
        "NIFTY BEES": "NIFTYBEES.NS",
        "NIPPON INDIA ETF NIFTY 50 BEES": "NIFTYBEES.NS",
        
        # --- Major Conglomerates & Auto Demergers ---
        "OIL AND NATURAL GAS CORP.": "ONGC.NS",
        "OIL & NATURAL GAS CORPORATION": "ONGC.NS",
        "TATA MOTORS LIMITED": "TMCV.NS",
        "TATA MOTORS PASS VEH LTD": "TMPV.NS",
        "TATA MOTORS PASSENGER VEHICLES LIMITED": "TMPV.NS",
        "RELIANCE INDUSTRIES LTD": "RELIANCE.NS",
        "RELIANCE INDUSTRIES LIMITED": "RELIANCE.NS",
        
        # --- Infrastructure & Engineering ---
        "AMARA RAJA ENERGY MOB LTD": "ARE&M.NS",
        "AMARA RAJA ENERGY MOBILITY LIMITED": "ARE&M.NS",
        "AMARA RAJA BATTERIES": "ARE&M.NS",
        "JKUMAR INFR.LTD.": "JKIL.NS",
        "J.KUMAR INFRAPROJECTS LIMITED": "JKIL.NS",
        "H.G. INFRA ENGINEERING LIMITED": "HGINFRA.NS",
        "H.G.INFRA ENGINEERING LTD": "HGINFRA.NS",
        
        # --- Banking, Financial Services & Insurance ---
        "HDFC BANK LTD": "HDFCBANK.NS",
        "HDFC BANK LIMITED": "HDFCBANK.NS",
        "ICICI BANK LTD": "ICICIBANK.NS",
        "ICICI BANK LIMITED": "ICICIBANK.NS",
        "STATE BANK OF INDIA": "SBIN.NS",
        "AXIS BANK LTD": "AXISBANK.NS",
        "KOTAK MAHINDRA BANK LTD": "KOTAKBANK.NS",
        
        # --- IT & Technology ---
        "TATA CONSULTANCY SERVICES LTD": "TCS.NS",
        "TCS": "TCS.NS",
        "INFOSYS LTD": "INFY.NS",
        "INFOSYS LIMITED": "INFY.NS",
        "WIPRO LTD": "WIPRO.NS",
        "HCL TECHNOLOGIES LTD": "HCLTECH.NS",
        "TECH MAHINDRA LTD": "TECHM.NS",
        
        # --- Pharma & Healthcare ---
        "SUN PHARMACEUTICAL INDUSTRIES LTD": "SUNPHARMA.NS",
        "SUN PHARMA": "SUNPHARMA.NS",
        "DR. REDDY'S LABORATORIES LTD": "DRREDDY.NS",
        "CIPLA LTD": "CIPLA.NS",
        "DIVI'S LABORATORIES LTD": "DIVISLAB.NS",
        
        # --- Metals, Energy & Commodities ---
        "TATA STEEL LTD": "TATASTEEL.NS",
        "JSW STEEL LTD": "JSWSTEEL.NS",
        "HINDALCO INDUSTRIES LTD": "HINDALCO.NS",
        "COAL INDIA LTD": "COALINDIA.NS",
        "NTPC LTD": "NTPC.NS",
        "POWER GRID CORP OF INDIA LTD": "POWERGRID.NS"
    }
    
    if original_name in KNOWN_ALIASES:
        return KNOWN_ALIASES[original_name]

    equity_df = load_equity_master()
    if equity_df is not None:
        # A. Check if the name matches the SYMBOL directly
        match_sym = equity_df[equity_df["SYMBOL"] == original_name]
        if not match_sym.empty:
            return f"{original_name}.NS"

        # B. Check exact match with NAME OF COMPANY
        match_name = equity_df[equity_df["NAME OF COMPANY"].str.strip().str.upper() == original_name]
        if not match_name.empty:
            return f"{match_name.iloc[0]['SYMBOL']}.NS"

        # C. Normalized match (handling AND vs &, stripping stopwords/punctuation)
        def clean(txt):
            txt = str(txt).upper().replace(" AND ", " & ").replace(".", "")
            txt = re.sub(r"[^A-Z0-9\s&]", " ", txt)
            stopwords = {"LTD", "LIMITED", "INC", "CORP", "CORPORATION", "CO", "COMPANY"}
            return "".join([w for w in txt.split() if w not in stopwords])

        clean_original = clean(original_name)
        
        for _, row in equity_df.iterrows():
            official_name = str(row["NAME OF COMPANY"])
            if clean(official_name) == clean_original:
                return f"{row['SYMBOL']}.NS"

    # Fallback to local cache files if present
    if os.path.exists(cache_dir):
        cached_files = [f.replace(".parquet", "") for f in os.listdir(cache_dir) if f.endswith(".parquet")]
        if original_name in cached_files:
            return original_name

    # Ultimate fallback guess
    words = original_name.split()
    return f"{words[0]}.NS" if words else f"{original_name}.NS"


def parse_broker_file(file_obj):
    """Smart parser that prioritizes Symbol/ISIN over arbitrary company names."""
    try:
        if file_obj.name.lower().endswith(".xlsx"):
            df_raw = pd.read_excel(file_obj, header=None)
        else:
            df_raw = pd.read_csv(file_obj, header=None)

        header_row_idx = 0
        for i, row in df_raw.iterrows():
            row_str = " ".join([str(cell).lower() for cell in row])
            if "qty" in row_str or "quantity" in row_str or "shares" in row_str:
                header_row_idx = i
                break

        if file_obj.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file_obj, header=header_row_idx)
        else:
            df = pd.read_csv(file_obj, header=header_row_idx)

        cols = {str(c).strip().lower(): c for c in df.columns}

        t_col = (
            cols.get("symbol")
            or cols.get("instrument")
            or cols.get("stock name")
            or cols.get("company name")
        )
        q_col = (
            cols.get("qty")
            or cols.get("qty.")
            or cols.get("quantity")
            or cols.get("shares")
        )
        p_col = (
            cols.get("avg. cost")
            or cols.get("average price")
            or cols.get("avg price")
            or cols.get("avg. price")
            or cols.get("average buy price")
        )
        c_col = (
            cols.get("closing price")
            or cols.get("ltp")
            or cols.get("current market price")
            or cols.get("cmp")
        )

        if not (t_col and q_col and p_col):
            return None, f"Could not map required columns. Found headers: {list(df.columns)}"

        norm_df = pd.DataFrame()
        norm_df["Ticker"] = df[t_col].astype(str).str.strip().str.upper()
        norm_df["Quantity"] = pd.to_numeric(df[q_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
        norm_df["Average Price"] = pd.to_numeric(df[p_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

        if c_col:
            norm_df["Broker_LTP"] = pd.to_numeric(df[c_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
        else:
            norm_df["Broker_LTP"] = 0.0

        return norm_df[norm_df["Quantity"] > 0].copy(), ""
    except Exception as e:
        return None, str(e)


def compute_trade_plan(ohlcv_df: pd.DataFrame, atr_multiplier: float, hold_days: int):
    """Derive a mechanical entry/exit plan from the SAME rules used in backtest.py.

    Returns None if there isn't enough history for a 14-period ATR. This never
    predicts price or issues a recommendation - it is a deterministic function
    of (last close, ATR-14, ATR_MULTIPLIER, HOLD_DAYS), all already disclosed
    in the Trade Execution Guide / README.
    """
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 15:
        return None

    atr_series = ta.atr(ohlcv_df["High"], ohlcv_df["Low"], ohlcv_df["Close"], length=14)
    if atr_series is None or atr_series.dropna().empty:
        return None

    last_close = float(ohlcv_df["Close"].iloc[-1])
    atr_val = float(atr_series.iloc[-1])
    if not (last_close > 0) or not (atr_val > 0):
        return None

    stop_price = last_close - atr_multiplier * atr_val
    if stop_price <= 0:
        return None

    risk_per_share = last_close - stop_price
    risk_pct = (risk_per_share / last_close) * 100

    last_date = ohlcv_df.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.Timestamp(last_date)
    # Approximate: counts business days only, does not account for market holidays.
    future_sessions = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=hold_days)
    exit_by_date = future_sessions[-1] if len(future_sessions) else None

    return {
        "reference_date": last_date,
        "entry": last_close,
        "atr": atr_val,
        "stop": stop_price,
        "risk_per_share": risk_per_share,
        "risk_pct": risk_pct,
        "exit_by_date": exit_by_date,
    }


def metric_card(label, value, note="", state="", tooltip=""):
    state_html = f'<span class="dot"></span>' if state else ""
    tooltip_attr = f' title="{tooltip}" style="cursor:help;"' if tooltip else ""
    return (
        f'<div class="card"{tooltip_attr}><div class="card-label">{label}</div><div'
        f' class="card-value {state}">{state_html}{value}</div><div'
        f' class="card-note">{note}</div></div>'
    )


screener_data = load_screener_data()
updated_at = "Waiting for data"
if screener_data:
    updated_at = datetime.fromisoformat(
        screener_data.get("updated_at", datetime.now().isoformat())
    ).strftime("%d %b %Y · %I:%M %p")

st.markdown(
    f"""
<div class="hero"><div><div class="eyebrow">Nifty Quant</div><h1>Quantitative Market Screener</h1><p>Technical momentum and breakout opportunities across the Nifty 500.</p></div><div class="scan">LAST SCAN<br><b style="color:#dfe6ec">{updated_at}</b></div></div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Screener", "Portfolio Health", "📚 TA Glossary"])

# ==========================================
# TAB 1: SCREENER
# ==========================================
with tab1:
    if not screener_data:
        st.error("No screener data found. Run `python src/screener.py` first.")
    else:
        market_regime = screener_data.get("market_regime", "Unknown")
        signals = screener_data.get("signals", [])
        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()
        trade_ready_count = (
            int((signals_df.get("status") == "Trade-Ready").sum())
            if not signals_df.empty
            else 0
        )
        regime_state = (
            "good"
            if market_regime == "Bullish"
            else "bad"
            if market_regime == "Bearish"
            else "warn"
        )

        with st.expander("🧭 New here? 60-second guide to this screen", expanded=False):
            st.markdown(
                """
This tool scans 500 large Indian stocks every day and flags the ones showing signs of
upward momentum, based on price trend and trading activity — not tips or predictions.

- **Market Regime** — the overall mood of the market. When it's Bearish, this tool
  hides all setups, since buying into a falling market is historically much riskier.
- **Score (0-6)** — how many technical checks a stock passes today (trend, momentum,
  volume). Higher isn't a promise, just more confirmation.
- **Status** — *Trade-Ready* stocks pass the most checks; *Watchlist* stocks are
  earlier / less confirmed.
- **Entry & Exit Plan** — click any stock below to see a suggested stop-loss level
  and time-based exit, calculated the same way this strategy was historically tested.

Every technical term (RSI, EMA, ATR, MACD...) has a **hover tooltip** — just rest your
cursor over any column header, card, or badge. For full definitions, see the
**📚 TA Glossary** tab above.

**Nothing here is financial advice** — it's a rules-based educational tool. Always do
your own research before trading.
                """
            )

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            metric_card(
                "Market Regime",
                market_regime.upper(),
                "Nifty 50 vs 200-day EMA · 3-session confirmation",
                regime_state,
                "Is the overall Indian market currently trending up (Bullish), down "
                "(Bearish), or unclear? This tool only looks for buy setups when the "
                "market itself is healthy.",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card(
                "Stocks Scanned",
                f'{screener_data.get("total_scanned",0):,}',
                "Nifty 500 universe",
                "",
                "How many of India's top 500 stocks were checked in today's scan.",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            metric_card(
                "Opportunities",
                screener_data.get("total_signals", 0),
                f"Quant score ≥ {WATCHLIST_MIN_SCORE}",
                "",
                "Stocks that passed at least "
                f"{WATCHLIST_MIN_SCORE} of the 6 technical checks today — see the "
                "table below.",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            metric_card(
                "Trade Ready",
                trade_ready_count,
                f"Quant score {TRADE_READY_MIN_SCORE}–6",
                "good" if trade_ready_count else "",
                "The strongest subset of Opportunities — stocks confirming "
                f"{TRADE_READY_MIN_SCORE} or more of the 6 checks at once.",
            ),
            unsafe_allow_html=True,
        )

        if market_regime == "Bearish":
            st.warning(
                "📉 **Market regime is bearish** — long (buy) setups are hidden right now "
                "because Nifty 50 has closed below its 200-day trend average for "
                "three sessions in a row. Historically, buying into a falling market "
                "is riskier, so this screener pauses rather than force a signal."
            )
        elif market_regime == "Unknown":
            st.warning("Market regime data is unavailable — long setup classification is suspended.")

        st.markdown(
            '<div class="section-kicker">Opportunity scanner</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Signals")

        if not signals_df.empty:
            f1, f2, f3 = st.columns([1.2, 1.2, 2.6])
            status_filter = f1.multiselect(
                "Status",
                ["Trade-Ready", "Watchlist"],
                default=["Trade-Ready", "Watchlist"],
                help=(
                    "Trade-Ready = strongest setups (score "
                    f"{TRADE_READY_MIN_SCORE}-6). Watchlist = still forming, worth "
                    f"keeping an eye on (score {WATCHLIST_MIN_SCORE}-{TRADE_READY_MIN_SCORE - 1})."
                ),
            )
            
            # Dynamically build score dropdown starting from WATCHLIST_MIN_SCORE
            score_options = list(range(WATCHLIST_MIN_SCORE, 7))
            min_score = f2.selectbox(
                "Minimum score",
                score_options,
                index=0,
                help=(
                    "Each stock earns up to 6 points, one per technical rule it "
                    "passes today. Raise this to see only stocks confirming more "
                    "rules at once — historically stronger setups, but fewer results."
                ),
            )
            search = f3.text_input(
                "Find ticker",
                placeholder="Search RELIANCE, INFY…",
                help="Type any part of an NSE symbol to narrow the list below.",
            )

            filtered = signals_df[
                signals_df["status"].isin(status_filter)
                & (signals_df["score"] >= min_score)
            ].copy()
            if search:
                filtered = filtered[
                    filtered["ticker"].str.contains(search, case=False, na=False)
                ]
            filtered = filtered.sort_values(["score", "volume_ratio"], ascending=[False, False])

            # Safety check: generate triggers_str if missing from JSON
            if (
                "triggers_str" not in filtered.columns
                and "triggers" in filtered.columns
            ):
                filtered["triggers_str"] = filtered["triggers"].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else ""
                )

            desired_cols = [
                "ticker",
                "status",
                "score",
                "close",
                "rsi_14",
                "volume_ratio",
                "pct_above_50",
                "triggers_str",
            ]
            display_cols = [col for col in desired_cols if col in filtered.columns]

            col_config = {
                "ticker": st.column_config.TextColumn(
                    "Ticker", help="The stock's NSE trading symbol — search for this on your broker app."
                ),
                "status": st.column_config.TextColumn(
                    "Status",
                    help=f"Trade-Ready = strongest setup (Score {TRADE_READY_MIN_SCORE}-6). Watchlist = still forming (Score {WATCHLIST_MIN_SCORE}-{TRADE_READY_MIN_SCORE-1}).",
                ),
                "score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=6,
                    format="%d / 6",
                    help="How many of the 6 technical rules (trend, momentum, volume) this stock passes today. Higher = more confirmation, not a guarantee.",
                ),
                "close": st.column_config.NumberColumn(
                    "Price", format="₹%.2f", help="Most recent closing price on the NSE."
                ),
                "rsi_14": st.column_config.NumberColumn(
                    "RSI",
                    format="%.1f",
                    help="Relative Strength Index (0-100): measures recent buying vs. selling pressure. This screener looks for 60-70, a 'strong but not overheated' zone. Above 70 is often called overbought.",
                ),
                "volume_ratio": st.column_config.NumberColumn(
                    "Volume",
                    format="%.2fx",
                    help="Today's shares traded vs. the normal 20-day average. 2x+ means unusually high interest in the stock today.",
                ),
                "pct_above_50": st.column_config.NumberColumn(
                    "Above 50 EMA",
                    format="%+.2f%%",
                    help="How far the price sits above its 50-day trend average. The screener skips anything over 15% above, to avoid chasing a stock that's already run up a lot.",
                ),
                "triggers_str": st.column_config.TextColumn(
                    "Technical triggers", help="The specific rules this stock passed today. Hover a badge in 'Stock analysis' below for a plain-English explanation of each one.",
                ),
            }
            st.caption(
                "💡 Not sure what a column means? Hover over any column header for an "
                "explanation, or check the **📚 TA Glossary** tab above."
            )
            st.dataframe(
                filtered[display_cols],
                column_config=col_config,
                hide_index=True,
                width="stretch",
                height=min(500, 74 + 35 * max(len(filtered), 1)),
            )

            st.markdown(
                '<div class="section-kicker">Stock analysis</div>',
                unsafe_allow_html=True,
            )
            a, b = st.columns([2, 1])
            selected_stock = a.selectbox(
                "Inspect stock",
                options=(
                    filtered["ticker"].tolist()
                    if not filtered.empty
                    else signals_df["ticker"].tolist()
                ),
                help="Pick any stock from the filtered list to see its chart, triggers, and entry/exit plan below.",
            )
            period = b.segmented_control(
                "Chart range",
                options=["3M", "6M", "1Y"],
                default="6M",
                help="How much price history to display in the chart below.",
            )

            row = signals_df.loc[signals_df["ticker"] == selected_stock].iloc[0]
            status_class = "good" if row.get("status") == "Trade-Ready" else "warn"
            status_tooltip = (
                f"Confirms {TRADE_READY_MIN_SCORE}+ of 6 checks — the strongest setups."
                if row.get("status") == "Trade-Ready"
                else f"Confirms {WATCHLIST_MIN_SCORE}-{TRADE_READY_MIN_SCORE-1} of 6 checks — still forming."
            )
            triggers = row.get("triggers", [])
            trigger_html = "".join(
                f'<span class="pill" title="{TRIGGER_HELP.get(t, t)}" '
                f'style="cursor:help;">✓ {t}</span>'
                for t in triggers
            )
            st.markdown(
                f"""<div class="stock-head"><div class="stock-title">{selected_stock}"""
                f""" <span class="{status_class}" title="{status_tooltip}" """
                f"""style="font-size:.8rem;margin-left:.5rem;cursor:help;">● {row.get("status","")}</span></div>"""
                f"""<div class="stock-meta">₹{row.get("close",0):,.2f} &nbsp; · &nbsp;"""
                f""" <span title="How many of the 6 technical checks this stock passes today." style="cursor:help;">Score"""
                f""" {int(row.get("score",0))}/6</span> &nbsp; · &nbsp; <span title="Relative Strength Index: momentum gauge 0-100. This screener looks for the 60-70 'strong but not overheated' zone." style="cursor:help;">RSI"""
                f""" {row.get("rsi_14",0):.1f}</span> &nbsp; · &nbsp; <span title="Today's trading volume vs. the normal 20-day average." style="cursor:help;">Volume"""
                f""" {row.get("volume_ratio",0):.2f}×</span> &nbsp; · &nbsp;"""
                f""" <span title="Distance of price above its 50-day trend average." style="cursor:help;">{row.get("pct_above_50",0):+.2f}% above 50 EMA</span></div><div"""
                f""" style="margin-top:.65rem">{trigger_html}</div></div>""",
                unsafe_allow_html=True,
            )

            ohlcv_df = load_stock_ohlcv(selected_stock)
            if ohlcv_df is not None:
                ohlcv_df = ohlcv_df.copy()
                ohlcv_df["EMA_50"] = ohlcv_df["Close"].ewm(span=50, adjust=False).mean()
                ohlcv_df["EMA_200"] = (
                    ohlcv_df["Close"].ewm(span=200, adjust=False).mean()
                )
                days = {"3M": 90, "6M": 180, "1Y": 365}.get(period, 180)
                chart_df = ohlcv_df.tail(days)

                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(
                        x=chart_df.index,
                        open=chart_df["Open"],
                        high=chart_df["High"],
                        low=chart_df["Low"],
                        close=chart_df["Close"],
                        name="Price",
                        increasing_line_color="#38d996",
                        decreasing_line_color="#ff6470",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=chart_df.index,
                        y=chart_df["EMA_50"],
                        mode="lines",
                        name="50 EMA",
                        line=dict(color="#f2b84b", width=1.4),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=chart_df.index,
                        y=chart_df["EMA_200"],
                        mode="lines",
                        name="200 EMA",
                        line=dict(color="#6aa9ff", width=1.6),
                    )
                )
                trade_plan = compute_trade_plan(ohlcv_df, ATR_MULTIPLIER, HOLD_DAYS)

                if trade_plan:
                    fig.add_hline(
                        y=trade_plan["stop"],
                        line=dict(color="#ff6672", width=1.2, dash="dash"),
                        annotation_text="Model stop",
                        annotation_position="bottom right",
                        annotation_font_color="#ff6672",
                    )
                    fig.add_hline(
                        y=trade_plan["entry"],
                        line=dict(color="#73aaff", width=1, dash="dot"),
                        annotation_text="Reference entry",
                        annotation_position="top right",
                        annotation_font_color="#73aaff",
                    )

                fig.update_layout(
                    height=560,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor="#0b0f14",
                    plot_bgcolor="#0b0f14",
                    font=dict(color="#aeb9c4"),
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.03, x=0),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(title="Price (INR)", gridcolor="#1b2530"),
                )
                st.plotly_chart(
                    fig, width="stretch", config={"displayModeBar": False}
                )

                st.markdown(
                    '<div class="section-kicker" style="margin-top:'
                    ' 1.5rem;">Entry &amp; Exit Plan</div>',
                    unsafe_allow_html=True,
                )

                if not trade_plan:
                    st.info(
                        "Not enough cached history to compute a 14-period ATR for"
                        f" {selected_stock}, so no entry/exit plan is shown."
                    )
                else:
                    st.caption(
                        f"Mechanically derived from the {selected_stock} cache as of "
                        f"{trade_plan['reference_date'].strftime('%d %b %Y')} using the same "
                        f"rules tested in backtest.py — not a recommendation to trade."
                    )
                    p1, p2, p3, p4 = st.columns(4)
                    p1.markdown(
                        metric_card(
                            "Reference Entry",
                            f"₹{trade_plan['entry']:,.2f}",
                            "Last close · tested model enters at the next session's open.",
                            "",
                            "The backtest enters at tomorrow's open, which isn't known yet — this is the closest available reference.",
                        ),
                        unsafe_allow_html=True,
                    )
                    p2.markdown(
                        metric_card(
                            "Model Stop-Loss",
                            f"₹{trade_plan['stop']:,.2f}",
                            f"{ATR_MULTIPLIER}× ATR(14) = ₹{trade_plan['atr']:.2f} below entry",
                            "bad",
                            "Gap-through opens are filled at the weaker opening price in the backtest, not this exact level.",
                        ),
                        unsafe_allow_html=True,
                    )
                    p3.markdown(
                        metric_card(
                            "Risk / Share",
                            f"₹{trade_plan['risk_per_share']:,.2f}",
                            f"{trade_plan['risk_pct']:.2f}% of entry price",
                            "warn",
                            "Distance from reference entry to the model stop-loss.",
                        ),
                        unsafe_allow_html=True,
                    )
                    exit_label = (
                        trade_plan["exit_by_date"].strftime("%d %b %Y")
                        if trade_plan["exit_by_date"] is not None
                        else "N/A"
                    )
                    p4.markdown(
                        metric_card(
                            "Time-Exit By",
                            exit_label,
                            f"If the stop isn't hit within {HOLD_DAYS} sessions (approx., excludes holidays)",
                            "",
                            "The tested model force-exits at the close of the Nth holding session.",
                        ),
                        unsafe_allow_html=True,
                    )

                    with st.expander("Position-size calculator"):
                        st.caption(
                            "Answers 'how many shares should I buy?' based on how much "
                            "money you're okay losing if the stop-loss is hit — a core "
                            "risk-management habit, not a return prediction."
                        )
                        cc1, cc2, cc3 = st.columns(3)
                        capital = cc1.number_input(
                            "Capital allocated (₹)",
                            min_value=0.0,
                            value=100000.0,
                            step=5000.0,
                            help="The total money you're setting aside for this trade idea (not your whole portfolio).",
                        )
                        risk_pct_input = cc2.number_input(
                            "Risk per trade (%)",
                            min_value=0.1,
                            max_value=100.0,
                            value=1.0,
                            step=0.25,
                            help="Common guidance is to risk only 1-2% of capital per trade, so a handful of losses in a row won't hurt badly.",
                        )
                        risk_amount = capital * (risk_pct_input / 100)
                        shares = int(risk_amount // trade_plan["risk_per_share"]) if trade_plan["risk_per_share"] > 0 else 0
                        position_value = shares * trade_plan["entry"]
                        cc3.metric(
                            "Suggested Qty",
                            f"{shares:,} shares",
                            help="Risk amount ÷ risk-per-share, rounded down. Not a recommendation to buy — just the math for your inputs.",
                        )
                        st.caption(
                            f"₹{risk_amount:,.0f} at risk (stop hit) · position size ≈ ₹{position_value:,.0f} "
                            f"({(position_value / capital * 100) if capital else 0:.1f}% of allocated capital)."
                        )

                st.markdown(
                    '<div class="section-kicker" style="margin-top:'
                    ' 1.5rem;">Trade Execution Guide</div>',
                    unsafe_allow_html=True,
                )
                st.info(
                    f"Tested model: enter at the next session's open after a score ≥ {WATCHLIST_MIN_SCORE}; "
                    f"use a {ATR_MULTIPLIER} × ATR stop (with gap-through stops filled at the open); "
                    f"otherwise exit at the close of the {HOLD_DAYS}th holding session. "
                    "Backtest assumptions include configurable slippage and transaction costs."
                )
            else:
                st.info(f"No OHLCV cache found for {selected_stock}.")
        else:
            st.info("No actionable signals found for this scan.")

# ==========================================
# TAB 2: PORTFOLIO HEALTH
# ==========================================
with tab2:
    st.markdown(
        '<div class="section-kicker">Portfolio intelligence</div>',
        unsafe_allow_html=True,
    )
    st.header("Portfolio Health Analyzer")
    st.caption(
        "Upload your holdings and this checks each stock against its 200-day trend "
        "average — a simple, widely-used way to see if a stock is in a long-term "
        "uptrend or downtrend. Hover any column header below for details."
    )

    st.info(
        "🔒 **Privacy First:** Upload your Groww (or standard broker) Holdings"
        " CSV or Excel file here. It is processed for this session and is not"
        " intentionally saved by this application."
    )

    uploaded_file = st.file_uploader(
        "Upload Holdings (CSV or XLSX)",
        type=["csv", "xlsx"],
        help=(
            "Export your holdings from your broker app (usually under "
            "'Portfolio' or 'Holdings' → Export/Download) and upload that file here."
        ),
    )

    if uploaded_file is not None:
        port_df, err = parse_broker_file(uploaded_file)

        if port_df is None or port_df.empty:
            st.error(
                "Failed to parse file. Ensure it has Ticker/Stock Name, Quantity, and"
                f" Average Price columns. Details: {err}"
            )
        else:
            port_stats = []
            for _, row in port_df.iterrows():
                company_name = row["Ticker"]
                avg_price = row["Average Price"]
                qty = row["Quantity"]
                broker_ltp = row.get("Broker_LTP", 0.0)

                matched_ticker = find_cached_ticker(company_name, CACHE_DIR)

                stats = {
                    "Stock": company_name,
                    "Matched Symbol": (
                        matched_ticker
                        if matched_ticker != company_name
                        else "Not in Cache"
                    ),
                    "Quantity": qty,
                    "Buy Price": avg_price,
                    "LTP": broker_ltp if broker_ltp > 0 else None,
                    "Status": "Unknown (No Data)",
                    "P&L %": None,
                }

                close_data = pd.Series(dtype=float)
                ohlcv = load_stock_ohlcv(matched_ticker)
                if ohlcv is not None and not ohlcv.empty:
                    close_data = ohlcv["Close"]
                    if isinstance(close_data, pd.DataFrame):
                        close_data = close_data.iloc[:, 0]
                    close_data = pd.to_numeric(close_data, errors="coerce").dropna()
                    latest_close = float(close_data.iloc[-1]) if not close_data.empty else float("nan")
                    ema_200 = close_data.ewm(span=200, adjust=False).mean()
                    latest_ema_200 = float(ema_200.iloc[-1]) if not ema_200.empty else float("nan")

                    stats["LTP"] = latest_close

                    if latest_close < latest_ema_200:
                        stats["Status"] = "⚠️ Below 200 EMA"
                    else:
                        stats["Status"] = "✅ Healthy Trend"

                if close_data.empty:
                    stats["LTP"] = broker_ltp if broker_ltp > 0 else None
                    stats["Status"] = "Unknown (Invalid Price Data)"

                if stats["LTP"] is not None and avg_price > 0:
                    stats["P&L %"] = ((stats["LTP"] - avg_price) / avg_price) * 100

                port_stats.append(stats)

            res_df = pd.DataFrame(port_stats)

            col_conf = {
                "Stock": st.column_config.TextColumn(
                    "Stock Name", help="The name exactly as it appeared in your uploaded file."
                ),
                "Matched Symbol": st.column_config.TextColumn(
                    "Ticker",
                    help="The official NSE trading symbol we matched this holding to, using EQUITY_L.csv. Shows 'Not in Cache' if we couldn't confidently match it.",
                ),
                "Quantity": st.column_config.NumberColumn(
                    "Qty", help="Number of shares you hold, as read from your file."
                ),
                "Buy Price": st.column_config.NumberColumn(
                    "Buy Price", format="₹%.2f", help="Your average purchase price per share, as read from your file."
                ),
                "LTP": st.column_config.NumberColumn(
                    "LTP", format="₹%.2f", help="Last Traded Price — the most recent closing price we found for this stock."
                ),
                "P&L %": st.column_config.NumberColumn(
                    "P&L (%)",
                    format="%+.2f%%",
                    help="Your unrealized profit or loss: (Current Price − Buy Price) ÷ Buy Price. Green is a gain, red is a loss.",
                ),
                "Status": st.column_config.TextColumn(
                    "Technical Health",
                    help="✅ Healthy Trend = price is above its 200-day average (long-term uptrend). ⚠️ Below 200 EMA = price is below it (long-term downtrend, worth reviewing).",
                ),
            }
            st.caption(
                "🟢 **Healthy Trend** = price above its 200-day average · 🟠 **Below 200 EMA** "
                "= price below it, historically a weaker long-term trend. Hover any column "
                "header for details."
            )

            breakdowns = res_df[res_df["Status"] == "⚠️ Below 200 EMA"]
            if not breakdowns.empty:
                st.error(
                    f"🚨 **Worth a look:** {len(breakdowns)} of your holdings are trading"
                    " below their 200-day average — a simple sign of a long-term"
                    " downtrend, not a signal to sell."
                )
                st.dataframe(
                    breakdowns, column_config=col_conf, hide_index=True, width="stretch"
                )
            else:
                st.success(
                    "✅ None of your recognized holdings are trading below"
                    " their 200-day average — all in long-term uptrends by this measure."
                )

            st.divider()

            signals_df = (
                pd.DataFrame(screener_data.get("signals", []))
                if screener_data
                else pd.DataFrame()
            )
            if not signals_df.empty:
                bullish_held = signals_df[
                    signals_df["ticker"].isin(res_df["Matched Symbol"].tolist())
                ]
                st.subheader("🔥 Active Screener Setups in Your Portfolio")
                if not bullish_held.empty:
                    st.write(
                        "These stocks you currently own are triggering fresh quantitative"
                        " breakout signals today:"
                    )
                    st.dataframe(
                        bullish_held[["ticker", "status", "score", "triggers_str"]],
                        column_config={
                            "ticker": st.column_config.TextColumn(
                                "Ticker", help="NSE trading symbol."
                            ),
                            "status": st.column_config.TextColumn(
                                "Status",
                                help=f"Trade-Ready = strongest setup (Score {TRADE_READY_MIN_SCORE}-6). Watchlist = still forming.",
                            ),
                            "score": st.column_config.ProgressColumn(
                                "Score",
                                min_value=0,
                                max_value=6,
                                format="%d/6",
                                help="How many of the 6 technical checks this stock passes today.",
                            ),
                            "triggers_str": st.column_config.TextColumn(
                                "Triggers Met", help="The specific rules this stock passed today."
                            ),
                        },
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info(
                        "None of your current holdings triggered a new screener setup"
                        " today."
                    )

            st.divider()

            # Align the title and slider side-by-side
            c_title, c_slider = st.columns([3, 1])
            
            with c_title:
                st.subheader("All Parsed Holdings")
                st.caption("Every row we read from your uploaded file, matched and priced where possible.")
                
            with c_slider:
                table_height = st.slider(
                    "Table Height (px)",
                    min_value=300,
                    max_value=900,
                    value=400,
                    step=50,
                    key="parsed_holdings_height",
                    help="Drag to show more or fewer rows without scrolling.",
                )
            
            # Pass the slider value directly to the height parameter
            st.dataframe(
                res_df, 
                column_config=col_conf, 
                hide_index=True, 
                width="stretch",
                height=table_height
            )

# ==========================================
# TAB 3: TA GLOSSARY (Educational)
# ==========================================
with tab3:
    st.markdown(
        '<div class="section-kicker">Educational Reference</div>',
        unsafe_allow_html=True,
    )
    st.header("Technical Analysis (TA) Glossary")
    st.write(
        "A beginner-friendly guide to understanding the quantitative metrics"
        " used in this screener."
    )

    col_g1, col_g2 = st.columns(2, gap="large")

    with col_g1:
        st.subheader("📈 Trend Indicators")
        st.markdown(
            """
            <div class="glossary-term">EMA (Exponential Moving Average)</div>
            <div class="glossary-def">A type of moving average that places a greater weight on recent data points. We use the <b>50 EMA</b> for short/medium-term trends and the <b>200 EMA</b> to determine the overall long-term trend of a stock.</div>

            <div class="glossary-term">Breakout</div>
            <div class="glossary-def">When a stock's price moves above a major resistance level (like the 200 EMA) with increased volume. This often indicates the start of a new upward trend.</div>

            <div class="glossary-term">Volume</div>
            <div class="glossary-def">The number of shares traded in a day. We flag a <b>2x Volume Spike</b> (double the normal 20-day average) as unusually high participation; volume alone does not identify the participants or their direction.</div>
            """,
            unsafe_allow_html=True,
        )

    with col_g2:
        st.subheader("⚡ Momentum Indicators")
        st.markdown(
            """
            <div class="glossary-term">RSI (Relative Strength Index)</div>
            <div class="glossary-def">A momentum oscillator that ranges from 0 to 100.
            <ul>
                <li><b>Above 70:</b> Traditionally considered overbought; this is not a prediction of an immediate drop.</li>
                <li><b>Below 30:</b> Traditionally considered oversold; this is not a prediction of an immediate bounce.</li>
                <li><b>60 to 70:</b> The range used by this screener as one confirmation of recent upward momentum.</li>
            </ul>
            </div>

            <div class="glossary-term">MACD (Moving Average Convergence Divergence)</div>
            <div class="glossary-def">A trend-following momentum indicator. A <b>Bullish Cross</b> happens when the MACD line crosses above the signal line, acting as a buy signal.</div>
            
            <div class="glossary-term">Overextended (% Above 50 EMA)</div>
            <div class="glossary-def">The screener excludes stocks more than 15% above their 50 EMA to avoid chasing extended prices. This is a risk filter, not a forecast of a correction.</div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("🛡️ Risk Management & Trading Rules")
    st.markdown(
        f"""
        <div class="glossary-term">1. Tested Entry</div>
        <div class="glossary-def">
        The historical model enters at the next trading session's open after a qualifying six-point screener score of at least {WATCHLIST_MIN_SCORE}. It does not test pullback or breakout-retest entries.
        </div>
        
        <div class="glossary-term">2. Tested Stop-Loss</div>
        <div class="glossary-def">
        The model places a stop at <b>{ATR_MULTIPLIER} × ATR</b> below the executed entry price. If the next session opens below that stop, the simulation exits at the weaker opening price to account for a gap:
        <ul>
            <li>ATR is calculated from the signal day.</li>
            <li>Intraday stop hits exit at the stop price.</li>
            <li>Gap-through stops exit at that session's open.</li>
        </ul>
        </div>

        <div class="glossary-term">3. Tested Holding Period & Costs</div>
        <div class="glossary-def">
        A position that does not stop out exits at the close of the <b>{HOLD_DAYS}th holding session</b>. Backtest returns include configurable entry/exit slippage and round-trip transaction costs.
        <ul>
            <li>No 7–10 day time stop is modeled.</li>
            <li>No partial-profit or +10% target rule is modeled.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
