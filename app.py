import os
import re
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

st.set_page_config(
    page_title="Nifty Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CACHE_DIR = "data/ohlcv_cache"
SCREENER_JSON = "data/latest_screener_results.json"

# Dictionary to map technical triggers to plain-English explanations for the UI
TRIGGER_HELP = {
    "ABOVE_EMA_200": (
        "Long-term trend is positive (Price is trading above its 200-day"
        " average)."
    ),
    "EMA_200_BREAKOUT": (
        "Fresh Breakout: Price just crossed above the 200-day average today."
        " Strong reversal signal."
    ),
    "ABOVE_EMA_50": (
        "Medium-term trend is positive (Price is trading above its 50-day"
        " average)."
    ),
    "VOLUME_SPIKE_2X": (
        "Unusual activity: today's volume is more than double its 20-day"
        " average. Volume alone does not identify who is buying or selling."
    ),
    "RSI_60_TO_70": (
        "Goldilocks Momentum: RSI is between 60-70 (Strong momentum, but not"
        " dangerously overbought)."
    ),
    "MACD_BULLISH_CROSS": (
        "MACD line crossed above the signal line today, indicating an immediate"
        " shift to upward momentum."
    ),
}

st.markdown(
    """
<style>
:root { --bg:#0b0f14; --card:#111821; --line:#24303d; --muted:#8b98a7; --text:#f2f5f7; --green:#38d996; --amber:#f2b84b; --red:#ff6470; }
.stApp { background:var(--bg); color:var(--text); }
.block-container { max-width:1480px; padding-top:1.5rem; padding-bottom:3rem; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { letter-spacing:-.025em; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; margin:.2rem 0 1.2rem; }
.eyebrow { color:var(--green); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { margin:.25rem 0 0; font-size:2rem; }
.hero p { color:var(--muted); margin:.3rem 0 0; }
.scan { color:var(--muted); font-size:.82rem; text-align:right; }
.card { background:linear-gradient(180deg,#121a24,#0f161e); border:1px solid var(--line); border-radius:14px; padding:1rem 1.05rem; min-height:112px; }
.card-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
.card-value { font-size:1.65rem; font-weight:750; margin:.45rem 0 .15rem; }
.card-note { color:var(--muted); font-size:.78rem; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; background:currentColor; }
.good { color:var(--green); } .warn { color:var(--amber); } .bad { color:var(--red); }
.section-kicker { color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-top:.6rem; }
.stock-head { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; margin:.5rem 0 1rem; }
.stock-title { font-size:1.35rem; font-weight:750; }
.stock-meta { color:var(--muted); font-size:.82rem; margin-top:.25rem; }
.pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.28rem .55rem; margin:.15rem .25rem .15rem 0; color:#cbd5df; font-size:.76rem; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding:.65rem 0; background:transparent; }
.stTabs [aria-selected="true"] { color:var(--green)!important; }
div[data-baseweb="select"] > div { background:#111821; border-color:var(--line); }
[data-testid="stAlert"] { border-radius:12px; }
.glossary-term { color:var(--green); font-weight:700; font-size:1.1rem; margin-bottom:0.2rem; }
.glossary-def { color:var(--text); font-size:0.95rem; line-height:1.5; margin-bottom:1.5rem; }
.glossary-def ul { margin-top: 0.3rem; margin-bottom: 0.3rem; }
.glossary-def li { margin-bottom: 0.3rem; }
/* .legacy-execution-guide { display:none !important; } */
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
        # Fallback for Streamlit Cloud: fetch on-the-fly using yfinance if cache is missing
        try:
            yf_ticker = (
                ticker
                if ticker.endswith(".NS") or "." in ticker
                else f"{ticker}.NS"
            )
            df = yf.download(yf_ticker, period="1y", progress=False)
            if df.empty:
                df = yf.download(ticker, period="1y", progress=False)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Save it temporarily so it's cached for the rest of this session
                os.makedirs(CACHE_DIR, exist_ok=True)
                df.to_parquet(file_path)
                
                return df
            return None
        except Exception:
            return None


def find_cached_ticker(company_name, cache_dir):
    """Fuzzy matching to link full broker company names to local YF ticker symbols."""
    if not os.path.exists(cache_dir):
        return company_name
        
    cached_files = [
        f.replace(".parquet", "")
        for f in os.listdir(cache_dir)
        if f.endswith(".parquet")
    ]

    # Clean the input company name, removing common corporate suffixes
    clean_name = re.sub(r"[^a-zA-Z0-9\s]", "", str(company_name)).upper()
    suffixes = [" LIMITED", " LTD", " INC", " CORP", " CORPORATION", " LTD."]
    for suffix in suffixes:
        if clean_name.endswith(suffix):
            clean_name = clean_name[: -len(suffix)].strip()
    
    clean_name_no_space = clean_name.replace(" ", "")

    # 1. Exact Match Check
    if company_name in cached_files:
        return company_name

    # 2. Heuristic Matching
    best_match = None
    for ticker in cached_files:
        clean_ticker = ticker.replace(".NS", "").upper()
        
        # A. Exact match with the cleaned name
        if clean_ticker == clean_name_no_space or clean_ticker == clean_name.replace(" ", ""):
            return ticker
            
        # B. Ticker is fully contained within the cleaned company name
        if clean_ticker in clean_name_no_space and len(clean_ticker) >= 3:
             if best_match is None or len(clean_ticker) > len(best_match[1]):
                 best_match = (ticker, clean_ticker)

    # 3. Fallback: If we can't map it to a cached Nifty 500 symbol, 
    # we should NOT return the full company name to yfinance.
    # We will try to guess the NSE symbol by taking the first word of the company name.
    if best_match:
         return best_match[0]
    else:
         # Guess ticker: Take the first word, remove non-alphanumeric, append .NS
         guessed_ticker = company_name.split(" ")[0]
         guessed_ticker = re.sub(r"[^a-zA-Z0-9]", "", guessed_ticker).upper()
         return f"{guessed_ticker}.NS"


def parse_broker_file(file_obj):
    """Smart parser that skips metadata rows and captures closing prices."""
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
            cols.get("instrument")
            or cols.get("stock name")
            or cols.get("symbol")
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
            return (
                None,
                f"Could not map required columns. Found headers: {list(df.columns)}",
            )

        norm_df = pd.DataFrame()
        norm_df["Ticker"] = df[t_col].astype(str).str.strip().str.upper()
        norm_df["Quantity"] = (
            pd.to_numeric(
                df[q_col].astype(str).str.replace(",", ""), errors="coerce"
            )
            .fillna(0)
        )
        norm_df["Average Price"] = (
            pd.to_numeric(
                df[p_col].astype(str).str.replace(",", ""), errors="coerce"
            )
            .fillna(0)
        )

        if c_col:
            norm_df["Broker_LTP"] = (
                pd.to_numeric(
                    df[c_col].astype(str).str.replace(",", ""), errors="coerce"
                )
                .fillna(0)
            )
        else:
            norm_df["Broker_LTP"] = 0.0

        return norm_df[norm_df["Quantity"] > 0].copy(), ""
    except Exception as e:
        return None, str(e)


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

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            metric_card(
                "Market Regime",
                market_regime.upper(),
                "Nifty 50 vs 200-day EMA · 3-session confirmation",
                regime_state,
                "Shows if the overall market is healthy.",
            ),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card(
                "Stocks Scanned",
                f'{screener_data.get("total_scanned",0):,}',
                "Nifty 500 universe",
                "",
                "Total stocks evaluated today.",
            ),
            unsafe_allow_html=True,
        )
        c3.markdown(
            metric_card(
                "Opportunities",
                screener_data.get("total_signals", 0),
                "Quant score ≥ 3",
                "",
                "Stocks passing strict technical filters.",
            ),
            unsafe_allow_html=True,
        )
        c4.markdown(
            metric_card(
                "Trade Ready",
                trade_ready_count,
                "Quant score 5–6",
                "good" if trade_ready_count else "",
                "High-conviction breakout setups.",
            ),
            unsafe_allow_html=True,
        )

        if market_regime == "Bearish":
            st.warning(
                "Market regime is bearish — long setups are filtered while Nifty 50"
                " has closed below its 200-day EMA for three consecutive sessions."
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
                "Status", ["Trade-Ready", "Watchlist"], default=["Trade-Ready", "Watchlist"]
            )
            min_score = f2.selectbox("Minimum score", [3, 4, 5, 6], index=0)
            search = f3.text_input("Find ticker", placeholder="Search RELIANCE, INFY…")

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
                    "Ticker", help="NSE trading symbol."
                ),
                "status": st.column_config.TextColumn(
                    "Status",
                    help="Trade-Ready (Score 5-6) or Watchlist (Score 3-4).",
                ),
                "score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=6,
                    format="%d / 6",
                    help="Quantitative strength score.",
                ),
                "close": st.column_config.NumberColumn(
                    "Price", format="₹%.2f", help="Last closing price."
                ),
                "rsi_14": st.column_config.NumberColumn(
                    "RSI",
                    format="%.1f",
                    help="Momentum indicator (60-70 is the sweet spot).",
                ),
                "volume_ratio": st.column_config.NumberColumn(
                    "Volume",
                    format="%.2fx",
                    help="Today's volume compared to the 20-day average.",
                ),
                "pct_above_50": st.column_config.NumberColumn(
                    "Above 50 EMA",
                    format="%+.2f%%",
                    help="Distance from 50-day average.",
                ),
                "triggers_str": st.column_config.TextColumn(
                    "Technical triggers", help="The rules the stock passed today."
                ),
            }
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
            )
            period = b.segmented_control(
                "Chart range", options=["3M", "6M", "1Y"], default="6M"
            )

            row = signals_df.loc[signals_df["ticker"] == selected_stock].iloc[0]
            status_class = "good" if row.get("status") == "Trade-Ready" else "warn"
            triggers = row.get("triggers", [])
            trigger_html = "".join(
                f'<span class="pill" title="{TRIGGER_HELP.get(t, t)}" '
                f'style="cursor:help;">✓ {t}</span>'
                for t in triggers
            )
            st.markdown(
                f"""<div class="stock-head"><div class="stock-title">{selected_stock}"""
                f""" <span class="{status_class}" """
                f"""style="font-size:.8rem;margin-left:.5rem">● {row.get("status","")}</span></div>"""
                f"""<div class="stock-meta">₹{row.get("close",0):,.2f} &nbsp; · &nbsp;"""
                f""" Score {int(row.get("score",0))}/6 &nbsp; · &nbsp; RSI"""
                f""" {row.get("rsi_14",0):.1f} &nbsp; · &nbsp; Volume"""
                f""" {row.get("volume_ratio",0):.2f}× &nbsp; · &nbsp;"""
                f""" {row.get("pct_above_50",0):+.2f}% above 50 EMA</div><div"""
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

                latest_close = float(chart_df["Close"].squeeze().iloc[-1])
                latest_low = float(chart_df["Low"].squeeze().iloc[-1])
                latest_ema50 = float(chart_df["EMA_50"].squeeze().iloc[-1])
                latest_ema200 = float(chart_df["EMA_200"].squeeze().iloc[-1])
                swing_low = float(chart_df["Low"].tail(10).min())
                target_10 = latest_close * 1.10

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
                    ' 1.5rem;">Trade Execution Guide</div>',
                    unsafe_allow_html=True,
                )
                st.info(
                    "Tested model: enter at the next session's open after a score ≥ 3; "
                    "use a 2 × ATR stop (with gap-through stops filled at the open); "
                    "otherwise exit at the close of the 20th holding session. "
                    "Backtest assumptions include configurable slippage and transaction costs."
                )
                st.markdown(
                    f"""
<div class="legacy-execution-guide" style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.2rem; margin-top: 1rem;">
    <!-- Card 1: Entry -->
    <div class="card" style="min-height: auto; border-top: 3px solid var(--green); display: flex; flex-direction: column;">
        <div class="card-label" style="color:var(--green); display:flex; align-items:center; gap:0.4rem;">
            <span style="font-size:1rem;">🎯</span> 1. SUGGESTED ENTRY ZONE
        </div>
        <div class="card-note" style="margin-top:0.8rem; line-height:1.5; flex-grow:1;">
            Wait for the next trading session's open. The ideal accumulation zone is between a slight pullback and major support.
        </div>
        <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--line);">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 0.3rem;">
                <span style="font-size:0.85rem; color:var(--muted);">Aggressive (-3% dip)</span>
                <span style="font-size:1.1rem; color:var(--text); font-weight:600;">₹{latest_close * 0.97:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-size:0.85rem; color:var(--muted);">Conservative (200 EMA)</span>
                <span style="font-size:1.1rem; color:var(--text); font-weight:600;">₹{latest_ema200:,.2f}</span>
            </div>
        </div>
    </div>
    <!-- Card 2: Stops -->
    <div class="card" style="min-height: auto; border-top: 3px solid var(--red); display: flex; flex-direction: column;">
        <div class="card-label" style="color:var(--red); display:flex; align-items:center; gap:0.4rem;">
            <span style="font-size:1rem;">🛑</span> 2. STRUCTURAL STOP-LOSSES
        </div>
        <div class="card-note" style="margin-top:0.8rem; line-height:1.5; flex-grow:1;">
            Consider these logical structural levels (calculated at a 1% buffer below support) to invalidate the setup.
        </div>
        <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--line);">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 0.3rem;">
                <span style="font-size:0.85rem; color:var(--muted);">Below 50 EMA</span>
                <span style="color:var(--text); font-weight:600;">₹{latest_ema50 * 0.99:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 0.3rem;">
                <span style="font-size:0.85rem; color:var(--muted);">Breakout Bottom</span>
                <span style="color:var(--text); font-weight:600;">₹{latest_low * 0.99:,.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-size:0.85rem; color:var(--muted);">Recent Swing Low</span>
                <span style="color:var(--text); font-weight:600;">₹{swing_low * 0.99:,.2f}</span>
            </div>
        </div>
    </div>
    <!-- Card 3: Targets -->
    <div class="card" style="min-height: auto; border-top: 3px solid var(--amber); display: flex; flex-direction: column;">
        <div class="card-label" style="color:var(--amber); display:flex; align-items:center; gap:0.4rem;">
            <span style="font-size:1rem;">⏳</span> 3. TRADE MANAGEMENT
        </div>
        <div class="card-note" style="margin-top:0.8rem; line-height:1.5; flex-grow:1;">
            For discretionary swing trading, secure partial profits into strength rather than holding strictly for time.
        </div>
        <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--line); padding-bottom: 0.3rem;">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 0.4rem;">
                <span style="font-size:0.85rem; color:var(--muted);">Take Profit (+10%)</span>
                <span style="font-size:1.15rem; color:var(--green); font-weight:700;">₹{target_10:,.2f}</span>
            </div>
            <span style="font-size:0.8rem; color:var(--muted); line-height: 1.4; display:block;">
                <b>Rule:</b> Sell 50% of the position at the target, then immediately move the stop-loss on the remainder to your entry price.
            </span>
        </div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
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

    st.info(
        "🔒 **Privacy First:** Upload your Groww (or standard broker) Holdings"
        " CSV or Excel file here. It is processed for this session and is not"
        " intentionally saved by this application."
    )

    uploaded_file = st.file_uploader(
        "Upload Holdings (CSV or XLSX)", type=["csv", "xlsx"]
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
                    # yfinance can return a DataFrame for Close when its columns are
                    # multi-level. Reduce it to one numeric Series before scalar access.
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
                    # Keep the row visible and use the broker price when Yahoo/cache data
                    # contains no usable Close observations.
                    stats["LTP"] = broker_ltp if broker_ltp > 0 else None
                    stats["Status"] = "Unknown (Invalid Price Data)"

                if stats["LTP"] is not None and avg_price > 0:
                    stats["P&L %"] = ((stats["LTP"] - avg_price) / avg_price) * 100

                port_stats.append(stats)

            res_df = pd.DataFrame(port_stats)

            col_conf = {
                "Stock": "Stock Name",
                "Matched Symbol": st.column_config.TextColumn(
                    "Ticker",
                    help=(
                        "The NSE symbol matched in your local Nifty 500 database."
                    ),
                ),
                "Quantity": st.column_config.NumberColumn("Qty"),
                "Buy Price": st.column_config.NumberColumn(
                    "Buy Price", format="₹%.2f"
                ),
                "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                "P&L %": st.column_config.NumberColumn("P&L (%)", format="%+.2f%%"),
                "Status": "Technical Health",
            }

            breakdowns = res_df[res_df["Status"] == "⚠️ Below 200 EMA"]
            if not breakdowns.empty:
                st.error(
                    f"🚨 **Action Required:** {len(breakdowns)} of your holdings have"
                    " broken below their 200-day EMA. These are mathematically in a"
                    " long-term downtrend."
                )
                st.dataframe(
                    breakdowns, column_config=col_conf, hide_index=True, width="stretch"
                )
            else:
                st.success(
                    "✅ Excellent! None of your recognized holdings are trading below"
                    " their 200-day EMA."
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
                            "ticker": "Ticker",
                            "status": "Status",
                            "score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=6, format="%d/6"
                            ),
                            "triggers_str": "Triggers Met",
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

            st.subheader("All Parsed Holdings")
            st.dataframe(
                res_df, column_config=col_conf, hide_index=True, width="stretch"
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
        """
        <div class="glossary-term">1. Tested Entry</div>
        <div class="glossary-def">
        The historical model enters at the next trading session's open after a qualifying six-point screener score of at least 3. It does not test pullback or breakout-retest entries.
        </div>
        
        <div class="glossary-term">2. Tested Stop-Loss</div>
        <div class="glossary-def">
        The model places a stop at <b>2 × ATR</b> below the executed entry price. If the next session opens below that stop, the simulation exits at the weaker opening price to account for a gap:
        <ul>
            <li>ATR is calculated from the signal day.</li>
            <li>Intraday stop hits exit at the stop price.</li>
            <li>Gap-through stops exit at that session's open.</li>
        </ul>
        </div>

        <div class="glossary-term">3. Tested Holding Period & Costs</div>
        <div class="glossary-def">
        A position that does not stop out exits at the close of the <b>20th holding session</b>. Backtest returns include configurable entry/exit slippage and round-trip transaction costs.
        <ul>
            <li>No 7–10 day time stop is modeled.</li>
            <li>No partial-profit or +10% target rule is modeled.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
