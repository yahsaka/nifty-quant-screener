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
        "Institutional Interest: Today's trading volume is more than double the"
        " normal 20-day average."
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
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      return df if not df.empty else None
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

  if company_name in cached_files:
    return company_name

  clean_name = re.sub(r"[^a-zA-Z0-9]", "", company_name).lower()

  best_match = None
  for ticker in cached_files:
    clean_ticker = re.sub(r"[^a-zA-Z0-9]", "", ticker).lower()
    if clean_ticker.endswith("ns") and len(clean_ticker) > 2:
      clean_ticker_base = clean_ticker[:-2]
    else:
      clean_ticker_base = clean_ticker

    if len(clean_ticker_base) < 3:
      continue

    if clean_ticker_base in clean_name:
      if best_match is None or len(clean_ticker_base) > len(best_match[1]):
        best_match = (ticker, clean_ticker_base)

  return best_match[0] if best_match else company_name


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
            "Nifty 50 vs 50-day EMA",
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
          " remains below its 50-day EMA."
      )

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
          use_container_width=True,
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
            fig, use_container_width=True, config={"displayModeBar": False}
        )

        latest_close = chart_df["Close"].iloc[-1]
        latest_low = chart_df["Low"].iloc[-1]
        latest_ema50 = chart_df["EMA_50"].iloc[-1]
        latest_ema200 = chart_df["EMA_200"].iloc[-1]
        swing_low = chart_df["Low"].tail(10).min()
        target_10 = latest_close * 1.10

        st.markdown(
            '<div class="section-kicker" style="margin-top:'
            ' 1.5rem;">Trade Execution Guide</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 0.5rem;">
                    <div class="card" style="min-height: auto;">
                        <div class="card-label" style="color:var(--green);">1. Entry Timing (Breakout Retest)</div>
                        <div class="card-note" style="margin-top:0.8rem; line-height:1.5;">
                            Wait 3-5 days for a pullback on lower volume before buying.
                            <br><br><b>Suggested Entry Zone:</b><br>
                            <span style="font-size:1.1rem; color:var(--text); font-weight:600;">₹{latest_ema200:,.2f}</span> <span style="font-size:0.8rem;">(200 EMA)</span> to <span style="font-size:1.1rem; color:var(--text); font-weight:600;">₹{latest_close * 0.97:,.2f}</span> <span style="font-size:0.8rem;">(-3% dip)</span>
                        </div>
                    </div>
                    <div class="card" style="min-height: auto;">
                        <div class="card-label" style="color:var(--red);">2. Structural Stop-Loss</div>
                        <div class="card-note" style="margin-top:0.8rem; line-height:1.5;">
                            Place your stop-loss manually below a structural support level:
                            <br><br>
                            • <b>Below 50 EMA:</b> <span style="color:var(--text); font-weight:600;">₹{latest_ema50 * 0.99:,.2f}</span><br>
                            • <b>Recent Swing Low:</b> <span style="color:var(--text); font-weight:600;">₹{swing_low * 0.99:,.2f}</span><br>
                            • <b>Breakout Bottom:</b> <span style="color:var(--text); font-weight:600;">₹{latest_low * 0.99:,.2f}</span>
                        </div>
                    </div>
                    <div class="card" style="min-height: auto;">
                        <div class="card-label" style="color:var(--amber);">3. Holding Period & Profit</div>
                        <div class="card-note" style="margin-top:0.8rem; line-height:1.5;">
                            <b>Time Stop:</b> If it doesn't move in 7-10 days, sell it.
                            <br><br><b>Scaling Out (+10% Target):</b><br>
                            Sell half your position at <span style="font-size:1.1rem; color:var(--green); font-weight:600;">₹{target_10:,.2f}</span>, then move stop-loss to entry price.
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
      " CSV or Excel file here. It is processed 100% in your browser memory and"
      " never saved."
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

        ohlcv = load_stock_ohlcv(matched_ticker)
        if ohlcv is not None and not ohlcv.empty:
          latest_close = float(ohlcv["Close"].iloc[-1])
          ohlcv["EMA_200"] = ohlcv["Close"].ewm(span=200, adjust=False).mean()
          latest_ema_200 = float(ohlcv["EMA_200"].iloc[-1])

          stats["LTP"] = latest_close

          if latest_close < latest_ema_200:
            stats["Status"] = "⚠️ Below 200 EMA"
          else:
            stats["Status"] = "✅ Healthy Trend"

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
            breakdowns, column_config=col_conf, hide_index=True, use_container_width=True
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
              use_container_width=True,
          )
        else:
          st.info(
              "None of your current holdings triggered a new screener setup"
              " today."
          )

      st.divider()

      st.subheader("All Parsed Holdings")
      st.dataframe(
          res_df, column_config=col_conf, hide_index=True, use_container_width=True
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
        <div class="glossary-def">The number of shares traded in a day. We look for a <b>2x Volume Spike</b> (double the normal 20-day average). This is a massive footprint left by institutional buyers.</div>
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
            <li><b>Above 70:</b> Considered "Overbought" (due for a drop).</li>
            <li><b>Below 30:</b> Considered "Oversold" (due for a bounce).</li>
            <li><b>60 to 70:</b> The "Goldilocks Zone" where a stock has strong upward momentum but hasn't become dangerously over-extended yet.</li>
        </ul>
        </div>

        <div class="glossary-term">MACD (Moving Average Convergence Divergence)</div>
        <div class="glossary-def">A trend-following momentum indicator. A <b>Bullish Cross</b> happens when the MACD line crosses above the signal line, acting as a buy signal.</div>
        
        <div class="glossary-term">Overextended (% Above 50 EMA)</div>
        <div class="glossary-def">If a stock is trading more than 15% above its 50 EMA, the rubber band is stretched too tight, and a sharp correction is highly likely. The screener filters these out automatically.</div>
        """,
        unsafe_allow_html=True,
    )

  st.divider()

  st.subheader("🛡️ Risk Management & Trading Rules")
  st.markdown(
      """
    <div class="glossary-term">1. Entry Timing (The "Breakout Retest")</div>
    <div class="glossary-def">
    Our backtesting data proved that buying the exact top of a massive volume breakout often leads to immediate losses (whipsaws). Why? Because the stock is temporarily exhausted and will likely pull back. 
    <br><b>Action:</b> Do not buy immediately. Add the "Trade-Ready" stock to your watchlist and wait 3-5 days for it to "retest" the breakout level on lower volume before entering.
    </div>
    
    <div class="glossary-term">2. Setting a Stop-Loss (Where to exit a losing trade)</div>
    <div class="glossary-def">
    Never use a rigid percentage (like -5%) or a pure volatility calculation to set a stop-loss in the Indian mid-cap market; normal intraday noise will randomly shake you out. Instead, place your stop-loss manually below a <b>structural support level</b> based on the chart:
    <ul>
        <li>Just below the 50-day EMA line.</li>
        <li>Just below the recent "swing low" (the bottom of the most recent dip).</li>
        <li>Just below the absolute bottom of the massive green breakout candle.</li>
    </ul>
    </div>

    <div class="glossary-term">3. Holding Period & Taking Profit</div>
    <div class="glossary-def">
    Swing trades typically last anywhere from <b>5 to 20 trading sessions</b> (1 to 4 weeks). 
    <ul>
        <li><b>Time Stops:</b> If a stock doesn't move in your favor within 7-10 days of a breakout, the momentum is dead. Sell it and free up your capital for a better setup.</li>
        <li><b>Scaling Out:</b> If a stock shoots up 10% in just a few days, sell half of your position to lock in gains, and move your stop-loss up to your entry price for the remaining shares. This guarantees a risk-free trade.</li>
    </ul>
    </div>
    """,
      unsafe_allow_html=True,
  )