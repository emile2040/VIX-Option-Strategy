import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import io
import traceback
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import anthropic as _anthropic
from pricing.futures import vix_futures_price, vix_futures_term_structure
from pricing.options import black76
from data.ibkr import fetch_vix_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIX Option Strategy",
    page_icon="📈",
    layout="wide",
)

st.title("VIX Option Strategy Dashboard")

tab_pricer, tab_spreads, tab_sandbox = st.tabs(["Futures & Options Pricer", "VIX Fixed Put Strikes", "🧪 Playing Around"])

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("live", None), ("vix_history", None), ("sim_results", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Vol-of-vol mapping ────────────────────────────────────────────────────────
SIGMA_BREAKPOINTS = [(10, 0.75), (15, 0.90), (22, 1.25), (35, 1.80), (50, 3.00)]

def sigma_from_vix(v):
    if v <= SIGMA_BREAKPOINTS[0][0]:
        return SIGMA_BREAKPOINTS[0][1]
    if v >= SIGMA_BREAKPOINTS[-1][0]:
        return SIGMA_BREAKPOINTS[-1][1]
    for i in range(len(SIGMA_BREAKPOINTS) - 1):
        x0, y0 = SIGMA_BREAKPOINTS[i]
        x1, y1 = SIGMA_BREAKPOINTS[i + 1]
        if x0 <= v <= x1:
            return y0 + (y1 - y0) * (v - x0) / (x1 - x0)

def next_wednesday(trade_date, min_days):
    """First Wednesday at least min_days calendar days after trade_date."""
    target     = trade_date + datetime.timedelta(days=int(min_days))
    days_ahead = (2 - target.weekday()) % 7   # 2 = Wednesday
    return target + datetime.timedelta(days=days_ahead)

SIGMA_MAPPING_TABLE = pd.DataFrame([
    {"VIX range": "≤ 10",   "σ (auto)": "75%"},
    {"VIX range": "10–15",  "σ (auto)": "75% → 90%"},
    {"VIX range": "15–22",  "σ (auto)": "90% → 125%"},
    {"VIX range": "22–35",  "σ (auto)": "125% → 180%"},
    {"VIX range": "35–50",  "σ (auto)": "180% → 300%"},
    {"VIX range": "> 50",   "σ (auto)": "300%"},
])

REGIME_TABLE = pd.DataFrame([
    {"Regime": "Calm / suppressed",  "σ range": "70%–90%",   "VIX spot": "10–15",  "Example periods": "2017, early 2020",  "Notes": "Very low realized vol-of-vol"},
    {"Regime": "Normal",             "σ range": "100%–130%", "VIX spot": "15–22",  "Example periods": "2019, 2021–23",     "Notes": "Base case; use 110% as default"},
    {"Regime": "Elevated stress",    "σ range": "150%–180%", "VIX spot": "22–35",  "Example periods": "Late 2018, 2022",   "Notes": "Rising uncertainty, no full panic"},
    {"Regime": "Crisis spike",       "σ range": "200%–300%", "VIX spot": "35–80+", "Example periods": "Mar 2020, GFC",     "Notes": "Use extreme care — skew matters most"},
])

# ── Sidebar — Model Parameters ────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")

    st.subheader("Live Data (IBKR)")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Fetch Live VIX from IBKR", use_container_width=True):
            with st.spinner("Connecting to IB Gateway..."):
                st.session_state.live = fetch_vix_data()
    with col_btn2:
        if st.button("Clear Live Data", use_container_width=True):
            st.session_state.live = None

    live = st.session_state.live
    if live:
        if live["error"]:
            st.error(f"IBKR error: {live['error']}")
            if "_debug" in live:
                with st.expander("Debug info"):
                    st.json(live["_debug"])
        else:
            st.success(f"Live data fetched at {live['timestamp']}")

    st.divider()
    st.subheader("Log-Mean-Reverting Model")

    default_V0 = float(round(live["spot"], 2)) if (live and live["spot"]) else 16.0
    V0        = st.number_input("Spot VIX  (V₀)",           min_value=5.0,  max_value=100.0, value=default_V0, step=0.5,   format="%.2f")
    kappa     = st.slider("Mean-reversion speed  (κ)",       min_value=2.50, max_value=12.50, value=5.0,        step=0.1,   format="%.2f")
    theta_bar = st.number_input("Long-run VIX mean  (θ̄)",   min_value=5.0,  max_value=80.0,  value=19.0,       step=0.5,   format="%.2f")

    sigma_auto = sigma_from_vix(V0)
    override   = st.checkbox("Override σ (vol-of-vol)", value=False)
    if override:
        sigma = st.number_input("Vol-of-vol  (σ) — manual", min_value=0.01, max_value=5.0,
                                value=round(sigma_auto, 2), step=0.05, format="%.2f")
    else:
        sigma = sigma_auto
        st.info(f"σ auto-mapped from V₀: **{sigma * 100:.1f}%**")

    r     = st.number_input("Risk-free rate  (r)", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    today = datetime.date.today()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Futures & Options Pricer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pricer:

    SCHEDULE = [
        ("1w",  1/52), ("2w",  2/52), ("3w",  3/52),
        ("1m",  1/12), ("2m",  2/12), ("3m",  3/12),
        ("6m",  6/12), ("12m", 1.0),  ("24m", 2.0),
    ]
    labels     = [s[0] for s in SCHEDULE]
    maturities = [s[1] for s in SCHEDULE]

    rows = vix_futures_term_structure(V0, kappa, theta_bar, sigma, r, maturities)
    for i, row in enumerate(rows):
        row["Tenor"] = labels[i]

    df = pd.DataFrame(rows)[["Tenor", "Time to Expiry (yrs)", "Futures Price"]]
    df["vs. Spot"] = df["Futures Price"] - V0

    _opt_expiry_val = st.session_state.get("opt_expiry", today + datetime.timedelta(days=30))
    if not isinstance(_opt_expiry_val, datetime.date):
        _opt_expiry_val = today + datetime.timedelta(days=30)
    opt_tau_preview = (_opt_expiry_val - today).days / 365.0
    F_model_preview = vix_futures_price(V0, kappa, theta_bar, sigma, r, opt_tau_preview)

    # ── Futures Term Structure ────────────────────────────────────────────────
    st.header("Futures Term Structure")
    col_table, col_chart = st.columns([1, 2], gap="large")

    with col_table:
        st.subheader("Term Structure Table")
        st.dataframe(
            df.style.format({
                "Time to Expiry (yrs)": "{:.4f}",
                "Futures Price": "{:.4f}",
                "vs. Spot": "{:+.4f}",
            }, na_rep="—").map(
                lambda v: "color: #ff6b6b" if isinstance(v, float) and v < 0 else "color: #4caf7d",
                subset=["vs. Spot"],
            ),
            use_container_width=True, hide_index=True, height=370,
        )

    with col_chart:
        st.subheader("Term Structure Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=maturities, y=df["Futures Price"],
            mode="lines+markers", marker=dict(size=8), line=dict(width=2.5),
            name="Futures Price",
        ))
        opt_label = _opt_expiry_val.strftime("%d %b %y")
        fig.add_trace(go.Scatter(
            x=[opt_tau_preview], y=[F_model_preview],
            mode="markers", marker=dict(size=12, symbol="diamond", color="gold"),
            name=f"Option expiry ({opt_label})",
        ))
        fig.add_hline(y=theta_bar, line_dash="dash", line_color="tomato",
                      annotation_text=f"θ̄ = {theta_bar:.2f}", annotation_position="right")
        fig.add_hline(y=V0, line_dash="dot", line_color="lightgreen",
                      annotation_text=f"V₀ = {V0:.2f}", annotation_position="right")
        fig.update_layout(
            xaxis=dict(title="Tenor", tickvals=maturities, ticktext=labels),
            yaxis_title="Futures Price", template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=80, t=40, b=40), height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Live Futures vs Model ─────────────────────────────────────────────────
    live = st.session_state.live
    if live and not live["error"] and live["futures"]:
        st.divider()
        st.subheader("Live Futures vs Model  ·  " + (live.get("note") or f"fetched {live['timestamp']}"))
        live_df = pd.DataFrame(live["futures"])
        live_df["Model Price"]   = live_df["tau"].apply(lambda t: round(vix_futures_price(V0, kappa, theta_bar, sigma, r, t), 4))
        live_df["vs. Spot"]      = live_df["price"] - V0
        live_df["Model vs Live"] = live_df["Model Price"] - live_df["price"]
        live_df = live_df.rename(columns={"expiry": "Expiry", "tau": "τ (yrs)", "price": "Live Price"})[
            ["Expiry", "τ (yrs)", "Live Price", "vs. Spot", "Model Price", "Model vs Live"]]
        st.dataframe(
            live_df.style.format({
                "τ (yrs)": "{:.4f}", "Live Price": "{:.4f}",
                "vs. Spot": "{:+.4f}", "Model Price": "{:.4f}", "Model vs Live": "{:+.4f}",
            }).map(
                lambda v: "color: #ff6b6b" if isinstance(v, float) and v < 0 else "color: #4caf7d",
                subset=["vs. Spot", "Model vs Live"],
            ),
            use_container_width=True, hide_index=True,
        )

    # ── Options Pricer (Black-76) ─────────────────────────────────────────────
    st.divider()
    st.header("Options Pricer — Black-76")

    oc1, oc2, oc3 = st.columns(3, gap="large")
    with oc1:
        opt_expiry = st.date_input(
            "Option Expiry Date",
            value=today + datetime.timedelta(days=30),
            min_value=today + datetime.timedelta(days=1),
            key="opt_expiry",
        )
        opt_type = st.selectbox("Option Type", ["Call", "Put"], index=1, key="opt_type")
    with oc2:
        opt_tau = (opt_expiry - today).days / 365.0
        F_model = vix_futures_price(V0, kappa, theta_bar, sigma, r, opt_tau)
        st.metric("Model Futures Price at Expiry", f"{F_model:.4f}",
                  help=f"τ = {opt_tau:.4f} yrs  ({(opt_expiry - today).days} days)")
        strike = st.number_input("Strike  (K)", min_value=1.0, max_value=200.0,
                                 value=round(F_model, 1), step=0.5, format="%.2f", key="opt_strike")
    with oc3:
        result = black76(F_model, strike, r, sigma, opt_tau, opt_type.lower())
        st.metric("Option Price", f"{result['price']:.4f}")

    st.subheader("Greeks")
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("Delta  (Δ)", f"{result['delta']:.4f}"  if result["delta"]  is not None else "—")
    gc2.metric("Gamma  (Γ)", f"{result['gamma']:.6f}"  if result["gamma"]  is not None else "—")
    gc3.metric("Vega  (ν)",  f"{result['vega']:.4f}"   if result["vega"]   is not None else "—", help="per 1% change in σ")
    gc4.metric("Theta  (Θ)", f"{result['theta']:.4f}"  if result["theta"]  is not None else "—", help="per calendar day")
    gc5.metric("Rho  (ρ)",   f"{result['rho']:.4f}"    if result["rho"]    is not None else "—")

    with st.expander("Model details", expanded=False):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Futures Price  (F) | {F_model:.4f} |
| Strike  (K) | {strike:.4f} |
| Time to Expiry  (T) | {opt_tau:.4f} yrs  ({(opt_expiry - today).days} days) |
| Vol-of-Vol  (σ) | {sigma * 100:.2f}% |
| Risk-free rate  (r) | {r * 100:.2f}% |
| d1 | {result['d1'] if result['d1'] is not None else '—'} |
| d2 | {result['d2'] if result['d2'] is not None else '—'} |
""")

    st.divider()
    with st.expander("Vol-of-Vol Reference", expanded=False):
        col_map, col_regime = st.columns(2, gap="large")
        with col_map:
            st.markdown("**σ Mapping (auto from Spot VIX)**")
            st.dataframe(SIGMA_MAPPING_TABLE, hide_index=True, use_container_width=True)
            st.caption(f"Current V₀ = {V0:.2f}  →  σ = {sigma_auto * 100:.1f}%"
                       + ("  *(overridden)*" if override else ""))
        with col_regime:
            st.markdown("**Regime Reference**")
            st.dataframe(REGIME_TABLE, hide_index=True, use_container_width=True)

    # ── VIX Historical Data (download) ───────────────────────────────────────
    st.divider()

    VIX_CSV_URL   = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    VIX_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "VIX_History.csv")

    if st.session_state.vix_history is None and os.path.exists(VIX_DATA_PATH):
        st.session_state.vix_history = pd.read_csv(VIX_DATA_PATH, parse_dates=["Date"])

    dl1, dl2 = st.columns([2, 3])
    with dl1:
        if st.button("Download and Import VIX Historical Data", use_container_width=True):
            with st.spinner("Downloading from CBOE..."):
                try:
                    vix_hist = pd.read_csv(VIX_CSV_URL, parse_dates=["DATE"])
                    vix_hist.columns = [c.strip().upper() for c in vix_hist.columns]
                    vix_hist = vix_hist.rename(columns={"DATE": "Date"}).sort_values("Date").reset_index(drop=True)
                    vix_hist.to_csv(VIX_DATA_PATH, index=False)
                    st.session_state.vix_history = vix_hist
                    st.success(f"Downloaded {len(vix_hist):,} rows  ({vix_hist['Date'].iloc[0].strftime('%d %b %Y')} → {vix_hist['Date'].iloc[-1].strftime('%d %b %Y')})")
                except Exception as e:
                    st.error(f"Download failed: {e}")
    with dl2:
        _loaded = st.session_state.vix_history
        if _loaded is not None:
            _dm1, _dm2, _dm3 = st.columns(3)
            _dm1.metric("From",        _loaded["Date"].iloc[0].strftime("%d %b %Y"))
            _dm2.metric("To",          _loaded["Date"].iloc[-1].strftime("%d %b %Y"))
            _dm3.metric("Data points", f"{len(_loaded):,}")

    # ── VIX Historical Analysis ───────────────────────────────────────────────
    st.divider()
    st.header("VIX Historical Analysis")

    _vh = st.session_state.vix_history
    if _vh is None:
        st.info("Load VIX historical data in the **Short Put Spreads on VIX** tab to see this section.")
    else:
        _close  = _vh["CLOSE"]
        _dates  = pd.to_datetime(_vh["Date"])
        _yr_min = _dates.dt.year.min()
        _yr_max = _dates.dt.year.max()

        _atl_val  = _close.min()
        _atl_date = _vh.loc[_close.idxmin(), "Date"]
        _atl_date = pd.to_datetime(_atl_date).strftime("%b %Y")
        _ath_val  = _close.max()
        _ath_date = _vh.loc[_close.idxmax(), "Date"]
        _ath_date = pd.to_datetime(_ath_date).strftime("%b %Y")
        _mean_val   = _close.mean()
        _median_val = _close.median()
        _normal_pct = ((_close >= 12) & (_close <= 25)).mean() * 100
        _n_days     = len(_vh)

        # Metric cards
        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("All-time low",    f"{_atl_val:.2f}", delta=_atl_date,    delta_color="off")
        hm2.metric("All-time high",   f"{_ath_val:.2f}", delta=_ath_date,    delta_color="off")
        hm3.metric("Long-run mean",   f"{_mean_val:.2f}", delta=f"Median: {_median_val:.2f}", delta_color="off")
        hm4.metric("Normal range",    "12–25",  delta=f"~{_normal_pct:.0f}% of trading days", delta_color="off")

        # Regime bar chart
        VH_REGIMES = [
            (0,  12, "< 12  Ultra low",   "#5a9e6f"),
            (12, 15, "12–15  Low",        "#a8d5a2"),
            (15, 20, "15–20  Normal",     "#5b7bbf"),
            (20, 25, "20–25  Elevated",   "#a0b4d9"),
            (25, 30, "25–30  High",       "#c8a84b"),
            (30, 40, "30–40  Stress",     "#b06030"),
            (40, 9999, "> 40  Crisis",    "#7a2e10"),
        ]
        _reg_pcts   = []
        _reg_labels = []
        _reg_colors = []
        for lo, hi, label, color in VH_REGIMES:
            pct = ((_close >= lo) & (_close < hi)).mean() * 100
            _reg_pcts.append(pct)
            _reg_labels.append(label)
            _reg_colors.append(color)

        fig_reg = go.Figure(go.Bar(
            x=_reg_labels, y=_reg_pcts,
            marker_color=_reg_colors,
            text=[f"{p:.1f}%" for p in _reg_pcts],
            textposition="outside", textfont=dict(size=11),
        ))
        fig_reg.update_layout(
            title=f"Days spent in each VIX regime ({_yr_min}–{_yr_max})",
            xaxis_title=None,
            yaxis=dict(title="% of trading days", ticksuffix="%", range=[0, max(_reg_pcts) * 1.2]),
            template="plotly_dark",
            margin=dict(l=40, r=40, t=50, b=40), height=360,
            showlegend=False,
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # Annual average VIX chart with min–max shading
        _vh2 = _vh.copy()
        _vh2["Year"] = pd.to_datetime(_vh2["Date"]).dt.year
        _ann = _vh2.groupby("Year")["CLOSE"].agg(avg="mean", lo="min", hi="max").reset_index()

        fig_ann = go.Figure()
        fig_ann.add_trace(go.Scatter(
            x=_ann["Year"], y=_ann["hi"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_ann.add_trace(go.Scatter(
            x=_ann["Year"], y=_ann["lo"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(210, 150, 130, 0.25)",
            showlegend=False, hoverinfo="skip",
        ))
        fig_ann.add_trace(go.Scatter(
            x=_ann["Year"], y=_ann["avg"],
            mode="lines+markers",
            line=dict(color="#5b7bbf", width=2),
            marker=dict(size=6, color="#5b7bbf"),
            name="Annual avg VIX",
        ))
        fig_ann.add_hline(y=_mean_val, line_dash="dash", line_color="tomato",
                          annotation_text=f"Mean {_mean_val:.2f}",
                          annotation_position="right")
        fig_ann.update_layout(
            title=f"Annual average VIX ({_yr_min}–{_yr_max})",
            xaxis=dict(title=None, tickmode="linear", dtick=3),
            yaxis=dict(title="VIX level", rangemode="tozero"),
            template="plotly_dark",
            margin=dict(l=40, r=80, t=50, b=40), height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_ann, use_container_width=True)

        st.caption(f"Source: CBOE VIX daily closing levels {_yr_min}–{_yr_max}  ·  n={_n_days:,} trading days")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Short Put Spreads on VIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab_spreads:
    st.header("VIX Put Strategy — Historical Simulation")

    # ── Strategy selector ─────────────────────────────────────────────────────
    strategy = st.selectbox(
        "Strategy",
        ["Short Put Spread", "Long Put Spread", "Short Put", "Long Put"],
        key="strategy",
        help=(
            "Short Put Spread: sell higher-K put, buy lower-K put as hedge  |  "
            "Long Put Spread: buy higher-K put, sell lower-K put as hedge  |  "
            "Short Put: sell a single put  |  "
            "Long Put: buy a single put"
        ),
    )
    is_spread = strategy in ["Short Put Spread", "Long Put Spread"]
    is_short  = strategy in ["Short Put Spread", "Short Put"]

    # Leg labels
    LEG_LABELS = {
        "Short Put Spread": ("Short Put  (sell)",          "Long Put  (buy — hedge)"),
        "Long Put Spread":  ("Long Put  (buy)",            "Short Put  (sell — hedge)"),
        "Short Put":        ("Put  (sell)",                None),
        "Long Put":         ("Put  (buy)",                 None),
    }
    leg1_label, leg2_label = LEG_LABELS[strategy]

    # ── Strategy Parameters ───────────────────────────────────────────────────
    st.subheader("Strategy Parameters")
    sp1, sp2 = st.columns(2, gap="large")
    with sp1:
        st.markdown(f"**{leg1_label}**")
        short_strike    = st.number_input("Strike",           min_value=1.0,  max_value=100.0, value=19.0,  step=0.5,  format="%.2f", key="short_strike")
        short_vol_shift = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,   value=-0.05, step=0.05, format="%.2f", key="short_vol_shift",
                                          help="Additive shift on σ for this leg")
    with sp2:
        if is_spread:
            st.markdown(f"**{leg2_label}**")
            long_strike    = st.number_input("Strike",           min_value=1.0,  max_value=100.0, value=15.0,  step=0.5,  format="%.2f", key="long_strike")
            long_vol_shift = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,   value=0.05,  step=0.05, format="%.2f", key="long_vol_shift",
                                             help="Additive shift on σ for this leg")
        else:
            long_strike    = None
            long_vol_shift = 0.0

    st.divider()
    st.markdown("**Trade Execution**")
    tc1, tc2, tc3 = st.columns(3, gap="large")
    with tc1:
        min_days_to_expiry = st.number_input(
            "Minimum calendar days before a Wednesday expiry",
            min_value=1, max_value=60, value=10, step=1, key="min_dte",
            help="Expiry used = first Wednesday at least this many days ahead",
        )
    with tc2:
        trading_cost = st.number_input(
            "Trading cost on premium  ($)",
            min_value=0.0, max_value=10.0, value=0.05, step=0.01, format="%.2f", key="trading_cost",
            help="Added to premium paid (long) or subtracted from premium received (short)",
        )
    with tc3:
        n_contracts = st.number_input(
            "Number of options  (lot size 100)",
            min_value=1, max_value=10000, value=1, step=1, key="n_contracts",
            help="Number of options per trade (lot size = 100)",
        )

    # ── VIX Historical Data (status) ──────────────────────────────────────────
    st.divider()
    vh = st.session_state.vix_history
    if vh is not None:
        st.caption(
            f"VIX data loaded: {vh['Date'].iloc[0].strftime('%d %b %Y')} → "
            f"{vh['Date'].iloc[-1].strftime('%d %b %Y')}  ·  {len(vh):,} rows  "
            f"(download / refresh in the **Futures & Options Pricer** tab)"
        )
    else:
        st.warning("No VIX data loaded — go to the **Futures & Options Pricer** tab to download it.")

    # ── Simulation Date Range ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Simulation Date Range")

    vh = st.session_state.vix_history
    data_start = vh["Date"].iloc[0].date()  if vh is not None else datetime.date(1990, 1, 2)
    data_end   = vh["Date"].iloc[-1].date() if vh is not None else today

    dr1, dr2 = st.columns(2, gap="large")
    with dr1:
        sim_start = st.date_input("Start date", value=data_start,
                                  min_value=data_start, max_value=data_end, key="sim_start")
    with dr2:
        sim_end = st.date_input("End date", value=data_end,
                                min_value=data_start, max_value=data_end, key="sim_end")

    if sim_start >= sim_end:
        st.warning("Start date must be before end date.")

    # ── VIX Entry Level Filter ────────────────────────────────────────────────
    st.divider()
    vf_col1, vf_col2, vf_col3 = st.columns([2, 1, 1], gap="large")
    with vf_col1:
        no_vix_filter = st.toggle(
            "No VIX entry filter — trade at all VIX levels",
            value=False, key="no_vix_filter",
        )
    with vf_col2:
        vix_entry_min = st.number_input(
            "Min Entry VIX", min_value=0.0, max_value=200.0, value=10.0,
            step=0.5, format="%.1f", key="vix_entry_min",
            disabled=no_vix_filter,
            help="Only enter a trade if Spot VIX ≥ this value",
        )
    with vf_col3:
        vix_entry_max = st.number_input(
            "Max Entry VIX", min_value=0.0, max_value=200.0, value=40.0,
            step=0.5, format="%.1f", key="vix_entry_max",
            disabled=no_vix_filter,
            help="Only enter a trade if Spot VIX ≤ this value",
        )
    if no_vix_filter:
        vix_entry_min, vix_entry_max = 0.0, 9999.0
    elif vix_entry_min >= vix_entry_max:
        st.warning("Min Entry VIX must be less than Max Entry VIX.")

    # ── Premium Filter ────────────────────────────────────────────────────────
    st.divider()
    mp_col1, mp_col2 = st.columns([3, 1], gap="large")
    with mp_col1:
        no_min_premium = st.toggle(
            "No premium filter — enter all trades",
            value=False, key="no_min_premium",
        )
    with mp_col2:
        if is_short:
            min_premium = st.number_input(
                "Minimum premium to receive  ($)",
                min_value=0.0, max_value=10.0, value=0.15, step=0.01, format="%.2f",
                key="min_premium_short", disabled=no_min_premium,
                help="Skip if premium received is below this threshold",
            )
        else:
            min_premium = st.number_input(
                "Maximum premium to pay  ($)",
                min_value=0.0, max_value=10.0, value=2.0, step=0.01, format="%.2f",
                key="min_premium_long", disabled=no_min_premium,
                help="Skip if premium paid exceeds this threshold",
            )
    if no_min_premium:
        min_premium = 9999.0 if not is_short else 0.0

    # ── Run Simulation ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Put Spread Premium Simulation")

    if vh is None:
        st.info("Download VIX historical data first.")
    elif sim_start >= sim_end:
        st.warning("Fix the date range first.")
    else:
        if st.button("Run Simulation", type="primary", use_container_width=False):
            cutoff = sim_end - datetime.timedelta(days=int(min_days_to_expiry))
            mask   = (vh["Date"].dt.date >= sim_start) & (vh["Date"].dt.date <= cutoff)
            sim_df = vh[mask].copy().reset_index(drop=True)

            if sim_df.empty:
                st.warning("No data in range after applying minimum DTE cutoff.")
            else:
                # Build a fast date → closing VIX lookup for P&L at expiry
                vix_lookup = {
                    row["Date"].date(): float(row["CLOSE"])
                    for _, row in vh.iterrows()
                }
                # Helper: nearest available VIX date on or after a given date
                all_dates_sorted = sorted(vix_lookup.keys())

                def lookup_expiry_vix(exp_date):
                    if exp_date in vix_lookup:
                        return vix_lookup[exp_date]
                    # Find next available trading day
                    for d in all_dates_sorted:
                        if d >= exp_date:
                            return vix_lookup[d]
                    return None

                results_rows = []
                progress = st.progress(0, text="Computing…")
                total = len(sim_df)

                for i, (_, row) in enumerate(sim_df.iterrows()):
                    date     = row["Date"].date()
                    spot_vix = float(row["CLOSE"])
                    expiry   = next_wednesday(date, int(min_days_to_expiry))
                    tau      = (expiry - date).days / 365.0

                    base_sigma = sigma_from_vix(spot_vix)
                    sigma1     = max(base_sigma + short_vol_shift, 0.01)
                    sigma2     = max(base_sigma + long_vol_shift,  0.01) if is_spread else None

                    F      = vix_futures_price(spot_vix, kappa, theta_bar, base_sigma, r, tau)
                    price1 = black76(F, short_strike, r, sigma1, tau, "put")["price"]
                    price2 = (black76(F, long_strike, r, sigma2, tau, "put")["price"]
                              if is_spread else None)

                    # Net premium (received for short, paid for long)
                    if strategy == "Short Put Spread":
                        net_premium = max(price1 - price2 - trading_cost, 0.0)
                    elif strategy == "Long Put Spread":
                        net_premium = price1 - price2 + trading_cost
                    elif strategy == "Short Put":
                        net_premium = max(price1 - trading_cost, 0.0)
                    else:  # Long Put
                        net_premium = price1 + trading_cost

                    # Trade entry check
                    in_vix_range = vix_entry_min <= spot_vix <= vix_entry_max
                    if is_short:
                        trade_entered = net_premium >= min_premium and in_vix_range
                    else:
                        trade_entered = net_premium <= min_premium and in_vix_range

                    # P&L at expiry
                    expiry_vix = lookup_expiry_vix(expiry) if trade_entered else None
                    if trade_entered and expiry_vix is not None:
                        if is_spread:
                            expiry_value = (max(short_strike - expiry_vix, 0.0)
                                            - max(long_strike  - expiry_vix, 0.0))
                        else:
                            expiry_value = max(short_strike - expiry_vix, 0.0)
                        if is_short:
                            pnl = round(n_contracts * 100 * (net_premium - expiry_value), 2)
                        else:
                            pnl = round(n_contracts * 100 * (expiry_value - net_premium), 2)
                    else:
                        expiry_value = None
                        pnl          = None

                    results_rows.append({
                        "Date":             date,
                        "Expiry (Wed)":     expiry,
                        "Days to Expiry":   (expiry - date).days,
                        "Spot VIX":         round(spot_vix, 2),
                        "Base σ":           f"{base_sigma * 100:.1f}%",
                        "F (model)":        round(F, 4),
                        "Leg 1 σ":          f"{sigma1 * 100:.1f}%",
                        "Leg 1 Price":      round(price1, 4),
                        "Leg 2 σ":          f"{sigma2 * 100:.1f}%" if sigma2 else None,
                        "Leg 2 Price":      round(price2, 4) if price2 else None,
                        "Net Premium":      round(net_premium, 4),
                        "Trade Entered":    trade_entered,
                        "Expiry VIX":       round(expiry_vix, 2)   if expiry_vix   is not None else None,
                        "Expiry Value":     round(expiry_value, 4)  if expiry_value is not None else None,
                        "P&L ($)":          pnl,
                    })

                    if i % 200 == 0:
                        progress.progress(i / total, text=f"Computing… {i:,}/{total:,}")

                progress.empty()
                st.session_state.sim_results = pd.DataFrame(results_rows)

        # Display results if available
        if st.session_state.sim_results is not None:
            results  = st.session_state.sim_results
            entered  = results[results["Trade Entered"]].copy()
            pnl_vals = entered["P&L ($)"].dropna()
            n_trades = len(pnl_vals)

            total_pnl   = pnl_vals.sum()
            avg_pnl     = pnl_vals.mean()     if n_trades else None
            win_rate    = (pnl_vals > 0).mean() * 100 if n_trades else None
            max_win     = pnl_vals.max()      if n_trades else None
            max_loss    = pnl_vals.min()      if n_trades else None
            avg_win     = pnl_vals[pnl_vals > 0].mean() if (pnl_vals > 0).any() else None
            avg_loss    = pnl_vals[pnl_vals < 0].mean() if (pnl_vals < 0).any() else None
            profit_factor = (pnl_vals[pnl_vals > 0].sum() /
                             abs(pnl_vals[pnl_vals < 0].sum())
                             ) if (pnl_vals < 0).any() else None
            sharpe      = (pnl_vals.mean() / pnl_vals.std() * np.sqrt(252)
                           ) if n_trades > 1 else None

            if entered.empty:
                st.warning("No trades were entered with the current filters. "
                           "Try relaxing the premium threshold or VIX entry range.")
                st.stop()

            # ── Summary statistics ────────────────────────────────────────────
            st.divider()
            st.subheader("Strategy Statistics")

            row1 = st.columns(4)
            row1[0].metric("Total P&L",          f"${total_pnl:,.0f}")
            row1[1].metric("Trades entered",      f"{n_trades:,}")
            row1[2].metric("Win rate",            f"{win_rate:.1f}%" if win_rate is not None else "—")
            row1[3].metric("Avg P&L / trade",     f"${avg_pnl:,.0f}"  if avg_pnl  is not None else "—")

            row2 = st.columns(4)
            row2[0].metric("Max win",             f"${max_win:,.0f}"  if max_win  is not None else "—")
            row2[1].metric("Max loss",            f"${max_loss:,.0f}" if max_loss is not None else "—")
            row2[2].metric("Avg win / Avg loss",
                           f"{abs(avg_win/avg_loss):.2f}x"
                           if avg_win is not None and avg_loss is not None else "—")
            row2[3].metric("Profit factor",       f"{profit_factor:.2f}" if profit_factor is not None else "∞")

            # ── Annual P&L bar chart ──────────────────────────────────────────
            st.divider()
            st.subheader("Annual P&L")

            entered["Year"] = pd.to_datetime(entered["Date"]).dt.year
            annual = entered.groupby("Year")["P&L ($)"].sum().reset_index()
            annual.columns = ["Year", "P&L ($)"]

            bar_colors = ["#4caf7d" if v >= 0 else "#ff6b6b" for v in annual["P&L ($)"]]
            fig_annual = go.Figure(go.Bar(
                x=annual["Year"].astype(str),
                y=annual["P&L ($)"],
                marker_color=bar_colors,
                text=[f"${v:,.0f}" for v in annual["P&L ($)"]],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig_annual.add_hline(y=0, line_color="white", line_width=1)
            fig_annual.update_layout(
                xaxis_title="Year", yaxis_title="P&L ($)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=30, b=40), height=380,
                showlegend=False,
                yaxis=dict(tickprefix="$", tickformat=",.0f"),
            )
            st.plotly_chart(fig_annual, use_container_width=True)

            # ── P&L vs Premium scatter ────────────────────────────────────────
            st.divider()
            st.subheader("P&L vs Spread Premium")

            fig_scatter = go.Figure(go.Scatter(
                x=entered["Net Premium"],
                y=entered["P&L ($)"],
                mode="markers",
                marker=dict(
                    color=entered["P&L ($)"],
                    colorscale=[[0, "#ff6b6b"], [0.5, "#888888"], [1, "#4caf7d"]],
                    cmin=entered["P&L ($)"].min(),
                    cmid=0,
                    cmax=entered["P&L ($)"].max(),
                    size=5,
                    opacity=0.6,
                    colorbar=dict(title="P&L ($)", tickprefix="$", tickformat=",.0f"),
                ),
                customdata=np.stack([
                    entered["Date"].astype(str),
                    entered["Spot VIX"],
                    entered["Expiry VIX"].fillna(0),
                ], axis=1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Premium: $%{x:.4f}<br>"
                    "P&L: $%{y:,.2f}<br>"
                    "Entry VIX: %{customdata[1]:.2f}<br>"
                    "Expiry VIX: %{customdata[2]:.2f}<extra></extra>"
                ),
            ))
            fig_scatter.add_hline(y=0, line_color="white", line_dash="dot", line_width=1)
            fig_scatter.update_layout(
                xaxis=dict(title="Spread Premium ($)", tickprefix="$"),
                yaxis=dict(title="P&L ($)", tickprefix="$", tickformat=",.0f"),
                template="plotly_dark",
                margin=dict(l=40, r=40, t=30, b=40), height=420,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # ── VIX bucket charts ────────────────────────────────────────────
            st.divider()
            st.subheader("Analysis by Entry VIX Level")

            vix_min = int(np.floor(entered["Spot VIX"].min()))
            vix_max = int(np.ceil(entered["Spot VIX"].max()))
            # Width-1 up to 25, width-5 from 25–40, width-10 above 40
            VIX_BINS   = list(range(vix_min, min(25, vix_max) + 1))
            VIX_LABELS = [f"{v}–{v+1}" for v in range(vix_min, min(25, vix_max))]
            if vix_max > 25:
                mid_end = min(40, int(np.ceil(vix_max / 5) * 5))
                for v in range(25, mid_end, 5):
                    VIX_BINS.append(v + 5)
                    VIX_LABELS.append(f"{v}–{v+5}")
            if vix_max > 40:
                coarse_end = int(np.ceil(vix_max / 10) * 10)
                for v in range(40, coarse_end, 10):
                    VIX_BINS.append(v + 10)
                    VIX_LABELS.append(f"{v}–{v+10}")

            entered["VIX Bucket"] = pd.cut(
                entered["Spot VIX"], bins=VIX_BINS, labels=VIX_LABELS,
                right=False, include_lowest=True
            )
            bucket_stats = (
                entered.groupby("VIX Bucket", observed=True)["P&L ($)"]
                .agg(
                    count="count",
                    win_rate=lambda x: (x > 0).mean() * 100,
                    avg_pnl="mean",
                    avg_win=lambda x: x[x > 0].mean() if (x > 0).any() else np.nan,
                    avg_loss=lambda x: x[x < 0].mean() if (x < 0).any() else np.nan,
                    n_wins=lambda x: (x > 0).sum(),
                    n_losses=lambda x: (x < 0).sum(),
                )
                .reset_index()
            )
            bucket_stats = bucket_stats[bucket_stats["count"] > 0]
            bx = bucket_stats["VIX Bucket"].astype(str)

            # Chart 1 — Win Rate %
            fig_wr = go.Figure(go.Bar(
                x=bx, y=bucket_stats["win_rate"],
                marker_color=["#4caf7d" if v >= 50 else "#ff6b6b" for v in bucket_stats["win_rate"]],
                text=[f"{v:.1f}%<br><sub>{int(c)}</sub>" for v, c in zip(bucket_stats["win_rate"], bucket_stats["count"])],
                textposition="outside", textfont=dict(size=11),
            ))
            fig_wr.add_hline(y=50, line_dash="dash", line_color="white", line_width=1,
                             annotation_text="50%", annotation_position="right")
            fig_wr.update_layout(
                title="Win Rate % by Entry VIX", xaxis_title="Entry VIX range",
                yaxis=dict(title="Win rate (%)", range=[0, 115], ticksuffix="%"),
                template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40),
                height=370, showlegend=False,
            )
            st.plotly_chart(fig_wr, use_container_width=True)

            # Chart 2 — Average Trade P&L
            fig_ap = go.Figure(go.Bar(
                x=bx, y=bucket_stats["avg_pnl"],
                marker_color=["#4caf7d" if v >= 0 else "#ff6b6b" for v in bucket_stats["avg_pnl"]],
                text=[f"${v:,.0f}<br><sub>{int(c)}</sub>" for v, c in zip(bucket_stats["avg_pnl"], bucket_stats["count"])],
                textposition="outside", textfont=dict(size=11),
            ))
            fig_ap.add_hline(y=0, line_color="white", line_width=1)
            fig_ap.update_layout(
                title="Average Trade P&L by Entry VIX", xaxis_title="Entry VIX range",
                yaxis=dict(title="Avg P&L ($)", tickprefix="$", tickformat=",.0f"),
                template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40),
                height=370, showlegend=False,
            )
            st.plotly_chart(fig_ap, use_container_width=True)

            # Chart 3 — Number of wins vs losses
            fig_wl = go.Figure()
            fig_wl.add_trace(go.Bar(
                name="Wins", x=bx, y=bucket_stats["n_wins"],
                marker_color="#4caf7d",
                text=bucket_stats["n_wins"].astype(int),
                textposition="outside", textfont=dict(size=11),
            ))
            fig_wl.add_trace(go.Bar(
                name="Losses", x=bx, y=bucket_stats["n_losses"],
                marker_color="#ff6b6b",
                text=bucket_stats["n_losses"].astype(int),
                textposition="outside", textfont=dict(size=11),
            ))
            fig_wl.update_layout(
                title="Number of Wins vs Losses by Entry VIX", xaxis_title="Entry VIX range",
                yaxis_title="Number of trades", barmode="group",
                template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40),
                height=370, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_wl, use_container_width=True)

            # Chart 4 — Avg win P&L vs avg loss P&L
            fig_wlp = go.Figure()
            fig_wlp.add_trace(go.Bar(
                name="Avg Win", x=bx, y=bucket_stats["avg_win"],
                marker_color="#4caf7d",
                text=[f"${v:,.0f}" if pd.notna(v) else "" for v in bucket_stats["avg_win"]],
                textposition="outside", textfont=dict(size=11),
            ))
            fig_wlp.add_trace(go.Bar(
                name="Avg Loss", x=bx, y=bucket_stats["avg_loss"],
                marker_color="#ff6b6b",
                text=[f"${v:,.0f}" if pd.notna(v) else "" for v in bucket_stats["avg_loss"]],
                textposition="outside", textfont=dict(size=11),
            ))
            fig_wlp.add_hline(y=0, line_color="white", line_width=1)
            fig_wlp.update_layout(
                title="Average Win P&L vs Average Loss P&L by Entry VIX", xaxis_title="Entry VIX range",
                yaxis=dict(title="Avg P&L ($)", tickprefix="$", tickformat=",.0f"),
                barmode="group", template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40), height=370,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_wlp, use_container_width=True)

            # Trade count caption
            count_caption = "  ·  ".join(
                f"{row['VIX Bucket']}: {int(row['count'])}"
                for _, row in bucket_stats.iterrows()
            )
            st.caption(f"Trades per VIX bucket  —  {count_caption}")

            # ── Shared helpers for trade tables ──────────────────────────────
            def colour_pnl(val):
                if not isinstance(val, (int, float)) or pd.isna(val):
                    return ""
                return "color: #4caf7d" if val >= 0 else "color: #ff6b6b"

            TRADE_COL_CONFIG = {
                "Date":           st.column_config.DateColumn(     "Date",      width=90),
                "Expiry (Wed)":   st.column_config.DateColumn(     "Expiry",    width=90),
                "Days to Expiry": st.column_config.NumberColumn(   "DTE",       width=50),
                "Spot VIX":       st.column_config.NumberColumn(   "VIX",       width=60),
                "Base σ":         st.column_config.TextColumn(     "Base σ",    width=65),
                "F (model)":      st.column_config.NumberColumn(   "F",         width=65),
                "Leg 1 σ":        st.column_config.TextColumn(     "L1 σ",      width=60),
                "Leg 1 Price":    st.column_config.NumberColumn(   "L1 Price",  width=70),
                "Leg 2 σ":        st.column_config.TextColumn(     "L2 σ",      width=60),
                "Leg 2 Price":    st.column_config.NumberColumn(   "L2 Price",  width=70),
                "Net Premium":    st.column_config.NumberColumn(   "Premium",   width=75),
                "Trade Entered":  st.column_config.CheckboxColumn( "Trade?",    width=60),
                "Expiry VIX":     st.column_config.NumberColumn(   "Exp VIX",   width=70),
                "Expiry Value":   st.column_config.NumberColumn(   "Exp Value", width=80),
                "P&L ($)":        st.column_config.TextColumn(     "P&L ($)",   width=80),
            }
            TRADE_FMT = {
                "Net Premium":  "{:.4f}",
                "Expiry Value": "{:.4f}",
                "P&L ($)":      lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
            }

            def render_trade_table(df):
                st.dataframe(
                    df.style.format(TRADE_FMT, na_rep="—").map(colour_pnl, subset=["P&L ($)"]),
                    use_container_width=True, hide_index=True,
                    column_config=TRADE_COL_CONFIG,
                )

            sim_caption = (
                f"Min DTE = {min_days_to_expiry}d (next Wed)  ·  "
                f"Strategy: {strategy}  ·  "
                f"Leg 1 K={short_strike} (σ {short_vol_shift:+.2f})"
                + (f"  ·  Leg 2 K={long_strike} (σ {long_vol_shift:+.2f})" if is_spread else "")
                + f"  ·  Cost ${trading_cost:.2f}  ·  n={n_contracts}"
            )

            # ── Spot check ───────────────────────────────────────────────────
            st.divider()
            st.markdown("**Spot check — 5 random example dates**")
            sample_size = min(5, len(results))
            ex_df = results.sample(n=sample_size).sort_values("Date").reset_index(drop=True)
            render_trade_table(ex_df)
            st.caption(sim_caption)

            # ── Full trade list ───────────────────────────────────────────────
            st.divider()
            if st.button(
                f"Show full trade list  ({n_trades:,} entered trades)",
                key="show_full_trades",
            ):
                full_df = entered.drop(columns=["VIX Bucket"], errors="ignore").sort_values("Date").reset_index(drop=True)
                render_trade_table(full_df)
                st.caption(sim_caption)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Playing Around (AI sandbox)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sandbox:
    st.header("🧪 Playing Around")
    st.caption("Describe what you'd like to explore and Claude will build it for you.")

    # ── API key — reads from .streamlit/secrets.toml, falls back to text input ─
    _api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not _api_key:
        _api_key = st.text_input(
            "Anthropic API key", type="password", key="anthropic_api_key",
            help="Paste your key here, or save it permanently in .streamlit/secrets.toml",
        )

    # ── Context Claude will know about ───────────────────────────────────────
    _SYSTEM_PROMPT = f"""You are a data analyst assistant embedded in a VIX options strategy dashboard.
You write concise Python/Streamlit code that will be executed with exec() in a sandbox.
The following objects are already available (do NOT import or redefine them):
  - pd, np, go, st, datetime
  - vix_futures_price(V0, kappa, theta_bar, sigma, r, tau) → float
  - vix_futures_term_structure(V0, kappa, theta_bar, sigma, r, maturities) → list of dicts
  - black76(F, K, r, sigma, T, option_type) → dict with price, delta, gamma, vega, theta, rho
  - sigma_from_vix(vix) → float  (auto vol-of-vol)
  - next_wednesday(date, min_days) → date
  - V0={V0}, kappa={kappa}, theta_bar={theta_bar}, sigma={sigma:.4f}, r={r}  (current model params)
  - vix_history  (pandas DataFrame with columns: Date, OPEN, HIGH, LOW, CLOSE — full CBOE history, may be None)
  - today  (datetime.date)

Rules:
- Use st.plotly_chart(), st.dataframe(), st.metric(), st.write() etc. to display output.
- Use plotly_dark template for all charts.
- Do NOT use st.sidebar, st.tabs, or st.set_page_config.
- Return ONLY the executable Python code block — no markdown fences, no explanation.
- Keep it concise and readable."""

    # ── Session state ─────────────────────────────────────────────────────────
    for _k, _v in [("sandbox_history", []), ("sandbox_running", False)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Chat history display ──────────────────────────────────────────────────
    for _msg in st.session_state.sandbox_history:
        with st.chat_message(_msg["role"]):
            if _msg["role"] == "assistant":
                # Re-execute stored code so outputs re-render on page reload
                _sandbox_ns = {
                    "pd": pd, "np": np, "go": go, "st": st, "datetime": datetime,
                    "vix_futures_price": vix_futures_price,
                    "vix_futures_term_structure": vix_futures_term_structure,
                    "black76": black76,
                    "sigma_from_vix": sigma_from_vix,
                    "next_wednesday": next_wednesday,
                    "V0": V0, "kappa": kappa, "theta_bar": theta_bar,
                    "sigma": sigma, "r": r,
                    "vix_history": st.session_state.vix_history,
                    "today": today,
                }
                try:
                    exec(_msg["code"], _sandbox_ns)  # noqa: S102
                except Exception as _e:
                    st.error(f"Error re-rendering: {_e}")
            else:
                st.markdown(_msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    _prompt = st.chat_input("What would you like to explore?  e.g. 'Plot put delta vs VIX for K=18'")

    if _prompt:
        if not _api_key:
            st.warning("Please enter your Anthropic API key above.")
        else:
            # Show user message
            st.session_state.sandbox_history.append({"role": "user", "content": _prompt})
            with st.chat_message("user"):
                st.markdown(_prompt)

            # Build messages for Claude
            _messages = [
                {"role": m["role"], "content": m["content"] if m["role"] == "user" else m["code"]}
                for m in st.session_state.sandbox_history
            ]

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        _client = _anthropic.Anthropic(api_key=_api_key)
                        _resp = _client.messages.create(
                            model="claude-haiku-4-5",
                            max_tokens=2048,
                            system=_SYSTEM_PROMPT,
                            messages=_messages,
                        )
                        _code = _resp.content[0].text.strip()
                        # Strip markdown fences if Claude added them
                        if _code.startswith("```"):
                            _code = "\n".join(_code.split("\n")[1:])
                        if _code.endswith("```"):
                            _code = "\n".join(_code.split("\n")[:-1])
                        _code = _code.strip()
                    except Exception as _e:
                        st.error(f"API error: {_e}")
                        _code = None

                if _code:
                    _sandbox_ns = {
                        "pd": pd, "np": np, "go": go, "st": st, "datetime": datetime,
                        "vix_futures_price": vix_futures_price,
                        "vix_futures_term_structure": vix_futures_term_structure,
                        "black76": black76,
                        "sigma_from_vix": sigma_from_vix,
                        "next_wednesday": next_wednesday,
                        "V0": V0, "kappa": kappa, "theta_bar": theta_bar,
                        "sigma": sigma, "r": r,
                        "vix_history": st.session_state.vix_history,
                        "today": today,
                    }
                    _stdout_cap = io.StringIO()
                    _old_out = sys.stdout
                    sys.stdout = _stdout_cap
                    try:
                        exec(_code, _sandbox_ns)  # noqa: S102
                        sys.stdout = _old_out
                        _printed = _stdout_cap.getvalue()
                        if _printed:
                            st.text(_printed)
                        st.session_state.sandbox_history.append(
                            {"role": "assistant", "code": _code}
                        )
                    except Exception:
                        sys.stdout = _old_out
                        st.error(traceback.format_exc())

    # Clear button
    if st.session_state.sandbox_history:
        if st.button("🗑 Clear conversation", key="sandbox_clear"):
            st.session_state.sandbox_history = []
            st.rerun()
