import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

tab_pricer, tab_spreads = st.tabs(["Futures & Options Pricer", "Short Put Spreads on VIX"])

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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Short Put Spreads on VIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab_spreads:
    st.header("Short Put Spreads on VIX — Historical Simulation")

    # ── Strategy Parameters ───────────────────────────────────────────────────
    st.subheader("Strategy Parameters")
    sp1, sp2 = st.columns(2, gap="large")
    with sp1:
        st.markdown("**Short Put**")
        short_strike    = st.number_input("Strike",           min_value=1.0,  max_value=100.0, value=14.0,  step=0.5,  format="%.2f", key="short_strike")
        short_vol_shift = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,   value=-0.10, step=0.05, format="%.2f", key="short_vol_shift",
                                          help="Additive shift on σ for this leg  (e.g. −0.10 = σ − 10%)")
    with sp2:
        st.markdown("**Long Put**")
        long_strike    = st.number_input("Strike",           min_value=1.0,  max_value=100.0, value=1.0, step=0.5,  format="%.2f", key="long_strike")
        long_vol_shift = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,   value=0.0, step=0.05, format="%.2f", key="long_vol_shift",
                                         help="Additive shift on σ for this leg  (e.g. +0.10 = σ + 10%)")

    st.divider()
    st.markdown("**Trade Execution**")
    tc1, tc2 = st.columns(2, gap="large")
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
            help="Subtracted from net premium received",
        )

    # ── VIX Historical Data ───────────────────────────────────────────────────
    st.divider()
    st.subheader("VIX Historical Data")

    VIX_CSV_URL   = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    VIX_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "VIX_History.csv")

    # Auto-load from disk on first run
    if st.session_state.vix_history is None and os.path.exists(VIX_DATA_PATH):
        st.session_state.vix_history = pd.read_csv(VIX_DATA_PATH, parse_dates=["Date"])

    col_dl1, col_dl2 = st.columns([2, 3])
    with col_dl1:
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

    with col_dl2:
        vh = st.session_state.vix_history
        if vh is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("From",         vh["Date"].iloc[0].strftime("%d %b %Y"))
            m2.metric("To",           vh["Date"].iloc[-1].strftime("%d %b %Y"))
            m3.metric("Data points",  f"{len(vh):,}")

    if st.session_state.vix_history is not None:
        with st.expander("Preview data", expanded=False):
            vh = st.session_state.vix_history
            st.dataframe(pd.concat([vh.head(5), vh.tail(5)]), use_container_width=True, hide_index=True)

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

    # ── Minimum Premium Filter ────────────────────────────────────────────────
    st.divider()
    min_premium = st.number_input(
        "Minimum spread premium to enter trade  ($)",
        min_value=0.0, max_value=10.0, value=0.15, step=0.01, format="%.2f", key="min_premium",
        help="Skip the trade if the floored spread premium is below this threshold",
    )

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
                results_rows = []
                progress = st.progress(0, text="Computing…")
                total = len(sim_df)

                for i, (_, row) in enumerate(sim_df.iterrows()):
                    date     = row["Date"].date()
                    spot_vix = float(row["CLOSE"])
                    expiry   = next_wednesday(date, int(min_days_to_expiry))
                    tau      = (expiry - date).days / 365.0

                    base_sigma  = sigma_from_vix(spot_vix)
                    sigma_short = max(base_sigma + short_vol_shift, 0.01)
                    sigma_long  = max(base_sigma + long_vol_shift,  0.01)

                    F          = vix_futures_price(spot_vix, kappa, theta_bar, base_sigma, r, tau)
                    short_put  = black76(F, short_strike, r, sigma_short, tau, "put")["price"]
                    long_put   = black76(F, long_strike,  r, sigma_long,  tau, "put")["price"]

                    spread_premium = max(short_put - long_put - trading_cost, 0.0)

                    results_rows.append({
                        "Date":            date,
                        "Expiry (Wed)":    expiry,
                        "Days to Expiry":  (expiry - date).days,
                        "Spot VIX":        round(spot_vix, 2),
                        "Base σ":          f"{base_sigma * 100:.1f}%",
                        "F (model)":       round(F, 4),
                        "Short Put σ":     f"{sigma_short * 100:.1f}%",
                        "Short Put Price": round(short_put, 4),
                        "Long Put σ":      f"{sigma_long * 100:.1f}%",
                        "Long Put Price":  round(long_put, 4),
                        "Spread Premium":  round(spread_premium, 4),
                        "Trade Entered":   spread_premium >= min_premium,
                    })

                    if i % 200 == 0:
                        progress.progress(i / total, text=f"Computing… {i:,}/{total:,}")

                progress.empty()
                st.session_state.sim_results = pd.DataFrame(results_rows)

        # Display results if available
        if st.session_state.sim_results is not None:
            results = st.session_state.sim_results
            st.markdown("**Spot check — 5 random example dates**")
            sample_size = min(5, len(results))
            ex_df = results.sample(n=sample_size).sort_values("Date").reset_index(drop=True)
            st.dataframe(ex_df, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(results):,} rows computed  ·  "
                f"Min DTE = {min_days_to_expiry}d (next Wed)  ·  "
                f"Short K={short_strike} (σ {short_vol_shift:+.2f})  ·  "
                f"Long K={long_strike} (σ {long_vol_shift:+.2f})  ·  "
                f"Cost ${trading_cost:.2f}  ·  Min premium ${min_premium:.2f}"
            )
