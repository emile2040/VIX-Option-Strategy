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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIX Option Strategy",
    page_icon="📈",
    layout="wide",
)

st.title("VIX Option Strategy Dashboard")

# ── Session state for live data ───────────────────────────────────────────────
if "live" not in st.session_state:
    st.session_state.live = None

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
    st.subheader("Log-Mean-Reverting Model")

    # ── Live data fetch ───────────────────────────────────────────────────────
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
    st.subheader("Model Parameters")

    # Pre-fill V0 from live spot if available
    default_V0 = float(round(live["spot"], 2)) if (live and live["spot"]) else 16.0
    V0        = st.number_input("Spot VIX  (V₀)",           min_value=5.0,  max_value=100.0, value=default_V0,  step=0.5,   format="%.2f")
    kappa     = st.slider("Mean-reversion speed  (κ)",       min_value=2.50, max_value=12.50, value=5.0,   step=0.1,   format="%.2f")
    theta_bar = st.number_input("Long-run VIX mean  (θ̄)",   min_value=5.0,  max_value=80.0,  value=19.0,  step=0.5,   format="%.2f")

    # σ: auto-mapped from V0, with optional override
    sigma_auto = sigma_from_vix(V0)
    override   = st.checkbox("Override σ (vol-of-vol)", value=False)
    if override:
        sigma = st.number_input("Vol-of-vol  (σ) — manual", min_value=0.01, max_value=5.0,
                                value=round(sigma_auto, 2), step=0.05, format="%.2f")
    else:
        sigma = sigma_auto
        st.info(f"σ auto-mapped from V₀: **{sigma * 100:.1f}%**")

    r = st.number_input("Risk-free rate  (r)", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")

    today = datetime.date.today()

# ── Fixed maturity schedule ───────────────────────────────────────────────────
SCHEDULE = [
    ("1w",  1 / 52),
    ("2w",  2 / 52),
    ("3w",  3 / 52),
    ("1m",  1 / 12),
    ("2m",  2 / 12),
    ("3m",  3 / 12),
    ("6m",  6 / 12),
    ("12m", 12 / 12),
    ("24m", 24 / 12),
]

labels     = [s[0] for s in SCHEDULE]
maturities = [s[1] for s in SCHEDULE]

rows = vix_futures_term_structure(V0, kappa, theta_bar, sigma, r, maturities)
for i, row in enumerate(rows):
    row["Tenor"] = labels[i]

df = pd.DataFrame(rows)[["Tenor", "Time to Expiry (yrs)", "Futures Price"]]
df["vs. Spot"] = df["Futures Price"] - V0

# ── Layout ────────────────────────────────────────────────────────────────────
st.header("Futures Term Structure")

col_table, col_chart = st.columns([1, 2], gap="large")

with col_table:
    st.subheader("Term Structure Table")
    styled = df.style.format({
        "Time to Expiry (yrs)": "{:.4f}",
        "Futures Price": "{:.4f}",
        "vs. Spot": "{:+.4f}",
    }, na_rep="—").map(
        lambda v: "color: #ff6b6b" if isinstance(v, float) and v < 0 else "color: #4caf7d",
        subset=["vs. Spot"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=370)

with col_chart:
    st.subheader("Term Structure Chart")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=maturities,
        y=df["Futures Price"],
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(width=2.5),
        name="Futures Price",
    ))

    fig.add_hline(
        y=theta_bar,
        line_dash="dash", line_color="tomato",
        annotation_text=f"θ̄ = {theta_bar:.2f}",
        annotation_position="right",
    )
    fig.add_hline(
        y=V0,
        line_dash="dot", line_color="lightgreen",
        annotation_text=f"V₀ = {V0:.2f}",
        annotation_position="right",
    )

    fig.update_layout(
        xaxis=dict(title="Tenor", tickvals=maturities, ticktext=labels),
        yaxis_title="Futures Price",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=80, t=40, b=40),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

# ── Live Futures vs Model ─────────────────────────────────────────────────────
live = st.session_state.live
if live and not live["error"] and live["futures"]:
    st.divider()
    st.subheader("Live Futures vs Model  ·  " + (live.get("note") or f"fetched {live['timestamp']}"))

    live_df = pd.DataFrame(live["futures"])

    # For each live contract compute model price at its exact τ
    live_df["Model Price"] = live_df["tau"].apply(
        lambda t: round(vix_futures_price(V0, kappa, theta_bar, sigma, r, t), 4)
    )
    live_df["vs. Spot"]      = live_df["price"] - V0
    live_df["Model vs Live"] = live_df["Model Price"] - live_df["price"]

    live_df = live_df.rename(columns={
        "expiry": "Expiry", "tau": "τ (yrs)", "price": "Live Price"
    })[["Expiry", "τ (yrs)", "Live Price", "vs. Spot", "Model Price", "Model vs Live"]]

    styled_live = live_df.style.format({
        "τ (yrs)":       "{:.4f}",
        "Live Price":    "{:.4f}",
        "vs. Spot":      "{:+.4f}",
        "Model Price":   "{:.4f}",
        "Model vs Live": "{:+.4f}",
    }).map(
        lambda v: "color: #ff6b6b" if isinstance(v, float) and v < 0 else "color: #4caf7d",
        subset=["vs. Spot", "Model vs Live"],
    )
    st.dataframe(styled_live, use_container_width=True, hide_index=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
st.divider()
price_by_tenor = {row["Tenor"]: row["Futures Price"] for row in rows}
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spot VIX",    f"{V0:.2f}")
col2.metric("1w Future",   f"{price_by_tenor['1w']:.2f}")
col3.metric("1m Future",   f"{price_by_tenor['1m']:.2f}")
col4.metric("6m Future",   f"{price_by_tenor['6m']:.2f}")
col5.metric("12m Future",  f"{price_by_tenor['12m']:.2f}")

# ── Options Pricer (Black-76) ─────────────────────────────────────────────────
st.divider()
st.header("Options Pricer — Black-76")

with st.container():
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
        st.metric(
            label="Model Futures Price at Expiry",
            value=f"{F_model:.4f}",
            help=f"τ = {opt_tau:.4f} yrs  ({(opt_expiry - today).days} days)",
        )
        strike = st.number_input(
            "Strike  (K)",
            min_value=1.0, max_value=200.0,
            value=round(F_model, 1),
            step=0.5, format="%.2f",
            key="opt_strike",
        )

    with oc3:
        result = black76(F_model, strike, r, sigma, opt_tau, opt_type.lower())
        st.metric("Option Price", f"{result['price']:.4f}")

st.subheader("Greeks")
gc1, gc2, gc3, gc4, gc5 = st.columns(5)
gc1.metric("Delta  (Δ)", f"{result['delta']:.4f}"  if result["delta"]  is not None else "—")
gc2.metric("Gamma  (Γ)", f"{result['gamma']:.6f}"  if result["gamma"]  is not None else "—")
gc3.metric("Vega  (ν)",  f"{result['vega']:.4f}"   if result["vega"]   is not None else "—",
           help="per 1% change in σ")
gc4.metric("Theta  (Θ)", f"{result['theta']:.4f}"  if result["theta"]  is not None else "—",
           help="per calendar day")
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

# ── Vol-of-Vol Reference ──────────────────────────────────────────────────────
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
