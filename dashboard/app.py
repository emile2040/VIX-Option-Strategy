import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import io
import math
import traceback
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import anthropic as _anthropic
from scipy.stats import norm as _norm
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

tab_pricer, tab_spreads, tab_dynamic, tab_optimizer, tab_sandbox = st.tabs([
    "Futures & Options Pricer",
    "VIX Fixed Put Strikes",
    "Dynamic Put Spreads Strikes",
    "🔍 Optimizer",
    "🧪 Playing Around",
])

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("live", None), ("vix_history", None), ("sim_results", None),
                  ("dyn_sim_results", None), ("opt_results", None)]:
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

def _black76_put_vec(F, K, r_val, sigma, T):
    """Vectorised Black-76 put price. F, K, sigma, T are numpy arrays of the same shape."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrtT  = np.sqrt(np.maximum(T, 0.0))
        denom  = sigma * sqrtT
        log_fk = np.log(np.where(K > 0, F / np.maximum(K, 1e-10), 1.0))
        d1 = np.where(denom > 1e-10,
                      (log_fk + 0.5 * sigma ** 2 * T) / denom, 0.0)
        d2  = d1 - denom
        df  = np.exp(-r_val * np.maximum(T, 0.0))
        price = np.where(
            T > 0,
            df * (K * _norm.cdf(-d2) - F * _norm.cdf(-d1)),
            np.maximum(K - F, 0.0),
        )
    return np.maximum(price, 0.0)


def build_vix_buckets(vix_min_val, vix_max_val):
    """
    Return (bins, labels) for pd.cut that robustly span [vix_min_val, vix_max_val].
    Grid: width-1 up to 25, width-5 from 25–40, width-10 above 40.
    Always anchors the first edge at vix_min_val and the last at vix_max_val.
    """
    # Full grid of edges across all regimes
    _grid = sorted(set(
        list(range(0, 26)) +          # width-1: 0–25
        list(range(25, 45, 5)) +      # width-5: 25–40
        list(range(40, 210, 10))      # width-10: 40+
    ))
    # Inner edges strictly between vix_min and vix_max
    _inner = [e for e in _grid if vix_min_val < e < vix_max_val]
    bins   = sorted(set([vix_min_val] + _inner + [vix_max_val]))
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
    return bins, labels

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

    # ── VIX data status (outside form — always visible) ───────────────────────
    vh = st.session_state.vix_history
    data_start = vh["Date"].iloc[0].date()  if vh is not None else datetime.date(1990, 1, 2)
    data_end   = vh["Date"].iloc[-1].date() if vh is not None else today
    if vh is not None:
        st.caption(
            f"VIX data loaded: {vh['Date'].iloc[0].strftime('%d %b %Y')} → "
            f"{vh['Date'].iloc[-1].strftime('%d %b %Y')}  ·  {len(vh):,} rows  "
            f"(download / refresh in the **Futures & Options Pricer** tab)"
        )
    else:
        st.warning("No VIX data loaded — go to the **Futures & Options Pricer** tab to download it.")

    # ── Strategy selector (outside form — updates labels & legs immediately) ────
    strategy = st.selectbox(
        "Strategy",
        ["Short Put Spread", "Long Put Spread", "Short Put", "Long Put"],
        key="strategy",
        help=(
            "Short Put Spread: sell higher-K put, buy lower-K put as hedge  |  "
            "Long Put Spread: buy higher-K put, sell lower-K put as hedge  |  "
            "Short Put: sell a single put  |  Long Put: buy a single put"
        ),
    )
    is_spread = strategy in ["Short Put Spread", "Long Put Spread"]
    is_short  = strategy in ["Short Put Spread", "Short Put"]
    LEG_LABELS = {
        "Short Put Spread": ("Short Put  (sell)",  "Long Put  (buy — hedge)"),
        "Long Put Spread":  ("Long Put  (buy)",    "Short Put  (sell — hedge)"),
        "Short Put":        ("Put  (sell)",        None),
        "Long Put":         ("Put  (buy)",         None),
    }
    leg1_label, leg2_label = LEG_LABELS[strategy]

    # ── All other inputs wrapped in a form — no re-run until Run Simulation ───
    with st.form("sim_form"):
        st.subheader("Strategy Parameters")
        sp1, sp2 = st.columns(2, gap="large")
        with sp1:
            st.markdown(f"**{leg1_label}**")
            short_strike    = st.number_input("Strike",           min_value=1.0,  max_value=100.0, value=19.0,  step=0.5,  format="%.2f", key="short_strike")
            short_vol_shift = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,   value=-0.05, step=0.05, format="%.2f", key="short_vol_shift")
        with sp2:
            if is_spread:
                st.markdown(f"**{leg2_label}**")
                long_strike    = st.number_input("Strike",           min_value=1.0,  max_value=100.0, value=15.0, step=0.5,  format="%.2f", key="long_strike")
                long_vol_shift = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,   value=0.05, step=0.05, format="%.2f", key="long_vol_shift")
            else:
                long_strike, long_vol_shift = None, 0.0

        st.divider()
        st.markdown("**Trade Execution**")
        tc1, tc2, tc3 = st.columns(3, gap="large")
        with tc1:
            min_days_to_expiry = st.number_input(
                "Minimum calendar days before a Wednesday expiry",
                min_value=1, max_value=60, value=10, step=1, key="min_dte")
        with tc2:
            trading_cost = st.number_input(
                "Trading cost on premium  ($)",
                min_value=0.0, max_value=10.0, value=0.05, step=0.01, format="%.2f", key="trading_cost")
        with tc3:
            n_contracts = st.number_input(
                "Number of options  (lot size 100)",
                min_value=1, max_value=10000, value=1, step=1, key="n_contracts")

        st.divider()
        st.subheader("Simulation Date Range")
        dr1, dr2 = st.columns(2, gap="large")
        with dr1:
            sim_start = st.date_input("Start date", value=data_start,
                                      min_value=data_start, max_value=data_end, key="sim_start")
        with dr2:
            sim_end = st.date_input("End date", value=data_end,
                                    min_value=data_start, max_value=data_end, key="sim_end")

        st.divider()
        run_sim = st.form_submit_button("▶ Run Simulation", type="primary")

    # ── Filters (outside form — toggle flips immediately grey/ungrey inputs) ──
    st.divider()
    # VIX entry range filter
    apply_vix_filter = st.toggle(
        "Filter by VIX entry level  (only trade when VIX is within the range below)",
        value=True, key="apply_vix_filter")
    vf_col1, vf_col2 = st.columns(2, gap="large")
    with vf_col1:
        vix_entry_min = st.number_input(
            "Min Entry VIX", min_value=0.0, max_value=200.0,
            value=10.0, step=0.5, format="%.1f",
            key="vix_entry_min", disabled=not apply_vix_filter)
    with vf_col2:
        vix_entry_max = st.number_input(
            "Max Entry VIX", min_value=0.0, max_value=200.0,
            value=40.0, step=0.5, format="%.1f",
            key="vix_entry_max", disabled=not apply_vix_filter)
    if not apply_vix_filter:
        vix_entry_min, vix_entry_max = 0.0, 9999.0

    st.divider()
    # Premium filter
    apply_min_premium = st.toggle(
        "Filter by premium  (only trade when premium meets the threshold below)",
        value=True, key="apply_min_premium")
    if is_short:
        min_premium = st.number_input(
            "Minimum premium to receive  ($)",
            min_value=0.0, max_value=10.0, value=0.15, step=0.01, format="%.2f",
            key="min_premium_short", disabled=not apply_min_premium)
    else:
        min_premium = st.number_input(
            "Maximum premium to pay  ($)",
            min_value=0.0, max_value=10.0, value=2.0, step=0.01, format="%.2f",
            key="min_premium_long", disabled=not apply_min_premium)
    if not apply_min_premium:
        min_premium = 0.0 if is_short else 9999.0

    st.divider()

    # ── Simulation (runs only when form is submitted) ─────────────────────────
    st.subheader("Put Spread Premium Simulation")

    if vh is None:
        st.info("Download VIX historical data first.")
    elif run_sim and sim_start >= sim_end:
        st.warning("Start date must be before end date.")
    else:
        if run_sim:
            cutoff = sim_end - datetime.timedelta(days=int(min_days_to_expiry))
            mask   = (vh["Date"].dt.date >= sim_start) & (vh["Date"].dt.date <= cutoff)
            sim_df = vh[mask].copy().reset_index(drop=True)

            if sim_df.empty:
                st.warning("No data in range after applying minimum DTE cutoff.")
            else:
                # Build a fast date → closing VIX lookup for P&L at expiry
                import bisect
                _dates_arr  = [d.date() for d in vh["Date"]]
                _closes_arr = vh["CLOSE"].tolist()
                vix_lookup  = dict(zip(_dates_arr, _closes_arr))

                # Binary search: nearest trading day on or after exp_date
                def lookup_expiry_vix(exp_date):
                    if exp_date in vix_lookup:
                        return vix_lookup[exp_date]
                    idx = bisect.bisect_left(_dates_arr, exp_date)
                    if idx < len(_dates_arr):
                        return _closes_arr[idx]
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

                    if i % 500 == 0:
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

            # ── P&L distribution ──────────────────────────────────────────────
            st.divider()
            st.subheader("P&L Distribution")

            _pnl = entered["P&L ($)"].dropna()
            _gains  = _pnl[_pnl >= 0]
            _losses = _pnl[_pnl <  0]

            _bin_size = max(1.0, round((_pnl.max() - _pnl.min()) / 60 / 5) * 5)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=_gains,
                xbins=dict(size=_bin_size),
                marker_color="#4caf7d",
                name="Gains",
                opacity=0.85,
            ))
            fig_dist.add_trace(go.Histogram(
                x=_losses,
                xbins=dict(size=_bin_size),
                marker_color="#ff6b6b",
                name="Losses",
                opacity=0.85,
            ))
            fig_dist.add_vline(x=0, line_color="white", line_dash="dot", line_width=1.5)
            fig_dist.update_layout(
                barmode="overlay",
                xaxis=dict(title="P&L ($)", tickprefix="$", tickformat=",.0f"),
                yaxis=dict(title="Number of trades"),
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=30, b=40), height=380,
            )
            _pct_wins = 100 * len(_gains) / len(_pnl)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption(
                f"Gains: {len(_gains):,} trades ({_pct_wins:.1f}%)  ·  "
                f"Losses: {len(_losses):,} trades ({100-_pct_wins:.1f}%)  ·  "
                f"Bin size: ${_bin_size:,.0f}"
            )

            # ── VIX bucket charts ────────────────────────────────────────────
            st.divider()
            st.subheader("Analysis by Entry VIX Level")

            vix_min = int(np.floor(entered["Spot VIX"].min()))
            vix_max = int(np.ceil( entered["Spot VIX"].max()))
            VIX_BINS, VIX_LABELS = build_vix_buckets(vix_min, vix_max)

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
                "P&L ($)":        st.column_config.NumberColumn(   "P&L ($)",   width=80, format="$%.2f"),
            }
            TRADE_FMT = {
                "Spot VIX":     "{:.3f}",
                "F (model)":    "{:.3f}",
                "Leg 1 Price":  "{:.3f}",
                "Leg 2 Price":  "{:.3f}",
                "Net Premium":  "{:.3f}",
                "Expiry VIX":   "{:.3f}",
                "Expiry Value": "{:.3f}",
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
# TAB 3: Dynamic Put Spreads Strikes
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dynamic:
    st.header("Dynamic Put Spreads Strikes — Historical Simulation")
    st.caption(
        "Short Put Spread where the strikes are chosen automatically each day "
        "based on the VIX entry level."
    )

    # ── VIX data status ───────────────────────────────────────────────────────
    _dvh = st.session_state.vix_history
    _d_data_start = _dvh["Date"].iloc[0].date()  if _dvh is not None else datetime.date(1990, 1, 2)
    _d_data_end   = _dvh["Date"].iloc[-1].date() if _dvh is not None else today
    if _dvh is not None:
        st.caption(
            f"VIX data loaded: {_dvh['Date'].iloc[0].strftime('%d %b %Y')} → "
            f"{_dvh['Date'].iloc[-1].strftime('%d %b %Y')}  ·  {len(_dvh):,} rows  "
            f"(download / refresh in the **Futures & Options Pricer** tab)"
        )
    else:
        st.warning("No VIX data loaded — go to the **Futures & Options Pricer** tab to download it.")

    # ── Inputs (form) ─────────────────────────────────────────────────────────
    with st.form("dyn_sim_form"):
        st.subheader("Strike Selection")
        dk1, dk2, dk3 = st.columns(3, gap="large")
        with dk1:
            dyn_higher_dist = st.select_slider(
                "Higher strike distance from VIX",
                options=list(range(-10, 0)) + list(range(1, 11)),
                value=-1,
                key="dyn_higher_dist",
                help=(
                    "−1 = nearest integer strictly below VIX entry level  |  "
                    "+1 = nearest integer strictly above VIX entry level  |  "
                    "−2 / +2 = next one further down / up, etc."
                ),
            )
        with dk2:
            dyn_spread_distance = st.number_input(
                "Distance between strikes  (higher − lower)",
                min_value=1, max_value=50, value=4, step=1,
                key="dyn_spread_distance",
                help="Lower strike = higher strike − this value",
            )
        with dk3:
            st.markdown("&nbsp;")   # spacer
            st.markdown(
                "_Example: VIX = 18.4, distance = −1, width = 4  →  "
                "higher strike = **18**, lower strike = **14**_"
            )

        st.divider()
        st.subheader("Volatility Shifts")
        dv1, dv2 = st.columns(2, gap="large")
        with dv1:
            st.markdown("**Short Put  (sell — higher strike)**")
            dyn_short_vol_shift = st.number_input(
                "Volatility Shift", min_value=-2.0, max_value=2.0,
                value=-0.05, step=0.05, format="%.2f", key="dyn_short_vol_shift")
        with dv2:
            st.markdown("**Long Put  (buy — lower strike / hedge)**")
            dyn_long_vol_shift = st.number_input(
                "Volatility Shift", min_value=-2.0, max_value=2.0,
                value=0.05, step=0.05, format="%.2f", key="dyn_long_vol_shift")

        st.divider()
        st.markdown("**Trade Execution**")
        dtc1, dtc2, dtc3 = st.columns(3, gap="large")
        with dtc1:
            dyn_min_dte = st.number_input(
                "Minimum calendar days before a Wednesday expiry",
                min_value=1, max_value=60, value=10, step=1, key="dyn_min_dte")
        with dtc2:
            dyn_trading_cost = st.number_input(
                "Trading cost on premium  ($)",
                min_value=0.0, max_value=10.0, value=0.05, step=0.01,
                format="%.2f", key="dyn_trading_cost")
        with dtc3:
            dyn_n_contracts = st.number_input(
                "Number of options  (lot size 100)",
                min_value=1, max_value=10000, value=1, step=1, key="dyn_n_contracts")

        st.divider()
        st.subheader("Simulation Date Range")
        ddr1, ddr2 = st.columns(2, gap="large")
        with ddr1:
            dyn_sim_start = st.date_input(
                "Start date", value=_d_data_start,
                min_value=_d_data_start, max_value=_d_data_end, key="dyn_sim_start")
        with ddr2:
            dyn_sim_end = st.date_input(
                "End date", value=_d_data_end,
                min_value=_d_data_start, max_value=_d_data_end, key="dyn_sim_end")

        st.divider()
        dyn_run_sim = st.form_submit_button("▶ Run Simulation", type="primary")

    # ── Filters (outside form — toggle flips immediately grey/ungrey inputs) ──
    st.divider()
    # VIX entry range filter
    dyn_apply_vix_filter = st.toggle(
        "Filter by VIX entry level  (only trade when VIX is within the range below)",
        value=True, key="dyn_apply_vix_filter")
    _dyn_vf1, _dyn_vf2 = st.columns(2, gap="large")
    with _dyn_vf1:
        dyn_vix_entry_min = st.number_input(
            "Min Entry VIX", min_value=0.0, max_value=200.0,
            value=10.0, step=0.5, format="%.1f",
            key="dyn_vix_entry_min", disabled=not dyn_apply_vix_filter)
    with _dyn_vf2:
        dyn_vix_entry_max = st.number_input(
            "Max Entry VIX", min_value=0.0, max_value=200.0,
            value=40.0, step=0.5, format="%.1f",
            key="dyn_vix_entry_max", disabled=not dyn_apply_vix_filter)
    if not dyn_apply_vix_filter:
        dyn_vix_entry_min, dyn_vix_entry_max = 0.0, 9999.0

    st.divider()
    # Premium filter
    dyn_apply_premium = st.toggle(
        "Filter by premium  (only trade when premium meets the threshold below)",
        value=True, key="dyn_apply_premium")
    dyn_min_premium = st.number_input(
        "Minimum premium to receive  ($)",
        min_value=0.0, max_value=10.0, value=0.15, step=0.01, format="%.2f",
        key="dyn_min_premium", disabled=not dyn_apply_premium)
    if not dyn_apply_premium:
        dyn_min_premium = 0.0

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────────
    st.subheader("Dynamic Put Spread Simulation")

    if _dvh is None:
        st.info("Download VIX historical data first.")
    elif dyn_run_sim and dyn_sim_start >= dyn_sim_end:
        st.warning("Start date must be before end date.")
    else:
        if dyn_run_sim:
            import bisect as _bisect

            _d_cutoff = dyn_sim_end - datetime.timedelta(days=int(dyn_min_dte))
            _d_mask   = (
                (_dvh["Date"].dt.date >= dyn_sim_start) &
                (_dvh["Date"].dt.date <= _d_cutoff)
            )
            _d_sim_df = _dvh[_d_mask].copy().reset_index(drop=True)

            if _d_sim_df.empty:
                st.warning("No data in range after applying minimum DTE cutoff.")
            else:
                # Fast expiry-VIX lookup
                _d_dates_arr  = [d.date() for d in _dvh["Date"]]
                _d_closes_arr = _dvh["CLOSE"].tolist()
                _d_vix_lookup = dict(zip(_d_dates_arr, _d_closes_arr))

                def _dyn_lookup_expiry_vix(exp_date):
                    if exp_date in _d_vix_lookup:
                        return _d_vix_lookup[exp_date]
                    idx = _bisect.bisect_left(_d_dates_arr, exp_date)
                    if idx < len(_d_dates_arr):
                        return _d_closes_arr[idx]
                    return None

                def _dyn_higher_strike(spot_vix, dist):
                    """
                    dist < 0 → nearest integer strictly below spot_vix, then go dist steps further down.
                    dist > 0 → nearest integer strictly above spot_vix, then go dist steps further up.
                    """
                    if dist < 0:
                        return float(math.ceil(spot_vix) + dist)   # ceil - |dist|
                    else:
                        return float(math.floor(spot_vix) + dist)  # floor + dist

                _d_rows = []
                _d_progress = st.progress(0, text="Computing…")
                _d_total = len(_d_sim_df)

                for _d_i, (_, _d_row) in enumerate(_d_sim_df.iterrows()):
                    _d_date     = _d_row["Date"].date()
                    _d_spot_vix = float(_d_row["CLOSE"])
                    _d_expiry   = next_wednesday(_d_date, int(dyn_min_dte))
                    _d_tau      = (_d_expiry - _d_date).days / 365.0

                    _d_base_sigma = sigma_from_vix(_d_spot_vix)
                    _d_sigma1     = max(_d_base_sigma + dyn_short_vol_shift, 0.01)
                    _d_sigma2     = max(_d_base_sigma + dyn_long_vol_shift,  0.01)

                    _d_F = vix_futures_price(_d_spot_vix, kappa, theta_bar, _d_base_sigma, r, _d_tau)

                    _d_k_high = _dyn_higher_strike(_d_spot_vix, dyn_higher_dist)
                    _d_k_low  = max(_d_k_high - dyn_spread_distance, 1.0)

                    _d_price1 = black76(_d_F, _d_k_high, r, _d_sigma1, _d_tau, "put")["price"]
                    _d_price2 = black76(_d_F, _d_k_low,  r, _d_sigma2, _d_tau, "put")["price"]

                    _d_net_premium = max(_d_price1 - _d_price2 - dyn_trading_cost, 0.0)

                    _d_in_vix_range  = dyn_vix_entry_min <= _d_spot_vix <= dyn_vix_entry_max
                    _d_trade_entered = _d_net_premium >= dyn_min_premium and _d_in_vix_range

                    _d_expiry_vix   = _dyn_lookup_expiry_vix(_d_expiry) if _d_trade_entered else None
                    if _d_trade_entered and _d_expiry_vix is not None:
                        _d_expiry_value = (max(_d_k_high - _d_expiry_vix, 0.0)
                                           - max(_d_k_low  - _d_expiry_vix, 0.0))
                        _d_pnl = round(dyn_n_contracts * 100 * (_d_net_premium - _d_expiry_value), 2)
                    else:
                        _d_expiry_value = None
                        _d_pnl          = None

                    _d_rows.append({
                        "Date":             _d_date,
                        "Expiry (Wed)":     _d_expiry,
                        "Days to Expiry":   (_d_expiry - _d_date).days,
                        "Spot VIX":         round(_d_spot_vix, 2),
                        "Higher Strike":    _d_k_high,
                        "Lower Strike":     _d_k_low,
                        "Base σ":           f"{_d_base_sigma * 100:.1f}%",
                        "F (model)":        round(_d_F, 4),
                        "Leg 1 σ":          f"{_d_sigma1 * 100:.1f}%",
                        "Leg 1 Price":      round(_d_price1, 4),
                        "Leg 2 σ":          f"{_d_sigma2 * 100:.1f}%",
                        "Leg 2 Price":      round(_d_price2, 4),
                        "Net Premium":      round(_d_net_premium, 4),
                        "Trade Entered":    _d_trade_entered,
                        "Expiry VIX":       round(_d_expiry_vix, 2)    if _d_expiry_vix    is not None else None,
                        "Expiry Value":     round(_d_expiry_value, 4)   if _d_expiry_value  is not None else None,
                        "P&L ($)":          _d_pnl,
                    })

                    if _d_i % 500 == 0:
                        _d_progress.progress(_d_i / _d_total, text=f"Computing… {_d_i:,}/{_d_total:,}")

                _d_progress.empty()
                st.session_state.dyn_sim_results = pd.DataFrame(_d_rows)

        # ── Display results ───────────────────────────────────────────────────
        if st.session_state.dyn_sim_results is not None:
            _dr = st.session_state.dyn_sim_results
            _de = _dr[_dr["Trade Entered"]].copy()
            _dpnl_vals = _de["P&L ($)"].dropna()
            _dn_trades = len(_dpnl_vals)

            _d_total_pnl    = _dpnl_vals.sum()
            _d_avg_pnl      = _dpnl_vals.mean()              if _dn_trades else None
            _d_win_rate     = (_dpnl_vals > 0).mean() * 100  if _dn_trades else None
            _d_max_win      = _dpnl_vals.max()                if _dn_trades else None
            _d_max_loss     = _dpnl_vals.min()                if _dn_trades else None
            _d_avg_win      = _dpnl_vals[_dpnl_vals > 0].mean() if (_dpnl_vals > 0).any() else None
            _d_avg_loss     = _dpnl_vals[_dpnl_vals < 0].mean() if (_dpnl_vals < 0).any() else None
            _d_profit_factor = (
                _dpnl_vals[_dpnl_vals > 0].sum() / abs(_dpnl_vals[_dpnl_vals < 0].sum())
                if (_dpnl_vals < 0).any() else None
            )
            _d_sharpe = (
                _dpnl_vals.mean() / _dpnl_vals.std() * np.sqrt(252)
                if _dn_trades > 1 else None
            )

            if _de.empty:
                st.warning("No trades were entered. Try relaxing the premium threshold.")
                st.stop()

            # ── Statistics ───────────────────────────────────────────────────
            st.divider()
            st.subheader("Strategy Statistics")
            _dr1 = st.columns(4)
            _dr1[0].metric("Total P&L",      f"${_d_total_pnl:,.0f}")
            _dr1[1].metric("Trades entered", f"{_dn_trades:,}")
            _dr1[2].metric("Win rate",        f"{_d_win_rate:.1f}%" if _d_win_rate is not None else "—")
            _dr1[3].metric("Avg P&L / trade", f"${_d_avg_pnl:,.0f}"  if _d_avg_pnl  is not None else "—")
            _dr2 = st.columns(4)
            _dr2[0].metric("Max win",  f"${_d_max_win:,.0f}"  if _d_max_win  is not None else "—")
            _dr2[1].metric("Max loss", f"${_d_max_loss:,.0f}" if _d_max_loss is not None else "—")
            _dr2[2].metric("Avg win / Avg loss",
                           f"{abs(_d_avg_win/_d_avg_loss):.2f}x"
                           if _d_avg_win is not None and _d_avg_loss is not None else "—")
            _dr2[3].metric("Profit factor", f"{_d_profit_factor:.2f}" if _d_profit_factor is not None else "∞")

            # ── Annual P&L ────────────────────────────────────────────────────
            st.divider()
            st.subheader("Annual P&L")
            _de["Year"] = pd.to_datetime(_de["Date"]).dt.year
            _d_annual = _de.groupby("Year")["P&L ($)"].sum().reset_index()
            _d_annual.columns = ["Year", "P&L ($)"]
            _d_fig_annual = go.Figure(go.Bar(
                x=_d_annual["Year"].astype(str),
                y=_d_annual["P&L ($)"],
                marker_color=["#4caf7d" if v >= 0 else "#ff6b6b" for v in _d_annual["P&L ($)"]],
                text=[f"${v:,.0f}" for v in _d_annual["P&L ($)"]],
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_annual.add_hline(y=0, line_color="white", line_width=1)
            _d_fig_annual.update_layout(
                xaxis_title="Year", yaxis_title="P&L ($)",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=30, b=40), height=380,
                showlegend=False,
                yaxis=dict(tickprefix="$", tickformat=",.0f"),
            )
            st.plotly_chart(_d_fig_annual, use_container_width=True)

            # ── P&L vs Premium scatter ────────────────────────────────────────
            st.divider()
            st.subheader("P&L vs Spread Premium")
            _d_fig_scatter = go.Figure(go.Scatter(
                x=_de["Net Premium"],
                y=_de["P&L ($)"],
                mode="markers",
                marker=dict(
                    color=_de["P&L ($)"],
                    colorscale=[[0, "#ff6b6b"], [0.5, "#888888"], [1, "#4caf7d"]],
                    cmin=_de["P&L ($)"].min(), cmid=0, cmax=_de["P&L ($)"].max(),
                    size=5, opacity=0.6,
                    colorbar=dict(title="P&L ($)", tickprefix="$", tickformat=",.0f"),
                ),
                customdata=np.stack([
                    _de["Date"].astype(str),
                    _de["Spot VIX"],
                    _de["Higher Strike"],
                    _de["Lower Strike"],
                    _de["Expiry VIX"].fillna(0),
                ], axis=1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Premium: $%{x:.3f}<br>"
                    "P&L: $%{y:,.2f}<br>"
                    "Entry VIX: %{customdata[1]:.2f}<br>"
                    "Strikes: %{customdata[2]:.0f} / %{customdata[3]:.0f}<br>"
                    "Expiry VIX: %{customdata[4]:.2f}<extra></extra>"
                ),
            ))
            _d_fig_scatter.add_hline(y=0, line_color="white", line_dash="dot", line_width=1)
            _d_fig_scatter.update_layout(
                xaxis=dict(title="Spread Premium ($)", tickprefix="$"),
                yaxis=dict(title="P&L ($)", tickprefix="$", tickformat=",.0f"),
                template="plotly_dark",
                margin=dict(l=40, r=40, t=30, b=40), height=420,
            )
            st.plotly_chart(_d_fig_scatter, use_container_width=True)

            # ── P&L Distribution ──────────────────────────────────────────────
            st.divider()
            st.subheader("P&L Distribution")
            _d_pnl_s  = _de["P&L ($)"].dropna()
            _d_gains  = _d_pnl_s[_d_pnl_s >= 0]
            _d_losses = _d_pnl_s[_d_pnl_s <  0]
            _d_bin_sz = max(1.0, round((_d_pnl_s.max() - _d_pnl_s.min()) / 60 / 5) * 5)
            _d_fig_dist = go.Figure()
            _d_fig_dist.add_trace(go.Histogram(
                x=_d_gains,  xbins=dict(size=_d_bin_sz),
                marker_color="#4caf7d", name="Gains",  opacity=0.85))
            _d_fig_dist.add_trace(go.Histogram(
                x=_d_losses, xbins=dict(size=_d_bin_sz),
                marker_color="#ff6b6b", name="Losses", opacity=0.85))
            _d_fig_dist.add_vline(x=0, line_color="white", line_dash="dot", line_width=1.5)
            _d_fig_dist.update_layout(
                barmode="overlay",
                xaxis=dict(title="P&L ($)", tickprefix="$", tickformat=",.0f"),
                yaxis=dict(title="Number of trades"),
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=30, b=40), height=380,
            )
            _d_pct_wins = 100 * len(_d_gains) / len(_d_pnl_s)
            st.plotly_chart(_d_fig_dist, use_container_width=True)
            st.caption(
                f"Gains: {len(_d_gains):,} ({_d_pct_wins:.1f}%)  ·  "
                f"Losses: {len(_d_losses):,} ({100-_d_pct_wins:.1f}%)  ·  "
                f"Bin size: ${_d_bin_sz:,.0f}"
            )

            # ── Analysis by Entry VIX level ───────────────────────────────────
            st.divider()
            st.subheader("Analysis by Entry VIX Level")
            _d_vix_min = int(np.floor(_de["Spot VIX"].min()))
            _d_vix_max = int(np.ceil( _de["Spot VIX"].max()))
            _D_BINS, _D_LABELS = build_vix_buckets(_d_vix_min, _d_vix_max)

            _de["VIX Bucket"] = pd.cut(
                _de["Spot VIX"], bins=_D_BINS, labels=_D_LABELS,
                right=False, include_lowest=True)
            _d_bstats = (
                _de.groupby("VIX Bucket", observed=True)["P&L ($)"]
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
            _d_bstats = _d_bstats[_d_bstats["count"] > 0]
            _d_bx = _d_bstats["VIX Bucket"].astype(str)

            _d_fig_wr = go.Figure(go.Bar(
                x=_d_bx, y=_d_bstats["win_rate"],
                marker_color=["#4caf7d" if v >= 50 else "#ff6b6b" for v in _d_bstats["win_rate"]],
                text=[f"{v:.1f}%<br><sub>{int(c)}</sub>" for v, c in zip(_d_bstats["win_rate"], _d_bstats["count"])],
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_wr.add_hline(y=50, line_dash="dash", line_color="white", line_width=1,
                                annotation_text="50%", annotation_position="right")
            _d_fig_wr.update_layout(
                title="Win Rate % by Entry VIX", xaxis_title="Entry VIX range",
                yaxis=dict(title="Win rate (%)", range=[0, 115], ticksuffix="%"),
                template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40),
                height=370, showlegend=False,
            )
            st.plotly_chart(_d_fig_wr, use_container_width=True)

            _d_fig_ap = go.Figure(go.Bar(
                x=_d_bx, y=_d_bstats["avg_pnl"],
                marker_color=["#4caf7d" if v >= 0 else "#ff6b6b" for v in _d_bstats["avg_pnl"]],
                text=[f"${v:,.0f}<br><sub>{int(c)}</sub>" for v, c in zip(_d_bstats["avg_pnl"], _d_bstats["count"])],
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_ap.add_hline(y=0, line_color="white", line_width=1)
            _d_fig_ap.update_layout(
                title="Average Trade P&L by Entry VIX", xaxis_title="Entry VIX range",
                yaxis=dict(title="Avg P&L ($)", tickprefix="$", tickformat=",.0f"),
                template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40),
                height=370, showlegend=False,
            )
            st.plotly_chart(_d_fig_ap, use_container_width=True)

            _d_fig_wl = go.Figure()
            _d_fig_wl.add_trace(go.Bar(
                name="Wins",   x=_d_bx, y=_d_bstats["n_wins"],
                marker_color="#4caf7d",
                text=_d_bstats["n_wins"].astype(int),
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_wl.add_trace(go.Bar(
                name="Losses", x=_d_bx, y=_d_bstats["n_losses"],
                marker_color="#ff6b6b",
                text=_d_bstats["n_losses"].astype(int),
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_wl.update_layout(
                title="Number of Wins vs Losses by Entry VIX", xaxis_title="Entry VIX range",
                yaxis_title="Number of trades", barmode="group",
                template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40),
                height=370, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(_d_fig_wl, use_container_width=True)

            _d_fig_wlp = go.Figure()
            _d_fig_wlp.add_trace(go.Bar(
                name="Avg Win",  x=_d_bx, y=_d_bstats["avg_win"],
                marker_color="#4caf7d",
                text=[f"${v:,.0f}" if pd.notna(v) else "" for v in _d_bstats["avg_win"]],
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_wlp.add_trace(go.Bar(
                name="Avg Loss", x=_d_bx, y=_d_bstats["avg_loss"],
                marker_color="#ff6b6b",
                text=[f"${v:,.0f}" if pd.notna(v) else "" for v in _d_bstats["avg_loss"]],
                textposition="outside", textfont=dict(size=11),
            ))
            _d_fig_wlp.add_hline(y=0, line_color="white", line_width=1)
            _d_fig_wlp.update_layout(
                title="Average Win P&L vs Average Loss P&L by Entry VIX", xaxis_title="Entry VIX range",
                yaxis=dict(title="Avg P&L ($)", tickprefix="$", tickformat=",.0f"),
                barmode="group", template="plotly_dark",
                margin=dict(l=40, r=40, t=50, b=40), height=370,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(_d_fig_wlp, use_container_width=True)

            _d_count_cap = "  ·  ".join(
                f"{row['VIX Bucket']}: {int(row['count'])}" for _, row in _d_bstats.iterrows()
            )
            st.caption(f"Trades per VIX bucket  —  {_d_count_cap}")

            # ── Trade tables ─────────────────────────────────────────────────
            def _d_colour_pnl(val):
                if not isinstance(val, (int, float)) or pd.isna(val):
                    return ""
                return "color: #4caf7d" if val >= 0 else "color: #ff6b6b"

            _D_COL_CFG = {
                "Date":           st.column_config.DateColumn(     "Date",      width=80),
                "Expiry (Wed)":   st.column_config.DateColumn(     "Expiry",    width=80),
                "Days to Expiry": st.column_config.NumberColumn(   "DTE",       width=40),
                "Spot VIX":       st.column_config.NumberColumn(   "VIX",       width=55),
                "Higher Strike":  st.column_config.NumberColumn(   "K↑",        width=45),
                "Lower Strike":   st.column_config.NumberColumn(   "K↓",        width=45),
                "Base σ":         st.column_config.TextColumn(     "Bσ",        width=50),
                "F (model)":      st.column_config.NumberColumn(   "F",         width=55),
                "Leg 1 σ":        st.column_config.TextColumn(     "L1σ",       width=50),
                "Leg 1 Price":    st.column_config.NumberColumn(   "L1 $",      width=60),
                "Leg 2 σ":        st.column_config.TextColumn(     "L2σ",       width=50),
                "Leg 2 Price":    st.column_config.NumberColumn(   "L2 $",      width=60),
                "Net Premium":    st.column_config.NumberColumn(   "Prem",      width=60),
                "Trade Entered":  st.column_config.CheckboxColumn( "In?",       width=40),
                "Expiry VIX":     st.column_config.NumberColumn(   "Exp VIX",   width=60),
                "Expiry Value":   st.column_config.NumberColumn(   "Exp Val",   width=60),
                "P&L ($)":        st.column_config.NumberColumn(   "P&L",       width=70, format="$%.2f"),
            }
            _D_FMT = {
                "Spot VIX":     "{:.3f}",
                "Higher Strike":"{:.0f}",
                "Lower Strike": "{:.0f}",
                "F (model)":    "{:.3f}",
                "Leg 1 Price":  "{:.3f}",
                "Leg 2 Price":  "{:.3f}",
                "Net Premium":  "{:.3f}",
                "Expiry VIX":   "{:.3f}",
                "Expiry Value": "{:.3f}",
            }

            def _d_render_table(df):
                st.dataframe(
                    df.style.format(_D_FMT, na_rep="—").map(_d_colour_pnl, subset=["P&L ($)"]),
                    use_container_width=True, hide_index=True,
                    column_config=_D_COL_CFG,
                )

            _d_sim_caption = (
                f"Min DTE = {dyn_min_dte}d  ·  "
                f"Higher strike dist = {dyn_higher_dist:+d}  ·  "
                f"Spread width = {dyn_spread_distance}  ·  "
                f"Short σ {dyn_short_vol_shift:+.2f}  ·  Long σ {dyn_long_vol_shift:+.2f}  ·  "
                f"Cost ${dyn_trading_cost:.2f}  ·  n={dyn_n_contracts}"
            )

            st.divider()
            st.markdown("**Spot check — 5 random example dates**")
            _d_sample = min(5, len(_dr))
            _d_ex_df  = _dr.sample(n=_d_sample).sort_values("Date").reset_index(drop=True)
            _d_render_table(_d_ex_df)
            st.caption(_d_sim_caption)

            st.divider()
            if st.button(
                f"Show full trade list  ({_dn_trades:,} entered trades)",
                key="dyn_show_full_trades",
            ):
                _d_full_df = _de.drop(columns=["VIX Bucket"], errors="ignore").sort_values("Date").reset_index(drop=True)
                _d_render_table(_d_full_df)
                st.caption(_d_sim_caption)

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Dynamic Strategy Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_optimizer:
    st.header("🔍 Dynamic Strategy Optimizer")
    st.caption(
        "Grid search over **Higher Strike Distance** and **Spread Width** to find "
        "the best parameters for each VIX entry level range. "
        "Uses the same model as the Dynamic Put Spreads tab."
    )

    _ov = st.session_state.vix_history
    _o_data_start = _ov["Date"].iloc[0].date()  if _ov is not None else datetime.date(1990, 1, 2)
    _o_data_end   = _ov["Date"].iloc[-1].date() if _ov is not None else today
    if _ov is not None:
        st.caption(
            f"VIX data loaded: {_ov['Date'].iloc[0].strftime('%d %b %Y')} → "
            f"{_ov['Date'].iloc[-1].strftime('%d %b %Y')}  ·  {len(_ov):,} rows"
        )
    else:
        st.warning("No VIX data loaded — go to the **Futures & Options Pricer** tab to download it.")

    # ── Form ──────────────────────────────────────────────────────────────────
    with st.form("opt_form"):
        st.subheader("Volatility Shifts")
        _ov1, _ov2 = st.columns(2, gap="large")
        with _ov1:
            st.markdown("**Short Put  (sell — higher strike)**")
            opt_short_vs = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,
                                           value=-0.05, step=0.05, format="%.2f", key="opt_short_vs")
        with _ov2:
            st.markdown("**Long Put  (buy — lower strike)**")
            opt_long_vs = st.number_input("Volatility Shift", min_value=-2.0, max_value=2.0,
                                          value=0.05, step=0.05, format="%.2f", key="opt_long_vs")

        st.divider()
        st.markdown("**Trade Execution**")
        _otc1, _otc2, _otc3 = st.columns(3, gap="large")
        with _otc1:
            opt_min_dte = st.number_input("Minimum calendar days before a Wednesday expiry",
                                          min_value=1, max_value=60, value=10, step=1, key="opt_min_dte")
        with _otc2:
            opt_cost = st.number_input("Trading cost on premium  ($)", min_value=0.0, max_value=10.0,
                                       value=0.05, step=0.01, format="%.2f", key="opt_cost")
        with _otc3:
            opt_n = st.number_input("Number of options  (lot size 100)",
                                    min_value=1, max_value=10000, value=1, step=1, key="opt_n")

        st.divider()
        st.subheader("Simulation Date Range")
        _odr1, _odr2 = st.columns(2, gap="large")
        with _odr1:
            opt_start = st.date_input("Start date", value=_o_data_start,
                                      min_value=_o_data_start, max_value=_o_data_end, key="opt_start")
        with _odr2:
            opt_end = st.date_input("End date", value=_o_data_end,
                                    min_value=_o_data_start, max_value=_o_data_end, key="opt_end")

        st.divider()
        st.subheader("Search Grid")
        _og1, _og2, _og3, _og4 = st.columns(4, gap="large")
        with _og1:
            opt_h_min = st.number_input("Min H-Strike Distance", min_value=-10, max_value=-1,
                                        value=-5, step=1, key="opt_h_min",
                                        help="Most negative higher-strike distance to test (e.g. −5)")
        with _og2:
            opt_h_max = st.number_input("Max H-Strike Distance", min_value=1, max_value=10,
                                        value=5, step=1, key="opt_h_max",
                                        help="Most positive higher-strike distance to test (e.g. +5)")
        with _og3:
            opt_s_min = st.number_input("Min Spread Width", min_value=1, max_value=50,
                                        value=1, step=1, key="opt_s_min")
        with _og4:
            opt_s_max = st.number_input("Max Spread Width", min_value=1, max_value=50,
                                        value=15, step=1, key="opt_s_max")

        st.divider()
        opt_min_trades = st.number_input(
            "Minimum trades per bucket to include  (removes buckets with too few data points)",
            min_value=1, max_value=200, value=10, step=1, key="opt_min_trades")

        st.divider()
        opt_run = st.form_submit_button("▶ Run Optimisation", type="primary")

    # ── Filters (outside form — live toggle) ──────────────────────────────────
    st.divider()
    opt_apply_vix = st.toggle(
        "Filter by VIX entry level  (only include days when VIX is within the range below)",
        value=True, key="opt_apply_vix")
    _ovf1, _ovf2 = st.columns(2, gap="large")
    with _ovf1:
        opt_vix_lo = st.number_input("Min Entry VIX", min_value=0.0, max_value=200.0,
                                     value=10.0, step=0.5, format="%.1f",
                                     key="opt_vix_lo", disabled=not opt_apply_vix)
    with _ovf2:
        opt_vix_hi = st.number_input("Max Entry VIX", min_value=0.0, max_value=200.0,
                                     value=40.0, step=0.5, format="%.1f",
                                     key="opt_vix_hi", disabled=not opt_apply_vix)
    if not opt_apply_vix:
        opt_vix_lo, opt_vix_hi = 0.0, 9999.0

    st.divider()
    opt_apply_prem = st.toggle(
        "Filter by premium  (only count trades when premium meets the threshold below)",
        value=True, key="opt_apply_prem")
    opt_min_prem = st.number_input("Minimum premium to receive  ($)",
                                   min_value=0.0, max_value=10.0, value=0.15, step=0.01,
                                   format="%.2f", key="opt_min_prem",
                                   disabled=not opt_apply_prem)
    if not opt_apply_prem:
        opt_min_prem = 0.0

    st.divider()

    # ── Engine ────────────────────────────────────────────────────────────────
    if _ov is None:
        st.info("Download VIX historical data first.")
    elif opt_run and opt_start >= opt_end:
        st.warning("Start date must be before end date.")
    else:
        if opt_run:
            import bisect as _obisect

            # ── Phase 1: precompute per-day invariants ─────────────────────
            _o_cutoff = opt_end - datetime.timedelta(days=int(opt_min_dte))
            _o_mask   = (_ov["Date"].dt.date >= opt_start) & (_ov["Date"].dt.date <= _o_cutoff)
            _o_sim_df = _ov[_o_mask].copy().reset_index(drop=True)

            if _o_sim_df.empty:
                st.warning("No data in range after applying minimum DTE cutoff.")
                st.stop()

            _o_dates_arr  = [d.date() for d in _ov["Date"]]
            _o_closes_arr = _ov["CLOSE"].tolist()
            _o_vix_lkup   = dict(zip(_o_dates_arr, _o_closes_arr))

            def _o_lkup_evix(exp_date):
                if exp_date in _o_vix_lkup:
                    return _o_vix_lkup[exp_date]
                idx = _obisect.bisect_left(_o_dates_arr, exp_date)
                return _o_closes_arr[idx] if idx < len(_o_closes_arr) else None

            _ph1 = st.progress(0, text="Phase 1 — precomputing daily data…")
            _o_rows1, _o_n1 = [], len(_o_sim_df)

            for _oi, (_, _or) in enumerate(_o_sim_df.iterrows()):
                _o_date  = _or["Date"].date()
                _o_spot  = float(_or["CLOSE"])
                _o_exp   = next_wednesday(_o_date, int(opt_min_dte))
                _o_tau   = (_o_exp - _o_date).days / 365.0
                _o_bsig  = sigma_from_vix(_o_spot)
                _o_rows1.append({
                    "spot":  _o_spot,
                    "F":     vix_futures_price(_o_spot, kappa, theta_bar, _o_bsig, r, _o_tau),
                    "sig1":  max(_o_bsig + opt_short_vs, 0.01),
                    "sig2":  max(_o_bsig + opt_long_vs,  0.01),
                    "tau":   _o_tau,
                    "evix":  _o_lkup_evix(_o_exp) or np.nan,
                })
                if _oi % 500 == 0:
                    _ph1.progress(_oi / _o_n1, text=f"Phase 1 — {_oi:,}/{_o_n1:,}")
            _ph1.empty()

            _o_pre = pd.DataFrame(_o_rows1)

            # Apply VIX entry filter
            _o_pre = _o_pre[
                (_o_pre["spot"] >= opt_vix_lo) & (_o_pre["spot"] <= opt_vix_hi)
            ].reset_index(drop=True)

            if _o_pre.empty:
                st.warning("No trading days in the VIX entry range. Widen the filter.")
                st.stop()

            # Extract numpy arrays
            _oa_spot = _o_pre["spot"].values
            _oa_F    = _o_pre["F"].values
            _oa_sig1 = _o_pre["sig1"].values
            _oa_sig2 = _o_pre["sig2"].values
            _oa_tau  = _o_pre["tau"].values
            _oa_evix = _o_pre["evix"].values

            # Assign VIX bucket to each day (fixed, independent of parameters)
            _o_vlo = int(np.floor(_oa_spot.min()))
            _o_vhi = int(np.ceil( _oa_spot.max()))
            _o_bins, _o_lbls = build_vix_buckets(_o_vlo, _o_vhi)
            _oa_bkt = pd.cut(_oa_spot, bins=_o_bins, labels=_o_lbls,
                             right=False, include_lowest=True).astype(str)

            # ── Phase 2: vectorised grid search ───────────────────────────
            _o_h_vals  = list(range(opt_h_min, 0)) + list(range(1, opt_h_max + 1))
            _o_s_vals  = list(range(opt_s_min, opt_s_max + 1))
            _o_total   = len(_o_h_vals) * len(_o_s_vals)
            _ph2 = st.progress(0, text=f"Phase 2 — searching {_o_total:,} combinations…")
            _o_cnt, _o_res = 0, []

            for _o_hd in _o_h_vals:
                # k_high per day (vectorised)
                _o_khi = (np.ceil(_oa_spot) + _o_hd if _o_hd < 0
                          else np.floor(_oa_spot) + _o_hd)
                for _o_sd in _o_s_vals:
                    _o_klo  = np.maximum(_o_khi - _o_sd, 1.0)
                    _o_p1   = _black76_put_vec(_oa_F, _o_khi, r, _oa_sig1, _oa_tau)
                    _o_p2   = _black76_put_vec(_oa_F, _o_klo, r, _oa_sig2, _oa_tau)
                    _o_prem = np.maximum(_o_p1 - _o_p2 - opt_cost, 0.0)
                    _o_in   = _o_prem >= opt_min_prem

                    _o_ev  = (np.maximum(_o_khi - _oa_evix, 0.0)
                              - np.maximum(_o_klo - _oa_evix, 0.0))
                    _o_pnl = np.where(
                        _o_in & ~np.isnan(_oa_evix),
                        opt_n * 100 * (_o_prem - _o_ev),
                        np.nan,
                    )

                    # Stats per VIX bucket
                    for _o_lbl in _o_lbls:
                        _o_bm  = (_oa_bkt == _o_lbl) & ~np.isnan(_o_pnl)
                        _o_bp  = _o_pnl[_o_bm]
                        if len(_o_bp) < int(opt_min_trades):
                            continue
                        _o_wins   = _o_bp[_o_bp > 0]
                        _o_losses = _o_bp[_o_bp < 0]
                        _o_res.append({
                            "VIX Bucket":    _o_lbl,
                            "H-Dist":        int(_o_hd),
                            "Spread Width":  int(_o_sd),
                            "Trades":        len(_o_bp),
                            "Avg P&L":       float(np.mean(_o_bp)),
                            "Total P&L":     float(np.sum(_o_bp)),
                            "Win Rate":      float((_o_bp > 0).mean() * 100),
                            "Max Win":       float(np.max(_o_bp)),
                            "Max Loss":      float(np.min(_o_bp)),
                            "Profit Factor": (float(_o_wins.sum() / abs(_o_losses.sum()))
                                              if len(_o_losses) > 0 else np.nan),
                        })

                    _o_cnt += 1
                    if _o_cnt % 20 == 0:
                        _ph2.progress(_o_cnt / _o_total,
                                      text=f"Phase 2 — {_o_cnt:,}/{_o_total:,} combinations…")

            _ph2.empty()

            if not _o_res:
                st.warning("No results — try lowering the minimum trades threshold or widening filters.")
                st.stop()

            st.session_state.opt_results = pd.DataFrame(_o_res)

        # ── Display ──────────────────────────────────────────────────────────
        if st.session_state.opt_results is not None:
            _opt = st.session_state.opt_results

            # Natural bucket order
            def _o_bkt_sort(s):
                try:    return float(s.split("–")[0])
                except: return 0.0

            _o_bkt_order = sorted(_opt["VIX Bucket"].unique(), key=_o_bkt_sort)

            _O_CFG = {
                "VIX Bucket":    st.column_config.TextColumn(   "VIX Range",   width=90),
                "H-Dist":        st.column_config.NumberColumn( "H-Dist",      width=65,  format="%d"),
                "Spread Width":  st.column_config.NumberColumn( "Width",       width=55,  format="%d"),
                "Trades":        st.column_config.NumberColumn( "Trades",      width=60,  format="%d"),
                "Avg P&L":       st.column_config.NumberColumn( "Avg P&L",     width=85,  format="$%.0f"),
                "Total P&L":     st.column_config.NumberColumn( "Total P&L",   width=90,  format="$%.0f"),
                "Win Rate":      st.column_config.NumberColumn( "Win %",       width=65,  format="%.1f"),
                "Max Win":       st.column_config.NumberColumn( "Max Win",     width=80,  format="$%.0f"),
                "Max Loss":      st.column_config.NumberColumn( "Max Loss",    width=80,  format="$%.0f"),
                "Profit Factor": st.column_config.NumberColumn( "P.Factor",    width=75,  format="%.2f"),
            }

            def _o_clr(val):
                if not isinstance(val, (int, float)) or pd.isna(val): return ""
                return "color: #4caf7d" if val >= 0 else "color: #ff6b6b"

            # ── Summary: best combo per bucket ────────────────────────────
            st.divider()
            st.subheader("Best Combination per VIX Bucket")
            # Deduplicate across the whole grid before picking the best per bucket
            _opt_dedup = (
                _opt.sort_values(["Avg P&L", "Spread Width"], ascending=[False, True])
                    .assign(_avg_key=(_opt["Avg P&L"] * 100).round(0))
                    .groupby(["VIX Bucket", "H-Dist", "_avg_key"], sort=False)
                    .first()
                    .reset_index()
                    .drop(columns=["_avg_key"])
            )
            _opt_best = (
                _opt_dedup.sort_values("Avg P&L", ascending=False)
                    .groupby("VIX Bucket", sort=False)
                    .first()
                    .reset_index()
            )
            _opt_best = _opt_best.assign(
                _s=_opt_best["VIX Bucket"].map(_o_bkt_sort)
            ).sort_values("_s").drop(columns=["_s"]).reset_index(drop=True)

            st.dataframe(
                _opt_best.style.map(_o_clr, subset=["Avg P&L", "Total P&L"]),
                use_container_width=True, hide_index=True,
                column_config=_O_CFG,
            )

            # ── Per-bucket top 5 + heatmap ────────────────────────────────
            st.divider()
            st.subheader("Top 5 per VIX Bucket")

            for _o_bkt in _o_bkt_order:
                _o_bdf = _opt[_opt["VIX Bucket"] == _o_bkt].sort_values(
                    ["Avg P&L", "Spread Width"], ascending=[False, True]
                )
                # Deduplicate: within each (H-Dist, rounded Avg P&L) keep only
                # the smallest Spread Width — eliminates floor-clamped duplicates
                _o_bdf_dedup = (
                    _o_bdf
                    .assign(_avg_key=(_o_bdf["Avg P&L"] * 100).round(0))
                    .groupby(["H-Dist", "_avg_key"], sort=False)
                    .first()
                    .reset_index()
                    .drop(columns=["_avg_key"])
                    .sort_values("Avg P&L", ascending=False)
                )
                _o_top5  = _o_bdf_dedup.head(5).reset_index(drop=True)
                _o_best  = _o_top5["Avg P&L"].iloc[0]
                _o_nhd   = _o_top5["H-Dist"].iloc[0]
                _o_nsw   = _o_top5["Spread Width"].iloc[0]

                with st.expander(
                    f"VIX {_o_bkt}  ·  best avg P&L = ${_o_best:,.0f}  "
                    f"(H-Dist = {_o_nhd:+d}, Width = {_o_nsw})  "
                    f"·  {len(_o_bdf):,} combinations tested"
                ):
                    st.markdown("**Top 5 by Average P&L per Trade**  *(floor duplicates removed)*")
                    st.dataframe(
                        _o_top5.drop(columns=["VIX Bucket"])
                               .style.map(_o_clr, subset=["Avg P&L", "Total P&L"]),
                        use_container_width=True, hide_index=True,
                        column_config={k: v for k, v in _O_CFG.items() if k != "VIX Bucket"},
                    )

                    # Heatmap: rows = Spread Width, cols = H-Dist
                    st.markdown("**Avg P&L Heatmap across all combinations**")
                    _o_piv = _o_bdf.pivot_table(
                        index="Spread Width", columns="H-Dist",
                        values="Avg P&L", aggfunc="first",
                    )
                    _o_z    = _o_piv.values
                    _o_zabs = max(abs(np.nanmin(_o_z)), abs(np.nanmax(_o_z)), 1.0)
                    _o_fig  = go.Figure(go.Heatmap(
                        x=[f"{v:+d}" for v in _o_piv.columns.tolist()],
                        y=_o_piv.index.tolist(),
                        z=_o_z,
                        colorscale=[[0.0, "#ff6b6b"], [0.5, "#2a2a2a"], [1.0, "#4caf7d"]],
                        zmid=0, zmin=-_o_zabs, zmax=_o_zabs,
                        colorbar=dict(title="Avg P&L ($)", tickprefix="$", tickformat=",.0f"),
                        hovertemplate=(
                            "H-Dist: %{x}<br>Spread Width: %{y}<br>"
                            "Avg P&L: $%{z:,.0f}<extra></extra>"
                        ),
                    ))
                    _o_fig.update_layout(
                        xaxis_title="Higher Strike Distance",
                        yaxis_title="Spread Width",
                        template="plotly_dark",
                        margin=dict(l=40, r=40, t=20, b=40),
                        height=max(300, 35 * len(_o_piv) + 100),
                    )
                    st.plotly_chart(_o_fig, use_container_width=True)

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
    _sim = st.session_state.sim_results
    _sim_info = "None — no simulation has been run yet"
    if _sim is not None:
        _entered_sim = _sim[_sim["Trade Entered"]]
        _sim_info = (
            f"DataFrame with {len(_sim):,} rows, {len(_entered_sim):,} trades entered. "
            f"Columns: {list(_sim.columns)}. "
            f"Date range: {_sim['Date'].min()} to {_sim['Date'].max()}."
        )

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
  - sim_results  (full simulation output — {_sim_info})
  - sim_entered  (subset of sim_results where Trade Entered == True, ready to use directly)
  - today  (datetime.date)

The sim_results / sim_entered columns are:
  Date, Expiry (Wed), Days to Expiry, Spot VIX, Base σ, F (model),
  Leg 1 σ, Leg 1 Price, Leg 2 σ, Leg 2 Price, Net Premium,
  Trade Entered, Expiry VIX, Expiry Value, P&L ($)

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
                    "sim_results": st.session_state.sim_results,
                    "sim_entered": (st.session_state.sim_results[st.session_state.sim_results["Trade Entered"]].copy()
                                    if st.session_state.sim_results is not None else None),
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
                        "sim_results": st.session_state.sim_results,
                        "sim_entered": (st.session_state.sim_results[st.session_state.sim_results["Trade Entered"]].copy()
                                        if st.session_state.sim_results is not None else None),
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
