"""
VIX Futures Pricing - Log-Mean-Reverting Model

Model:
    d(ln V) = κ (ln θ̄ - ln V) dt + σ dW

    ln V_T is normally distributed with:
        mean:     m(τ) = e^{-κτ} ln(V₀) + (1 - e^{-κτ}) ln(θ̄)
        variance: v²(τ) = σ² / (2κ) * (1 - e^{-2κτ})

    Futures price:
        F(0, T) = E[V_T] = exp( m(τ) + ½ v²(τ) )

Parameters
----------
V0        : Spot VIX level
kappa     : Mean-reversion speed κ (annualized)
theta_bar : Long-run VIX mean θ̄
sigma     : Vol-of-vol σ (annualized)
r         : Risk-free rate (annualized) — reserved for extensions (e.g. discounting)
tau       : Time to expiry in years
"""

import numpy as np


def vix_futures_price(V0: float, kappa: float, theta_bar: float,
                      sigma: float, r: float, tau: float) -> float:
    """
    Price a single VIX futures contract.

    Returns
    -------
    float : Futures price
    """
    if tau <= 0:
        return float(V0)

    alpha = np.log(theta_bar)
    decay = np.exp(-kappa * tau)

    m  = decay * np.log(V0) + (1.0 - decay) * alpha
    v2 = (sigma ** 2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * tau))

    return float(np.exp(m + 0.5 * v2))


def vix_futures_term_structure(V0: float, kappa: float, theta_bar: float,
                                sigma: float, r: float,
                                maturities: list[float]) -> list[dict]:
    """
    Compute the VIX futures term structure for a list of maturities.

    Parameters
    ----------
    maturities : list of float
        Times to expiry in years (e.g. [1/12, 2/12, ..., 12/12])

    Returns
    -------
    list of dict with keys:
        Maturity (months), Time to Expiry (yrs), Futures Price
    """
    rows = []
    for tau in maturities:
        price = vix_futures_price(V0, kappa, theta_bar, sigma, r, tau)
        rows.append({
            "Maturity (months)": round(tau * 12, 1),
            "Time to Expiry (yrs)": round(tau, 4),
            "Futures Price": round(price, 4),
        })
    return rows
