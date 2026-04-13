"""
VIX Options Pricing — Black-76 Model

Black-76 prices European options on futures contracts.

Model:
    d1 = [ ln(F/K) + ½σ²T ] / (σ√T)
    d2 = d1 − σ√T

    Call = e^{−rT} [ F·N(d1) − K·N(d2) ]
    Put  = e^{−rT} [ K·N(−d2) − F·N(−d1) ]

Parameters
----------
F     : float — Futures price at expiry (from the log-mean-reverting model)
K     : float — Strike price
r     : float — Risk-free rate (annualised)
sigma : float — Vol-of-vol σ (annualised)
T     : float — Time to expiry in years
option_type : str — 'call' or 'put'
"""

import numpy as np
from scipy.stats import norm


def black76(F: float, K: float, r: float, sigma: float, T: float,
            option_type: str = "call") -> dict:
    """
    Price a European option on a VIX futures contract using Black-76.

    Returns
    -------
    dict with keys:
        price, delta, gamma, vega, theta, rho, d1, d2
    """
    option_type = option_type.lower()

    # At or past expiry — intrinsic value only
    if T <= 0:
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        return {
            "price": intrinsic, "delta": None, "gamma": None,
            "vega": None, "theta": None, "rho": None,
            "d1": None, "d2": None,
        }

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)   # discount factor

    if option_type == "call":
        price = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        delta = df * norm.cdf(d1)
    else:
        price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -df * norm.cdf(-d1)

    gamma = df * norm.pdf(d1) / (F * sigma * np.sqrt(T))
    vega  = F * df * norm.pdf(d1) * np.sqrt(T) / 100        # per 1% move in σ
    theta = -(F * sigma * df * norm.pdf(d1)) / (2 * np.sqrt(T)) / 365  # per calendar day
    rho   = -T * price                                        # approximate for futures options

    return {
        "price": round(float(price), 4),
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 6),
        "vega":  round(float(vega),  4),
        "theta": round(float(theta), 4),
        "rho":   round(float(rho),   4),
        "d1":    round(float(d1),    4),
        "d2":    round(float(d2),    4),
    }
