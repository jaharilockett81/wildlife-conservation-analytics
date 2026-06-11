"""
Black-Scholes Options Pricing Model

Author: Jahari Lockett
Florida Atlantic University
B.S. Data Science & Analytics

Project Overview:
This project uses the Black-Scholes model to estimate the fair value of
European call and put options for NVIDIA (NVDA).

The program:
1. Retrieves real market data using yfinance.
2. Calculates historical volatility from price movements.
3. Prices sample option contracts.
4. Computes the five primary option Greeks.
5. Visualizes how option values respond to changing market conditions.

Why this project matters:
Options pricing is a core concept in quantitative finance. This project
demonstrates how mathematical models can be used to estimate risk,
value derivative contracts, and support investment decision-making.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------
# MODEL INPUTS
# ------------------------------------------------------------
# These assumptions drive all pricing calculations.
# Change the ticker, interest rate, or fallback values
# to test the model on different market conditions.

TICKER = "NVDA"
RISK_FREE_RATE = 0.045  # 4.5% risk-free rate (US Treasury Jun 2026)
FALLBACK_PRICE = 204.04  # Real price as of Jun 7, 2026

# Historical NVDA returns used when live data is unavailable.
# These published returns provide a backup estimate for
# long-term volatility calculations.
#
# Source:
# FinanceCharts.com
# Macrotrends
NVDA_ANNUAL_RETURNS = [1.2226, 1.2548, -0.5026, 2.3901, 1.7125]


# =============================================================================
# STEP 1: GET REAL NVDA DATA
# =============================================================================


def get_nvda_data():
    """
    Downloads market data for NVIDIA and estimates annualized volatility.
    Returns:
       current_price
       annualized_volatility
       data_source
    """

    try:
        import yfinance as yf

        print("Trying yfinance download...")

        hist = yf.Ticker(TICKER).history(period="5y")

        if hist is not None and len(hist) > 200:
            # Convert daily price movements into logarithmic returns.
            # Log returns are commonly used in quantitative finance
            # because they are mathematically stable and additive.
            #
            # Annualized volatility represents the stock's expected
            # level of price fluctuation over a full year.

            log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            sigma = float(log_returns.std() * np.sqrt(252))  # annualized
            price = float(hist["Close"].iloc[-1])

            print(f"LIVE] NVDA price: ${price:.2f}  historical vol (5Y): {sigma:.1%}")
            return price, sigma, "yfinance (Yahoo Finance) — 5Y daily closes"
    except Exception as e:
        print(f" yfinance unavailable: {e}")

    # If live market data cannot be retrieved,
    # estimate volatility using published historical returns.
    # This keeps the project functional even without
    # an internet connection or API access.Fallback: use verified published annual returns

    print("FALLBACK] Using published NVDA returns 2020-2024")
    print("Source: FinanceCharts.com / Macrotrends")
    log_r = np.log(1 + np.array(NVDA_ANNUAL_RETURNS))
    sigma = float(log_r.std(ddof=1))  # annualized from annual returns
    print(f"  NVDA price: ${FALLBACK_PRICE:.2f}  historical vol (5Y): {sigma:.1%}")
    return (
        FALLBACK_PRICE,
        sigma,
        "Published annual returns — FinanceCharts / Macrotrends",
    )


print("=" * 55)
print("  Black-Scholes Options Pricing — NVDA")
print("  Jahari Lockett — Florida Atlantic University")
print("=" * 55)
print("\n[1/5] Loading NVDA data...")
S0, SIGMA, DATA_SOURCE = get_nvda_data()

print(f"\n  Underlying price  : ${S0:.2f}")
print(f"  Historical vol    : {SIGMA:.1%} (annualized)")
print(f"  Risk-free rate    : {RISK_FREE_RATE:.1%}")
print(f"  Data source       : {DATA_SOURCE}")


# ============================================================
# STEP 2: BLACK-SCHOLES OPTION PRICING MODEL
# ============================================================
# The Black-Scholes model estimates the fair value of European options.
#
# A call option gives the buyer the right to BUY a stock at a set strike price.
# A put option gives the buyer the right to SELL a stock at a set strike price.
#
# For this project, I am using NVDA as the underlying stock and calculating
# option prices based on stock price, strike price, time, volatility, and rates.


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the theoretical price of a European call or put option
    using the Black-Scholes formula.

    Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate
        sigma: Annualized volatility
        option_type: "call" or "put"

    Returns:
        Estimated fair value of the option
    """
    # How the formula works:
    # d1 and d2 help measure how far the option is in or out of the money,
    # while adjusting for time, volatility, and interest rates.
    #
    # norm.cdf() converts those values into probabilities under a normal curve.
    # The final formula discounts the strike price back to today's value.
    # Make sure inputs are valid

    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute the five option Greeks.

    Delta: How much the option price changes per $1 move in the stock.
    Gamma: How fast Delta itself changes (the "acceleration").
    Theta: How much value the option loses each day (time decay).
    Vega:  How much the price changes per 1% move in volatility.
    Rho:   How much the price changes per 1% move in interest rates.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    sqrtT = np.sqrt(T)
    disc = np.exp(-r * T)

    # Delta — hedge ratio
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

    # Gamma — same for call and put
    gamma = pdf_d1 / (S * sigma * sqrtT)

    # Theta — daily time decay
    common = -(S * pdf_d1 * sigma) / (2 * sqrtT)
    if option_type == "call":
        theta = (common - r * K * disc * norm.cdf(d2)) / 365
    else:
        theta = (common + r * K * disc * norm.cdf(-d2)) / 365

    # Vega — per 1% change in vol
    vega = S * pdf_d1 * sqrtT / 100

    # Rho — per 1% change in rate
    if option_type == "call":
        rho = K * T * disc * norm.cdf(d2) / 100
    else:
        rho = -K * T * disc * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


# =============================================================================
# STEP 3: PRICE SOME OPTIONS
# =============================================================================
print("\n[2/5] Pricing NVDA options...")

# Define a few contracts to price
# ATM = at-the-money (strike ≈ current price)
# OTM = out-of-the-money (call strike above price / put strike below)
contracts = [
    {"K": round(S0), "T": 0.5, "type": "call", "label": "ATM Call (6M)"},
    {"K": round(S0), "T": 0.5, "type": "put", "label": "ATM Put (6M)"},
    {"K": round(S0 * 1.2), "T": 0.5, "type": "call", "label": "OTM Call +20% (6M)"},
    {"K": round(S0 * 0.8), "T": 0.5, "type": "put", "label": "OTM Put -20% (6M)"},
    {"K": round(S0), "T": 1.0, "type": "call", "label": "ATM Call (1Y)"},
]

print(f"\n  NVDA Options — S=${S0:.2f}  σ={SIGMA:.1%}  r={RISK_FREE_RATE:.1%}")
print("  " + "-" * 72)
print(
    f"  {'Label':<24} {'Strike':<8} {'Price':<10} "
    f"{'Delta':<8} {'Theta/day':<12} {'Vega/1%vol'}"
)
print("  " + "-" * 72)

rows = []
for c in contracts:
    price = black_scholes_price(S0, c["K"], c["T"], RISK_FREE_RATE, SIGMA, c["type"])
    greeks = compute_greeks(S0, c["K"], c["T"], RISK_FREE_RATE, SIGMA, c["type"])
    print(
        f"  {c['label']:<24} ${c['K']:<7.0f} ${price:<9.2f} "
        f"{greeks['delta']:<8.3f} ${greeks['theta']:<11.3f} ${greeks['vega']:.3f}"
    )
    rows.append(
        {
            "label": c["label"],
            "strike": c["K"],
            "T": c["T"],
            "type": c["type"],
            "price": round(price, 4),
            **{k: round(v, 6) for k, v in greeks.items()},
        }
    )

pd.DataFrame(rows).to_csv("outputs/nvda_options.csv", index=False)
print("\n  Saved → outputs/nvda_options.csv")

# Quick put-call parity check
# This identity must hold: Call - Put = Stock - Strike * e^(-rT)
call_price = black_scholes_price(S0, round(S0), 0.5, RISK_FREE_RATE, SIGMA, "call")
put_price = black_scholes_price(S0, round(S0), 0.5, RISK_FREE_RATE, SIGMA, "put")
lhs = call_price - put_price
rhs = S0 - round(S0) * np.exp(-RISK_FREE_RATE * 0.5)
print(f"\n  Put-Call Parity check: C - P = {lhs:.4f}  |  S - Ke^(-rT) = {rhs:.4f}")
print(
    f"  Error: {abs(lhs - rhs):.2e} ✓"
    if abs(lhs - rhs) < 1e-6
    else "  WARNING: parity violated"
)

# ============================================================
# STEP 4: VISUALIZING OPTION BEHAVIOR
# ============================================================
#
# Mathematical formulas are useful, but visualizations help
# explain how option values respond to changing market conditions.
#
# These charts explore:
# • Volatility sensitivity
# • Time decay (Theta)
# • Stock price sensitivity (Delta)
# • The behavior of all five Greeks
#
# Together, they provide intuition behind the Black-Scholes model.
print("\n[3/5] Creating charts...")

BG = "#0d0d1a"  # deep navy background
PANEL = "#13132b"  # dark navy panel ( dark colors improve contrast)
BLUE = "#93c5fd"  # pastel blue
GREEN = "#a5f3d4"  # soft mint
RED = "#f4a7c3"  # pastel pink
AMBER = "#d8b4fe"  # soft lilac
PUR = "#c084fc"  # pastel purple
MUTED = "#9ca3c8"  # muted lavender
WHITE = "#f0eeff"  # warm white


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, pad=9, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


# Range of stock prices for sensitivity charts
S_range = np.linspace(S0 * 0.40, S0 * 1.60, 300)
K_atm = round(S0)


# ── Figure 1: NVDA Price & Volatility Context ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

# Left: how the 5-year annual returns look
ax1 = axes[0]
style(
    ax1,
    "NVDA Annual Returns 2020-2024\n(Real data — basis for σ estimate)",
    "Year",
    "Annual Return (%)",
)
rets = [r * 100 for r in NVDA_ANNUAL_RETURNS]
years = [2020, 2021, 2022, 2023, 2024]
colors = [GREEN if r >= 0 else RED for r in rets]
bars = ax1.bar(years, rets, color=colors, width=0.6, alpha=0.85)
ax1.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
for bar, r in zip(bars, rets):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        r + (8 if r >= 0 else -14),
        f"{r:+.1f}%",
        ha="center",
        color=WHITE,
        fontsize=8,
    )
ax1.set_xticks(years)

# Right: how option price changes with stock price
ax2 = axes[1]
style(
    ax2,
    f"Option Price vs Stock Price\n(K=${K_atm}, T=0.5yr, σ={SIGMA:.0%})",
    "Stock Price ($)",
    "Option Price ($)",
)
call_prices = [
    black_scholes_price(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "call") for s in S_range
]
put_prices = [
    black_scholes_price(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "put") for s in S_range
]
ax2.plot(S_range, call_prices, color=BLUE, linewidth=2, label="Call")
ax2.plot(S_range, put_prices, color=AMBER, linewidth=2, label="Put")
ax2.axvline(
    S0, color=GREEN, linewidth=1, linestyle=":", alpha=0.7, label=f"S₀=${S0:.0f}"
)
ax2.axvline(
    K_atm, color=MUTED, linewidth=1, linestyle="--", alpha=0.6, label=f"K=${K_atm}"
)
ax2.legend(fontsize=7.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

fig.suptitle(
    "Figure 1 — NVDA Data & Basic Option Pricing",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout(pad=2)
plt.savefig(
    "figures/fig1_nvda_overview.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("  Saved → figures/fig1_nvda_overview.png")
plt.close()


# ── Figure 2: Sensitivity Dashboard ───────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.34)

sigma_range = np.linspace(0.05, 1.50, 300)
T_range = np.linspace(0.01, 1.0, 300)[::-1]  # countdown to expiry

# A — price vs. vol at 3 maturities
ax1 = fig.add_subplot(gs[0, 0])
style(
    ax1, "Call Price vs Volatility\n(3 maturities)", "Volatility (%)", "Call Price ($)"
)
for T_val, col, lbl in [
    (0.25, GREEN, "3 months"),
    (0.50, BLUE, "6 months"),
    (1.00, PUR, "1 year"),
]:
    prices = [
        black_scholes_price(S0, K_atm, T_val, RISK_FREE_RATE, s, "call")
        for s in sigma_range
    ]
    ax1.plot(sigma_range * 100, prices, color=col, linewidth=2, label=lbl)
ax1.axvline(
    SIGMA * 100,
    color=AMBER,
    linewidth=1.2,
    linestyle="--",
    alpha=0.8,
    label=f"σ_hist={SIGMA:.0%}",
)
ax1.legend(fontsize=7.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

# B — theta decay
ax2 = fig.add_subplot(gs[0, 1])
style(
    ax2,
    "Theta Decay — Value Erodes as Expiry Approaches",
    "Days to Expiry",
    "Option Price ($)",
)
call_decay = [
    black_scholes_price(S0, K_atm, t, RISK_FREE_RATE, SIGMA, "call") for t in T_range
]
put_decay = [
    black_scholes_price(S0, K_atm, t, RISK_FREE_RATE, SIGMA, "put") for t in T_range
]
ax2.plot(T_range * 365, call_decay, color=BLUE, linewidth=2, label="Call")
ax2.plot(T_range * 365, put_decay, color=AMBER, linewidth=2, label="Put")
ax2.invert_xaxis()  # show time running toward zero (expiry)
ax2.legend(fontsize=7.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

# C — delta vs. stock price
ax3 = fig.add_subplot(gs[1, 0])
style(
    ax3,
    "Delta vs Stock Price\n(0.5 = ATM, 1.0 = deep in-the-money)",
    "Stock Price ($)",
    "Delta",
)
call_deltas = [
    compute_greeks(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "call")["delta"]
    for s in S_range
]
put_deltas = [
    compute_greeks(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "put")["delta"]
    for s in S_range
]
ax3.plot(S_range, call_deltas, color=BLUE, linewidth=2, label="Call Δ")
ax3.plot(S_range, put_deltas, color=AMBER, linewidth=2, label="Put Δ")
ax3.axhline(0, color=MUTED, linewidth=0.5, linestyle=":", alpha=0.5)
ax3.axhline(0.5, color=MUTED, linewidth=0.5, linestyle=":", alpha=0.4)
ax3.axhline(-0.5, color=MUTED, linewidth=0.5, linestyle=":", alpha=0.4)
ax3.axvline(K_atm, color=MUTED, linewidth=0.9, linestyle="--", alpha=0.5)
ax3.legend(fontsize=7.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

# D — vega vs. stock price
ax4 = fig.add_subplot(gs[1, 1])
style(
    ax4,
    "Vega vs Stock Price\n($ gain per 1% increase in volatility)",
    "Stock Price ($)",
    "Vega ($)",
)
vegas = [
    compute_greeks(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "call")["vega"]
    for s in S_range
]
ax4.plot(S_range, vegas, color=PUR, linewidth=2)
ax4.axvline(K_atm, color=MUTED, linewidth=0.9, linestyle="--", alpha=0.5)
ax4.fill_between(S_range, vegas, alpha=0.12, color=PUR)

fig.suptitle(
    f"Figure 2 — Black-Scholes Sensitivity Analysis | NVDA\n"
    f"S₀=${S0:.2f}  K=${K_atm}  σ={SIGMA:.0%} (5Y historical)  r={RISK_FREE_RATE:.1%}",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.savefig("figures/fig2_sensitivity.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved → figures/fig2_sensitivity.png")
plt.close()


# ── Figure 3: All 5 Greeks ─────────────────────────────────────────────────
greek_names = ["delta", "gamma", "theta", "vega", "rho"]
greek_colors = [BLUE, GREEN, RED, AMBER, PUR]

fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor=BG)
fig.subplots_adjust(wspace=0.32)

for ax, gk, col in zip(axes, greek_names, greek_colors):
    call_vals = [
        compute_greeks(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "call")[gk]
        for s in S_range
    ]
    put_vals = [
        compute_greeks(s, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "put")[gk] for s in S_range
    ]
    style(ax, title=gk.capitalize(), xlabel="Stock Price ($)", ylabel=gk.capitalize())
    ax.plot(S_range, call_vals, color=col, linewidth=1.8, label="Call")
    ax.plot(
        S_range,
        put_vals,
        color=col,
        linewidth=1.8,
        linestyle="--",
        alpha=0.6,
        label="Put",
    )
    ax.axvline(K_atm, color=MUTED, linewidth=0.7, linestyle=":", alpha=0.5)
    ax.axhline(0, color=AMBER, linewidth=0.5, linestyle=":", alpha=0.4)
    ax.legend(fontsize=6.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

fig.suptitle(
    f"Figure 3 — All 5 Greeks | NVDA | σ={SIGMA:.0%}  K=${K_atm}",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout(pad=1.5)
plt.savefig("figures/fig3_greeks.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved → figures/fig3_greeks.png")
plt.close()


# =============================================================================
# STEP 5: PRINT FINAL SUMMARY
# =============================================================================
print("\n[4/5] Summary")
print(f"\n  NVDA ATM Call (6 months, K=${K_atm}):")
atm_price = black_scholes_price(S0, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "call")
atm_greeks = compute_greeks(S0, K_atm, 0.5, RISK_FREE_RATE, SIGMA, "call")
print(f"    Price         : ${atm_price:.2f}")
print(f"    Intrinsic     : ${max(0, S0 - K_atm):.2f}")
print(f"    Time Value    : ${atm_price - max(0, S0 - K_atm):.2f}")
print(
    f"    Delta         : {atm_greeks['delta']:.3f}  (moves ${atm_greeks['delta']:.2f} per $1 in NVDA)"
)
print(
    f"    Theta         : {atm_greeks['theta']:.3f}  (loses ${abs(atm_greeks['theta']):.2f}/day)"
)
print(
    f"    Vega          : {atm_greeks['vega']:.3f}  (gains ${atm_greeks['vega']:.2f} per 1% vol increase)"
)

print(f"\n  Note: NVDA's {SIGMA:.0%} historical volatility makes these")
print("  options very expensive compared to a typical stock (~20% vol).")

print("\n[5/5] Done!")
print("\n" + "=" * 55)
print("  Done! Check figures/ and outputs/ folders.")
print("=" * 55)
