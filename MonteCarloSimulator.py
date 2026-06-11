"""
Monte Carlo Stock Price Simulation

Author: Jahari Lockett
Florida Atlantic University
Data Science & Analytics

Project Overview:
This project uses Monte Carlo simulation to model possible future stock prices
for NVDA, AAPL, MSFT, SPY, and QQQ.

The program:
1. Downloads real 5-year price history using yfinance.
2. Calculates expected return and volatility from historical data.
3. Simulates 10,000 possible future price paths for each stock.
4. Calculates risk metrics such as Value at Risk.
5. Creates visualizations comparing return, volatility, and downside risk.

Why this project matters:
Monte Carlo simulation is widely used in quantitative finance because it helps
estimate uncertainty. Instead of predicting one future stock price, this project
models many possible outcomes and measures the risk around them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# STEP 1: COLLECT STOCK MARKET DATA
# ============================================================
#
# The simulation begins by collecting historical stock data.
# Historical returns are used to estimate each stock's expected
# return and volatility.
#
# If live data is unavailable, the model falls back to published
# annual return data so the project can still run consistently.
REAL_ANNUAL_RETURNS = {
    "NVDA": {
        "name": "NVIDIA",
        "returns": [1.2226, 1.2548, -0.5026, 2.3901, 1.7125],
        "price": 204.04,  # Jun 7, 2026
        "color": "#76b900",
    },
    "AAPL": {
        "name": "Apple",
        "returns": [0.8075, 0.3465, -0.2641, 0.4902, 0.3071],
        "price": 258.90,
        "color": "#60a5fa",
    },
    "MSFT": {
        "name": "Microsoft",
        "returns": [0.4104, 0.5119, -0.2802, 0.5819, 0.1293],
        "price": 449.00,
        "color": "#f59e0b",
    },
    "SPY": {
        "name": "S&P 500 ETF",
        "returns": [0.1840, 0.2875, -0.1817, 0.2629, 0.2502],
        "price": 610.00,
        "color": "#a78bfa",
    },
    "QQQ": {
        "name": "Nasdaq-100 ETF",
        "returns": [0.4860, 0.2742, -0.3258, 0.5486, 0.2558],
        "price": 540.00,
        "color": "#34d399",
    },
}

TICKERS = list(REAL_ANNUAL_RETURNS.keys())


def get_stock_data():
    """
    Collects stock price data and estimates return/volatility assumptions.

    Returns:
        Dictionary containing each ticker's:
        - expected annual return
        - annualized volatility
        - current price
        - data source
    """
    stats = {}

    # First attempt: use live Yahoo Finance data through yfinance.
    # This gives the model recent market prices and daily return history.
    try:
        import yfinance as yf  # pip install yfinance if Pylance flags this

        print("  Trying yfinance download...")

        for ticker in TICKERS:
            hist = yf.Ticker(ticker).history(period="5y")

            if hist is not None and len(hist) > 200:
                # Calculate daily log-returns from real price data
                log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                mu_daily = log_returns.mean()
                sigma_daily = log_returns.std()
                current_price = hist["Close"].iloc[-1]

                stats[ticker] = {
                    "mu_daily": mu_daily,
                    "sigma_daily": sigma_daily,
                    "mu_annual": mu_daily * 252,
                    "sigma_annual": sigma_daily * np.sqrt(252),
                    "price": current_price,
                    "source": "yfinance (live)",
                    "color": REAL_ANNUAL_RETURNS[ticker]["color"],
                    "name": REAL_ANNUAL_RETURNS[ticker]["name"],
                }
                print(
                    f"  [LIVE] {ticker}: price=${current_price:.2f} "
                    f"annual vol={sigma_daily * np.sqrt(252):.1%}"
                )
            else:
                raise ValueError(f"Not enough data for {ticker}")

        return stats

    except Exception as e:
        print(f"  yfinance unavailable ({e})")
        print("  Using verified published annual returns instead...")

    # Backup method:
    # If yfinance is unavailable, estimate return and volatility from
    # verified annual returns. This keeps the simulation reproducible
    # even without a live data connection.
    for ticker, d in REAL_ANNUAL_RETURNS.items():
        # Convert annual % returns to log-returns
        log_returns = np.log(1 + np.array(d["returns"]))
        mu_annual = log_returns.mean()
        sigma_annual = log_returns.std(ddof=1)

        stats[ticker] = {
            "mu_daily": mu_annual / 252,
            "sigma_daily": sigma_annual / np.sqrt(252),
            "mu_annual": mu_annual,
            "sigma_annual": sigma_annual,
            "price": d["price"],
            "source": "Published annual returns (FinanceCharts/Macrotrends)",
            "color": d["color"],
            "name": d["name"],
        }
        print(
            f"  [FALLBACK] {ticker}: price=${d['price']:.2f} "
            f"annual vol={sigma_annual:.1%}"
        )

    return stats


print("=" * 55)
print("  Monte Carlo Stock Price Simulation")
print("  Jahari Lockett — Florida Atlantic University")
print("=" * 55)
print("\n[1/4] Loading stock data...")
stats = get_stock_data()


# =============================================================================
# STEP 2: PRINT SUMMARY TABLE
# =============================================================================
print("\n\n── Empirical Statistics ────────────────────────────")
print(f"  {'Ticker':<6} {'Name':<18} {'Price':<10} {'CAGR':<10} {'Volatility'}")
print("  " + "-" * 55)
for t, s in stats.items():
    cagr = (np.exp(s["mu_annual"]) - 1) * 100
    vol = s["sigma_annual"] * 100
    print(
        f"  {t:<6} {s['name']:<18} ${s['price']:<9.2f} {cagr:+.1f}%{'':<5} {vol:.1f}%"
    )

# Save stats table
rows = [
    {
        "ticker": t,
        "name": s["name"],
        "price": s["price"],
        "cagr_pct": round((np.exp(s["mu_annual"]) - 1) * 100, 2),
        "vol_pct": round(s["sigma_annual"] * 100, 2),
        "source": s["source"],
    }
    for t, s in stats.items()
]
pd.DataFrame(rows).to_csv("outputs/stock_statistics.csv", index=False)


# =============================================================================
# STEP 3: MONTE CARLO SIMULATION
# =============================================================================
# For each stock we simulate 10,000 possible price paths over one year.
#
# Each day the price changes by:
#     price_tomorrow = price_today * exp(daily_drift + daily_noise)
#
# This is called Geometric Brownian Motion (GBM) — the standard model
# used in finance for simulating stock prices.

print("\n[2/4] Running simulations (10,000 paths per stock)...")


def run_simulation(price, mu_daily, sigma_daily, n_sims=10_000, n_days=252):
    """
    Simulate stock prices using Geometric Brownian Motion.

    Parameters:
        price      : Starting price (today's price)
        mu_daily   : Average daily return (from historical data)
        sigma_daily: Daily volatility (from historical data)
        n_sims     : Number of simulations to run
        n_days     : How many trading days to simulate (252 = 1 year)

    Returns a matrix of shape (n_days+1, n_sims) — one column per simulation.
    """
    np.random.seed(42)

    # The -0.5*sigma^2 term is called the Ito correction.
    # Without it, the average simulated price would be too high.
    drift = mu_daily - 0.5 * sigma_daily**2
    daily_log_returns = drift + sigma_daily * np.random.randn(n_days, n_sims)

    # Stack a row of zeros at the top (day 0 = today)
    log_paths = np.vstack([np.zeros(n_sims), np.cumsum(daily_log_returns, axis=0)])
    return price * np.exp(log_paths)


simulations = {}
for ticker, s in stats.items():
    paths = run_simulation(s["price"], s["mu_daily"], s["sigma_daily"])
    final = paths[-1]  # final prices after 1 year
    returns = (final - s["price"]) / s["price"]  # 1-year returns

    simulations[ticker] = {
        "paths": paths,
        "final": final,
        "returns": returns,
        "mean_return": returns.mean(),
        "var_95": np.percentile(returns, 5),  # Value at Risk 95%
        "var_99": np.percentile(returns, 1),  # Value at Risk 99%
        "p_profit": (final > s["price"]).mean(),  # P(making money)
    }
    print(
        f"  {ticker}: E[return]={returns.mean():+.1%}  "
        f"VaR(95%)={np.percentile(returns, 5):.1%}  "
        f"P(profit)={(final > s['price']).mean():.1%}"
    )


# =============================================================================
# STEP 4: VISUALIZATIONS
# =============================================================================

print("\n[3/4] Creating charts...")

BG = "#0d0d1a"  # deep navy background
PANEL = "#13132b"  # dark navy panel
PINK = "#f4a7c3"  # pastel pink
PURPLE = "#c084fc"  # pastel purple
BLUE = "#93c5fd"  # pastel blue
LILAC = "#d8b4fe"  # soft lilac
MINT = "#a5f3d4"  # soft mint
MUTED = "#9ca3c8"  # muted lavender
WHITE = "#f0eeff"  # warm white
RED = "#f4a7c3"  # alias → pastel pink for VaR lines
AMBER = "#d8b4fe"  # alias → lilac for secondary VaR
GREEN = "#a5f3d4"  # alias → mint for positive


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


# ── Figure 1: Fan Charts ──────────────────────────────────────────────────
# Shows the range of possible future prices for each stock.
# The darker band is the "middle 50%" of outcomes (25th to 75th percentile).
# The lighter band is the "middle 90%" (5th to 95th percentile).

fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
fig.subplots_adjust(hspace=0.45, wspace=0.32)
axes = axes.flatten()
days_x = np.arange(253)

for ax, ticker in zip(axes, TICKERS):
    sim = simulations[ticker]
    col = stats[ticker]["color"]
    S0 = stats[ticker]["price"]

    # Calculate percentile bands
    p5 = np.percentile(sim["paths"], 5, axis=1)
    p25 = np.percentile(sim["paths"], 25, axis=1)
    p50 = np.percentile(sim["paths"], 50, axis=1)
    p75 = np.percentile(sim["paths"], 75, axis=1)
    p95 = np.percentile(sim["paths"], 95, axis=1)

    # Draw 150 faint paths for visual effect
    sample_idx = np.random.choice(10000, 150, replace=False)
    for path in sim["paths"][:, sample_idx].T:
        ax.plot(days_x, path, color=col, alpha=0.02, linewidth=0.5)

    ax.fill_between(days_x, p5, p95, alpha=0.14, color=col)
    ax.fill_between(days_x, p25, p75, alpha=0.28, color=col)
    ax.plot(days_x, p50, color=col, linewidth=2, label="Median")
    ax.axhline(
        S0,
        color=MUTED,
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label=f"Start: ${S0:.0f}",
    )

    style(
        ax,
        title=f"{ticker} — {stats[ticker]['name']}",
        xlabel="Trading Days",
        ylabel="Price ($)",
    )
    ax.legend(fontsize=7, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

axes[5].set_visible(False)
fig.suptitle(
    "Monte Carlo Price Simulations (10,000 paths, 1-year horizon)\n"
    "Parameters from real 5-year return history",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.savefig("figures/fig1_fan_charts.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved → figures/fig1_fan_charts.png")
plt.close()


# ── Figure 2: Return Distributions ────────────────────────────────────────
# Shows the full distribution of 10,000 simulated 1-year returns.
# The red line is the 95% Value at Risk — the worst loss in 95% of scenarios.

fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
fig.subplots_adjust(hspace=0.45, wspace=0.32)
axes = axes.flatten()

for ax, ticker in zip(axes, TICKERS):
    sim = simulations[ticker]
    col = stats[ticker]["color"]

    ax.hist(
        sim["returns"] * 100,
        bins=100,
        color=col,
        alpha=0.7,
        edgecolor="none",
        density=True,
    )
    ax.axvline(
        sim["var_95"] * 100,
        color=RED,
        linewidth=1.5,
        linestyle="--",
        label=f"VaR 95%: {sim['var_95']:.1%}",
    )
    ax.axvline(
        sim["mean_return"] * 100,
        color=MINT,
        linewidth=1.5,
        label=f"Mean: {sim['mean_return']:+.1%}",
    )
    ax.axvline(0, color=MUTED, linewidth=0.8, alpha=0.5)

    style(
        ax,
        title=f"{ticker} — 1-Year Return Distribution",
        xlabel="Return (%)",
        ylabel="Density",
    )
    ax.legend(fontsize=7, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

axes[5].set_visible(False)
fig.suptitle(
    "Simulated 1-Year Return Distributions with Value at Risk\n"
    "10,000 Monte Carlo paths per stock",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.savefig(
    "figures/fig2_return_distributions.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("  Saved → figures/fig2_return_distributions.png")
plt.close()


# ── Figure 3: Risk Comparison ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

# Left: VaR bar chart
ax1 = axes[0]
style(
    ax1,
    title="Value at Risk Comparison\n(1-Year Simulation)",
    xlabel="Ticker",
    ylabel="Loss (%)",
)
x = np.arange(len(TICKERS))
var95 = [-simulations[t]["var_95"] * 100 for t in TICKERS]
var99 = [-simulations[t]["var_99"] * 100 for t in TICKERS]
ax1.bar(x - 0.2, var95, 0.35, label="95% VaR", color=LILAC, alpha=0.85)
ax1.bar(x + 0.2, var99, 0.35, label="99% VaR", color=PINK, alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(TICKERS, color=MUTED)
for i, (v5, v9) in enumerate(zip(var95, var99)):
    ax1.text(i - 0.2, v5 + 0.3, f"{v5:.1f}%", ha="center", color=WHITE, fontsize=7)
    ax1.text(i + 0.2, v9 + 0.3, f"{v9:.1f}%", ha="center", color=WHITE, fontsize=7)
ax1.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

# Right: Risk-return scatter
ax2 = axes[1]
style(
    ax2,
    title="Risk vs. Return\n(Empirical — 5-Year Data)",
    xlabel="Annualized Volatility (%)",
    ylabel="CAGR (%)",
)
for t in TICKERS:
    s = stats[t]
    cagr = (np.exp(s["mu_annual"]) - 1) * 100
    vol = s["sigma_annual"] * 100
    ax2.scatter(vol, cagr, color=s["color"], s=120, zorder=5)
    ax2.annotate(f"  {t}", (vol, cagr), color=WHITE, fontsize=8)
ax2.axhline(0, color=LILAC, linewidth=0.6, linestyle="--", alpha=0.45)

fig.suptitle(
    "Risk Analysis — NVDA · AAPL · MSFT · SPY · QQQ",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout(pad=2)
plt.savefig(
    "figures/fig3_risk_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("  Saved → figures/fig3_risk_comparison.png")
plt.close()


# ============================================================
# STEP 5: SAVE SIMULATION RESULTS
# ============================================================
#
# The final results table summarizes each stock's simulated performance:
# price, expected return, volatility, downside risk, and probability of profit.
#
# Saving the table as a CSV makes the project easier to review,
# share, and include in a portfolio.
print("\n[4/4] Saving results...")

out = pd.DataFrame(
    [
        {
            "ticker": t,
            "price": round(stats[t]["price"], 2),
            "cagr_pct": round((np.exp(stats[t]["mu_annual"]) - 1) * 100, 2),
            "vol_pct": round(stats[t]["sigma_annual"] * 100, 2),
            "mean_return": round(simulations[t]["mean_return"] * 100, 2),
            "var_95_pct": round(simulations[t]["var_95"] * 100, 2),
            "var_99_pct": round(simulations[t]["var_99"] * 100, 2),
            "p_profit": round(simulations[t]["p_profit"] * 100, 1),
        }
        for t in TICKERS
    ]
)

print("\n" + out.to_string(index=False))
out.to_csv("outputs/simulation_results.csv", index=False)
print("\nSaved → outputs/simulation_results.csv")
print("\n" + "=" * 55)
print("  Done! Check figures/ and outputs/ folders.")
print("=" * 55)
