# AI Coding Agent Instructions

## Project Overview
This is a quantitative finance and data science portfolio consisting of three independent Python applications. Each is a self-contained analysis that can run offline using fallback data.

**Key Pattern:** All projects follow the same structural template—load data, compute metrics, simulate futures, visualize results, export CSVs.

---

## Architecture & Data Flow

### Projects
1. **Black-ScholesOptionPricing.py** — European option valuation with Greeks
2. **MonteCarloSimulator.py** — Stock price simulations (5 tickers: NVDA, AAPL, MSFT, SPY, QQQ)
3. **WildlifeConservationDataAnalysis.py** — Real species population analysis with ML classification

### Common Pattern
```
[Step 1] Fetch live data (yfinance/custom)
    ↓ [falls back to hardcoded verified data]
[Step 2] Compute statistics (volatility, returns, change rates)
    ↓
[Step 3] Run simulations (Geometric Brownian Motion or Monte Carlo)
    ↓
[Step 4] Visualize (matplotlib with consistent dark theme)
    ↓
[Step 5] Export CSV to outputs/
```

### Data Dependencies
- **Black-Scholes**: Uses NVDA 5Y annual returns (`NVDA_ANNUAL_RETURNS`) + live yfinance data
- **MonteCarlo**: Uses `REAL_ANNUAL_RETURNS` dict (5 stocks) + live yfinance data
- **Wildlife**: Hardcoded real species data (`SPECIES`) tuple—no external API

**Fallback Strategy**: All projects gracefully degrade to published/hardcoded data if yfinance unavailable (important for reproducibility).

---

## Critical Conventions

### 1. Data Processing Patterns
- **Log returns**: Use `np.log(price_t / price_t-1)` for daily returns; annualize with `* 252` (trading days)
- **Annualization**: `sigma_annual = sigma_daily * sqrt(252)`; `mu_annual = mu_daily * 252`
- **DataFrame columns**: Use snake_case; create derived columns for new metrics (e.g., `pct_change`, `p_collapse`)

### 2. Visualization Standards
All three projects use **identical dark theme**:
```python
BG = "#0d0d1a"      # deep navy background
PANEL = "#13132b"   # chart panel background
BLUE = "#93c5fd"    # primary data series
PINK = "#f4a7c3"    # negative/decline/risk
MINT = "#a5f3d4"    # positive/recovery
PURPLE = "#c084fc"  # alternative accent
WHITE = "#f0eeff"   # text
MUTED = "#9ca3c8"   # axis labels
```

Common `style()` helper function appears in each file—use it for all axes to maintain consistency.

### 3. Mathematical Patterns
- **Black-Scholes inputs**: S (spot), K (strike), T (time years), r (risk-free rate), σ (volatility), option_type
- **Greeks**: delta, gamma, theta (daily decay), vega, rho (all from `compute_greeks()`)
- **Simulation**: Geometric Brownian Motion with Ito correction: `drift = μ - 0.5σ²`
- **Risk metrics**: Value at Risk (5th/1st percentile), P(profit), probability of collapse

### 4. Output Formats
- **Figures**: Save to `figures/` with descriptive names (`fig1_nvda_overview.png`, etc.)
- **Data**: Save to `outputs/` as CSV with lowercase headers
- **Print output**: Use formatted tables (72 chars wide) with dashed separators for readability

---

## Key Implementation Examples

### Load Data with Fallback
```python
# Try live data first, fall back to verified historical data
try:
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="5y")
    # ... calculate sigma from real data
except:
    # Use hardcoded annual returns instead
    log_returns = np.log(1 + np.array(PUBLISHED_RETURNS))
    sigma = log_returns.std(ddof=1)
```

### Compute Annualized Volatility
```python
log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
sigma_annual = float(log_returns.std() * np.sqrt(252))
```

### Run Monte Carlo Simulation
```python
def simulate_gbm(price, mu_daily, sigma_daily, n_sims=10_000, n_days=252):
    drift = mu_daily - 0.5 * sigma_daily**2
    daily_returns = drift + sigma_daily * np.random.randn(n_days, n_sims)
    log_paths = np.vstack([np.zeros(n_sims), np.cumsum(daily_returns, axis=0)])
    return price * np.exp(log_paths)
```

### Style a Subplot
```python
ax.set_facecolor(PANEL)
ax.tick_params(colors=MUTED, labelsize=8)
ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold")
ax.spines['right'].set_edgecolor("#2a2a4a")  # same for all spines
```

---

## Workflows

### Adding a New Stock to MonteCarloSimulator
1. Add entry to `REAL_ANNUAL_RETURNS` dict with ticker, name, historical returns (5Y), current price, and hex color
2. Ticker automatically included via `TICKERS = list(REAL_ANNUAL_RETURNS.keys())`
3. Simulation loop in Step 3 processes all tickers—no loop changes needed
4. Both Figure 1 and Figure 2 use `zip(axes, TICKERS)` so they auto-scale

### Adding a New Visualization
1. Create figure after Step 3 (after simulations complete)
2. Use consistent color palette and `style()` helper
3. Save with `plt.savefig("figures/figX_descriptive.png", dpi=150, facecolor=BG)`
4. Print confirmation message: `print("  Saved → figures/figX_descriptive.png")`
5. Add to results table in Step 5 summary if applicable

### Debugging Data Loading
1. Check if yfinance is installed: `import yfinance; print(yfinance.__version__)`
2. Verify fallback data is used: Print statements distinguish `[LIVE]` vs `[FALLBACK]`
3. For species data: Data is hardcoded in tuple—verify tuple syntax if data won't load
4. All outputs validated: CSV exports show shape in print statement

---

## File Organization
```
/CodingProjects/
├── Black-ScholesOptionPricing.py     # Options pricing (NVDA)
├── MonteCarloSimulator.py              # Stock simulations (5 tickers)
├── WildlifeConservationDataAnalysis.py # Species risk classification
├── figures/                            # PNG exports (auto-created)
├── outputs/                            # CSV exports (auto-created)
│   ├── nvda_options.csv
│   ├── stock_statistics.csv
│   ├── simulation_results.csv
│   └── species_data.csv
└── .github/copilot-instructions.md     # This file
```

---

## Common Pitfalls & Solutions

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| yfinance import fails | Module not installed | `pip install yfinance` |
| Simulation looks wrong | Random seed not set | Ensure `np.random.seed(42)` at top of simulation function |
| Colors don't match | Typo in hex code | Copy from constants block at top of file |
| CSV export has wrong column names | Didn't use snake_case | Check dict key names when creating DataFrame |
| Plots not displaying | Matplotlib backend issue | Use `plt.savefig()` instead of `plt.show()` for headless environments |

---

## Commands for Agents

- **Run single script**: `python Black-ScholesOptionPricing.py`
- **Run all three**: `for f in *.py; do python "$f"; done`
- **Check outputs**: `ls -la outputs/ figures/`
- **Install dependencies**: `pip install yfinance pandas numpy scipy scikit-learn matplotlib`

---

## Document Sources & References
- **Black-Scholes formula**: Standard quantitative finance textbook implementation
- **Greeks calculations**: All five Greeks derived from closed-form BS equations
- **Monte Carlo GBM**: Industry-standard for stock price simulation (Ito correction essential)
- **Put-Call parity check**: Validation mechanism—error should be < 1e-6
- **Cross-validation**: Used for ML model evaluation (5-fold holdout)
- **Species data**: Real IUCN Red List, WWF, and published government census data
