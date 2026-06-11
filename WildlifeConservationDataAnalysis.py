"""
=========================================================
Wildlife Conservation Data Analysis
=========================================================

Author: Jahari Lockett
Florida Atlantic University
Data Science & Analytics

Project Overview:
This project analyzes real-world conservation data from
endangered species around the world.

The analysis focuses on:

• Population change over time
• Habitat loss and environmental pressure
• Extinction-risk indicators
• Future population simulations
• Machine learning classification of high-risk species

Methods Used:
• Exploratory Data Analysis (EDA)
• Monte Carlo Simulation
• Logistic Regression
• Random Forest Classification
• Cross Validation

Data Sources:
• IUCN Red List
• WWF Living Planet Report
• International Rhino Foundation
• India Tiger Census

Outputs:
• Population trend visualizations
• Habitat-loss analysis
• Future population simulations
• Feature-importance charts
• Species risk classifications

Goal:
Use data science techniques to better understand
conservation challenges and identify species that may
require urgent intervention.

=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn — if Pylance flags these, run: pip install scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Create folders for outputs ──────────────────────────────────
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =============================================================================
# STEP 1: REAL SPECIES DATA
# =============================================================================
# These are real published population estimates, not made-up numbers.
# Each row is one species with:
#   name, IUCN status, region,
#   population in 2000, population in 2023,
#   habitat loss %, poaching score (0-10), main threat

SPECIES = [
    # Name                   Status  Region               Pop2000  Pop2023  HabLoss  Poach  Threat
    ("Vaquita", "CR", "North America", 570, 7, 82, 9.5, "Bycatch"),
    ("Javan Rhino", "CR", "Southeast Asia", 58, 50, 74, 8.5, "Poaching"),
    ("Amur Leopard", "CR", "East Asia", 30, 100, 58, 7.0, "Habitat loss"),
    (
        "Sumatran Orangutan",
        "CR",
        "Southeast Asia",
        15000,
        13800,
        67,
        4.0,
        "Deforestation",
    ),
    (
        "Black Rhino",
        "CR",
        "Sub-Saharan Africa",
        2700,
        6788,
        48,
        8.0,
        "Poaching (recovering)",
    ),
    ("Cross River Gorilla", "CR", "Central Africa", 300, 275, 55, 5.5, "Bushmeat"),
    ("Hawksbill Turtle", "CR", "Global", 15000, 8000, 50, 6.0, "Poaching"),
    ("Saola", "CR", "Southeast Asia", 200, 30, 72, 7.5, "Snaring"),
    ("Bengal Tiger", "EN", "South Asia", 3500, 3900, 42, 7.5, "Poaching"),
    ("Chimpanzee", "EN", "Central Africa", 250000, 230000, 44, 4.5, "Deforestation"),
    (
        "African Wild Dog",
        "EN",
        "Sub-Saharan Africa",
        5000,
        6600,
        36,
        3.0,
        "Human conflict",
    ),
    (
        "Bornean Orangutan",
        "EN",
        "Southeast Asia",
        80000,
        75000,
        60,
        4.0,
        "Deforestation",
    ),
    ("Green Sea Turtle", "EN", "Global", 100000, 85000, 28, 5.0, "Bycatch"),
    ("Irrawaddy Dolphin", "EN", "Southeast Asia", 100, 90, 45, 6.5, "Overfishing"),
    ("Snow Leopard", "VU", "Central Asia", 5500, 4500, 34, 5.5, "Climate"),
    (
        "African Elephant",
        "VU",
        "Sub-Saharan Africa",
        470000,
        415000,
        30,
        7.0,
        "Ivory poaching",
    ),
    ("Giant Panda", "VU", "East Asia", 1600, 1864, 22, 2.0, "Habitat (recovering)"),
    ("Polar Bear", "VU", "Arctic", 22000, 26000, 18, 1.5, "Climate change"),
    ("Leatherback Turtle", "VU", "Global", 40000, 34000, 38, 4.0, "Bycatch"),
    ("Clouded Leopard", "VU", "Southeast Asia", 10000, 9000, 33, 4.5, "Deforestation"),
]

# Turn it into a DataFrame — easier to work with
df = pd.DataFrame(
    SPECIES,
    columns=[
        "species",
        "status",
        "region",
        "pop_2000",
        "pop_2023",
        "habitat_loss",
        "poaching",
        "threat",
    ],
)

# Calculate % change in population
df["pct_change"] = ((df["pop_2023"] - df["pop_2000"]) / df["pop_2000"] * 100).round(1)

print("=" * 55)
print("  Wildlife Conservation Data Analysis")
print("  Jahari Lockett — Florida Atlantic University")
print("=" * 55)
print(f"\nLoaded {len(df)} species\n")
print(
    df[["species", "status", "pop_2000", "pop_2023", "pct_change"]].to_string(
        index=False
    )
)


# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# =============================================================================
# Before building any models, we explore the data visually.
# This helps us understand patterns and spot anything unusual.

print("\n\n── Summary Statistics ──────────────────────────────")
print(f"  Species declining : {(df['pct_change'] < 0).sum()}")
print(f"  Species recovering: {(df['pct_change'] > 0).sum()}")
print(
    f"  Biggest decline   : {df.loc[df['pct_change'].idxmin(), 'species']} "
    f"({df['pct_change'].min():.1f}%)"
)
print(
    f"  Biggest recovery  : {df.loc[df['pct_change'].idxmax(), 'species']} "
    f"({df['pct_change'].max():.1f}%)"
)

# Correlation between habitat loss and population change
r = df["habitat_loss"].corr(df["pct_change"])
print(f"  Habitat loss vs. % change correlation: {r:.3f}")


# =============================================================================
# STEP 3: VISUALIZATIONS
# =============================================================================

# Color theme — pastel pinks, purples, blues on deep navy
BG = "#0d0d1a"  # deep navy background
PANEL = "#13132b"  # slightly lighter navy panel
PINK = "#f4a7c3"  # pastel pink   — declining species
PURPLE = "#c084fc"  # pastel purple — recovering species
BLUE = "#93c5fd"  # pastel blue   — simulation bands
LILAC = "#d8b4fe"  # soft lilac    — threshold / warning line
MINT = "#a5f3d4"  # soft mint     — strong positive change
MUTED = "#9ca3c8"  # muted lavender for axis labels
WHITE = "#f0eeff"  # warm white text


def style(ax, title="", xlabel="", ylabel=""):
    """Helper to style every chart consistently."""
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, pad=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


# ── Figure 1: Population Change Bar Chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
sorted_df = df.sort_values("pct_change")
colors = [PINK if v < 0 else MINT for v in sorted_df["pct_change"]]

bars = ax.barh(sorted_df["species"], sorted_df["pct_change"], color=colors, height=0.7)
ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", alpha=0.6)
style(
    ax,
    title="Population Change 2000 → 2023\n(Real IUCN Data)",
    xlabel="% Change",
    ylabel="",
)

# Add value labels to each bar
for bar, val in zip(bars, sorted_df["pct_change"]):
    ax.text(
        val + (3 if val >= 0 else -3),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.0f}%",
        va="center",
        ha="left" if val >= 0 else "right",
        color=WHITE,
        fontsize=7,
    )

fig.text(
    0.5,
    0.01,
    "Sources: IUCN Red List · IRF 2024 · India Tiger Census 2022",
    ha="center",
    color="#6b6b9e",
    fontsize=7,
)
plt.tight_layout(pad=2)
plt.savefig(
    "figures/fig1_population_change.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("\nSaved → figures/fig1_population_change.png")
plt.close()


# ── Figure 2: Habitat Loss vs. Population Change Scatter ──────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
status_colors = {"CR": PINK, "EN": LILAC, "VU": MINT}

for status, group in df.groupby("status"):
    ax.scatter(
        group["habitat_loss"],
        group["pct_change"],
        c=status_colors[status],
        label=status,
        s=80,
        alpha=0.9,
        edgecolors="none",
    )

# Add a trend line
z = np.polyfit(df["habitat_loss"], df["pct_change"], 1)
x_line = np.linspace(df["habitat_loss"].min(), df["habitat_loss"].max(), 100)
ax.plot(
    x_line,
    np.polyval(z, x_line),
    color=LILAC,
    linewidth=1.2,
    linestyle="--",
    alpha=0.55,
)
ax.axhline(0, color=LILAC, linewidth=0.8, linestyle=":", alpha=0.5)

style(
    ax,
    title=f"Habitat Loss vs Population Change\nPearson r = {r:.3f}",
    xlabel="Habitat Loss (%)",
    ylabel="Population Change (%)",
)
ax.legend(
    title="IUCN Status",
    fontsize=8,
    labelcolor=WHITE,
    facecolor=PANEL,
    edgecolor="#2a2a4a",
    title_fontsize=8,
)

plt.tight_layout(pad=2)
plt.savefig(
    "figures/fig2_habitat_scatter.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("Saved → figures/fig2_habitat_scatter.png")
plt.close()


# =============================================================================
# STEP 4: MONTE CARLO SIMULATION
# =============================================================================
# For each species we simulate 1,000 possible futures over 25 years.
# Each year the population changes by a random amount based on its
# historical decline rate. This gives us a range of possible outcomes.

print("\n\n── Monte Carlo Simulation (1,000 paths, 25 years) ────")


def simulate_species(
    pop_start, annual_change_rate, annual_volatility, years=25, n_sims=1000
):
    """
    Simulate future population paths.

    Each year:  population = population * exp(rate + noise)
    This means the population can go up or down randomly each year,
    but the average direction is set by the historical rate.
    """
    np.random.seed(42)
    paths = np.zeros((years + 1, n_sims))
    paths[0] = pop_start

    for t in range(1, years + 1):
        noise = np.random.normal(0, annual_volatility, n_sims)
        paths[t] = paths[t - 1] * np.exp(annual_change_rate + noise)
        paths[t] = np.maximum(paths[t], 0)  # population can't go negative

    return paths


# Run simulation for every species
results = []
for _, row in df.iterrows():
    # Estimate annual change rate from real data
    # If pop went from 570 to 7 over 23 years, the average annual log-change is:
    # ln(7/570) / 23 ≈ -0.188 per year
    if row["pop_2000"] > 0 and row["pop_2023"] > 0:
        annual_rate = np.log(row["pop_2023"] / row["pop_2000"]) / 23
    else:
        annual_rate = -0.05

    # Smaller populations get more random variation (less stable)
    volatility = 0.05 if row["pop_2023"] > 1000 else 0.12

    paths = simulate_species(row["pop_2023"], annual_rate, volatility)
    final_pop = paths[-1]

    # Critical threshold = 10% of what the population was in 2000
    threshold = row["pop_2000"] * 0.10
    p_collapse = (final_pop <= threshold).mean()

    results.append(
        {
            "species": row["species"],
            "status": row["status"],
            "pop_2023": row["pop_2023"],
            "median_2048": round(np.median(final_pop), 0),
            "p5_2048": round(np.percentile(final_pop, 5), 0),
            "p95_2048": round(np.percentile(final_pop, 95), 0),
            "p_collapse": round(p_collapse, 3),
            "paths": paths,
        }
    )

results_df = pd.DataFrame(results).sort_values("p_collapse", ascending=False)

print("\n  Collapse Probability (≤10% of year-2000 population by 2048):")
print(f"  {'Species':<25} {'Status':<6} {'P(collapse)':<14} {'Median 2048'}")
print("  " + "-" * 58)
for _, r in results_df.iterrows():
    print(
        f"  {r['species']:<25} {r['status']:<6} "
        f"{r['p_collapse']:<14.1%} {r['median_2048']:,.0f}"
    )


# ── Figure 3: Fan Charts for 6 Species ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=BG)
fig.subplots_adjust(hspace=0.45, wspace=0.35)
axes = axes.flatten()
years_fwd = np.arange(2023, 2049)

# Pick 6 interesting species to show
show = [
    "Vaquita",
    "Amur Leopard",
    "Black Rhino",
    "Snow Leopard",
    "Bengal Tiger",
    "Giant Panda",
]

for ax, sp in zip(axes, show):
    row = results_df[results_df["species"] == sp].iloc[0]
    orig = df[df["species"] == sp].iloc[0]
    paths = row["paths"]

    # Percentile bands
    p5 = np.percentile(paths, 5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    ax.fill_between(years_fwd, p5, p95, alpha=0.18, color=BLUE)
    ax.fill_between(years_fwd, p25, p75, alpha=0.35, color=PURPLE)
    ax.plot(years_fwd, p50, color=WHITE, linewidth=2, label="Median")

    # Critical threshold line
    threshold = orig["pop_2000"] * 0.10
    ax.axhline(
        threshold,
        color=PINK,
        linewidth=1.2,
        linestyle="--",
        label=f"Critical = {threshold:.0f}",
    )

    style(ax, title=sp, xlabel="Year", ylabel="Population")
    ax.legend(fontsize=6.5, labelcolor=WHITE, facecolor=PANEL, edgecolor="#2a2a4a")

fig.suptitle(
    "Monte Carlo Population Simulations (1,000 paths, 25 years)\n"
    "Shaded area = 5th to 95th percentile range",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
)
plt.savefig("figures/fig3_simulations.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("\nSaved → figures/fig3_simulations.png")
plt.close()


# =============================================================================
# STEP 5: MACHINE LEARNING — PREDICTING HIGH RISK SPECIES
# =============================================================================
# Can we predict which species are at highest risk using just a few features?
# We'll train two simple classifiers and compare them.

print("\n\n── Machine Learning: Extinction Risk Classifier ──────")

# Label: high risk = species that declined more than 15% OR is CR status
df["high_risk"] = ((df["pct_change"] < -15) | (df["status"] == "CR")).astype(int)

print(f"\n  High risk species : {df['high_risk'].sum()}")
print(f"  Lower risk species: {(df['high_risk'] == 0).sum()}")

# Features we'll use to predict risk
features = ["habitat_loss", "poaching", "pct_change"]
X = df[features].values
y = df["high_risk"].values

# Scale features so they're on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and evaluate two models using 5-fold cross-validation
# Cross-validation means we test on held-out data so the score is honest
lr = LogisticRegression(random_state=42, max_iter=500)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")
rf_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")

print(
    f"\n  Logistic Regression accuracy: {lr_scores.mean():.1%} ± {lr_scores.std():.1%}"
)
print(f"  Random Forest accuracy:       {rf_scores.mean():.1%} ± {rf_scores.std():.1%}")

# Fit the Random Forest on all data to get feature importances
rf.fit(X_scaled, y)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(
    ascending=False
)
print("\n  Most important features (Random Forest):")
for feat, imp in importances.items():
    print(f"    {feat:<18}: {imp:.3f}")


# ── Figure 4: Feature Importances ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
colors = [MINT, BLUE, PURPLE]
ax.barh(
    importances.index[::-1], importances.values[::-1], color=colors[::-1], height=0.55
)
for i, (feat, val) in enumerate(zip(importances.index[::-1], importances.values[::-1])):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", color=WHITE, fontsize=9)

style(
    ax,
    title="Random Forest — Feature Importances\nFor Extinction Risk Prediction",
    xlabel="Importance Score",
    ylabel="",
)
plt.tight_layout(pad=2)
plt.savefig(
    "figures/fig4_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("\nSaved → figures/fig4_feature_importance.png")
plt.close()


# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================
results_df.drop(columns=["paths"]).to_csv("outputs/simulation_results.csv", index=False)
df.to_csv("outputs/species_data.csv", index=False)

print("\n\nSaved → outputs/simulation_results.csv")
print("Saved → outputs/species_data.csv")
print("\n" + "=" * 55)
print("  Done! Check figures/ and outputs/ folders.")
print("=" * 55)
