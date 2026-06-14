# 🌿 Wildlife Conservation Data Analysis

A data science project analyzing real population data for 20 endangered species — exploring trends, running simulations, and building a machine learning classifier to identify extinction risk.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat)
![Data](https://img.shields.io/badge/Data-Real%20IUCN%20Baselines-2ea043?style=flat)

-----

## 🦏 What It Does

1. Loads real population data for 20 endangered species (IUCN Red List)
1. Explores trends with charts and statistics
1. Runs a Monte Carlo simulation to estimate future collapse probability
1. Trains two classifiers to predict which species are most at risk

-----

## 🔢 Real Data

Every population number is sourced from a published report — not made up.

|Species     |~2000|~2023    |Trend|Source                     |
|------------|-----|---------|-----|---------------------------|
|Vaquita     |570  |**7**    |−99% |IUCN Vaquita SRG 2024      |
|Javan Rhino |58   |**50**   |−14% |Int’l Rhino Foundation 2024|
|Amur Leopard|30   |**100**  |+233%|Land of Leopard NP 2023    |
|Black Rhino |2,700|**6,788**|+151%|IRF 2024                   |
|Bengal Tiger|3,500|**3,900**|+11% |India Tiger Census 2022    |

-----

## 🚀 How to Run

```bash
pip install -r requirements.txt
python WildlifeConservationDataAnalysis.py
```

-----

## 📊 Outputs

```
figures/
  fig1_population_change.png   # % change bar chart for all 20 species
  fig2_habitat_scatter.png     # Habitat loss vs. population change
  fig3_simulations.png         # Monte Carlo fan charts (6 species)
  fig4_feature_importance.png  # Random Forest feature importances

outputs/
  species_data.csv             # Full species dataset
  simulation_results.csv       # Collapse probabilities per species
```

-----

## 📂 Project Structure

```
WildlifeConservation/
├── WildlifeConservationDataAnalysis.py   # Main script
├── requirements.txt
├── README.md
├── figures/
└── outputs/
```

-----

## Sample Visualizations

### Population Change Analysis
![Population Change](fig1_population_change.png)

### Habitat Loss Analysis
![Habitat Analysis](fig2_habitat_scatter.png)

### Population Simulations
![Population Simulations](fig3_simulations.png)

### Feature Importance
![Feature Importance](fig4_feature_importance.png)

-----

## 👤 Author

**Jahari Lockett** — Data Science & Analytics, Florida Atlantic University
Conservation Volunteer @ Loggerhead Marine Life Center
[LinkedIn](https://www.linkedin.com/in/jahari-e-lockett-b4aa04246/) · [lockettj2023@fau.edu](mailto:lockettj2023@fau.edu)
