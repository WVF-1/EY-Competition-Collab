# EY-Competition-Collab
The GitHub Repo which will house the project(s) for the EY Data and AI competition.

---

## Results
The best methodology for modeling the three target variables was a random forest model, with a healthy amount of external data. We achieved a top 50 competitors (top 5%) score of 0.4689. This allowed us to earn a certificate for participating, as we reached a winning threshold.

## Certificate of High Achievment
<img width="485" height="340" alt="image" src="https://github.com/user-attachments/assets/c54d1d6e-a1c9-4e82-8363-2325ae3713a4" />

--- 

# 🌍 Water Quality Prediction — EY AI & Data Challenge (Top 5%)

**Final Ranking:** 49th out of ~1,000 teams (Top 5%)  
**Competition:** EY AI & Data Challenge 2026  
**Focus:** Predicting water quality across South African river systems using geospatial and environmental data

---

## Overview

This project was developed as part of the EY AI & Data Challenge, focused on predicting three critical water quality indicators:

- **Total Alkalinity**
- **Electrical Conductance**
- **Dissolved Reactive Phosphorus (DRP)**

The goal was to build a robust machine learning pipeline capable of generalizing across spatially distributed river systems, using both provided and externally sourced environmental data.

---

## Key Approach

This solution emphasizes feature engineering over model complexity, leveraging domain-aware transformations and geospatial reasoning.

### Model
- Random Forest Regressor (Scikit-learn)
- Parameter tuning for bias-variance balance
- Final models trained per target variable

### Core Techniques
- Spatial feature engineering (lat/lon interactions, clustering)
- Temporal encoding (cyclical date features, seasonal effects)
- Hydrology-informed features (flow averages, lag features, distance weighting)
- External data integration (climate, land cover, hydrology)
- Median imputation for incomplete joins
- Cross-validation with focus on generalization

---

## Feature Engineering Highlights

### Spatial Features
- Latitude / Longitude
- Interaction terms (lat², lon², lat×lon)
- Distance to nearest monitoring station
- Spatial clustering (regional grouping)

### Temporal Features
- Year, Day of Year
- Cyclical encoding:
  - `month_sin`, `month_cos`
  - `doy_sin`, `doy_cos`
- Seasonal regime classification (South Africa rainfall zones)
- Wet vs dry season indicators

### Environmental Features (External Data)
- Rainfall (1-day, 7-day, 30-day aggregates)
- Temperature and humidity trends
- Land cover classification (urban, cropland, water, vegetation)

### Hydrology Features
- Flow averages (7-day, 30-day)
- Lagged flow values
- Distance-weighted flow signals

---

## Results

| Model | Performance |
|------|--------|
| EY Benchmark | ~0.53–0.58 R² |
| Final Model | **~0.83–0.85 R² (TA & EC)** |
| DRP | Lower predictability (~0.68 R²) due to diffuse signal |

### Key Observations
- **Spatial features dominated model importance**, highlighting strong geographic dependence
- **Electrical Conductance** was the hardest to model initially, but improved with feature engineering
- **DRP remained the most challenging target**, likely due to:
  - localized chemical variability
  - missing upstream nutrient signals

---

## Challenges

- Sparse alignment between external datasets and training points
- Weak linear correlations in raw hydrology data
- High spatial autocorrelation (risk of overfitting)
- Limited upstream/downstream relationship data

---

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Geospatial data processing
- Custom EDA + feature engineering scripts

---

## Key Takeaways

- Real-world ML success often comes from feature engineering, not model complexity
- Geospatial problems require spatial awareness in both modeling and validation
- External data integration is powerful but requires careful alignment
- Domain knowledge (hydrology, climate) significantly improves results

---

## Acknowledgment

This project was completed as part of the EY AI & Data Challenge 2026 with my friend Adisyn Mooredhead.

---
