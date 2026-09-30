# House Price Prediction

Predicts the sale price of a house from its living area, number of bedrooms and number of bathrooms.

- **Data:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Ames, Iowa; 1,460 training houses) — `train.csv`, `test.csv`
- **Features:** `Area` (above-ground living area), `TotalBathrooms` (full + half baths), `TotalBedrooms`
- **Model:** scikit-learn `GradientBoostingRegressor`, saved with pickle as `model.h5`
- **Result:** training RMSE ≈ $41.7k (linear-regression baseline: ≈ $52.8k)

## Run

```bash
streamlit run app.py
```

## Notes / next steps

- The RMSE above is on the training data; a hold-out or cross-validated RMSE would give a fairer estimate of real-world error.
- Only 3 features are used to keep the app simple; the dataset has 79, so adding quality, year built and neighbourhood would improve accuracy substantially.

> **Credits:** the training notebooks come from a public GitHub repository by another author. This repository adds bug fixes, a working Streamlit app and this documentation.
