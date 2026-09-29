import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model.h5"  # pickled GradientBoostingRegressor


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()


def predict_sale_price(area, bedrooms, bathrooms):
    # Pass a named DataFrame so each value lands on the feature the model was trained on
    # (training order is Area, TotalBathrooms, TotalBedrooms).
    input_data = pd.DataFrame(
        [{"Area": area, "TotalBathrooms": bathrooms, "TotalBedrooms": bedrooms}]
    )[list(model.feature_names_in_)]
    return model.predict(input_data)[0]


# Streamlit app
st.title("🏠 House Sale Price Predictor")
st.caption("Gradient Boosting model trained on the Kaggle House Prices (Ames, Iowa) dataset")

area = st.number_input("Living area (sq ft)", min_value=100, max_value=10000, value=1500, step=50)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=6.0, value=2.0, step=0.5)

if st.button("Predict Sale Price"):
    predicted_price = predict_sale_price(area, bedrooms, bathrooms)
    st.success(f"Predicted sale price: ${predicted_price:,.0f}")

# Run with: streamlit run app.py
