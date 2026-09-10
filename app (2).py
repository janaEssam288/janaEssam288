import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, time

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide",
)

# ---------------------------------------------------------
# 1) Feature engineering — kept consistent with the notebook
# ---------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def clean_dataframe(df):
    """Same cleaning logic used in the training notebook."""
    df = df.copy()

    text_cols = df.select_dtypes(include="object").columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"NaN": np.nan, "nan": np.nan})

    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = df["Weatherconditions"].str.replace(
            "conditions ", "", regex=False
        )

    if "Time_taken(min)" in df.columns:
        df["Time_taken(min)"] = (
            df["Time_taken(min)"]
            .str.replace("(min) ", "", regex=False)
            .astype(float)
        )

    for c in [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def engineer_features(df):
    """Create the exact model-facing features used by the notebook."""
    df = df.copy()

    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    ).abs().clip(upper=30)

    order_date = pd.to_datetime(
        df["Order_Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )
    df["order_day_of_week"] = order_date.dt.dayofweek

    order_time = pd.to_datetime(
        df["Time_Orderd"],
        format="%H:%M:%S",
        errors="coerce",
    )
    df["order_hour"] = order_time.dt.hour

    drop_cols = [
        "ID",
        "Delivery_person_ID",
        "Order_Date",
        "Time_Orderd",
        "Time_Order_picked",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ---------------------------------------------------------
# 2) Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_model_from_path(path):
    return joblib.load(path)


# ---------------------------------------------------------
# 3) Styling
# ---------------------------------------------------------
st.title("🍔 Food Delivery Time Predictor")
st.caption("Machine Learning project — estimate delivery time in minutes.")

with st.sidebar:
    st.header("⚙️ Model")
    st.write(
        "Upload your trained `delivery_time_model.joblib` file "
        "or place it next to `app.py`."
    )

    default_model = "delivery_time_model.joblib"
    uploaded_model = st.file_uploader(
        "Upload .joblib model",
        type=["joblib"],
    )

    model = None

    if uploaded_model is not None:
        try:
            model = joblib.load(uploaded_model)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Could not load the model: {e}")

    elif __import__("os").path.exists(default_model):
        try:
            model = load_model_from_path(default_model)
            st.success("Local model loaded.")
        except Exception as e:
            st.error(f"Could not load local model: {e}")
    else:
        st.warning(
            "No model file found yet. Upload "
            "`delivery_time_model.joblib`."
        )

# ---------------------------------------------------------
# 4) Main input form
# ---------------------------------------------------------
st.subheader("📦 Enter delivery information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Delivery person age",
        min_value=15,
        max_value=60,
        value=30,
        step=1,
    )

    rating = st.number_input(
        "Delivery person rating",
        min_value=1.0,
        max_value=6.0,
        value=4.5,
        step=0.1,
    )

    vehicle_condition = st.selectbox(
        "Vehicle condition",
        options=[0, 1, 2, 3],
        index=2,
    )

    multiple_deliveries = st.selectbox(
        "Multiple deliveries",
        options=[0, 1, 2, 3],
        index=0,
    )

with col2:
    weather = st.selectbox(
        "Weather",
        [
            "Sunny",
            "Cloudy",
            "Windy",
            "Stormy",
            "Sandstorms",
            "Fog",
            "Windy",
        ],
        index=0,
    )

    traffic = st.selectbox(
        "Road traffic density",
        ["Low", "Medium", "High", "Jam"],
        index=1,
    )

    order_type = st.selectbox(
        "Type of order",
        ["Drinks", "Meal", "Snack", "Buffet"],
        index=1,
    )

    vehicle_type = st.selectbox(
        "Type of vehicle",
        ["bicycle", "scooter", "motorcycle", "electric_scooter"],
        index=1,
    )

with col3:
    festival = st.selectbox(
        "Festival",
        ["No", "Yes"],
        index=0,
    )

    city = st.selectbox(
        "City",
        ["Urban", "Metropolitian", "Semi-Urban"],
        index=1,
    )

    order_date = st.date_input(
        "Order date",
        value=date(2024, 1, 15),
    )

    order_time = st.time_input(
        "Order time",
        value=time(18, 0),
        step=300,
    )

st.markdown("---")
st.subheader("📍 Location")

loc1, loc2 = st.columns(2)

with loc1:
    restaurant_lat = st.number_input(
        "Restaurant latitude",
        value=19.0760,
        format="%.6f",
    )
    restaurant_lon = st.number_input(
        "Restaurant longitude",
        value=72.8777,
        format="%.6f",
    )

with loc2:
    delivery_lat = st.number_input(
        "Delivery latitude",
        value=19.0860,
        format="%.6f",
    )
    delivery_lon = st.number_input(
        "Delivery longitude",
        value=72.8877,
        format="%.6f",
    )

# ---------------------------------------------------------
# 5) Prediction
# ---------------------------------------------------------
st.markdown("---")

predict = st.button(
    "🚀 Predict Delivery Time",
    type="primary",
    use_container_width=True,
)

if predict:
    if model is None:
        st.error(
            "The app cannot predict yet because the trained model is missing. "
            "Upload `delivery_time_model.joblib` from the sidebar."
        )
    else:
        try:
            # Build raw input using the same column names as training data.
            raw_input = pd.DataFrame(
                [
                    {
                        "ID": "APP-1",
                        "Delivery_person_ID": "APP-PERSON",
                        "Delivery_person_Age": age,
                        "Delivery_person_Ratings": rating,
                        "Restaurant_latitude": restaurant_lat,
                        "Restaurant_longitude": restaurant_lon,
                        "Delivery_location_latitude": delivery_lat,
                        "Delivery_location_longitude": delivery_lon,
                        "Order_Date": order_date.strftime("%d-%m-%Y"),
                        "Time_Orderd": order_time.strftime("%H:%M:%S"),
                        "Time_Order_picked": order_time.strftime("%H:%M:%S"),
                        "Weatherconditions": weather,
                        "Road_traffic_density": traffic,
                        "Vehicle_condition": vehicle_condition,
                        "Type_of_order": order_type,
                        "Type_of_vehicle": vehicle_type,
                        "multiple_deliveries": multiple_deliveries,
                        "Festival": festival,
                        "City": city,
                    }
                ]
            )

            cleaned = clean_dataframe(raw_input)
            features = engineer_features(cleaned)

            prediction = float(model.predict(features)[0])
            prediction = max(0.0, prediction)

            distance = float(features["distance_km"].iloc[0])

            st.success("Prediction completed!")

            result1, result2 = st.columns(2)

            with result1:
                st.metric(
                    "Estimated delivery time",
                    f"{prediction:.1f} minutes",
                )

            with result2:
                st.metric(
                    "Estimated distance",
                    f"{distance:.2f} km",
                )

            st.info(
                f"Estimated delivery time: **{prediction:.1f} minutes**."
            )

            with st.expander("🔎 Model input after feature engineering"):
                st.dataframe(features, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# ---------------------------------------------------------
# 6) About
# ---------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ About this project"):
    st.write(
        """
        This app uses the trained Food Delivery Time Prediction model.

        The model predicts `Time_taken(min)` from delivery-person,
        order, traffic, weather, vehicle, city, time, and location features.

        The latitude/longitude pairs are converted into `distance_km`,
        while the order date/time are converted into
        `order_day_of_week` and `order_hour` before prediction.
        """
    )
