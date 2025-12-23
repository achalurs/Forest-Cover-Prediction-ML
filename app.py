import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "xgboost_forest_model.pkl"
SCALER_PATH = "xgboost_scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.info("⏳ Training model for first-time use. Please wait...")
    os.system("python xgboost_forest_cover.py")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Manual Prediction",
        "📁 File Upload Prediction",
        "📈 Charts & Analytics"
    ]
)

# ======================================================
# COVER TYPE MAPPING
# ======================================================
cover_types = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# ======================================================
# 🏠 HOME DASHBOARD
# ======================================================
if page == "🏠 Home":
    st.title("🌲 Forest Cover Type Prediction System")

    st.markdown("""
    ### 📌 Project Overview
    This application predicts **forest cover types** using an advanced
    **XGBoost Machine Learning model** trained on environmental data.

    ### 🔧 Technologies Used
    - Python
    - XGBoost
    - Scikit-learn
    - Streamlit

    ### 🚀 Features
    - Manual prediction
    - Bulk prediction via CSV upload
    - Analytics & visualization dashboard
    """)

    st.success("⬅️ Use the sidebar to explore the application")

# ======================================================
# 📊 MANUAL PREDICTION DASHBOARD
# ======================================================
elif page == "📊 Manual Prediction":
    st.title("📊 Manual Forest Cover Prediction")

    col1, col2 = st.columns(2)

    with col1:
        elevation = st.number_input("Elevation (meters)", 2000, 4000)
        aspect = st.number_input("Aspect (degrees)", 0, 360)
        slope = st.number_input("Slope (degrees)", 0, 60)
        h_dist_water = st.number_input("Horizontal Distance to Water", 0, 5000)
        v_dist_water = st.number_input("Vertical Distance to Water", -500, 500)

    with col2:
        h_dist_road = st.number_input("Horizontal Distance to Road", 0, 5000)
        hillshade_9am = st.slider("Hillshade at 9 AM", 0, 255)
        hillshade_noon = st.slider("Hillshade at Noon", 0, 255)
        hillshade_3pm = st.slider("Hillshade at 3 PM", 0, 255)
        h_dist_fire = st.number_input("Horizontal Distance to Fire Points", 0, 5000)

    if st.button("🔍 Predict Forest Cover"):
        input_data = np.zeros((1, model.n_features_in_))
        input_data[0, :10] = [
            elevation, aspect, slope,
            h_dist_water, v_dist_water,
            h_dist_road, hillshade_9am,
            hillshade_noon, hillshade_3pm,
            h_dist_fire
        ]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0] + 1

        st.success(f"🌳 Predicted Forest Cover Type: **{cover_types[prediction]}**")

# ======================================================
# 📁 FILE UPLOAD PREDICTION DASHBOARD
# ======================================================
elif page == "📁 File Upload Prediction":
    st.title("📁 CSV File Upload – Forest Cover Prediction")

    st.markdown("""
    **Instructions:**
    - Upload a CSV file with the same structure as training data
    - `Cover_Type` column is optional
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        if "Cover_Type" in df.columns:
            df = df.drop("Cover_Type", axis=1)
        if "Id" in df.columns:
            df = df.drop("Id", axis=1)

        if st.button("🚀 Predict for Uploaded File"):
            scaled_data = scaler.transform(df)
            predictions = model.predict(scaled_data) + 1

            df["Predicted_Cover_Type"] = predictions
            df["Forest_Type_Name"] = df["Predicted_Cover_Type"].map(cover_types)

            st.subheader("✅ Prediction Results")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="forest_cover_predictions.csv",
                mime="text/csv"
            )

# ======================================================
# 📈 CHARTS & ANALYTICS DASHBOARD
# ======================================================
elif page == "📈 Charts & Analytics":
    st.title("📈 Forest Cover Analytics Dashboard")

    data = pd.read_csv("train.csv")

    # -------------------------------
    # Chart 1: Cover Type Distribution
    # -------------------------------
    st.subheader("🌳 Forest Cover Type Distribution")

    cover_counts = data["Cover_Type"].value_counts().sort_index()
    cover_names = list(cover_types.values())

    dist_df = pd.DataFrame({
        "Forest Type": cover_names,
        "Count": cover_counts.values
    })

    st.bar_chart(dist_df.set_index("Forest Type"))

    # -------------------------------
    # Chart 2: Feature Importance
    # -------------------------------
    st.subheader("⭐ Top 15 Important Features")

    feature_names = data.drop(["Cover_Type", "Id"], axis=1).columns
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    st.dataframe(imp_df)
    st.bar_chart(imp_df.set_index("Feature"))

    # -------------------------------
    # Chart 3: Prediction Simulation
    # -------------------------------
    st.subheader("📊 Prediction Confidence Simulation")

    sample = data.drop(["Cover_Type", "Id"], axis=1).sample(500, random_state=42)
    sample_scaled = scaler.transform(sample)
    preds = model.predict(sample_scaled) + 1

    pred_counts = pd.Series(preds).value_counts().sort_index()

    pred_df = pd.DataFrame({
        "Forest Type": cover_names,
        "Predicted Count": pred_counts.values
    })

    st.bar_chart(pred_df.set_index("Forest Type"))

    st.success("✅ Analytics generated successfully")
