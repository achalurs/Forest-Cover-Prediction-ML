import streamlit as st
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

# ======================================================
# MODEL TRAINING WITH PROGRESS (CACHED)
# ======================================================
@st.cache_resource
def load_model_with_metrics():
    status = st.status("🚀 Initializing model training...", expanded=True)

    status.write("📂 Loading dataset...")
    data = pd.read_csv("train.csv")

    status.write("🧹 Preparing features and target...")
    X = data.drop(["Cover_Type", "Id"], axis=1)
    y = data["Cover_Type"] - 1

    status.write("✂️ Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    status.write("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    status.write("🧠 Training XGBoost model (first run only)...")
    model = XGBClassifier(
        n_estimators=120,        # optimized for cloud
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        objective="multi:softmax",
        num_class=7,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    status.write("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    status.update(label="✅ Model training completed!", state="complete")

    return model, scaler, X.columns, accuracy


with st.spinner("⏳ Setting things up. Please wait..."):
    model, scaler, feature_columns, model_accuracy = load_model_with_metrics()

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📈 Charts & Analytics",
        "📊 Manual Prediction",
        "📁 File Upload Prediction"
    ]
)

# ======================================================
# COVER TYPE MAP
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
# 🏠 HOME PAGE (WITH ACCURACY)
# ======================================================
if page == "🏠 Home":
    st.title("🌲 Forest Cover Type Prediction System")

    st.markdown("""
    ### 📌 Project Overview
    This application predicts **forest cover types** using an
    **XGBoost Machine Learning model** trained on environmental data.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("🎯 Model Accuracy", f"{model_accuracy*100:.2f}%")
    col2.metric("📊 Classes", "7")
    col3.metric("📂 Dataset Size", "15,120 records")

    st.success("⬅️ Use the sidebar to explore predictions and analytics")

# ======================================================
# 📊 MANUAL PREDICTION
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
        h_dist_fire = st.number_input("Horizontal Distance to Fire Point", 0, 5000)

    if st.button("🔍 Predict"):
        input_data = np.zeros((1, len(feature_columns)))
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
# 📁 FILE UPLOAD
# ======================================================
elif page == "📁 File Upload Prediction":
    st.title("📁 CSV File Upload – Forest Cover Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if "Cover_Type" in df.columns:
            df = df.drop("Cover_Type", axis=1)
        if "Id" in df.columns:
            df = df.drop("Id", axis=1)

        if st.button("🚀 Predict for File"):
            scaled_data = scaler.transform(df)
            preds = model.predict(scaled_data) + 1

            df["Predicted_Cover_Type"] = preds
            df["Forest_Type_Name"] = df["Predicted_Cover_Type"].map(cover_types)

            st.dataframe(df.head())

# ======================================================
# 📈 ANALYTICS
# ======================================================
elif page == "📈 Charts & Analytics":
    st.title("📈 Forest Cover Analytics Dashboard")

    data = pd.read_csv("train.csv")

    st.subheader("🌳 Forest Cover Distribution")
    counts = data["Cover_Type"].value_counts().sort_index()
    chart_df = pd.DataFrame({
        "Forest Type": list(cover_types.values()),
        "Count": counts.values
    })
    st.bar_chart(chart_df.set_index("Forest Type"))

    st.subheader("⭐ Top 15 Feature Importances")
    imp_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(15)

    st.dataframe(imp_df)
