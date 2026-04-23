import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained components
# -------------------------------
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
features = joblib.load("features.pkl")

# -------------------------------
# App Title
# -------------------------------
st.title("🌍 Country Development Clustering App")

st.write("Upload your cleaned dataset to classify countries into:")
st.write("👉 Developed | Developing | Underdeveloped")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

# -------------------------------
# Process File
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.write(df.head())

    # -------------------------------
    # Check Missing Columns
    # -------------------------------
    missing_cols = [col for col in features if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
    else:
        # -------------------------------
        # Select Correct Features
        # -------------------------------
        df_input = df[features]

        st.write("✅ Feature check passed")
        st.write("Expected features:", scaler.n_features_in_)
        st.write("Your features:", df_input.shape[1])

        # -------------------------------
        # Apply Model
        # -------------------------------
        scaled = scaler.transform(df_input)
        reduced = pca.transform(scaled)
        clusters = model.predict(reduced)

        # -------------------------------
        # Map Cluster Names
        # -------------------------------
        cluster_names = {
            0: "Developed",
            1: "Developing",
            2: "Underdeveloped"
        }

        df["Predicted_Cluster"] = clusters
        df["Category"] = df["Predicted_Cluster"].map(cluster_names)

        # -------------------------------
        # Show Results
        # -------------------------------
        st.subheader("✅ Prediction Results")
        st.write(df.head())

        # -------------------------------
        # Download Button
        # -------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Predicted Data",
            data=csv,
            file_name="predicted_countries.csv",
            mime="text/csv"
        )

        st.success("🎉 Prediction completed successfully!")