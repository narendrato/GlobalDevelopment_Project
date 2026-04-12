import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer

# --- Load the models ---
try:
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    kmeans_model = joblib.load("kmeans.joblib")
    columns = joblib.load("columns.joblib")   # saved training columns
except FileNotFoundError:
    st.error("Required files not found. Ensure all .joblib files are present.")
    st.stop()

st.title("Global Development Clustering App")
st.write("Upload dataset to classify countries into clusters")

uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # --- Read file ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write("### Original Data")
    st.dataframe(df_uploaded.head())
    st.write(f"Shape of uploaded data: {df_uploaded.shape}")
    st.write(f"Columns of uploaded data: {df_uploaded.columns.tolist()}")

    # --- Save country names ---
    country_names = df_uploaded['Country'].copy() if 'Country' in df_uploaded.columns else None

    # --- Drop non-feature columns ---
    df_processed = df_uploaded.drop('Country', axis=1, errors='ignore').copy()
    st.write(f"Shape after dropping Country: {df_processed.shape}")

    # --- Data Cleaning and Column Alignment ---
    original_feature_cols = columns
    st.write(f"Expected original features (from columns.joblib): {len(original_feature_cols)} columns.")

    # Add any missing columns from original_feature_cols to df_processed, filled with NaN
    for col in original_feature_cols:
        if col not in df_processed.columns:
            df_processed[col] = np.nan
    st.write(f"Shape after adding missing columns: {df_processed.shape}")

    # Now, iterate ONLY through original_feature_cols to clean and convert them
    for col in original_feature_cols:
        if df_processed[col].dtype == 'object':
            temp = df_processed[col].astype(str)\
                .str.replace('$', '', regex=False)\
                .str.replace(',', '', regex=False)\
                .str.strip()

            if temp.str.contains('%').any():
                df_processed[col] = pd.to_numeric(temp.str.replace('%', '', regex=False), errors='coerce') / 100
            else:
                df_processed[col] = pd.to_numeric(temp, errors='coerce')
        elif not pd.api.types.is_numeric_dtype(df_processed[col]):
             df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Finally, select and reorder columns to perfectly match the training features
    df_processed = df_processed[original_feature_cols]

    st.write(f"### Debugging Feature Alignment (Post-Processing):")
    st.write(f"Shape of df_processed before scaling: {df_processed.shape}")
    st.write(f"Columns of df_processed before scaling: {df_processed.columns.tolist()}")

    # Critical check: Ensure the number of features matches what PCA expects
    expected_features_count = len(original_feature_cols) # This should be 21
    if df_processed.shape[1] != expected_features_count:
        st.error(f"Feature count mismatch! Expected {expected_features_count} features but got {df_processed.shape[1]} after preprocessing.")
        st.stop()

    # --- Scaling ---
    try:
        X_scaled = scaler.transform(df_processed)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # --- Imputation ---
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)

    # --- PCA ---
    try:
        X_pca = pca.transform(X_imputed)
    except Exception as e:
        st.error(f"PCA error: {e}")
        st.stop()

    # --- Prediction ---
    try:
        clusters = kmeans_model.predict(X_pca)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # --- Map cluster names ---
    cluster_names = {
        0: "Developed",
        1: "Developing",
        2: "Underdeveloped"
    }

    df_uploaded['Cluster'] = clusters
    df_uploaded['Cluster_Name'] = df_uploaded['Cluster'].map(cluster_names)

    if country_names is not None:
        df_uploaded['Country'] = country_names

    # --- Show results ---
    st.write("### Cluster Results")
    if 'Country' in df_uploaded.columns:
        st.dataframe(df_uploaded[['Country', 'Cluster_Name']])
    else:
        st.dataframe(df_uploaded[['Cluster_Name']])

    # --- Visualization ---
    st.write("### Cluster Visualization")

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("K-Means Clusters")

    # Legend
    handles, _ = scatter.legend_elements()
    unique_clusters = sorted(np.unique(clusters))
    labels = [cluster_names.get(c, f"Cluster {c}") for c in unique_clusters]
    ax.legend(handles, labels, title="Clusters")

    st.pyplot(fig)
