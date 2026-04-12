import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# --- Load the models ---
try:
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    kmeans_model = joblib.load("kmeans.joblib")
except FileNotFoundError:
    st.error("Model files (scaler.joblib, pca.joblib, kmeans.joblib) not found. Please ensure they are in the same directory.")
    st.stop()

st.title("Global Development Clustering App")
st.write("Upload dataset to classify countries into clusters")

uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file based on its extension
    if uploaded_file.name.endswith('.csv'):
        df_uploaded = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df_uploaded = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or XLSX file.")
        st.stop()

    st.write("### Original Data (Uploaded)")
    st.dataframe(df_uploaded.head())

    # Save original country names for results
    country_names_uploaded = df_uploaded['Country'].copy() # Use .copy() to avoid SettingWithCopyWarning
    df_processed = df_uploaded.drop('Country', axis=1, errors='ignore').copy()

    # --- Data Cleaning (mimicking the notebook's preprocessing) ---
    # This part assumes `df` (the original training dataframe structure) is still accessible if needed for column alignment
    # In a real deployment, you might want to save and load the list of original columns.

    # Clean numeric columns
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            temp_col = df_processed[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
            if temp_col.str.contains('%').any():
                df_processed[col] = pd.to_numeric(temp_col.str.replace('%', '', regex=False), errors='coerce') / 100
            else:
                df_processed[col] = pd.to_numeric(temp_col, errors='coerce')

    df_processed = df_processed.select_dtypes(include=np.number)

    # To align columns with the training data, we need the column names from the original `df` used for training.
    # Assuming `df` from the notebook's global scope is implicitly available here or its columns are known.
    # For a robust solution, you would save the list of expected columns during training.
    # For this example, let's try to infer from the `scaler` or `pca` if possible, or assume `df` was the training DF.
    # As `df` is a global variable from the previous run, we can use it, but this is not ideal for standalone deployment.

    # IMPORTANT: In a real app, `original_training_columns` should be loaded from a saved list,
    # not rely on a global variable from a notebook run.
    # For now, we'll try to reconstruct based on the number of features the scaler expects.
    # This is a simplification; a production app needs to explicitly handle feature lists.

    # Let's assume the order of columns after initial preprocessing and dropping correlated ones
    # is consistent with `scaler.n_features_in_`.
    # A more robust solution would involve saving the column names that were fed to the scaler.

    # Create a dummy dataframe with the same columns as the training data to ensure consistent scaling
    # This is a workaround since original_training_columns was not explicitly saved.
    # We need to ensure `df_processed` has the same columns and order as the `df` that was used to train the scaler/pca.

    # Reconstructing the training dataframe's column list (this is brittle for deployment)
    # In a real app, save `df.columns.tolist()` that went into `StandardScaler`
    # For now, let's use the columns that `scaler` expects, if available
    try:
        # This assumes the scaler maintains feature names, which StandardScaler doesn't typically do by default
        # A safer approach is to explicitly save the feature names during training
        # For demonstration, we will rely on the `df` global if still in a notebook context.
        # If deploying as a standalone .py, the columns must be known a priori.

        # Dummy dataframe to align columns, assuming a certain number of features
        # This is a major simplification. Real deployment needs careful column management.
        # If running as a standalone .py, 'df' might not exist, so a more robust method is needed.
        # Let's assume `original_training_df_cols` was saved.
        # For now, let's use the current `df`'s numeric columns before clustering columns were added.

        # The `df` variable in the current notebook state contains clustering info, so we need to filter.
        # A better way would be to save `df.columns` _before_ adding cluster columns.
        # From the kernel state, 'df' has 'HC_Cluster', 'KMeans_Cluster', 'DBSCAN_Cluster', 'Cluster_Name'
        # Let's filter these out to get the original feature set used for scaling/PCA.
        known_cluster_cols = ['HC_Cluster', 'KMeans_Cluster', 'DBSCAN_Cluster', 'Cluster_Name']
        original_feature_cols = [col for col in df.columns if col not in known_cluster_cols]

        # Add missing columns with NaN values and select only the training columns
        for col in original_feature_cols:
            if col not in df_processed.columns:
                df_processed[col] = np.nan

        # Reorder columns to match the training data
        df_processed = df_processed[original_feature_cols]

    except NameError:
        st.error("Could not retrieve original training columns. Ensure the application can access them.")
        st.stop()

    # --- Scaling and Imputation ---
    X_uploaded_scaled = scaler.transform(df_processed)

    # NOTE: The SimpleImputer was not pickled. For consistent imputation,
    # the imputer object should have been saved during training. Here,
    # a new imputer is created and fitted on the *uploaded data's* scaled features,
    # which might lead to slightly different results if the missing value patterns/means
    # in the uploaded data differ significantly from the training data.
    imputer = SimpleImputer(strategy='mean')
    X_uploaded_scaled_imputed = imputer.fit_transform(X_uploaded_scaled)

    # --- PCA Transformation ---
    X_uploaded_pca = pca.transform(X_uploaded_scaled_imputed)

    # --- Prediction ---
    clusters_uploaded = kmeans_model.predict(X_uploaded_pca)

    cluster_names = {
        0: "Developed",
        1: "Developing",
        2: "Underdeveloped"
    }

    df_uploaded['KMeans_Cluster'] = clusters_uploaded
    df_uploaded['Cluster_Name'] = df_uploaded['KMeans_Cluster'].map(cluster_names)
    df_uploaded['Country'] = country_names_uploaded # Ensure country names are restored

    result_uploaded = df_uploaded[['Country', 'Cluster_Name']]
    st.write("### Cluster Results")
    st.dataframe(result_uploaded)

    st.write("### Cluster Visualization")
    fig, ax = plt.subplots()
    # Use the first two PCA components for visualization
    scatter = ax.scatter(X_uploaded_pca[:,0], X_uploaded_pca[:,1], c=clusters_uploaded, cmap='viridis')
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("K-Means Clusters (PCA)")

    # Create a legend
    handles, labels = scatter.legend_elements()
    unique_clusters = sorted(np.unique(clusters_uploaded))
    legend_labels = [cluster_names.get(c, f"Cluster {c}") for c in unique_clusters]
    ax.legend(handles, legend_labels, title="Clusters")

    st.pyplot(fig)
