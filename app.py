import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pycountry
from sklearn.impute import SimpleImputer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🌍 Global Development Clustering",
    layout="wide"
)

# =========================
# CUSTOM DARK UI
# =========================
st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
.block-container {padding: 1.5rem;}
.card {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.metric {
    font-size: 20px;
    font-weight: bold;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# FLAG FUNCTION
# =========================
def get_flag(country_name):
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        code = country.alpha_2
        return "".join(chr(127397 + ord(c)) for c in code)
    except:
        return ""

# =========================
# LOAD MODEL (PKL)
# =========================
@st.cache_resource
def load_models():
    try:
        with open("model_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)

        return (
            bundle["scaler"],
            bundle["pca"],
            bundle["kmeans"],
            bundle["columns"]
        )
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

scaler, pca, model, columns = load_models()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌍 Global Dev Clustering")
st.sidebar.caption("Unsupervised ML Project")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload dataset", type=["csv", "xlsx"]
)

menu = st.sidebar.radio("📊 Navigation", [
    "Overview & EDA",
    "Feature Analysis",
    "Clustering Models",
    "Model Comparison",
    "Country Explorer"
])

# =========================
# MAIN APP
# =========================
if uploaded_file:

    # LOAD DATA
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except:
        st.error("❌ Error reading file")
        st.stop()

    if "Country" not in df.columns:
        st.error("Dataset must contain 'Country' column")
        st.stop()

    # =========================
    # COUNTRY FILTER
    # =========================
    country_list = sorted(df["Country"].unique())

    selected_country = st.sidebar.selectbox(
        "🌐 Select Country",
        ["All Countries"] + country_list,
        format_func=lambda x: f"{get_flag(x)} {x}" if x != "All Countries" else x
    )

    # =========================
    # DATA CLEANING
    # =========================
    df_clean = df.drop("Country", axis=1).copy()

    for col in columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    df_clean = df_clean[columns]

    # Convert values
    for col in df_clean.columns:
        temp = df_clean[col].astype(str)\
            .str.replace('$', '', regex=False)\
            .str.replace(',', '', regex=False)

        if temp.str.contains('%').any():
            df_clean[col] = pd.to_numeric(
                temp.str.replace('%', '', regex=False),
                errors='coerce'
            ) / 100
        else:
            df_clean[col] = pd.to_numeric(temp, errors='coerce')

    # =========================
    # IMPUTATION
    # =========================
    imputer = SimpleImputer(strategy="mean")
    df_clean[:] = imputer.fit_transform(df_clean)

    # =========================
    # TRANSFORMATION
    # =========================
    X_scaled = scaler.transform(df_clean)
    X_pca = pca.transform(X_scaled)

    clusters = model.predict(X_pca)
    df["Cluster"] = clusters

    # =========================
    # CLUSTER LABELS
    # =========================
    cluster_data = df_clean.copy()
    cluster_data["Cluster"] = clusters

    cluster_data["GDP"] = pd.to_numeric(cluster_data["GDP"], errors="coerce")

    cluster_means = cluster_data.groupby("Cluster")["GDP"].mean().sort_values()

    labels = ["Low Income", "Middle Income", "High Income"]
    cluster_labels = {}

    for i, cluster_id in enumerate(cluster_means.index):
        cluster_labels[cluster_id] = labels[i] if i < len(labels) else f"Cluster {cluster_id}"

    df["Cluster Name"] = df["Cluster"].map(cluster_labels)

    # =========================
    # FILTER DATA
    # =========================
    if selected_country != "All Countries":
        df_filtered = df[df["Country"] == selected_country]
        df_clean_filtered = df_clean.loc[df_filtered.index]
    else:
        df_filtered = df
        df_clean_filtered = df_clean

    # =========================
    # HEADER
    # =========================
    st.title("🌍 Global Development Clustering")

    if selected_country != "All Countries":
        st.subheader(f"{get_flag(selected_country)} {selected_country}")
    else:
        st.subheader("All Countries Overview")

    # =========================
    # OVERVIEW
    # =========================
    if menu == "Overview & EDA":

        col1, col2, col3 = st.columns(3)

        col1.metric("🌍 Countries", df_filtered["Country"].nunique())
        col2.metric("📊 Features", df_clean_filtered.shape[1])
        col3.metric("🧠 Clusters", df["Cluster"].nunique())

        st.dataframe(df_filtered.head())

        # Missing values
        st.subheader("Missing Values")
        st.bar_chart(df_clean_filtered.isnull().sum())

        # Correlation heatmap
        st.subheader("Correlation Matrix")
        corr = df_clean_filtered.corr()

        fig, ax = plt.subplots()
        im = ax.imshow(corr)
        plt.colorbar(im)
        st.pyplot(fig)

    # =========================
    # FEATURE ANALYSIS
    # =========================
    elif menu == "Feature Analysis":

        feature = st.selectbox("Select Feature", df_clean_filtered.columns)

        st.metric("Mean", round(df_clean_filtered[feature].mean(), 2))
        st.metric("Max", round(df_clean_filtered[feature].max(), 2))
        st.metric("Min", round(df_clean_filtered[feature].min(), 2))

        fig, ax = plt.subplots()
        ax.hist(df_clean_filtered[feature], bins=30)
        st.pyplot(fig)

    # =========================
    # CLUSTERING
    # =========================
    elif menu == "Clustering Models":

        st.subheader("Cluster Distribution")
        st.bar_chart(pd.Series(clusters).value_counts())

        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
        st.pyplot(fig)

    # =========================
    # MODEL COMPARISON
    # =========================
    elif menu == "Model Comparison":

        cluster_data = df_clean_filtered.copy()
        cluster_data["Cluster"] = clusters

        st.dataframe(cluster_data.groupby("Cluster").mean())

    # =========================
    # COUNTRY EXPLORER
    # =========================
    elif menu == "Country Explorer":

        if selected_country == "All Countries":
            st.warning("Please select a country")
        else:
            row_clean = df_clean_filtered.loc[df_filtered.index].iloc[0]
            st.write(row_clean)

else:
    st.info("⬅️ Upload a dataset to begin")
