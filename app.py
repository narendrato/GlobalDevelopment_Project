import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Global Dev Clustering",
    layout="wide",
)

# =========================
# CUSTOM CSS (DARK UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
    color: white;
}
.block-container {
    padding: 2rem;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.metric {
    font-size: 20px;
    font-weight: bold;
}
.sidebar .sidebar-content {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌍 Global Dev Clustering")
st.sidebar.write("Unsupervised ML Project")

uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    model = joblib.load("kmeans.joblib")
    columns = joblib.load("columns.joblib")
    return scaler, pca, model, columns

try:
    scaler, pca, model, columns = load_models()
except:
    st.error("❌ Model files missing")
    st.stop()

# =========================
# MAIN APP
# =========================
if uploaded_file:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    country_col = df["Country"]

    # =========================
    # DATA CLEANING
    # =========================
    df_clean = df.drop("Country", axis=1, errors="ignore")

    for col in columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    df_clean = df_clean[columns]

    for col in df_clean.columns:
        temp = df_clean[col].astype(str)\
            .str.replace('$', '', regex=False)\
            .str.replace(',', '', regex=False)

        if temp.str.contains('%').any():
            df_clean[col] = pd.to_numeric(temp.str.replace('%', ''), errors='coerce') / 100
        else:
            df_clean[col] = pd.to_numeric(temp, errors='coerce')

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")

    X_scaled = scaler.transform(df_clean)
    X_imputed = imputer.fit_transform(X_scaled)
    X_pca = pca.transform(X_imputed)

    clusters = model.predict(X_pca)

    df["Cluster"] = clusters

    # =========================
    # COUNTRY SELECTOR
    # =========================
    st.markdown("## 📊 Per-Country Detail Card")

    selected_country = st.selectbox(
        "Select country to inspect:",
        country_col
    )

    row = df[df["Country"] == selected_country].iloc[0]

    # =========================
    # TOP CARD
    # =========================
    st.markdown(f"""
    <div class="card">
        <h3>🌐 {selected_country}</h3>
        <p>Cluster assigned: <b>{int(row['Cluster'])}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # METRICS GRID
    # =========================
    cols = st.columns(4)

    features = df_clean.columns

    for i, col_name in enumerate(features[:8]):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card">
                <p>{col_name}</p>
                <div class="metric">{row[col_name]}</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CLUSTER MEAN COMPARISON
    # =========================
    st.markdown("## 📉 Country vs Cluster Mean")

    cluster_id = row["Cluster"]
    cluster_data = df[df["Cluster"] == cluster_id]

    cluster_mean = cluster_data[features].mean()

    fig, ax = plt.subplots()

    ax.bar(features[:5], row[features[:5]], label="Country")
    ax.bar(features[:5], cluster_mean[:5], alpha=0.5, label="Cluster Mean")

    plt.xticks(rotation=45)
    plt.legend()

    st.pyplot(fig)

else:
    st.info("⬅️ Upload dataset to begin")
