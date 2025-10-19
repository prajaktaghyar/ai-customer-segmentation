import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="NeuraSegment", page_icon="🧠", layout="wide")

# ------------------------------------------------
# CUSTOM STYLING (NEON + GLASS)
# ------------------------------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top left, #0a0f1f, #000000);
    color: #e0e6ed;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4 {
    color: #00ffe1;
    font-weight: 700;
    text-shadow: 0 0 10px #00ffe1;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00ffe1, #0072ff);
    transform: scale(1.05);
}
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown("<h1 align='center'>🧠 NeuraSegment</h1>", unsafe_allow_html=True)
st.markdown("<h4 align='center' style='color:#38bdf8;'>AI-Powered Customer Intelligence & Smart Segmentation</h4>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------
# PROJECT INFO SECTION
# ------------------------------------------------
with st.expander("📘 Project Overview — Click to Expand"):
    st.markdown("""
    ### 💡 Project Title: **NeuraSegment — AI for Customer Intelligence**
    **Objective:**  
    To automatically segment customers into meaningful groups based on behavior, spending, or preferences using **unsupervised machine learning (K-Means)**.

    **Problem Statement:**  
    Businesses often struggle to identify and target high-value customers effectively.  
    NeuraSegment solves this by analyzing customer data and generating **AI-driven insights** for smarter marketing and decision-making.

    **Key Features:**  
    - 🧩 Real-time clustering with **K-Means & PCA visualization**  
    - 🎯 Automatic detection of **target customer segments**  
    - 💬 AI-generated **natural language insights** per cluster  
    - 📈 Works with **any dataset** (CSV/Excel)  
    - ⚙️ Downloadable results for business analysis

    **Impact:**  
    Helps organizations increase ROI by personalizing marketing, optimizing campaigns, and identifying top-performing customer groups instantly.

    ---
    """)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Customer Dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset uploaded successfully!")
        st.subheader("👀 Data Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric features found for segmentation.")
            st.stop()

        st.info(f"Detected numeric features for clustering: {', '.join(numeric_cols)}")

        # Choose K
        st.subheader("🔢 Select Number of Segments")
        k = st.slider("Number of clusters (K)", 2, 10, 4)

        if st.button("🚀 Run AI Segmentation"):
            with st.spinner("Analyzing data..."):
                X = df[numeric_cols].dropna()
                kmeans = KMeans(n_clusters=k, random_state=42)
                df["Cluster"] = kmeans.fit_predict(X)

                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X)
                df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

                st.success("✅ Segmentation complete!")

                # -------------------- Visualization --------------------
                st.subheader("🌐 Customer Segmentation Map (PCA 2D)")
                st.scatter_chart(df, x="PCA1", y="PCA2", color="Cluster")

                # -------------------- Cluster Summary --------------------
                st.subheader("📊 Cluster Summary (Average Values)")
                cluster_means = df.groupby("Cluster")[numeric_cols].mean()
                st.dataframe(cluster_means.style.highlight_max(axis=0, color="#00ffe1"))

                # -------------------- AI Insights --------------------
                st.subheader("🤖 AI Insights per Segment")
                for c in cluster_means.index:
                    desc = []
                    for f in numeric_cols:
                        mean_val = cluster_means.loc[c, f]
                        if mean_val > cluster_means[f].mean():
                            desc.append(f"higher {f}")
                        else:
                            desc.append(f"lower {f}")
                    insight = f"🧩 **Cluster {c}** customers show {', '.join(desc[:3])} tendencies."
                    st.markdown(insight)

                # -------------------- Targeted Segment --------------------
                st.subheader("🎯 Target Segment Analysis")
                target_features = [c for c in numeric_cols if any(k in c.lower() for k in ["spend", "income", "amount", "purchase", "score"])]
                if target_features:
                    target_col = target_features[0]
                    target_cluster = cluster_means[target_col].idxmax()
                    st.success(f"🏆 Target Segment: Cluster {target_cluster} — Highest {target_col}")
                    st.markdown(f"💡 Customers in this cluster have the greatest potential for conversion or revenue growth.")
                else:
                    avg_values = cluster_means.mean(axis=1)
                    target_cluster = avg_values.idxmax()
                    st.success(f"🏆 Target Segment: Cluster {target_cluster} — Highest overall metrics")

                st.write("👥 Example Customers from Target Segment:")
                st.dataframe(df[df["Cluster"] == target_cluster].head(8))

                # -------------------- Download --------------------
                st.subheader("💾 Download Results")
                output = BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)
                st.download_button(
                    label="📥 Download Segmented Dataset",
                    data=output,
                    file_name="neura_segments.csv",
                    mime="text/csv"
                )

                # -------------------- Footer --------------------
                st.markdown("---")
                st.markdown("<h5 align='center' style='color:#00ffe1;'>NeuraSegment | Built for Hackoverflow 9.0 — AI for Smarter Decisions</h5>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("👆 Upload a dataset to start your AI-driven customer segmentation.")


