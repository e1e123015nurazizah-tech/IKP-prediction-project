# ================================================
# Streamlit Dashboard Project Final
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import  GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import plotly.express as px
from sklearn.cluster import KMeans
import json
# ================================================

geojson_path = "Dataset/indonesia-province-simple.json"

# st.write("📂 Cek path GeoJSON:")
# st.write(os.path.abspath(geojson_path))
# st.write("Ada file?", os.path.exists(geojson_path))

# ===============================================
from model import (
    load_datasets,
    assess_dataset,
    apply_clean_kabkota_all,
    aggregate_pola_pangan,
    merge_all_datasets,
    post_merge_cleaning,
    merge_all_datasets, post_merge_cleaning,
    handle_missing_values,
    chart_ikp_vs_ipm,
    chart_akses_dasar_vs_ikp,
    chart_ikp_per_kelompok_pangan,
    chart_avg_kemiskinan_provinsi,
    chart_pdrb_tertinggi_provinsi,
    add_cluster_sosial_ekonomi,
    chart_cluster_sosial_ekonomi,
    chart_heatmap_correlation,      
    build_choropleth_ikp,
    chart_treemap_konsumsi_pangan,
    train_models,
    select_best_model,
    predict_manual,
    CLEANING_FUNCTIONS,
    FEATURES

)
# ===============================================

# ===============================================
# Streamlit Page Config & Custom CSS

st.set_page_config(page_title="Dashboard Ketahanan Pangan", layout="wide")

st.markdown("""
<style>
.card {
    min-height: 230px;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.card-blue { background-color: #f0f8ff; }
.card-gray { background-color: #f5f5f5; }
.card-green { background-color: #f0fff0; }
.card-orange { background-color: #fff5ee; }

.card:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

.desc-card {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# ===============================================
# Load Datasets Dari Folder Dataset
# ===============================================

# Unpack dataframes
df_IKP24, df_socio_econom, df_konsumsi_pangan, df_tingkat_kemiskinan, df_pola_pangan = load_datasets()


# ============================
# Sidebar Menu
# ============================

st.sidebar.title("📌 Navigasi Dashboard")

menu_utama = st.sidebar.radio(
    "Pilih Bagian:",
    [
        "🏠 Home",
        "📦 Data Wrangling",
        "⚙ Preprocessing",
        "📊 EDA",
        "📈 Visualisasi Lanjutan",
        "🤖 Pemodelan"
    ]
)


# =========================
# HOME
# =========================
if menu_utama == "🏠 Home":

    st.markdown(
        """
        <h1 style='text-align: center; color: #1f77b4;'>
        Prediksi Indeks Ketahanan Pangan (IKP) Kabupaten/Kota di Indonesia
        </h1>
        <h4 style='text-align: center;'>
        Melalui Evaluasi Algoritma LightGBM, CatBoost, dan Gradient Boosting
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("""
    <div class="desc-card">
    <h3>📌 Deskripsi Singkat</h3>

    Aplikasi ini dikembangkan untuk menganalisis dan memprediksi  
    <b>Indeks Ketahanan Pangan (IKP)</b> kabupaten/kota di Indonesia  
    berdasarkan <b>pertumbuhan ekonomi</b> dan <b>produktivitas pertanian</b>  
    dengan pendekatan <i>machine learning</i>.

    <ul>
    <li>🎯 Fokus pada ketahanan pangan nasional</li>
    <li>📊 Analisis berbasis data multi-tahun</li>
    <li>🤖 Pendekatan Machine Learning modern</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("## 📊 Ringkasan Proyek")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="card card-blue">
            <h4>📍 Cakupan Wilayah</h4>
            <h2>Kab/Kota Indonesia</h2>
            <p><b>38 Provinsi</b><br>514 Kabupaten/Kota</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card card-gray">
            <h4>📅 Periode Data</h4>
            <h2>Multi-Tahun</h2>
            <p>Data historis nasional</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card card-green">
            <h4>🤖 Algoritma</h4>
            <h2>3 Model</h2>
            <p>LightGBM<br>CatBoost<br>Gradient Boosting</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card card-orange">
            <h4>🎯 Target</h4>
            <h2>IKP</h2>
            <p>Indeks Ketahanan Pangan</p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")

    st.markdown("### 🚀 Fitur Utama Aplikasi")

    colA, colB = st.columns(2)

    with colA:
        st.success("✅ Data Wrangling Terstruktur")
        st.success("✅ Preprocessing & Normalisasi")
        st.success("✅ Visualisasi EDA Interaktif")

    with colB:
        st.info("📈 Perbandingan Performa Model")
        st.info("📊 Analisis Korelasi Variabel")
        st.info("🌍 Mendukung SDGs 2 & 8")

    st.markdown("---")

    st.markdown("## 📈 Rata-rata IKP per Provinsi")


    # Hitung rata-rata IKP per provinsi
    df_prov_ikp = (
        df_IKP24
        .groupby("Nama Provinsi", as_index=False)["IKP"]
        .mean()
        .sort_values("IKP", ascending=False)
    )

    fig = px.bar(
        df_prov_ikp,
        x="Nama Provinsi",
        y="IKP",
        color="IKP",
        color_continuous_scale="Viridis",
        title="Rata-rata Indeks Ketahanan Pangan (IKP) per Provinsi",
    )

    fig.update_layout(
        xaxis_title="Provinsi",
        yaxis_title="Rata-rata IKP",
        xaxis_tickangle=-45,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("---")

# Sub-menu otomatis
elif menu_utama == "📦 Data Wrangling":
    st.title("🧹 Data Wrangling")
    st.markdown("""
    Tahapan data wrangling dilakukan untuk memastikan kualitas dan konsistensi data
    sebelum masuk ke tahap preprocessing dan pemodelan.
    """)
    st.sidebar.markdown("### 📦 Data Wrangling Options")
    sub_wrang = st.sidebar.selectbox(
        "Pilih Proses:",
        ["📥 Data Gathering", "🔍 Data Assessing","🧼 Data Cleaning","🔗 Merge Data"]
    )

#     tab1, tab2, tab3, tab4 = st.tabs(
#     ["📥 Data Gathering", "🔍 Data Assessing", "🧼 Data Cleaning","🔗 Merge Data"]
# )
    if sub_wrang == "📥 Data Gathering":
        st.subheader("📥 Data Gathering")

        st.markdown("""
        Tahap ini bertujuan untuk mengumpulkan seluruh dataset yang digunakan
        dalam analisis prediksi **Indeks Ketahanan Pangan (IKP)**.
        """)

        st.markdown("### 📂 Dataset yang Digunakan")

        st.success("✅ df_IKP24 – Data Indeks Ketahanan Pangan")
        st.success("✅ df_socio_econom – Data Sosial & Ekonomi")
        st.success("✅ df_konsumsi_pangan – Data Konsumsi Pangan")
        st.success("✅ df_pola_pangan – Data Pola Pangan")
        st.success("✅ df_tingkat_kemiskinan – Data Kemiskinan")

        st.info("📌 Seluruh dataset dimuat dari folder **Dataset/** menggunakan Pandas.")
    elif sub_wrang == "🔍 Data Assessing":
        st.subheader("🔍 Data Assessing")

        dataset_dict = {
            "IKP": df_IKP24,
            "Sosial Ekonomi": df_socio_econom,
            "Konsumsi Pangan": df_konsumsi_pangan,
            "Kemiskinan": df_tingkat_kemiskinan,
            "Pola Pangan": df_pola_pangan
        }

        dataset_name = st.selectbox(
            "Pilih dataset",
            list(dataset_dict.keys())
        )

        df_selected = dataset_dict[dataset_name]

        # ===============================
        # PREVIEW DATA
        # ===============================
        st.markdown("### 👀 Preview Data")
        st.dataframe(df_selected.head())

        # ===============================
        # ASSESSING (CALL MODEL)
        # ===============================
        assess = assess_dataset(df_selected)

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", assess["shape"][0])
        col2.metric("Jumlah Kolom", assess["shape"][1])
        col3.metric("Missing Value", assess["missing"])

        col4, col5 = st.columns(2)
        col4.metric("Duplikasi Data", assess["duplicate"])
        col5.metric("Kolom Numerik", len(assess["numeric_cols"]))

        # ===============================
        # TIPE DATA
        # ===============================
        st.markdown("### 🧾 Tipe Data Kolom")
        st.dataframe(assess["dtypes"])

        # ===============================
        # STATISTIK DESKRIPTIF
        # ===============================
        st.markdown("### 📊 Statistik Deskriptif (Numerik)")
        st.dataframe(assess["describe"])

        # ===============================
        # OUTLIER VISUALIZATION (BOXPLOT)
        # ===============================
        st.markdown("### 📦 Deteksi Outlier (Boxplot)")

        assess = assess_dataset(df_selected)
        numeric_cols = assess["numeric_cols"]

        if len(numeric_cols) == 0:
            st.info("Tidak ada kolom numerik untuk dianalisis.")
        else:
            col_choice = st.selectbox(
                "Pilih kolom numerik",
                numeric_cols
            )

            fig = px.box(
                df_selected,
                y=col_choice,
                points="outliers",  
                title=f"Boxplot Outlier untuk Kolom: {col_choice}",
            )

            fig.update_layout(
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    elif sub_wrang == "🧼 Data Cleaning":
        st.subheader("🧼 Data Cleaning")

        st.markdown(
            """
            Tahap ini digunakan untuk membersihkan data **per dataset** secara **otomatis**.

            Proses yang dilakukan:
            - Penyamaan nama kolom & standarisasi teks
            - Pembersihan nama **kabupaten/kota**
            - **Agregasi otomatis (khusus Pola Pangan)**
            """
        )

        # ===============================
        # PILIH DATASET
        # ===============================
        dataset_dict = {
            "IKP": df_IKP24,
            "Sosial Ekonomi": df_socio_econom,
            "Konsumsi Pangan": df_konsumsi_pangan,
            "Kemiskinan": df_tingkat_kemiskinan,
            "Pola Pangan": df_pola_pangan
        }

        dataset_name = st.selectbox("📂 Pilih Dataset", dataset_dict.keys())
        df_before = dataset_dict[dataset_name]

        # ===============================
        # TAMPILAN SEBELUM CLEANING
        # ===============================
        st.markdown("### 📌 Data Sebelum Cleaning")
        st.dataframe(df_before.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df_before.shape[0])
        col2.metric("Jumlah Kolom", df_before.shape[1])
        col3.metric("Duplikasi", df_before.duplicated().sum())

        st.markdown("---")

        # ===============================
        # INIT SESSION STATE
        # ===============================
        if "dfs_clean" not in st.session_state:
            st.session_state.dfs_clean = {}

        # ===============================
        # CLEANING OTOMATIS
        # ===============================

        # 1️ Cleaning dasar (rename + standarisasi)
        df_step1 = CLEANING_FUNCTIONS[dataset_name](df_before)

        # 2️ Cleaning kabupaten/kota (AUTO APPLY)
        dfs_tmp = {dataset_name: df_step1}
        dfs_tmp = apply_clean_kabkota_all(dfs_tmp)
        df_step2 = dfs_tmp[dataset_name]

        # 3️ Agregasi otomatis (KHUSUS Pola Pangan)
        if dataset_name == "Pola Pangan":
            df_after = aggregate_pola_pangan(df_step2)
            st.info(
                "Dataset **Pola Pangan** otomatis diagregasi "
                "per provinsi dan kabupaten/kota menggunakan rata-rata skor PPH."
            )
        else:
            df_after = df_step2

        # ===============================
        # SIMPAN KE SESSION STATE
        # ===============================
        st.session_state.dfs_clean[dataset_name] = df_after

        # ===============================
        # TAMPILAN SESUDAH CLEANING
        # ===============================
        st.markdown("### ✅ Data Setelah Cleaning")
        st.dataframe(df_after.head())

        col4, col5, col6 = st.columns(3)
        col4.metric("Jumlah Baris", df_after.shape[0])
        col5.metric("Jumlah Kolom", df_after.shape[1])
        col6.metric(
            "Perubahan Baris",
            df_after.shape[0] - df_before.shape[0]
        )

        # ===============================
        # INFO TAMBAHAN
        # ===============================
        with st.expander("ℹ️ Informasi Tambahan"):
            st.write("Missing Value (sebelum):", df_before.isnull().sum().sum())
            st.write("Missing Value (sesudah):", df_after.isnull().sum().sum())
            st.write(
                "Kab/Kota unik (sebelum):",
                df_before["kabupaten/kota"].nunique()
                if "kabupaten/kota" in df_before.columns else "—"
            )
            st.write(
                "Kab/Kota unik (sesudah):",
                df_after["kabupaten/kota"].nunique()
                if "kabupaten/kota" in df_after.columns else "—"
            )

        st.success("Cleaning otomatis selesai dan data siap digunakan")

        st.info(
            """
            ✔ Cleaning kabupaten/kota diterapkan **otomatis**  
            ✔ Agregasi Pola Pangan dilakukan **di tahap cleaning**  
            ✔ Hasil disimpan dan digunakan langsung pada **Tab Merging**  
            """
        )

    elif sub_wrang == "🔗 Merge Data":
        st.subheader("🔗 Merge Data")

        st.markdown(
            """
            Tahap ini menggabungkan seluruh dataset yang telah melalui proses cleaning.
            Dataset **Pola Pangan telah diagregasi pada tahap Data Cleaning**.
            
            Setelah merge, dilakukan penanganan:
            - Penggabungan kolom bermakna sama
            - Penghapusan kolom redundan
            - Penghapusan kolom yang tidak digunakan
            """
        )

        # =====================================================
        # VALIDASI: CLEANING HARUS SUDAH DIJALANKAN
        # =====================================================
        required_datasets = [
            "IKP",
            "Sosial Ekonomi",
            "Konsumsi Pangan",
            "Kemiskinan",
            "Pola Pangan"
        ]

        if "dfs_clean" not in st.session_state:
            st.warning("⚠️ Jalankan **Data Cleaning** terlebih dahulu.")
            st.stop()

        missing = [d for d in required_datasets if d not in st.session_state.dfs_clean]
        if missing:
            st.error(f"❌ Dataset berikut belum dibersihkan: {', '.join(missing)}")
            st.stop()

        # =====================================================
        # INIT SESSION STATE
        # =====================================================
        for key in ["merge_1", "merge_2", "merge_3", "merge_final", "df_merge"]:
            if key not in st.session_state:
                st.session_state[key] = None

        # =====================================================
        # TOMBOL MERGE
        # =====================================================
        if st.button("🔗 Jalankan Merge Dataset", use_container_width=True):

            merge_1, merge_2, merge_3, merge_final = merge_all_datasets(
                st.session_state.dfs_clean["IKP"],
                st.session_state.dfs_clean["Sosial Ekonomi"],
                st.session_state.dfs_clean["Konsumsi Pangan"],
                st.session_state.dfs_clean["Kemiskinan"],
                st.session_state.dfs_clean["Pola Pangan"]  # SUDAH AGREGAT
            )

            # Simpan hasil merge
            st.session_state.merge_1 = merge_1
            st.session_state.merge_2 = merge_2
            st.session_state.merge_3 = merge_3
            st.session_state.merge_final = merge_final

            # =================================================
            # POST MERGE CLEANING
            # =================================================
            st.session_state.df_merge = post_merge_cleaning(merge_final)

            st.success("✅ Merge dan feature cleaning berhasil dijalankan")

        # =====================================================
        # DEBUG SHAPE SETIAP TAHAP MERGE
        # =====================================================
        if st.session_state.merge_1 is not None:
            st.markdown("### 🧪 Debug Shape Setiap Tahap Merge")
            st.write("Merge 1 (Sosial + Kemiskinan):", st.session_state.merge_1.shape)
            st.write("Merge 2 (+ Pola Pangan):", st.session_state.merge_2.shape)
            st.write("Merge 3 (+ Konsumsi Pangan):", st.session_state.merge_3.shape)
            st.write("Merge Final (+ IKP):", st.session_state.merge_final.shape)
            st.write(
                "Setelah Post-Merge Cleaning:",
                st.session_state.df_merge.shape
            )

        # =====================================================
        # TAMPILKAN DATASET AKHIR
        # =====================================================
        if st.session_state.df_merge is not None:
            df_merge = st.session_state.df_merge

            st.markdown("### 📊 Informasi Dataset Setelah Merge & Cleaning")

            col1, col2, col3 = st.columns(3)
            col1.metric("Jumlah Baris", df_merge.shape[0])
            col2.metric("Jumlah Kolom", df_merge.shape[1])
            col3.metric("Missing Value", df_merge.isnull().sum().sum())

            with st.expander("🔍 Struktur Kolom Dataset"):
                st.dataframe(
                    pd.DataFrame({
                        "Nama Kolom": df_merge.columns,
                        "Tipe Data": df_merge.dtypes.astype(str)
                    })
                )

            st.markdown("### 👀 Preview Dataset Akhir")
            st.dataframe(df_merge.head())

            st.info(
                """
                ✔ Dataset telah digabung menggunakan **outer join**  
                ✔ Kolom dengan makna sama telah disatukan  
                ✔ Kolom redundan dan tidak relevan telah dihapus  
                ✔ Dataset siap masuk ke tahap preprocessing & modeling
                """
            )

elif menu_utama == "⚙ Preprocessing":

    st.title("⚙ Tahap Preprocessing Data")

    # =================================================
    # CEK PRASYARAT
    # =================================================
    if "df_merge" not in st.session_state or st.session_state.df_merge is None:
        st.warning("⚠️ Jalankan **Data Merging** terlebih dahulu.")
        st.stop()

    df_base = st.session_state.df_merge

    # =================================================
    # SIDEBAR OPTION
    # =================================================
    st.sidebar.markdown("### ⚙ Preprocessing Options")
    sub_preproc = st.sidebar.selectbox(
        "Pilih Proses:",
        ["🧩 Missing Value", "📦 Outlier"]
    )

    # =================================================
    # 🧩 MISSING VALUE
    # =================================================
    if sub_preproc == "🧩 Missing Value":

        st.subheader("🧩 Penanganan Missing Value")

        st.markdown("""
        **Metode:**
        - Numerik → Median per provinsi → Median global  
        - Kategorik → Modus
        """)

        # ===============================
        # RINGKASAN DATA AWAL
        # ===============================
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df_base.shape[0])
        col2.metric("Jumlah Kolom", df_base.shape[1])
        col3.metric("Total Missing", int(df_base.isna().sum().sum()))

        # ===============================
        # VISUALISASI MISSING VALUE
        # ===============================
        st.markdown("### 📉 Distribusi Missing Value")

        missing_df = (
            df_base.isna()
            .sum()
            .reset_index()
            .rename(columns={"index": "Kolom", 0: "Jumlah Missing"})
        )

        missing_df = missing_df[missing_df["Jumlah Missing"] > 0]

        if not missing_df.empty:
            import plotly.express as px
            fig = px.bar(
                missing_df.sort_values("Jumlah Missing", ascending=False),
                x="Kolom",
                y="Jumlah Missing",
                text="Jumlah Missing"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 Tidak ada missing value")

        # ===============================
        # AUTO IMPUTASI
        # ===============================
        df_after, before_mv, after_mv = handle_missing_values(df_base)
        st.session_state.df_missing_handled = df_after

        st.success("✅ Missing value otomatis ditangani")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### ❌ Sebelum")
            st.dataframe(before_mv[before_mv > 0])
        with colB:
            st.markdown("#### ✅ Sesudah")
            st.dataframe(after_mv[after_mv > 0])

        st.markdown("### 👀 Preview Data")
        st.dataframe(df_after.head())

    # =================================================
    # 📦 OUTLIER 
    # =================================================
    elif sub_preproc == "📦 Outlier":

        st.subheader("📦 Penanganan Outlier (IQR Capping)")
        st.markdown(
            """
            Visualisasi **sebelum dan sesudah** penanganan outlier  
            menggunakan metode **Interquartile Range (IQR)**.
            """
        )

        # ===============================
        # VALIDASI DATA
        # ===============================
        if "df_missing_handled" not in st.session_state:
            st.warning("⚠️ Tangani missing value terlebih dahulu.")
            st.stop()

        df_out = st.session_state.df_missing_handled.copy()

        # ===============================
        # KOLOM NUMERIK (IKP DIKECUALIKAN)
        # ===============================
        num_cols = df_out.select_dtypes(include=["float64", "int64"]).columns.tolist()
        num_cols = [c for c in num_cols if c.lower() != "ikp"]

        if len(num_cols) == 0:
            st.warning("⚠️ Tidak ada kolom numerik yang bisa diproses.")
            st.stop()

        # ===============================
        # PILIH KOLOM
        # ===============================
        selected_col = st.selectbox(
            "📌 Pilih Kolom Numerik",
            num_cols
        )

        # ===============================
        # SIMPAN DATA SEBELUM
        # ===============================
        df_before = df_out.copy()

        # ===============================
        # HITUNG IQR
        # ===============================
        Q1 = df_out[selected_col].quantile(0.25)
        Q3 = df_out[selected_col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # ===============================
        # HITUNG OUTLIER SEBELUM
        # ===============================
        before_outliers = (
            (df_before[selected_col] < lower) |
            (df_before[selected_col] > upper)
        ).sum()

        # ===============================
        # IQR CAPPING
        # ===============================
        df_after = df_out.copy()
        df_after[selected_col] = df_after[selected_col].clip(lower, upper)

        # ===============================
        # HITUNG OUTLIER SESUDAH
        # ===============================
        after_outliers = (
            (df_after[selected_col] < lower) |
            (df_after[selected_col] > upper)
        ).sum()

        # SIMPAN KE SESSION (DATA SIAP EDA & MODEL)
        st.session_state.df_preprocessed = df_after


        col1, col2 = st.columns(2)

        with col1:
            fig_before = px.box(
                df_before,
                y=selected_col,
                points="outliers",
                title=f"📌 Sebelum Outlier Handling: {selected_col}"
            )
            st.plotly_chart(fig_before, use_container_width=True)

        with col2:
            fig_after = px.box(
                df_after,
                y=selected_col,
                points="outliers",
                title=f"✅ Sesudah Outlier Handling: {selected_col}"
            )
            st.plotly_chart(fig_after, use_container_width=True)

        # ===============================
        # METRIK
        # ===============================
        m1, m2, m3 = st.columns(3)
        m1.metric("Outlier Sebelum", int(before_outliers))
        m2.metric("Outlier Sesudah", int(after_outliers))
        m3.metric("Outlier Dihilangkan", int(before_outliers - after_outliers))

        st.success("✅ Penanganan outlier berhasil (IKP tidak diubah)")

        # ===============================
        # PREVIEW DATA
        # ===============================
        st.markdown("### 👀 Preview Data Setelah Outlier Handling")
        st.dataframe(df_after.head())


# ==============================
# EDA
# ==============================
elif menu_utama == "📊 EDA":

    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    Tahap **EDA** bertujuan untuk memahami pola, hubungan,
    dan karakteristik utama data **Indeks Ketahanan Pangan (IKP)**.
    """)

    # ===============================
    # GATE WAJIB PREPROCESSING
    # ===============================
    if "df_preprocessed" not in st.session_state:
        st.warning("⚠ Jalankan Cleaning → Merging → Preprocessing terlebih dahulu.")
        st.stop()

    df = st.session_state.df_preprocessed

    # =================================================
    # SIDEBAR OPTION
    # =================================================
    st.sidebar.markdown("### ⚙ Preprocessing Options")
    sub_eda = st.sidebar.selectbox(
        "Pilih Proses:",
        ["IKP vs IPM", "Akses Dasar", "Konsumsi Pangan",
         "Kemiskinan & PDRB", "Pola Sosial–Ekonomi"]
    )

    # ===============================
    # 1️⃣ IKP vs IPM
    # ===============================
    if sub_eda == "IKP vs IPM":
        st.subheader("📈 IKP vs Indeks Pembangunan Manusia (IPM)")
        st.altair_chart(
            chart_ikp_vs_ipm(df),
            use_container_width=True
        )
        st.info(
        """
        📌 **Insight Hubungan IKP dan IPM**
        - Tren titik membentuk garis naik → **hubungan positif yang kuat**
        - IPM meningkat → **IKP cenderung ikut meningkat**
        - Wilayah dengan IPM tinggi umumnya memiliki **ketahanan pangan lebih baik**
        - Warna lebih terang menunjukkan **pengeluaran per kapita tinggi** yang berkorelasi dengan IKP tinggi
        - Ukuran bubble lebih besar menandakan **konsumsi pangan lebih tinggi**
        - Terdapat beberapa outlier → menandakan **faktor lain di luar IPM masih berpengaruh**
        """
        )

    # ===============================
    # 2️⃣ AKSES DASAR
    # ===============================
    elif sub_eda == "Akses Dasar":
        st.subheader("🚰 Akses Air & Sanitasi vs IKP")
        st.altair_chart(
            chart_akses_dasar_vs_ikp(df),
            use_container_width=True
        )
        st.info(
            """
            📌 **Insight Hubungan Akses Dasar Rumah Tangga dengan IKP**
            - Pola sebaran titik membentuk tren naik → **korelasi positif kuat**
            - Semakin tinggi **akses air minum layak**, IKP cenderung **semakin tinggi**
            - Semakin tinggi **akses sanitasi layak**, IKP juga **meningkat secara konsisten**
            - Wilayah dengan akses air & sanitasi rendah didominasi **IKP rendah**
            - Akses sanitasi menunjukkan pola yang **lebih stabil** dibanding air layak
            - Terdapat beberapa outlier → menandakan **ketahanan pangan juga dipengaruhi faktor lain**
            """
        )

    # ===============================
    # 3️⃣ KONSUMSI PANGAN
    # ===============================
    elif sub_eda == "Konsumsi Pangan":
        st.subheader("🍚 IKP berdasarkan Kelompok Bahan Pangan")
        st.altair_chart(
            chart_ikp_per_kelompok_pangan(df),
            use_container_width=True
        )
        st.info(
                """
                📌 **Insight Konsumsi Pangan terhadap IKP**
                - Seluruh kelompok bahan pangan menunjukkan **median IKP yang tinggi (±75–85)**
                - Kelompok **Padi-padian, Sayur & Buah, serta Umbi-umbian** memiliki distribusi IKP paling stabil
                - Kelompok **Buah/Biji Berminyak dan Gula** menunjukkan **variasi IKP lebih lebar**
                - Outlier IKP rendah muncul pada hampir semua kelompok → indikasi **ketimpangan antar wilayah**
                - Konsumsi pangan yang beragam dan seimbang cenderung berkorelasi dengan **IKP yang lebih baik**
                - Tidak ada satu kelompok dominan → **ketahanan pangan bersifat multisektor**
                """
            )

    # ===============================
    # 4️⃣ KEMISKINAN & PDRB
    # ===============================
    elif sub_eda == "Kemiskinan & PDRB":
        st.subheader("💸 Kemiskinan & Kekuatan Ekonomi Provinsi")

        st.markdown("### 📉 Rata-Rata Persentase Penduduk Miskin")
        st.altair_chart(
            chart_avg_kemiskinan_provinsi(df),
            use_container_width=True
        )
        st.info(
            """
            📌 **Insight Penduduk Miskin Antar Provinsi**
            - Provinsi dengan persentase penduduk miskin tertinggi didominasi wilayah **Indonesia Timur**
            (Papua, Papua Barat, NTT)
            - Provinsi di wilayah **Jawa dan sekitarnya** cenderung memiliki persentase penduduk miskin lebih rendah
            - Terlihat **kesenjangan kemiskinan antar provinsi** yang cukup signifikan
            - Wilayah dengan akses ekonomi dan infrastruktur terbatas cenderung memiliki kemiskinan lebih tinggi
            - Pola ini menunjukkan perlunya **intervensi kebijakan yang lebih terfokus secara regional**
            """
        )

        st.divider()

        st.markdown("### 💰 PDRB Tertinggi di Kabupaten/Kota per Provinsi")
        st.altair_chart(
            chart_pdrb_tertinggi_provinsi(df),
            use_container_width=True
        )
        st.info(
            """
            📌 **Insight PDRB (Reg_GDP) Antar Provinsi**
            - PDRB tertinggi terkonsentrasi pada provinsi dengan **pusat kegiatan ekonomi utama**
            (Papua Barat, DKI Jakarta, Jawa Barat, Jawa Timur)
            - Provinsi di Indonesia Timur umumnya memiliki **PDRB relatif lebih rendah**
            - Terjadi **ketimpangan kapasitas ekonomi antar provinsi**
            - Tingginya PDRB mencerminkan kekuatan sektor industri, jasa, dan perdagangan
            - PDRB tinggi belum tentu mencerminkan **pemerataan kesejahteraan masyarakat**
            """
        )



    # ===============================
    # 5️⃣ MULTIVARIAT
    # ===============================
    elif sub_eda == "Pola Sosial–Ekonomi":
        st.subheader("📊 Pola IKP terhadap Faktor Sosial–Ekonomi")
        st.markdown(
            """
            Visualisasi clustering wilayah berdasarkan:
            **IPM, pengeluaran per kapita, dan rata-rata lama sekolah**
            untuk melihat pola ketahanan pangan.
            """
        )

        # Jalankan clustering (sekali, aman)
        df_clustered = add_cluster_sosial_ekonomi(df)

        st.altair_chart(
            chart_cluster_sosial_ekonomi(df_clustered),
            use_container_width=True
        )

        st.info(
        """
        📌 **Insight Pola IKP berdasarkan Cluster Sosial-Ekonomi Wilayah**
        - Terbentuk **beberapa cluster wilayah yang jelas** berdasarkan kombinasi IPM dan IKP
        - **Cluster IPM & IKP tinggi** didominasi wilayah dengan:
        • Pengeluaran per kapita tinggi  
        • Rata-rata lama sekolah lebih tinggi  
        • Ketahanan pangan sangat baik dan stabil
        - **Cluster menengah** menunjukkan IKP cukup baik meskipun IPM belum optimal,
        menandakan adanya **peran faktor lokal (produktivitas pangan, akses distribusi)**
        - **Cluster IPM & IKP rendah** umumnya memiliki:
        • Pengeluaran per kapita rendah  
        • Rata lama sekolah rendah  
        • Kerentanan ketahanan pangan lebih tinggi
        - Ukuran bubble yang lebih besar pada cluster atas menunjukkan
        **pengeluaran per kapita berkontribusi kuat terhadap peningkatan IKP**
        - Pola ini menegaskan bahwa **IKP dipengaruhi kombinasi faktor sosial-ekonomi,
        bukan oleh satu variabel tunggal**
        """
    )


# ==============================
# VISUALISASI LANJUTAN
# =============================

elif menu_utama == "📈 Visualisasi Lanjutan":

    st.title("📈 Visualisasi Lanjutan")
    st.markdown(
        """
        Tahap **Visualisasi Lanjutan** digunakan untuk menampilkan pola kompleks
        dan hubungan multivariat pada data **Indeks Ketahanan Pangan (IKP)**.
        """
    )

    # =================================================
    # GATE (SAMA DENGAN EDA)
    # =================================================
    if "df_preprocessed" not in st.session_state:
        st.warning(
            "⚠️ Silakan selesaikan tahap **Cleaning → Merging → Preprocessing** terlebih dahulu."
        )
        st.stop()

    df = st.session_state.df_preprocessed

    # =================================================
    # SIDEBAR PILIH VISUALISASI
    # =================================================
    st.sidebar.markdown("### 📈 Pilih Visualisasi")
    vis_type = st.sidebar.selectbox(
        "Tipe Visualisasi:",
        [
            "📦 Treemap Konsumsi Pangan",
            "🗺️ Peta Sebaran IKP",
            "🔥 Heatmap Korelasi",
        ]
    )

    st.divider()

    # =================================================
    # 📦 TREEMAP
    # =================================================
    if vis_type == "📦 Treemap Konsumsi Pangan":

        st.subheader("📦 Treemap Konsumsi Pangan")
        st.markdown(
            """
            Visualisasi **hierarkis** konsumsi pangan berdasarkan:
            - Provinsi
            - Kelompok Bahan Pangan
            
            Ukuran kotak menunjukkan **total konsumsi pangan**.
            """
        )

        fig = chart_treemap_konsumsi_pangan(df)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
            📌 **Insight**
            - Provinsi dengan konsumsi besar terlihat dominan
            - Perbedaan kontribusi antar kelompok bahan pangan jelas
            """
        )

    # =================================================
    # 🗺️ CHOROPLETH MAP
    # =================================================
    elif vis_type == "🗺️ Peta Sebaran IKP":

        st.subheader("🗺️ Peta Sebaran Rata-rata IKP Provinsi")

        st.markdown(
            """
            Visualisasi ini menampilkan **rata-rata Indeks Ketahanan Pangan (IKP)** 
            pada **38 provinsi di Indonesia** menggunakan **choropleth map**.
            """
        )

        # ======================================================
        # VALIDASI: pastikan preprocessing SUDAH SELESAI
        # ======================================================
        if "df_preprocessed" not in st.session_state:
            st.warning("⚠️ Silakan selesaikan tahap preprocessing terlebih dahulu.")
            st.stop()

        # ======================================================
        # AMBIL DATA FINAL (INI YANG BENAR)
        # ======================================================
        df = st.session_state.df_preprocessed.copy()

        # ======================================================
        # SLIDER RENTANG IKP (HANYA UNTUK INTERAKSI)
        # ======================================================
        min_ikp = float(df["ikp"].min())
        max_ikp = float(df["ikp"].max())

        min_val, max_val = st.slider(
            "🔎 Filter Rentang IKP",
            min_value=min_ikp,
            max_value=max_ikp,
            value=(min_ikp, max_ikp)
        )

        df_filtered = df[
            (df["ikp"] >= min_val) &
            (df["ikp"] <= max_val)
        ]

        # ======================================================
        # METRIK RINGKAS
        # ======================================================
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "📊 Rata-rata IKP Nasional",
            f"{df['ikp'].mean():.2f}"
        )

        c2.metric(
            "📈 IKP Tertinggi (Rata-rata Provinsi)",
            f"{df.groupby('provinsi')['ikp'].mean().max():.2f}"
        )

        c3.metric(
            "📉 IKP Terendah (Rata-rata Provinsi)",
            f"{df.groupby('provinsi')['ikp'].mean().min():.2f}"
        )

        st.markdown("---")

        # ======================================================
        # PETA CHOROPLETH
        # ======================================================
        geojson_path = "Dataset/indonesia-province-simple.json"

        chart = build_choropleth_ikp(
            df_preprocessed=df,      # DATA FINAL HASIL PREPROCESSING
            geojson_path=geojson_path
        )

        st.altair_chart(chart, use_container_width=True)

        # ======================================================
        # INFO
        # ======================================================
        st.info(
            """
            📌 **Catatan**
            - Data peta diambil dari **hasil preprocessing akhir (IQR Capping)**.
            - Slider hanya digunakan untuk interaksi, **bukan menghitung ulang IKP**.
            - IKP **tidak diubah** selama preprocessing (sesuai desain).
            """
        )

    # =================================================
    # 🔥 HEATMAP KORELASI
    # =================================================
    elif vis_type == "🔥 Heatmap Korelasi":

        st.subheader("🔥 Heatmap Korelasi Antar Variabel Numerik")
        st.markdown(
            """
            Heatmap ini menunjukkan **kekuatan dan arah hubungan**
            antar variabel numerik pada dataset IKP.
            """
        )

        fig = chart_heatmap_correlation(df)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
            📌 **Cara Membaca Heatmap**
            - Merah → korelasi positif kuat
            - Biru → korelasi negatif kuat
            - Mendekati 0 → hubungan lemah
            """
        )

# ==============================
# PEMODELAN
# ============================
elif menu_utama == "🤖 Pemodelan":

    st.title("🤖 Pemodelan Prediksi IKP")

    # ===============================
    # CEK DATA
    # ===============================
    if "df_preprocessed" not in st.session_state:
        st.warning("⚠️ Jalankan **Preprocessing** terlebih dahulu.")
        st.stop()

    df = st.session_state.df_preprocessed

    # ===============================
    # TRAIN MODEL
    # ===============================
    with st.spinner("⏳ Melatih model..."):
        results, scaler = train_models(df)

    best_model_name, best_info = select_best_model(results)
    best_model = best_info["model"]

    st.success(f"🏆 Model Terbaik: **{best_model_name}**")

    # ===============================
    # METRIK
    # ===============================
    c1, c2, c3,c4,c5,c6 = st.columns(6)
    c1.metric("RMSE Test", f"{best_info['RMSE_Test']:.3f}")
    c2.metric("R² Test", f"{best_info['R2_Test']:.3f}")
    c3.metric("MAE Test", f"{best_info['MAE_Test']:.3f}")
    c4.metric("RMSE Train", f"{best_info['RMSE_Train']:.3f}")
    c5.metric("R² Train", f"{best_info['R2_Train']:.3f}")
    c6.metric("MAE Train", f"{best_info['MAE_Train']:.3f}")

    st.divider()

    # ===============================
    # VISUALISASI R2 TRAIN vs TEST
    # ===============================
    r2_df = pd.DataFrame({
        "Model": results.keys(),
        "R2_Train": [v["R2_Train"] for v in results.values()],
        "R2_Test": [v["R2_Test"] for v in results.values()]
    })

    # melt data
    r2_melt = r2_df.melt(
        id_vars="Model",
        value_vars=["R2_Train", "R2_Test"],
        var_name="Dataset",
        value_name="R2"
    )

    # improve text & scale
    chart_r2 = alt.Chart(r2_melt).mark_line(point=alt.OverlayMarkDef(size=80)).encode(
        x=alt.X(
            "Model:N", 
            sort=list(results.keys()),
            axis=alt.Axis(labelAngle=0, labelFontSize=12, title="Model")
        ),
        y=alt.Y(
            "R2:Q",
            scale=alt.Scale(domain=[0.0, 1.05]),
            axis=alt.Axis(tickCount=6, labelFontSize=12, title="R² Score")
        ),
        color=alt.Color(
            "Dataset:N",
            scale=alt.Scale(
                domain=["R2_Train", "R2_Test"],
                range=["#0072B2", "#D55E00"]
            ),
            legend=alt.Legend(title="Jenis Dataset")
        ),
        tooltip=[
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("Dataset:N", title="Dataset"),
            alt.Tooltip("R2:Q", title="R²", format=".3f")
        ]
    ).properties(
        title={
            "text": "📊 Perbandingan Akurasi (R²) Train vs Test",
            "subtitle": "Model performance comparison (higher is better)",
            "fontSize": 16
        },
        width=750,
        height=450
    ).configure_title(
        fontSize=18,
        anchor="start"
    )
    st.altair_chart(chart_r2, use_container_width=True)


    # ===============================
    # AKTUAL vs PREDIKSI
    # ===============================
    df_pred = pd.DataFrame({
        "Actual": best_info["y_test"],
        "Predicted": best_info["y_test_pred"]
    })

    scatter = alt.Chart(df_pred).mark_circle(size=70, opacity=0.6).encode(
        x="Actual:Q",
        y="Predicted:Q",
        tooltip=["Actual", "Predicted"]
    )

    line = alt.Chart(df_pred).mark_line(color="red").encode(
        x="Actual:Q",
        y="Actual:Q"
    )

    st.altair_chart(
        (scatter + line).properties(
            title=f"Aktual vs Prediksi IKP ({best_model_name})",
            width=600,
            height=400
        ),
        use_container_width=True
    )

    st.divider()

    # ===============================
    # PREDIKSI MANUAL
    # ===============================
    st.subheader("🔮 Prediksi IKP Berdasarkan Input")

    input_data = {}
    cols = st.columns(2)

    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            input_data[feat] = st.number_input(
                feat.replace("_", " ").title(),
                value=float(df[feat].mean())
            )

    if st.button("🔮 Prediksi IKP"):
        pred = predict_manual(input_data, best_model, scaler)

        st.success(f"📊 Prediksi IKP: **{pred:.2f}**")

        if pred >= 80:
            st.info("🟢 Ketahanan Pangan Tinggi")
        elif pred >= 60:
            st.warning("🟡 Ketahanan Pangan Sedang")
        else:
            st.error("🔴 Ketahanan Pangan Rendah")
