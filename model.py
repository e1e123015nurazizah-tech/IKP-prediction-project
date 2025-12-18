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
import geopandas as gpd
# ===============================================

DATA_PATH = "Dataset"

@st.cache_data
def load_datasets():
    path_ikp24 = os.path.join(DATA_PATH, "2024_IKP_KabupatenKota.csv")
    path_socio = os.path.join(DATA_PATH, "2021socio_economic_indonesia.csv")
    path_konsum = os.path.join(DATA_PATH, "final_konsumsi_normalized_2024.csv")
    path_kemiskin = os.path.join(DATA_PATH, "Klasifikasi Tingkat Kemiskinan di Indonesia.csv")
    path_pola = os.path.join(DATA_PATH, "Skor Pola Pangan Harapan Konsumsi KabupatenKota update Tahun 2024.csv")

    df_IKP24 = pd.read_csv(path_ikp24)
    df_socio_econom = pd.read_csv(path_socio)
    df_konsumsi_pangan = pd.read_csv(path_konsum)
    df_tingkat_kemiskinan = pd.read_csv(path_kemiskin)
    df_pola_pangan = pd.read_csv(path_pola)

    return (
        df_IKP24,
        df_socio_econom,
        df_konsumsi_pangan,
        df_tingkat_kemiskinan,
        df_pola_pangan
    )

# ===============================================
# DATA ASSESSING FUNCTION
# ===============================================

def assess_dataset(df):
    """
    Melakukan data assessing:
    - info dataset
    - statistik deskriptif
    - missing value
    - duplikasi
    - kolom numerik
    - ringkasan outlier (IQR)
    """

    # Basic info
    n_rows, n_cols = df.shape
    missing_total = df.isnull().sum().sum()
    duplicate_total = df.duplicated().sum()

    # Tipe data kolom
    dtypes = df.dtypes.rename("Tipe Data")

    # Statistik deskriptif (numerik)
    desc = df.describe()

    # Kolom numerik
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Ringkasan outlier (IQR)
    outlier_summary = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        pct_outlier = (n_outlier / len(df)) * 100 if len(df) > 0 else 0

        outlier_summary.append({
            "Kolom": col,
            "Jumlah Outlier": n_outlier,
            "Persentase (%)": round(pct_outlier, 2)
        })

    outlier_df = pd.DataFrame(outlier_summary)

    return {
        "shape": (n_rows, n_cols),
        "missing": missing_total,
        "duplicate": duplicate_total,
        "dtypes": dtypes,
        "describe": desc,
        "numeric_cols": numeric_cols,
        "outlier": outlier_df
    }

# =================================================
# Cleaning
# =================================================
def clean_df_IKP24_streamlit(df_IKP24: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning df_IKP24
    Versi Streamlit
    (LOGIKA & URUTAN DISAMAKAN DENGAN COLAB)
    """

    # =========================================
    # COPY DF (WAJIB di Streamlit)
    # =========================================
    df_IKP24 = df_IKP24.copy()

    # --- RENAME KOLOM SPESIFIK (MANUAL) ---
    df_IKP24.columns = (
        df_IKP24.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    rename_map = {
        'nama_provinsi': 'provinsi',
        'nama_kabupaten': 'kabupaten/kota'
    }

    df_IKP24.rename(columns=rename_map, inplace=True)

    # --- STANDARISASI ISI DATA KATEGORI ---
    if 'provinsi' in df_IKP24.columns:
        df_IKP24['provinsi'] = (
            df_IKP24['provinsi']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    if 'kabupaten/kota' in df_IKP24.columns:
        df_IKP24['kabupaten/kota'] = (
            df_IKP24['kabupaten/kota']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    return df_IKP24

def clean_df_socio_econom_streamlit(df_socio_econom: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning df_socio_econom
    Versi Streamlit
    (LOGIKA & URUTAN DISAMAKAN DENGAN COLAB)
    """

    # =========================================
    # COPY DF (WAJIB di Streamlit)
    # =========================================
    df_socio_econom = df_socio_econom.copy()

    # --- RENAME KOLOM SPESIFIK (MANUAL) ---
    df_socio_econom.columns = (
        df_socio_econom.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    rename_map = {
        'province': 'provinsi',
        'cities_reg': 'kabupaten/kota'
    }

    df_socio_econom.rename(columns=rename_map, inplace=True)

    # --- STANDARISASI ISI DATA KATEGORI ---
    if 'provinsi' in df_socio_econom.columns:
        df_socio_econom['provinsi'] = (
            df_socio_econom['provinsi']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    if 'kabupaten/kota' in df_socio_econom.columns:
        df_socio_econom['kabupaten/kota'] = (
            df_socio_econom['kabupaten/kota']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    return df_socio_econom

def clean_df_konsumsi_pangan_streamlit(df_konsumsi_pangan: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning df_konsumsi_pangan
    Versi Streamlit
    (LOGIKA & URUTAN DISAMAKAN DENGAN COLAB)
    """

    # =========================================
    # COPY DF (WAJIB di Streamlit)
    # =========================================
    df_konsumsi_pangan = df_konsumsi_pangan.copy()

    # --- RENAME KOLOM SPESIFIK (MANUAL) ---
    df_konsumsi_pangan.columns = (
        df_konsumsi_pangan.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    rename_map = {
        'kabupaten_kota': 'kabupaten/kota'
    }

    df_konsumsi_pangan.rename(columns=rename_map, inplace=True)

    # --- STANDARISASI ISI DATA KATEGORI ---
    if 'provinsi' in df_konsumsi_pangan.columns:
        df_konsumsi_pangan['provinsi'] = (
            df_konsumsi_pangan['provinsi']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    if 'kabupaten/kota' in df_konsumsi_pangan.columns:
        df_konsumsi_pangan['kabupaten/kota'] = (
            df_konsumsi_pangan['kabupaten/kota']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
            .str.replace(r'\b(kabupaten|kota|kab|kot)\b', '', regex=True)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

    return df_konsumsi_pangan

def clean_df_kemiskinan_streamlit(df_tingkat_kemiskinan: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning df_tingkat_kemiskinan
    Versi Streamlit
    (LOGIKA & URUTAN DISAMAKAN DENGAN COLAB)
    """

    # =========================================
    # COPY DF (WAJIB di Streamlit)
    # =========================================
    df_tingkat_kemiskinan = df_tingkat_kemiskinan.copy()

    # --- RENAME KOLOM SPESIFIK (MANUAL) ---
    df_tingkat_kemiskinan.columns = (
        df_tingkat_kemiskinan.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    rename_map = {
        'kab/kota': 'kabupaten/kota'
    }

    df_tingkat_kemiskinan.rename(columns=rename_map, inplace=True)

    # --- STANDARISASI ISI DATA KATEGORI ---
    if 'provinsi' in df_tingkat_kemiskinan.columns:
        df_tingkat_kemiskinan['provinsi'] = (
            df_tingkat_kemiskinan['provinsi']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    if 'kabupaten/kota' in df_tingkat_kemiskinan.columns:
        df_tingkat_kemiskinan['kabupaten/kota'] = (
            df_tingkat_kemiskinan['kabupaten/kota']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    return df_tingkat_kemiskinan


def clean_df_pola_pangan_streamlit(df_pola_pangan: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning + agregasi df_pola_pangan
    Versi Streamlit
    (LOGIKA & URUTAN DISAMAKAN DENGAN COLAB)
    """

    # =========================================
    # COPY DF (WAJIB di Streamlit)
    # =========================================
    df_pola_pangan = df_pola_pangan.copy()

    # --- RENAME KOLOM SPESIFIK (MANUAL) ---
    df_pola_pangan.columns = (
        df_pola_pangan.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    rename_map = {
        'kabupaten_kota': 'kabupaten/kota'
    }

    df_pola_pangan.rename(columns=rename_map, inplace=True)

    # --- STANDARISASI ISI DATA KATEGORI ---
    if 'provinsi' in df_pola_pangan.columns:
        df_pola_pangan['provinsi'] = (
            df_pola_pangan['provinsi']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )

    if 'kabupaten/kota' in df_pola_pangan.columns:
        df_pola_pangan['kabupaten/kota'] = (
            df_pola_pangan['kabupaten/kota']
            .astype(str)
            .str.replace(r'\s*\([^)]*\)', '', regex=True)
            .str.strip()
            .str.lower()
            .str.strip()
        )
    return df_pola_pangan

def clean_kabkota_streamlit(df, col_name="kabupaten/kota"):
    df = df.copy()

    if col_name not in df.columns:
        return df

    kol = df[col_name].astype(str).str.lower()

    # 1. HAPUS KATA KABUPATEN / KOTA
    kol = kol.str.replace(r"\bkabupaten\b", "", regex=True)
    kol = kol.str.replace(r"\bkota\b", "", regex=True)
    kol = kol.str.replace(r"\bkab\b\.?", "", regex=True)
    kol = kol.str.replace(r"\bkot\b\.?", "", regex=True)

    # 2. HAPUS KARAKTER ASING
    kol = kol.str.replace(r"[\*\-_/()\']+", " ", regex=True)

    # 3. HAPUS ANGKA
    kol = kol.str.replace(r"\d+", " ", regex=True)

    # 4. RAPIKAN SPASI
    kol = kol.str.replace(r"\s+", " ", regex=True).str.strip()

    df[col_name] = kol
    return df

def apply_clean_kabkota_all(dfs: dict):
    """
    dfs = {
        'IKP': df_IKP_clean,
        'Sosial': df_socio_clean,
        'Konsumsi': df_konsumsi_clean,
        'Kemiskinan': df_kemiskinan_clean,
        'Pola': df_pola_clean
    }
    """
    cleaned = {}

    for name, df in dfs.items():
        cleaned[name] = clean_kabkota_streamlit(df, "kabupaten/kota")

    return cleaned

def aggregate_pola_pangan(df_pola_clean):
    """
    Agregasi df_pola_pangan
    (SESUAI DENGAN COLAB)
    """

    df_pola_clean = df_pola_clean.copy()

    df_pola_agg = (
        df_pola_clean
        .groupby(["provinsi", "kabupaten/kota"])
        .agg({"skor_pph": "mean"})
        .reset_index()
        .rename(columns={"skor_pph": "skor_pph_mean"})
    )

    return df_pola_agg


CLEANING_FUNCTIONS = {
    "IKP": clean_df_IKP24_streamlit,
    "Sosial Ekonomi": clean_df_socio_econom_streamlit,
    "Konsumsi Pangan": clean_df_konsumsi_pangan_streamlit,
    "Kemiskinan": clean_df_kemiskinan_streamlit,
    "Pola Pangan": clean_df_pola_pangan_streamlit   
}


# =====================================================
# MERGGE
# =====================================================
def merge_all_datasets(
    df_IKP24,
    df_socio_econom,
    df_konsumsi_pangan,
    df_tingkat_kemiskinan,
    df_pola_pangan
):
    # AMBIL HASIL CLEANING
    df_IKP = CLEANING_FUNCTIONS["IKP"](df_IKP24)
    df_SOSIAL = CLEANING_FUNCTIONS["Sosial Ekonomi"](df_socio_econom)
    df_KONSUMSI = CLEANING_FUNCTIONS["Konsumsi Pangan"](df_konsumsi_pangan)
    df_KEMISKINAN = CLEANING_FUNCTIONS["Kemiskinan"](df_tingkat_kemiskinan)
    df_POLA = CLEANING_FUNCTIONS["Pola Pangan"](df_pola_pangan)

    # 1️ Sosial Ekonomi + Kemiskinan
    merge_1 = pd.merge(
        df_SOSIAL,
        df_KEMISKINAN,
        on=["provinsi", "kabupaten/kota"],
        how="outer"
    )

    # 2️ + Pola Pangan
    merge_2 = pd.merge(
        merge_1,
        df_POLA,
        on=["provinsi", "kabupaten/kota"],
        how="outer"
    )

    # 3️ + Konsumsi Pangan
    merge_3 = pd.merge(
        merge_2,
        df_KONSUMSI,
        on=["provinsi", "kabupaten/kota"],
        how="outer"
    )

    # 4️ + IKP
    merge_final = pd.merge(
        merge_3,
        df_IKP,
        on=["provinsi", "kabupaten/kota"],
        how="outer"
    )

    return merge_1, merge_2, merge_3, merge_final 

def post_merge_cleaning(merge_final):
    # =====================================================
    # 0. Copy dataframe merge_final → df_merge
    # =====================================================
    df_merge = merge_final.copy()

    # =====================================================
    # 1. GABUNGKAN KOLUM YANG MAKNYA SAMA
    # =====================================================

    # 1.1 Tingkat kemiskinan
    if "persentase_penduduk_miskin_(p0)_menurut_kabupaten/kota_(persen)" in df_merge.columns:
        df_merge["persentase_penduduk_miskin"] = df_merge[
            "persentase_penduduk_miskin_(p0)_menurut_kabupaten/kota_(persen)"
        ].fillna(df_merge.get("poorpeople_percentage"))

    # 1.2 Rata-rata lama sekolah
    if "rata_rata_lama_sekolah_penduduk_15+_(tahun)" in df_merge.columns:
        df_merge["rata_rata_lama_sekolah"] = df_merge[
            "rata_rata_lama_sekolah_penduduk_15+_(tahun)"
        ].fillna(df_merge.get("avg_schooltime"))

    # 1.3 Pengeluaran per kapita
    if "pengeluaran_per_kapita_disesuaikan_(ribu_rupiah/orang/tahun)" in df_merge.columns:
        df_merge["pengeluaran_perkapita"] = df_merge[
            "pengeluaran_per_kapita_disesuaikan_(ribu_rupiah/orang/tahun)"
        ].fillna(df_merge.get("exp_percap"))

    # 1.4 Umur harapan hidup
    if "umur_harapan_hidup_(tahun)" in df_merge.columns:
        df_merge["umur_harapan_hidup"] = df_merge[
            "umur_harapan_hidup_(tahun)"
        ].fillna(df_merge.get("life_exp"))

    # =====================================================
    # 2. HAPUS KOLUM REDUNDAN
    # =====================================================
    kolom_redundan = [
        "persentase_penduduk_miskin_(p0)_menurut_kabupaten/kota_(persen)",
        "poorpeople_percentage",
        "rata_rata_lama_sekolah_penduduk_15+_(tahun)",
        "avg_schooltime",
        "pengeluaran_per_kapita_disesuaikan_(ribu_rupiah/orang/tahun)",
        "exp_percap",
        "umur_harapan_hidup_(tahun)",
        "life_exp"
    ]

    df_merge = df_merge.drop(
        columns=[c for c in kolom_redundan if c in df_merge.columns],
        errors="ignore"
    )

    # =====================================================
    # 3. HAPUS KOLUM YANG TIDAK DIGUNAKAN
    # =====================================================
    kolom_hapus = [
        "nomor_per_tahun",
        "kode_prov",
        "kode_kab/kota",
        "tipe",
        "tahun",
        "peringkat_kab/kota",
        "no",
        "tahun_x",
        "tahun_y",
        "kode_wilayah"
    ]

    df_merge = df_merge.drop(
        columns=[c for c in kolom_hapus if c in df_merge.columns],
        errors="ignore"
    )

    # =====================================================
    # 4. HAPUS STRING "nan" PADA KOLOM PROVINSI
    # =====================================================
    if "provinsi" in df_merge.columns:
        before = len(df_merge)

        df_merge = df_merge[
            df_merge["provinsi"].astype(str).str.strip().str.lower() != "nan"
        ]

        after = len(df_merge)

        print(
            f"[Post-Merge Cleaning] "
            f"Baris sebelum: {before}, "
            f"sesudah: {after}, "
            f"dihapus: {before - after}"
        )

    return df_merge

# ===============================
# Tangani Missing Value
# ===============================
def handle_missing_values(df_merge: pd.DataFrame):
    """
    Penanganan missing value
    - Numerik: median per provinsi
    - Fallback: median global
    - Kategorik: modus
    (SESUAI DENGAN KODE COLAB)
    """

    df = df_merge.copy()

    # ===============================
    # INFO AWAL
    # ===============================
    missing_before = df.isna().sum().sort_values(ascending=False)

    # ===============================
    # IDENTIFIKASI KOLOM
    # ===============================
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # ===============================
    # 1️ IMPUTASI NUMERIK: MEDIAN PER PROVINSI
    # ===============================
    for col in num_cols:
        df[col] = df.groupby("provinsi")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # ===============================
    # 2️ IMPUTASI NUMERIK FINAL: MEDIAN GLOBAL
    # ===============================
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # ===============================
    # 3️ IMPUTASI KATEGORIK: MODUS
    # ===============================
    for col in cat_cols:
        if not df[col].isnull().all():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # ===============================
    # INFO AKHIR
    # ===============================
    missing_after = df.isna().sum().sort_values(ascending=False)

    return df, missing_before, missing_after


# =====================================================
# OUTLIER FUNCTIONS
# =====================================================
def detect_outliers_per_provinsi(df, group_col, cols):
    outlier_results = {}

    for col in cols:
        total_outliers = 0
        contoh_outliers = []

        for prov, group in df.groupby(group_col):
            if group[col].isna().all():
                continue

            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = group[(group[col] < lower) | (group[col] > upper)]
            total_outliers += len(outliers)

            if len(outliers) > 0:
                contoh_outliers.append(
                    outliers[[col, "provinsi", "kabupaten/kota"]].head()
                )

        contoh_outliers = (
            pd.concat(contoh_outliers).head(10)
            if contoh_outliers else None
        )

        outlier_results[col] = {
            "jumlah": total_outliers,
            "persentase": (total_outliers / len(df)) * 100,
            "contoh": contoh_outliers
        }

    return outlier_results

def winsorize_per_provinsi(df, group_col, cols):
    df_new = df.copy()

    for col in cols:
        for prov, group in df.groupby(group_col):
            if group[col].isna().all():
                continue

            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask = df_new[group_col] == prov

            df_new.loc[mask, col] = np.where(
                df_new.loc[mask, col] < lower, lower,
                np.where(
                    df_new.loc[mask, col] > upper,
                    upper,
                    df_new.loc[mask, col]
                )
            )

    return df_new

# =====================================================
# EDA FUNCTIONS
# =====================================================

def summary_dataset(df: pd.DataFrame) -> dict:
    """
    Ringkasan dataset untuk EDA
    """
    return {
        "jumlah_provinsi": df["provinsi"].nunique(),
        "jumlah_kabupaten": df["kabupaten/kota"].nunique(),
        "jumlah_variabel": df.shape[1],
        "jumlah_baris": df.shape[0]
    }


# =====================================================
# 1️ IKP vs IPM
# =====================================================
def chart_ikp_vs_ipm(df: pd.DataFrame):
    prov_filter = alt.selection_multi(fields=["provinsi"], bind="legend")

    regression_line = alt.Chart(df).transform_regression(
        regression="ikp",
        on="indeks_pembangunan_manusia"
    ).mark_line(color="red", size=4).encode(
        x=alt.X("indeks_pembangunan_manusia:Q", title="Indeks Pembangunan Manusia"),
        y=alt.Y("ikp:Q", title="Indeks Ketahanan Pangan (IKP)")
    )

    points = alt.Chart(df).mark_circle().encode(
        x="indeks_pembangunan_manusia:Q",
        y="ikp:Q",
        color=alt.Color(
            "pengeluaran_perkapita:Q",
            scale=alt.Scale(scheme="viridis"),
            title="Pengeluaran Per Kapita"
        ),
        size=alt.Size(
            "konsumsi_pangan:Q",
            scale=alt.Scale(range=[40, 400]),
            title="Konsumsi Pangan"
        ),
        opacity=alt.condition(prov_filter, alt.value(1), alt.value(0.15)),
        tooltip=[
            "provinsi",
            "kabupaten/kota",
            "indeks_pembangunan_manusia",
            "ikp",
            "pengeluaran_perkapita",
            "konsumsi_pangan"
        ]
    ).add_selection(prov_filter)

    return (regression_line + points).properties(
        width=800,
        height=450,
        title="Hubungan IKP dan IPM"
    ).interactive()


def correlation_ikp_ipm(df: pd.DataFrame) -> float:
    """
    Korelasi IKP dan IPM
    """
    return df["indeks_pembangunan_manusia"].corr(df["ikp"])

# =====================================================
# EDA 2️ Akses Air & Sanitasi vs IKP
# =====================================================

def chart_akses_air_vs_ikp(df: pd.DataFrame):
    """
    Scatter IKP vs Akses Air Minum Layak
    (Interaktif, filter provinsi via legend)
    """

    prov_select = alt.selection_multi(fields=["provinsi"], bind="legend")

    chart_air = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X(
            "persentase_rumah_tangga_yang_memiliki_akses_terhadap_air_minum_layak:Q",
            title="Akses Air Minum Layak (%)"
        ),
        y=alt.Y(
            "ikp:Q",
            title="Indeks Ketahanan Pangan (IKP)"
        ),
        color=alt.Color("provinsi:N", title="Provinsi"),
        opacity=alt.condition(prov_select, alt.value(1), alt.value(0.15)),
        tooltip=[
            "provinsi",
            "kabupaten/kota",
            "ikp",
            "persentase_rumah_tangga_yang_memiliki_akses_terhadap_air_minum_layak"
        ]
    ).add_selection(prov_select).properties(
        title="IKP vs Akses Air Minum Layak",
        width=600,
        height=450
    )

    return chart_air


def chart_akses_sanitasi_vs_ikp(df: pd.DataFrame):
    """
    Scatter IKP vs Akses Sanitasi Layak
    (Interaktif, filter provinsi via legend)
    """

    prov_select = alt.selection_multi(fields=["provinsi"], bind="legend")

    chart_sanitasi = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X(
            "persentase_rumah_tangga_yang_memiliki_akses_terhadap_sanitasi_layak:Q",
            title="Akses Sanitasi Layak (%)"
        ),
        y=alt.Y(
            "ikp:Q",
            title="Indeks Ketahanan Pangan (IKP)"
        ),
        color=alt.Color("provinsi:N", title="Provinsi"),
        opacity=alt.condition(prov_select, alt.value(1), alt.value(0.15)),
        tooltip=[
            "provinsi",
            "kabupaten/kota",
            "ikp",
            "persentase_rumah_tangga_yang_memiliki_akses_terhadap_sanitasi_layak"
        ]
    ).add_selection(prov_select).properties(
        title="IKP vs Akses Sanitasi Layak",
        width=600,
        height=450
    )

    return chart_sanitasi


def chart_akses_dasar_vs_ikp(df: pd.DataFrame):
    """
    Gabungan visualisasi:
    IKP vs Akses Air  |  IKP vs Akses Sanitasi
    """
    return chart_akses_air_vs_ikp(df) | chart_akses_sanitasi_vs_ikp(df)


# =====================================================
# EDA 3️ IKP berdasarkan Kelompok Bahan Pangan
# =====================================================

def chart_ikp_per_kelompok_pangan(df: pd.DataFrame):
    """
    Boxplot distribusi IKP per kelompok bahan pangan
    (sesuai dengan visualisasi Colab)
    """

    boxplot_group = alt.Chart(df).mark_boxplot(size=50).encode(
        x=alt.X(
            "kelompok_bahan_pangan:N",
            title="Kelompok Bahan Pangan"
        ),
        y=alt.Y(
            "ikp:Q",
            title="Indeks Ketahanan Pangan (IKP)"
        ),
        color=alt.Color(
            "kelompok_bahan_pangan:N",
            title="Kelompok Bahan Pangan",
            legend=alt.Legend(orient="right")
        ),
        tooltip=[
            "kelompok_bahan_pangan:N",
            alt.Tooltip("median(ikp):Q", title="Median IKP"),
            alt.Tooltip("min(ikp):Q", title="Min IKP"),
            alt.Tooltip("max(ikp):Q", title="Max IKP")
        ]
    ).properties(
        title=alt.TitleParams(
            text="Distribusi IKP per Kelompok Bahan Pangan",
            fontSize=18,
            anchor="middle"
        ),
        width=900,
        height=450
    ).interactive()

    return boxplot_group

# =====================================================
# EDA 4️ Rata-rata Kemiskinan per Provinsi
# =====================================================
def chart_avg_kemiskinan_provinsi(df: pd.DataFrame):
    """
    Bar chart rata-rata persentase penduduk miskin per provinsi
    (IDENTIK dengan kode Colab)
    """

    df_prov_avg = (
        df.groupby("provinsi", as_index=False)
          .agg(avg_poor_percent=("persentase_penduduk_miskin", "mean"))
    )

    chart = (
        alt.Chart(df_prov_avg)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X(
                "provinsi:N",
                sort=alt.SortField(
                    field="avg_poor_percent",
                    order="descending"
                ),
                title="Provinsi"
            ),
            y=alt.Y(
                "avg_poor_percent:Q",
                title="Rata-Rata % Penduduk Miskin"
            ),
            tooltip=[
                alt.Tooltip("provinsi:N", title="Provinsi"),
                alt.Tooltip(
                    "avg_poor_percent:Q",
                    title="Avg % Miskin",
                    format=".2f"
                )
            ]
        )
        .properties(
            title="Perbandingan Rata-Rata Persentase Penduduk Miskin per Provinsi",
            width=900,
            height=400
        )
        .configure_axisX(
            labelAngle=-45,
            labelFontSize=11,
            titleFontSize=13
        )
    )

    return chart

# =====================================================
# EDA 4️ – PDRB Tertinggi per Provinsi
# =====================================================

def chart_pdrb_tertinggi_provinsi(df: pd.DataFrame):
    """
    Bar chart PDRB kabupaten/kota tertinggi di setiap provinsi
    (IDENTIK dengan kode Colab)
    """

    df_max_pdrb = (
        df.groupby("provinsi", as_index=False)
          .apply(
              lambda x: x.loc[x["reg_gdp"].idxmax()],
              include_groups=False
          )
          .reset_index(drop=True)
    )

    # konsistensi nama kolom
    df_max_pdrb["pdrb_percapita_largest"] = df_max_pdrb["reg_gdp"]

    chart = (
        alt.Chart(df_max_pdrb)
        .mark_bar(color="#2E7D32")
        .encode(
            x=alt.X(
                "provinsi:N",
                sort=alt.SortField(
                    field="pdrb_percapita_largest",
                    order="descending"
                ),
                title="Provinsi"
            ),
            y=alt.Y(
                "pdrb_percapita_largest:Q",
                title="Regional GDP (Kab/Kota Tertinggi)"
            ),
            tooltip=[
                alt.Tooltip("provinsi:N", title="Provinsi"),
                alt.Tooltip("kabupaten/kota:N", title="Kab/Kota"),
                alt.Tooltip(
                    "pdrb_percapita_largest:Q",
                    title="Regional GDP",
                    format=".2f"
                )
            ]
        )
        .properties(
            title="Regional GDP di Kabupaten/Kota Tertinggi per Provinsi",
            width=900,
            height=400
        )
        .configure_axisX(
            labelAngle=-45,
            labelFontSize=11,
            titleFontSize=13
        )
    )

    return chart

from sklearn.cluster import KMeans
import altair as alt
import pandas as pd

# =====================================================
# EDA 5️ – Clustering Sosial–Ekonomi & IKP
# =====================================================

def add_cluster_sosial_ekonomi(
    df: pd.DataFrame,
    n_clusters: int = 4
) -> pd.DataFrame:
    """
    Menambahkan kolom cluster berdasarkan:
    - IPM
    - Pengeluaran per kapita
    - Rata-rata lama sekolah

    (SESUAI kode Colab)
    """

    df_cluster = df.copy()

    features = df_cluster[
        [
            "indeks_pembangunan_manusia",
            "pengeluaran_perkapita",
            "rata_rata_lama_sekolah"
        ]
    ].dropna()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    kmeans.fit(features)

    # Default NaN dulu agar aman
    df_cluster["cluster"] = pd.NA

    # Isi cluster hanya untuk baris valid
    df_cluster.loc[features.index, "cluster"] = (
        kmeans.predict(features)
    )

    df_cluster["cluster"] = df_cluster["cluster"].astype("Int64")

    return df_cluster

def chart_cluster_sosial_ekonomi(df: pd.DataFrame):
    """
    Bubble chart:
    - X  : IPM
    - Y  : IKP
    - Warna : Cluster
    - Ukuran : Pengeluaran per kapita

    (IDENTIK dengan Colab)
    """

    chart = (
        alt.Chart(df.dropna(subset=["cluster"]))
        .mark_circle(opacity=0.8)
        .encode(
            x=alt.X(
                "indeks_pembangunan_manusia:Q",
                title="Indeks Pembangunan Manusia"
            ),
            y=alt.Y(
                "ikp:Q",
                title="Indeks Ketahanan Pangan (IKP)"
            ),
            color=alt.Color(
                "cluster:N",
                title="Cluster Wilayah"
            ),
            size=alt.Size(
                "pengeluaran_perkapita:Q",
                title="Pengeluaran Per Kapita",
                scale=alt.Scale(range=[50, 800])
            ),
            tooltip=[
                "provinsi",
                "kabupaten/kota",
                "cluster",
                "indeks_pembangunan_manusia",
                "pengeluaran_perkapita",
                "rata_rata_lama_sekolah",
                "ikp"
            ]
        )
        .properties(
            title="Cluster Wilayah Berdasarkan IPM, Pengeluaran & IKP",
            width=900,
            height=600
        )
        .interactive()
    )

    return chart


# =====================================================
# VISUALISASI LANJUTAN 1
# =====================================================
def chart_treemap_konsumsi_pangan(df: pd.DataFrame):
    treemap_data = (
        df.groupby(["provinsi", "kelompok_bahan_pangan"])
          .agg(total_konsumsi=("konsumsi_pangan", "sum"))
          .reset_index()
    )

    fig = px.treemap(
        treemap_data,
        path=["provinsi", "kelompok_bahan_pangan"],
        values="total_konsumsi",
        color="total_konsumsi",
        color_continuous_scale="Viridis",
        title="Treemap Konsumsi Pangan per Provinsi dan Kelompok Bahan Pangan"
    )

    fig.update_traces(
        textinfo="label+percent parent",
        textfont_size=12,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Total Konsumsi: %{value:,.0f}<br>"
            "Persentase: %{percentParent:.2%}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10)
    )

    return fig


# =====================================================
# VISUALISASI LANJUTAN 2
# =====================================================
def build_choropleth_ikp_plotly(df_merge: pd.DataFrame, geojson_path: str):
    """
    Choropleth Map IKP Provinsi Indonesia
    FIX FINAL — MATCH GEOJSON 100%
    """

    # ===============================
    # 1️ LOAD GEOJSON
    # ===============================
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # ===============================
    # 2️ STANDARISASI NAMA PROVINSI (IKUT GEOJSON)
    # ===============================
    mapping_prov = {
        "di yogyakarta": "daerah istimewa yogyakarta",
        "yogyakarta": "daerah istimewa yogyakarta",
        "banten": "probanten",
        "papua barat": "irian jaya barat",
        "papua tengah": "irian jaya tengah",
        "papua timur": "irian jaya timur",
        "nusa tenggara barat": "nusatenggara barat",
        "nusa tenggara timur": "nusa tenggara timur",
        "aceh" : "di. aceh"
    }

    df = df_merge.copy()

    df["provinsi_fix"] = (
        df["provinsi"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(mapping_prov)
        .str.upper()            
    )

    # ===============================
    # 3️ AGREGASI IKP PER PROVINSI
    # ===============================
    prov_ikp = (
        df.groupby("provinsi_fix", as_index=False)
          .agg(avg_ikp=("ikp", "mean"))
    )

    # ===============================
    # 4️ CHOROPLETH PLOTLY (KUNCI)
    # ===============================
    fig = px.choropleth(
        prov_ikp,
        geojson=geojson,
        locations="provinsi_fix",              # DATA (UPPERCASE)
        featureidkey="properties.Propinsi",    # GEOJSON (UPPERCASE)
        color="avg_ikp",
        color_continuous_scale="Viridis",
        hover_name="provinsi_fix",
        labels={"avg_ikp": "Rata-rata IKP"},
        title="Peta Sebaran Rata-rata IKP Provinsi Indonesia"
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )

    return fig

# =====================================================
# VISUALISASI LANJUTAN 3
# =====================================================
def chart_heatmap_correlation(df: pd.DataFrame):
    """
    Heatmap korelasi antar variabel numerik
    (SETARA dengan heatmap Seaborn di Colab)
    """

    # Ambil kolom numerik
    numerical_df = (
        df.select_dtypes(include=["float64", "int64"])
          .drop(columns=["kelompok_ikp"], errors="ignore")
    )

    # Matriks korelasi
    corr_matrix = numerical_df.corr()

    # Heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        height=700,
        width=900,
        coloraxis_colorbar=dict(title="Korelasi")
    )

    return fig

# ============================================
# MODELING IKP 
# ============================================

# ============================================
# KONFIGURASI FITUR
# ============================================

FEATURES = [
    "indeks_pembangunan_manusia",
    "pengeluaran_perkapita",
    "persentase_rumah_tangga_yang_memiliki_akses_terhadap_air_minum_layak",
    "persentase_rumah_tangga_yang_memiliki_akses_terhadap_sanitasi_layak",
    "umur_harapan_hidup",
    "rata_rata_lama_sekolah",
    "skor_pph_mean",
    "persentase_penduduk_miskin",
    "tingkat_partisipasi_angkatan_kerja",
    "tingkat_pengangguran_terbuka"
]

TARGET = "ikp"


# ============================================
# FUNGSI EVALUASI
# ============================================

def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# ============================================
# TRAIN SEMUA MODEL
# ============================================

def train_models(df_merge: pd.DataFrame):
    """
    Train GradientBoosting, LightGBM, CatBoost
    Return: results dict + scaler
    """

    df = df_merge.copy()

    # ===============================
    # SCALING
    # ===============================
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        # PREDIKSI
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # EVALUASI
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)

        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "MAE_Train": train_metrics["MAE"],
            "RMSE_Train": train_metrics["RMSE"],
            "R2_Train": train_metrics["R2"],
            "MAE_Test": test_metrics["MAE"],
            "RMSE_Test": test_metrics["RMSE"],
            "R2_Test": test_metrics["R2"],
        }

    return results, scaler


# ============================================
# PILIH MODEL TERBAIK
# ============================================

def select_best_model(results: dict):
    """
    Berdasarkan RMSE Test terendah
    Jika sama → R2 Test tertinggi
    """

    best_name = sorted(
        results.items(),
        key=lambda x: (x[1]["RMSE_Test"], -x[1]["R2_Test"])
    )[0][0]

    return best_name, results[best_name]


# ============================================
# PREDIKSI MANUAL
# ============================================

def predict_manual(input_data: dict, model, scaler):
    """
    input_data: dict dari dashboard
    """

    df_input = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df_input[FEATURES])

    pred = model.predict(df_scaled)[0]
    return pred
