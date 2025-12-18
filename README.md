# 🌾 IKP Prediction Project — Analisis & Prediksi Ketahanan Pangan Indonesia

## 🧾 Deskripsi Proyek
Proyek ini merupakan aplikasi **dashboard interaktif berbasis Streamlit** yang bertujuan untuk menganalisis dan memprediksi **Indeks Ketahanan Pangan (IKP)** pada tingkat **kabupaten/kota di Indonesia**.

Analisis dilakukan dengan mengintegrasikan berbagai **indikator sosial-ekonomi**, seperti:
- Indeks Pembangunan Manusia (IPM)
- Tingkat kemiskinan
- Produk Domestik Regional Bruto (PDRB)
- Konsumsi pangan
- Akses air minum dan sanitasi layak

Project ini menggabungkan pendekatan **data science, visual analytics, clustering wilayah, dan machine learning** untuk menghasilkan insight yang komprehensif dalam mendukung pengambilan keputusan berbasis data.

---

## 🎯 Tujuan
1. Menganalisis hubungan antara **IKP dan IPM** di berbagai wilayah.
2. Mengkaji pengaruh **akses air minum dan sanitasi layak** terhadap ketahanan pangan.
3. Menganalisis variasi **konsumsi pangan berdasarkan kelompok bahan pangan**.
4. Membandingkan **tingkat kemiskinan dan PDRB antar provinsi**.
5. Mengelompokkan wilayah menggunakan **clustering berbasis karakteristik sosial-ekonomi**.
6. Membangun model **prediksi IKP** berbasis machine learning.

---

## 📊 Dataset
Dataset yang digunakan merupakan data sekunder yang bersumber dari berbagai publikasi resmi dan dataset terbuka, meliputi:
- Data IKP kabupaten/kota
- Data IPM dan indikator sosial-ekonomi
- Data tingkat kemiskinan
- Data konsumsi pangan
- Data spasial provinsi Indonesia

### Proses Data Wrangling
- *Data gathering* dari berbagai sumber
- *Data cleaning* (missing values dan duplikasi)
- Normalisasi dan standarisasi data
- Penggabungan beberapa dataset lintas indikator

---

## 🔍 Exploratory Data Analysis (EDA)
Analisis eksploratif dilakukan untuk memahami pola dan hubungan antar variabel, antara lain:
- Hubungan **IKP vs IPM**
- IKP terhadap **akses air minum dan sanitasi**
- Distribusi IKP berdasarkan **kelompok bahan pangan**
- Perbandingan **kemiskinan dan PDRB antar provinsi**
- Analisis outlier dan ketimpangan wilayah

Visualisasi yang digunakan:
- Scatter plot dan bubble chart
- Bar chart dan boxplot
- Heatmap korelasi
- Peta sebaran IKP (choropleth map)

---

## 🧠 Pemodelan dan Prediksi
Model machine learning yang digunakan:
- **Gradient Boosting Regressor**
- **LightGBM Regressor**
- **CatBoost Regressor**

### Evaluasi Model
- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Visualisasi **aktual vs prediksi**
- Perbandingan performa data training dan testing  

Model terbaik digunakan untuk melakukan **prediksi IKP** secara interaktif melalui dashboard.

---
---

## 👥 Anggota Kelompok

| No | Nama | NIM |
|----|------------------------------|-----------|
| 1  | Nur Azizah                   | E1E123015 |
| 2  | Syuk Rina BTE Amiruddin       | E1E123018 |
| 3  | Cindy Rahmayanti              | E1E123002 |

---
## ⚙️ Cara Menjalankan Aplikasi

### 🟢 Menjalankan Secara Online (Streamlit Cloud)
1. Buka repository GitHub ini.
2. Akses aplikasi melalui link Streamlit (jika tersedia).
3. Dashboard dapat langsung digunakan tanpa instalasi tambahan.

---

### 💻 Menjalankan Secara Lokal
1. Clone repository:
   ```bash
   git clone https://github.com/e1e123015nurazizah-tech/IKP-prediction-project.git
   cd IKP-prediction-project
