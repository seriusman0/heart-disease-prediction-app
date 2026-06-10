# Panduan Lengkap Proyek UAS Data Mining: Prediksi Risiko Penyakit Jantung

**Judul Proyek**: Prediksi dan Segmentasi Risiko Penyakit Jantung Menggunakan Data Mining  
**Metode**: Classification + Clustering  
**Framework**: CRISP-DM  
**Web Framework**: Streamlit  
**Dataset**: Heart Failure Prediction (Kaggle)

## 1. Ringkasan Proyek

Proyek ini mengembangkan sistem untuk memprediksi risiko penyakit jantung pasien dan melakukan segmentasi pasien berdasarkan pola risiko. 

- **Business Problem**: Deteksi dini penyakit jantung untuk intervensi medis yang tepat.
- **Dataset**: 918 record, 11 fitur + target (mayoritas numerik).
- **Metode**:
  - **Classification**: Prediksi HeartDisease (0/1) menggunakan Random Forest / XGBoost.
  - **Clustering**: Segmentasi pasien dengan K-Means.
- **Output**: Aplikasi web Streamlit + Laporan lengkap.

Proyek ini memenuhi semua ketentuan UAS.

## 2. Struktur Folder Project

## 3. Tahapan Kerja (CRISP-DM)

### Phase 1: Business Understanding
- Rumuskan masalah, tujuan, dan metrik sukses.
- Tulis di laporan.

### Phase 2: Data Understanding & Preprocessing
1. Load dataset.
2. EDA lengkap (statistik, visualisasi, correlation).
3. Cleaning (Cholesterol = 0 → median).
4. One-Hot Encoding categorical features.
5. Scaling dengan StandardScaler.
6. Train-test split.

### Phase 3: Modeling
**Classification**:
- Train: Logistic Regression, Random Forest, XGBoost.
- Tuning + evaluasi (F1-Score, ROC-AUC).

**Clustering**:
- K-Means + Elbow/Silhouette.
- Analisis karakteristik cluster.

### Phase 4: Evaluation & Interpretation
- Pilih model terbaik.
- Gunakan SHAP untuk explainability.
- Interpretasi bisnis.

### Phase 5: Deployment (Streamlit)
- Buat semua halaman sesuai requirement dosen.
- Integrasikan model + preprocessing input user.

### Phase 6: Reporting & Presentasi
- Laporan PDF ≥15 halaman.
- Video 7–15 menit.
- Repository GitHub.

## 4. Checklist Tugas

### Minggu 1
- [ ] Download dataset + EDA + Preprocessing
- [ ] Buat requirements.txt

### Minggu 2
- [ ] Modeling + Tuning + Clustering + SHAP
- [ ] Simpan model

### Minggu 3
- [ ] Bangun + integrasikan Streamlit App
- [ ] Deploy (bonus)

### Minggu 4
- [ ] Laporan lengkap
- [ ] Video presentasi
- [ ] Review keseluruhan

## 5. Bonus & Tips
- Deploy online, SHAP, UI menarik, model advanced.
- Fokus interpretasi dan business value.
- Gunakan Plotly untuk visualisasi interaktif.

## 6. Resources
- Dataset: [Kaggle Heart Failure](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Streamlit Docs, SHAP Docs.

**Panduan ini adalah single source of truth.** Ikuti urutannya.
