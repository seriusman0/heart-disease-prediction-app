# UAS Data Mining — Prediksi Risiko Penyakit Jantung

Aplikasi prediksi dan segmentasi risiko penyakit jantung menggunakan teknik Data Mining.

**Metode:** Classification (Logistic Regression, Random Forest, XGBoost) + Clustering (K-Means)
**Framework:** CRISP-DM
**Dataset:** Heart Failure Prediction (Kaggle) — 918 pasien, 11 fitur

## Struktur Proyek

```
├── dataset/          # Data mentah (heart.csv)
├── notebook/         # Notebook analisis & training pipeline
├── model/            # Artifact model hasil training
├── app/              # Aplikasi Streamlit
│   ├── app.py        # Entry point
│   ├── pages/        # Halaman multi-page
│   └── assets/       # Aset statis
├── laporan/          # Laporan UAS (PDF)
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train model (jika belum ada artifact):
   ```bash
   python notebook/train_pipeline.py
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run app/app.py
   ```
