# NSL-KDD Multiclass NIDS (Random Forest + 1D-CNN)

Repo ini berisi:
- `train_nslkdd_models.py` untuk melatih ulang 2 model multiclass pada NSL-KDD:
  - Random Forest
  - 1D-CNN
- `app.py` untuk deploy uji-coba ke Streamlit Community Cloud.

## Struktur folder

```text
rfcnnnids_project/
├── app.py
├── requirements.txt
├── train_nslkdd_models.py
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
│   ├── rf_model.joblib
│   └── cnn_model.keras
└── artifacts/
    ├── preprocessor.joblib
    ├── label_encoder.joblib
    ├── class_names.json
    ├── feature_columns.json
    └── metrics_summary.json
```

## 1) Siapkan dataset

Unduh file:
- `KDDTrain+.txt`
- `KDDTest+.txt`

Lalu simpan ke folder `data/`.

## 2) Install dependency

```bash
pip install -r requirements.txt
```

## 3) Training model

```bash
python train_nslkdd_models.py
```

Atau:

```bash
python train_nslkdd_models.py --train data/KDDTrain+.txt --test data/KDDTest+.txt --epochs 20 --batch-size 256
```

## 4) Jalankan Streamlit lokal

```bash
streamlit run app.py
```

## 5) Deploy ke Streamlit Community Cloud

Pastikan file berikut sudah ada di repo GitHub:
- `app.py`
- `requirements.txt`
- folder `models/`
- folder `artifacts/`

Pada halaman deploy:
- **Repository**: repo GitHub Anda
- **Branch**: `main`
- **Main file path**: `app.py`

## Format file upload di app

App menerima:
- file NSL-KDD mentah tanpa header:
  - 41 kolom fitur
  - atau 42 kolom (fitur + label)
  - atau 43 kolom (fitur + label + difficulty)
- CSV dengan header 41 fitur mentah NSL-KDD

App akan menampilkan:
- prediksi Random Forest
- prediksi 1D-CNN
- prediksi ensemble (rata-rata probabilitas RF + CNN)
- confidence score
- probabilitas per kelas
- distribusi prediksi

## Kelas multiclass

Label serangan dipetakan menjadi 5 kelas:
- `normal`
- `dos`
- `probe`
- `r2l`
- `u2r`
