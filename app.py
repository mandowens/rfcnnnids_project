import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="NSL-KDD Multiclass NIDS", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RAW_FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]
LABEL_COL = "label"
DIFFICULTY_COL = "difficulty"

MODEL_FILES = {
    "rf": ["rf_model.joblib", "rf_model.pkl"],
    "cnn": ["cnn_model.keras", "cnn_model.h5"],
}
ARTIFACT_FILES = {
    "preprocessor": ["preprocessor.joblib", "preprocessor.pkl"],
    "label_encoder": ["label_encoder.joblib", "label_encoder.pkl"],
    "class_names": ["class_names.json"],
    "metrics": ["metrics_summary.json"],
    "feature_columns": ["feature_columns.json"],
}

def find_existing_file(folder: Path, candidates):
    for name in candidates:
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Tidak menemukan salah satu file berikut di {folder}: {candidates}")

@st.cache_resource(show_spinner=False)
def load_assets():
    rf_path = find_existing_file(MODELS_DIR, MODEL_FILES["rf"])
    cnn_path = find_existing_file(MODELS_DIR, MODEL_FILES["cnn"])
    preproc_path = find_existing_file(ARTIFACTS_DIR, ARTIFACT_FILES["preprocessor"])
    le_path = find_existing_file(ARTIFACTS_DIR, ARTIFACT_FILES["label_encoder"])
    class_names_path = find_existing_file(ARTIFACTS_DIR, ARTIFACT_FILES["class_names"])

    rf_model = joblib.load(rf_path)
    cnn_model = tf.keras.models.load_model(cnn_path)
    preprocessor = joblib.load(preproc_path)
    label_encoder = joblib.load(le_path)
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))

    metrics = {}
    try:
        metrics_path = find_existing_file(ARTIFACTS_DIR, ARTIFACT_FILES["metrics"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

    feature_columns = []
    try:
        feature_path = find_existing_file(ARTIFACTS_DIR, ARTIFACT_FILES["feature_columns"])
        feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    except Exception:
        feature_columns = []

    return rf_model, cnn_model, preprocessor, label_encoder, class_names, metrics, feature_columns

def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mendukung beberapa format upload:
    1) 41 kolom fitur tanpa header
    2) 42 kolom (41 fitur + label)
    3) 43 kolom (41 fitur + label + difficulty)
    4) CSV dengan header yang memuat kolom-kolom mentah NSL-KDD
    """
    original_cols = list(df.columns)

    # Kasus file tanpa header yang dibaca sebagai kolom integer
    if all(isinstance(c, int) for c in original_cols):
        if df.shape[1] == 41:
            df.columns = RAW_FEATURE_COLUMNS
            return df
        if df.shape[1] == 42:
            df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL]
            return df.drop(columns=[LABEL_COL])
        if df.shape[1] == 43:
            df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL, DIFFICULTY_COL]
            return df.drop(columns=[LABEL_COL, DIFFICULTY_COL])
        raise ValueError(
            f"Jumlah kolom {df.shape[1]} tidak cocok untuk format NSL-KDD. "
            "Gunakan 41, 42, atau 43 kolom."
        )

    # Kasus file dengan header
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    if all(col in lower_map for col in [c.lower() for c in RAW_FEATURE_COLUMNS]):
        renamed = df.rename(columns={lower_map[c.lower()]: c for c in RAW_FEATURE_COLUMNS})
        return renamed[RAW_FEATURE_COLUMNS]

    # Kasus header tapi jumlah kolom cocok
    if df.shape[1] == 41:
        df = df.copy()
        df.columns = RAW_FEATURE_COLUMNS
        return df
    if df.shape[1] == 42:
        df = df.copy()
        df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL]
        return df.drop(columns=[LABEL_COL])
    if df.shape[1] == 43:
        df = df.copy()
        df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL, DIFFICULTY_COL]
        return df.drop(columns=[LABEL_COL, DIFFICULTY_COL])

    raise ValueError(
        "Format file belum dikenali. Upload CSV dengan 41 fitur mentah NSL-KDD "
        "atau file KDDTest+/KDDTrain+ asli (42/43 kolom)."
    )

def cast_nslkdd_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in RAW_FEATURE_COLUMNS if c not in categorical_cols]

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def predict_all(df_raw: pd.DataFrame, rf_model, cnn_model, preprocessor, label_encoder, class_names):
    df_model = cast_nslkdd_types(df_raw)
    X = preprocessor.transform(df_model)
    X_dense = to_dense(X).astype("float32")

    rf_proba = rf_model.predict_proba(X_dense)
    rf_idx = np.argmax(rf_proba, axis=1)
    rf_pred = label_encoder.inverse_transform(rf_idx)
    rf_conf = np.max(rf_proba, axis=1)

    X_cnn = np.expand_dims(X_dense, axis=-1)
    cnn_proba = cnn_model.predict(X_cnn, verbose=0)
    cnn_idx = np.argmax(cnn_proba, axis=1)
    cnn_pred = label_encoder.inverse_transform(cnn_idx)
    cnn_conf = np.max(cnn_proba, axis=1)

    ensemble_proba = (rf_proba + cnn_proba) / 2.0
    ens_idx = np.argmax(ensemble_proba, axis=1)
    ens_pred = label_encoder.inverse_transform(ens_idx)
    ens_conf = np.max(ensemble_proba, axis=1)

    result = df_raw.copy()
    result["rf_prediction"] = rf_pred
    result["rf_confidence"] = np.round(rf_conf, 4)
    result["cnn_prediction"] = cnn_pred
    result["cnn_confidence"] = np.round(cnn_conf, 4)
    result["ensemble_prediction"] = ens_pred
    result["ensemble_confidence"] = np.round(ens_conf, 4)
    return result, rf_proba, cnn_proba, ensemble_proba

def proba_frame(proba: np.ndarray, class_names, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_proba_{name}" for name in class_names]
    return pd.DataFrame(np.round(proba, 4), columns=cols)

def show_metrics(metrics: dict):
    if not metrics:
        st.info("File metrik belum tersedia. Jalankan script training terlebih dahulu.")
        return

    c1, c2, c3, c4 = st.columns(4)
    rf_acc = metrics.get("rf", {}).get("accuracy")
    cnn_acc = metrics.get("cnn", {}).get("accuracy")
    rf_f1 = metrics.get("rf", {}).get("macro_f1")
    cnn_f1 = metrics.get("cnn", {}).get("macro_f1")

    c1.metric("RF Accuracy", f"{rf_acc:.4f}" if rf_acc is not None else "-")
    c2.metric("CNN Accuracy", f"{cnn_acc:.4f}" if cnn_acc is not None else "-")
    c3.metric("RF Macro F1", f"{rf_f1:.4f}" if rf_f1 is not None else "-")
    c4.metric("CNN Macro F1", f"{cnn_f1:.4f}" if cnn_f1 is not None else "-")

    with st.expander("Lihat ringkasan metrik lengkap"):
        st.json(metrics)

def main():
    st.title("NSL-KDD Multiclass NIDS")
    st.caption("Deploy uji-coba untuk klasifikasi multiclass dengan Random Forest dan 1D-CNN.")

    try:
        rf_model, cnn_model, preprocessor, label_encoder, class_names, metrics, feature_columns = load_assets()
    except Exception as e:
        st.error(
            "Gagal memuat model/artifact. Pastikan Anda sudah menjalankan script training "
            "dan menaruh file hasilnya di folder `models/` dan `artifacts/`.\n\n"
            f"Detail error: {e}"
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Prediksi", "Info Model", "Skema Input"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload CSV untuk prediksi",
            type=["csv", "txt"],
            help="Bisa berupa file KDDTest+/KDDTrain+ asli, atau CSV 41 fitur mentah NSL-KDD."
        )

        sample_note = st.info(
            "Untuk uji cepat, upload file `KDDTest+.txt` atau `KDDTrain+.txt`. "
            "App akan otomatis membaca 41 fitur mentah NSL-KDD."
        )

        if uploaded is not None:
            try:
                # Coba dua strategi pembacaan untuk mengakomodasi file tanpa header / dengan header
                try:
                    df_input = pd.read_csv(uploaded, header=None)
                except Exception:
                    uploaded.seek(0)
                    df_input = pd.read_csv(uploaded)

                df_norm = normalize_uploaded_dataframe(df_input)
                st.success(f"Berhasil membaca file: {df_norm.shape[0]} baris, {df_norm.shape[1]} fitur.")
                st.dataframe(df_norm.head(20), use_container_width=True)

                result_df, rf_proba, cnn_proba, ens_proba = predict_all(
                    df_norm, rf_model, cnn_model, preprocessor, label_encoder, class_names
                )

                out_df = pd.concat(
                    [
                        result_df.reset_index(drop=True),
                        proba_frame(rf_proba, class_names, "rf"),
                        proba_frame(cnn_proba, class_names, "cnn"),
                        proba_frame(ens_proba, class_names, "ensemble"),
                    ],
                    axis=1,
                )

                st.subheader("Hasil Prediksi")
                st.dataframe(out_df, use_container_width=True)

                st.subheader("Distribusi Prediksi Ensemble")
                dist = out_df["ensemble_prediction"].value_counts().rename_axis("class").reset_index(name="count")
                st.bar_chart(dist.set_index("class"))

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download hasil prediksi CSV",
                    data=csv_bytes,
                    file_name="nslkdd_multiclass_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Gagal memproses file: {e}")

    with tab2:
        show_metrics(metrics)
        st.write("**Kelas target:**", ", ".join(class_names))
        if feature_columns:
            with st.expander("Jumlah fitur setelah preprocessing"):
                st.write(f"{len(feature_columns)} fitur hasil one-hot encoding + scaling.")
        st.markdown(
            """
            **Model yang dimuat oleh app**
            - Random Forest multiclass
            - 1D-CNN multiclass
            - Ensemble sederhana berbasis rata-rata probabilitas RF dan CNN
            """
        )

    with tab3:
        st.write("**41 fitur mentah NSL-KDD yang diharapkan app:**")
        st.code(", ".join(RAW_FEATURE_COLUMNS))
        st.write(
            "App juga bisa membaca file NSL-KDD mentah tanpa header dengan 42 atau 43 kolom, "
            "lalu otomatis menghapus kolom label dan difficulty."
        )

if __name__ == "__main__":
    main()
