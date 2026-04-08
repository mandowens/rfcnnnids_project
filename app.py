import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # helps with older .h5 models on TF 2.16+

from pathlib import Path
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NIDS - RF & CNN Inference",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(".")
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

RF_MODEL_CANDIDATES = [
    MODELS_DIR / "rf_model.pkl",
    MODELS_DIR / "rf_model.joblib",
    MODELS_DIR / "random_forest_model.pkl",
    MODELS_DIR / "random_forest_model.joblib",
]

CNN_MODEL_CANDIDATES = [
    MODELS_DIR / "cnn_model.h5",
    MODELS_DIR / "cnn_model.keras",
    MODELS_DIR / "nids_cnn.h5",
    MODELS_DIR / "nids_cnn.keras",
]

SCALER_CANDIDATES = [
    ARTIFACTS_DIR / "scaler.pkl",
    ARTIFACTS_DIR / "scaler.joblib",
    ARTIFACTS_DIR / "standard_scaler.pkl",
    ARTIFACTS_DIR / "standard_scaler.joblib",
]

FEATURE_COLUMNS_CANDIDATES = [
    ARTIFACTS_DIR / "feature_columns.pkl",
    ARTIFACTS_DIR / "feature_columns.json",
    ARTIFACTS_DIR / "selected_features.pkl",
    ARTIFACTS_DIR / "selected_features.json",
]

LABEL_ENCODER_CANDIDATES = [
    ARTIFACTS_DIR / "label_encoder.pkl",
    ARTIFACTS_DIR / "label_encoder.joblib",
]

CLASS_NAMES_CANDIDATES = [
    ARTIFACTS_DIR / "class_names.json",
]

# Add your custom Keras objects here if your CNN model uses custom layers/losses/metrics.
CUSTOM_OBJECTS: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def load_pickle_or_joblib(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".joblib":
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_rf_model() -> Tuple[Optional[Any], Optional[str]]:
    path = find_first_existing(RF_MODEL_CANDIDATES)
    if path is None:
        return None, "RF model tidak ditemukan di folder models/."
    try:
        return load_pickle_or_joblib(path), str(path)
    except Exception as e:
        return None, f"Gagal memuat RF model: {e}"


@st.cache_resource(show_spinner=False)
def load_cnn_model() -> Tuple[Optional[tf.keras.Model], Optional[str]]:
    path = find_first_existing(CNN_MODEL_CANDIDATES)
    if path is None:
        return None, "CNN model tidak ditemukan di folder models/."
    try:
        model = tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
        return model, str(path)
    except Exception as e:
        return None, (
            "Gagal memuat CNN model. Jika model .h5 memakai custom metric/layer/loss, "
            f"tambahkan ke CUSTOM_OBJECTS. Detail error: {e}"
        )


@st.cache_resource(show_spinner=False)
def load_scaler() -> Tuple[Optional[Any], Optional[str]]:
    path = find_first_existing(SCALER_CANDIDATES)
    if path is None:
        return None, "Scaler tidak ditemukan di folder artifacts/."
    try:
        return load_pickle_or_joblib(path), str(path)
    except Exception as e:
        return None, f"Gagal memuat scaler: {e}"


@st.cache_resource(show_spinner=False)
def load_feature_columns() -> Tuple[Optional[List[str]], Optional[str]]:
    path = find_first_existing(FEATURE_COLUMNS_CANDIDATES)
    if path is None:
        return None, "Daftar feature columns tidak ditemukan di folder artifacts/."

    try:
        data = load_json(path) if path.suffix.lower() == ".json" else load_pickle_or_joblib(path)

        if isinstance(data, dict):
            if "feature_columns" in data:
                data = data["feature_columns"]
            elif "selected_features" in data:
                data = data["selected_features"]

        if isinstance(data, np.ndarray):
            data = data.tolist()

        if not isinstance(data, list):
            raise ValueError("Format feature columns harus list.")

        data = [str(col) for col in data]
        return data, str(path)
    except Exception as e:
        return None, f"Gagal memuat feature columns: {e}"


@st.cache_resource(show_spinner=False)
def load_label_encoder() -> Tuple[Optional[Any], Optional[str]]:
    path = find_first_existing(LABEL_ENCODER_CANDIDATES)
    if path is None:
        return None, None
    try:
        return load_pickle_or_joblib(path), str(path)
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def load_class_names() -> Tuple[Optional[List[str]], Optional[str]]:
    path = find_first_existing(CLASS_NAMES_CANDIDATES)
    if path is None:
        return None, None
    try:
        data = load_json(path)
        if isinstance(data, dict) and "class_names" in data:
            data = data["class_names"]
        if not isinstance(data, list):
            raise ValueError("class_names.json harus berisi list string.")
        return [str(x) for x in data], str(path)
    except Exception:
        return None, None


def get_effective_class_names(label_encoder: Any, class_names: Optional[List[str]], n_classes: Optional[int] = None) -> List[str]:
    if class_names:
        return class_names
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        return [str(x) for x in label_encoder.classes_]
    if n_classes is not None:
        if n_classes == 2:
            return ["normal", "attack"]
        return [f"class_{i}" for i in range(n_classes)]
    return ["normal", "attack"]


@st.cache_data(show_spinner=False)
def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def align_features(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    working_df = df.copy()

    missing_columns = [col for col in feature_columns if col not in working_df.columns]
    extra_columns = [col for col in working_df.columns if col not in feature_columns]

    for col in missing_columns:
        working_df[col] = 0

    aligned_df = working_df[feature_columns].copy()

    for col in aligned_df.columns:
        aligned_df[col] = pd.to_numeric(aligned_df[col], errors="coerce")

    aligned_df = aligned_df.replace([np.inf, -np.inf], np.nan)
    aligned_df = aligned_df.fillna(0)

    return aligned_df, missing_columns, extra_columns


def scale_features(df_features: pd.DataFrame, scaler: Any) -> np.ndarray:
    return scaler.transform(df_features)



def adapt_input_for_cnn(X_scaled: np.ndarray, cnn_model: tf.keras.Model) -> np.ndarray:
    input_shape = cnn_model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape is None:
        return X_scaled

    if len(input_shape) == 2:
        # (batch, features)
        return X_scaled

    if len(input_shape) == 3:
        # Common cases: (batch, features, 1) or (batch, 1, features)
        _, dim1, dim2 = input_shape
        n_samples, n_features = X_scaled.shape

        if dim1 == n_features and (dim2 in [1, None]):
            return X_scaled.reshape(n_samples, n_features, 1)

        if dim2 == n_features and (dim1 in [1, None]):
            return X_scaled.reshape(n_samples, 1, n_features)

        if dim1 not in [None] and dim2 not in [None] and dim1 * dim2 == n_features:
            return X_scaled.reshape(n_samples, dim1, dim2)

        raise ValueError(
            f"Input CNN tidak cocok. Shape model: {input_shape}, shape data: {X_scaled.shape}"
        )

    raise ValueError(
        f"Bentuk input CNN tidak didukung oleh app ini: {input_shape}. "
        "Sesuaikan fungsi adapt_input_for_cnn()."
    )



def decode_predictions_from_indices(indices: np.ndarray, class_names: List[str]) -> List[str]:
    decoded = []
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(class_names):
            decoded.append(class_names[idx])
        else:
            decoded.append(str(idx))
    return decoded



def rf_predict(
    X_scaled: np.ndarray,
    model: Any,
    label_encoder: Any,
    class_names: Optional[List[str]],
    threshold: float,
) -> Tuple[List[str], Optional[pd.DataFrame], np.ndarray]:
    prob_df = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        if proba.ndim == 2 and proba.shape[1] == 2:
            effective_names = get_effective_class_names(label_encoder, class_names, 2)
            positive_prob = proba[:, 1]
            pred_idx = (positive_prob >= threshold).astype(int)
            pred_labels = decode_predictions_from_indices(pred_idx, effective_names)
            confidence = np.where(pred_idx == 1, positive_prob, 1 - positive_prob)
            prob_df = pd.DataFrame(proba, columns=[f"prob_{name}" for name in effective_names])
            return pred_labels, prob_df, confidence

        if proba.ndim == 2 and proba.shape[1] > 2:
            effective_names = get_effective_class_names(label_encoder, class_names, proba.shape[1])
            pred_idx = np.argmax(proba, axis=1)
            pred_labels = decode_predictions_from_indices(pred_idx, effective_names)
            confidence = np.max(proba, axis=1)
            prob_df = pd.DataFrame(proba, columns=[f"prob_{name}" for name in effective_names])
            return pred_labels, prob_df, confidence

    raw_pred = model.predict(X_scaled)
    raw_pred = np.asarray(raw_pred)

    if raw_pred.dtype.kind in {"U", "S", "O"}:
        pred_labels = [str(x) for x in raw_pred]
        confidence = np.ones(len(pred_labels))
        return pred_labels, prob_df, confidence

    raw_pred = raw_pred.astype(int)
    effective_names = get_effective_class_names(label_encoder, class_names)
    pred_labels = decode_predictions_from_indices(raw_pred, effective_names)
    confidence = np.ones(len(pred_labels))
    return pred_labels, prob_df, confidence



def cnn_predict(
    X_scaled: np.ndarray,
    model: tf.keras.Model,
    label_encoder: Any,
    class_names: Optional[List[str]],
    threshold: float,
) -> Tuple[List[str], Optional[pd.DataFrame], np.ndarray]:
    X_cnn = adapt_input_for_cnn(X_scaled, model)
    proba = model.predict(X_cnn, verbose=0)
    proba = np.asarray(proba)

    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)

    if proba.ndim == 2 and proba.shape[1] == 1:
        positive_prob = proba[:, 0]
        effective_names = get_effective_class_names(label_encoder, class_names, 2)
        pred_idx = (positive_prob >= threshold).astype(int)
        pred_labels = decode_predictions_from_indices(pred_idx, effective_names)
        confidence = np.where(pred_idx == 1, positive_prob, 1 - positive_prob)
        prob_df = pd.DataFrame(
            {
                f"prob_{effective_names[0]}": 1 - positive_prob,
                f"prob_{effective_names[1]}": positive_prob,
            }
        )
        return pred_labels, prob_df, confidence

    if proba.ndim == 2 and proba.shape[1] >= 2:
        effective_names = get_effective_class_names(label_encoder, class_names, proba.shape[1])
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = decode_predictions_from_indices(pred_idx, effective_names)
        confidence = np.max(proba, axis=1)
        prob_df = pd.DataFrame(proba, columns=[f"prob_{name}" for name in effective_names])
        return pred_labels, prob_df, confidence

    raise ValueError(f"Output CNN tidak dikenali. Shape output: {proba.shape}")



def run_inference(
    model_choice: str,
    df_input: pd.DataFrame,
    rf_model: Any,
    cnn_model: tf.keras.Model,
    scaler: Any,
    feature_columns: List[str],
    label_encoder: Any,
    class_names: Optional[List[str]],
    threshold: float,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df_features, missing_columns, extra_columns = align_features(df_input, feature_columns)
    X_scaled = scale_features(df_features, scaler)

    result_df = df_input.copy().reset_index(drop=True)

    if model_choice == "Random Forest":
        pred_labels, prob_df, confidence = rf_predict(X_scaled, rf_model, label_encoder, class_names, threshold)
        result_df["model"] = "Random Forest"

    elif model_choice == "CNN":
        pred_labels, prob_df, confidence = cnn_predict(X_scaled, cnn_model, label_encoder, class_names, threshold)
        result_df["model"] = "CNN"

    elif model_choice == "Compare: RF vs CNN":
        rf_labels, rf_prob_df, rf_conf = rf_predict(X_scaled, rf_model, label_encoder, class_names, threshold)
        cnn_labels, cnn_prob_df, cnn_conf = cnn_predict(X_scaled, cnn_model, label_encoder, class_names, threshold)

        result_df["rf_prediction"] = rf_labels
        result_df["rf_confidence"] = np.round(rf_conf, 6)
        result_df["cnn_prediction"] = cnn_labels
        result_df["cnn_confidence"] = np.round(cnn_conf, 6)
        result_df["is_agree"] = result_df["rf_prediction"] == result_df["cnn_prediction"]

        if rf_prob_df is not None:
            rf_prob_df = rf_prob_df.add_prefix("rf_")
            result_df = pd.concat([result_df, rf_prob_df.reset_index(drop=True)], axis=1)
        if cnn_prob_df is not None:
            cnn_prob_df = cnn_prob_df.add_prefix("cnn_")
            result_df = pd.concat([result_df, cnn_prob_df.reset_index(drop=True)], axis=1)

        return result_df, missing_columns, extra_columns

    else:
        raise ValueError("Pilihan model tidak valid.")

    result_df["prediction"] = pred_labels
    result_df["confidence"] = np.round(confidence, 6)

    if prob_df is not None:
        result_df = pd.concat([result_df, prob_df.reset_index(drop=True)], axis=1)

    return result_df, missing_columns, extra_columns



def show_status_card(title: str, ok: bool, detail: Optional[str]) -> None:
    if ok:
        st.sidebar.success(f"{title}\n\n{detail}")
    else:
        st.sidebar.error(f"{title}\n\n{detail}")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🛡️ NIDS Inference App")
st.caption("Streamlit app untuk inferensi Network Intrusion Detection System berbasis Random Forest dan CNN.")

with st.expander("Struktur repo yang diharapkan", expanded=False):
    st.code(
        """.
├── app.py
├── requirements.txt
├── models/
│   ├── rf_model.pkl            # atau rf_model.joblib
│   └── cnn_model.h5            # atau cnn_model.keras
└── artifacts/
    ├── scaler.pkl              # atau .joblib
    ├── feature_columns.pkl     # atau .json
    ├── label_encoder.pkl       # opsional
    └── class_names.json        # opsional
""",
        language="text",
    )
    st.markdown(
        """
- **CSV input** sebaiknya sudah memakai kolom fitur yang sama seperti saat training.
- Jika ada kolom yang hilang, app akan menambahkannya dengan nilai **0**.
- Jika ada kolom tambahan, app akan mengabaikannya.
- Bila model CNN memakai custom layer/loss/metric, tambahkan objek tersebut ke `CUSTOM_OBJECTS` di `app.py`.
"""
    )

rf_model, rf_info = load_rf_model()
cnn_model, cnn_info = load_cnn_model()
scaler, scaler_info = load_scaler()
feature_columns, feature_info = load_feature_columns()
label_encoder, label_info = load_label_encoder()
class_names, class_info = load_class_names()

st.sidebar.header("Model & Artifacts Status")
show_status_card("RF Model", rf_model is not None, rf_info)
show_status_card("CNN Model", cnn_model is not None, cnn_info)
show_status_card("Scaler", scaler is not None, scaler_info)
show_status_card("Feature Columns", feature_columns is not None, feature_info)
show_status_card("Label Encoder (opsional)", label_encoder is not None, label_info or "Tidak ditemukan")
show_status_card("Class Names (opsional)", class_names is not None, class_info or "Tidak ditemukan")

model_choice = st.sidebar.selectbox(
    "Pilih mode inferensi",
    ["Random Forest", "CNN", "Compare: RF vs CNN"],
)

threshold = st.sidebar.slider(
    "Threshold binary classification",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Dipakai untuk output biner saat model menghasilkan probabilitas kelas positif.",
)

if feature_columns:
    st.sidebar.info(f"Jumlah fitur yang diharapkan: {len(feature_columns)}")

required_ready = scaler is not None and feature_columns is not None
rf_ready = rf_model is not None
cnn_ready = cnn_model is not None

if model_choice == "Random Forest" and not (required_ready and rf_ready):
    st.error("Artifact untuk mode Random Forest belum lengkap. Periksa sidebar.")
    st.stop()

if model_choice == "CNN" and not (required_ready and cnn_ready):
    st.error("Artifact untuk mode CNN belum lengkap. Periksa sidebar.")
    st.stop()

if model_choice == "Compare: RF vs CNN" and not (required_ready and rf_ready and cnn_ready):
    st.error("Artifact untuk mode compare belum lengkap. Periksa sidebar.")
    st.stop()

uploaded_file = st.file_uploader("Upload file CSV fitur NIDS", type=["csv"])

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("#### Petunjuk input")
    st.write(
        "Upload dataset `.csv` berisi record jaringan yang akan diprediksi. "
        "Kolom fitur harus konsisten dengan fitur saat training model."
    )
with col2:
    if feature_columns:
        st.metric("Expected Features", len(feature_columns))

if uploaded_file is not None:
    try:
        df_input = read_uploaded_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

    st.markdown("### Preview data input")
    st.dataframe(df_input.head(10), use_container_width=True)

    with st.expander("Lihat daftar feature columns yang dibutuhkan", expanded=False):
        if feature_columns:
            st.write(feature_columns)

    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        try:
            result_df, missing_columns, extra_columns = run_inference(
                model_choice=model_choice,
                df_input=df_input,
                rf_model=rf_model,
                cnn_model=cnn_model,
                scaler=scaler,
                feature_columns=feature_columns,
                label_encoder=label_encoder,
                class_names=class_names,
                threshold=threshold,
            )

            st.success("Prediksi berhasil dijalankan.")

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                if missing_columns:
                    st.warning(
                        f"Kolom yang tidak ditemukan di CSV dan diisi 0: {missing_columns[:20]}"
                        + (" ..." if len(missing_columns) > 20 else "")
                    )
                else:
                    st.info("Tidak ada kolom fitur yang hilang.")
            with info_col2:
                if extra_columns:
                    st.warning(
                        f"Kolom ekstra di CSV yang diabaikan: {extra_columns[:20]}"
                        + (" ..." if len(extra_columns) > 20 else "")
                    )
                else:
                    st.info("Tidak ada kolom ekstra di CSV.")

            st.markdown("### Hasil prediksi")
            st.dataframe(result_df, use_container_width=True)

            if "prediction" in result_df.columns:
                st.markdown("### Distribusi hasil prediksi")
                pred_counts = result_df["prediction"].value_counts().reset_index()
                pred_counts.columns = ["class", "count"]
                st.bar_chart(pred_counts.set_index("class"))

            if {"rf_prediction", "cnn_prediction"}.issubset(result_df.columns):
                st.markdown("### Ringkasan perbandingan model")
                compare_col1, compare_col2, compare_col3 = st.columns(3)
                compare_col1.metric("Jumlah Data", len(result_df))
                compare_col2.metric("Agreement", int(result_df["is_agree"].sum()))
                compare_col3.metric("Disagreement", int((~result_df["is_agree"]).sum()))

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download hasil prediksi (.csv)",
                data=csv_bytes,
                file_name="nids_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Terjadi error saat inferensi: {e}")
            st.exception(e)
else:
    st.info("Silakan upload file CSV untuk memulai prediksi.")
