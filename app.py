from pathlib import Path

code = """import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="NIDS Inference App",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics_summary.json"

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

RAW_FEATURE_COLUMNS = NSL_KDD_COLUMNS[:41]


def path_status(path: Path):
    if path.exists():
        return True, str(path.relative_to(BASE_DIR))
    return False, str(path.relative_to(BASE_DIR))


@st.cache_resource(show_spinner=False)
def load_rf_model():
    return joblib.load(RF_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_cnn_model():
    return tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_PATH)


@st.cache_data(show_spinner=False)
def load_json_file(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_assets():
    errors = []
    assets = {}

    try:
        assets["rf_model"] = load_rf_model()
    except Exception as e:
        errors.append(f"RF model gagal dimuat: {e}")

    try:
        assets["cnn_model"] = load_cnn_model()
    except Exception as e:
        errors.append(f"CNN model gagal dimuat: {e}")

    try:
        assets["preprocessor"] = load_preprocessor()
    except Exception as e:
        errors.append(f"Preprocessor gagal dimuat: {e}")

    try:
        assets["label_encoder"] = load_label_encoder()
    except Exception as e:
        errors.append(f"Label encoder gagal dimuat: {e}")

    assets["class_names"] = load_json_file(CLASS_NAMES_PATH, default=[])
    assets["feature_columns"] = load_json_file(FEATURE_COLUMNS_PATH, default=[])
    assets["metrics"] = load_json_file(METRICS_PATH, default={})

    if not assets["class_names"] and "label_encoder" in assets:
        try:
            assets["class_names"] = assets["label_encoder"].classes_.tolist()
        except Exception:
            pass

    return assets, errors


def try_read_delimited(file_bytes: bytes, sep: str, header):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), sep=sep, header=header)


def read_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    for sep in [",", r"\\s*,\\s*"]:
        try:
            df = try_read_delimited(file_bytes, sep=sep, header=0)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    for sep in [",", r"\\s*,\\s*"]:
        try:
            df = try_read_delimited(file_bytes, sep=sep, header=None)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    for header in [0, None]:
        try:
            df = try_read_delimited(file_bytes, sep=r"\\s+", header=header)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    raise ValueError(f"Gagal membaca file: {uploaded_file.name}")


def coerce_nsl_kdd_raw(df: pd.DataFrame):
    if set(RAW_FEATURE_COLUMNS).issubset(df.columns):
        raw_df = df[RAW_FEATURE_COLUMNS].copy()
        return raw_df, "raw_features_with_header"

    if all(isinstance(c, int) for c in df.columns):
        n_cols = df.shape[1]
        if n_cols >= 43:
            tmp = df.iloc[:, :43].copy()
            tmp.columns = NSL_KDD_COLUMNS
            return tmp[RAW_FEATURE_COLUMNS].copy(), "raw_43_no_header"
        if n_cols == 42:
            tmp = df.iloc[:, :42].copy()
            tmp.columns = NSL_KDD_COLUMNS[:42]
            return tmp[RAW_FEATURE_COLUMNS].copy(), "raw_42_no_header"
        if n_cols == 41:
            tmp = df.iloc[:, :41].copy()
            tmp.columns = RAW_FEATURE_COLUMNS
            return tmp.copy(), "raw_41_no_header"

    return df.copy(), "encoded_or_unknown"


def preprocess_for_models(input_df: pd.DataFrame, preprocessor, expected_feature_columns):
    raw_or_encoded_df, mode = coerce_nsl_kdd_raw(input_df)

    if mode.startswith("raw_") or mode == "raw_features_with_header":
        X_transformed = preprocessor.transform(raw_or_encoded_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_transformed = np.asarray(X_transformed, dtype=np.float32)
        display_df = raw_or_encoded_df.copy()
        return X_transformed, display_df, "raw"

    if not expected_feature_columns:
        raise ValueError("feature_columns.json tidak ditemukan, padahal dibutuhkan untuk mode encoded input.")

    aligned = raw_or_encoded_df.copy()

    for col in expected_feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0

    aligned = aligned[expected_feature_columns]

    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0)

    X_transformed = aligned.to_numpy(dtype=np.float32)
    return X_transformed, aligned.copy(), "encoded"


def predict_with_rf(model, X: np.ndarray, class_names):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        pred = model.predict(X)
        n_classes = len(class_names)
        proba = np.zeros((len(pred), n_classes), dtype=np.float32)
        for i, p in enumerate(pred):
            proba[i, int(p)] = 1.0
    return proba, np.argmax(proba, axis=1)


def predict_with_cnn(model, X: np.ndarray):
    X_cnn = np.expand_dims(X, axis=-1)
    proba = model.predict(X_cnn, verbose=0)

    if proba.ndim == 1:
        proba = np.expand_dims(proba, axis=1)

    if proba.shape[1] == 1:
        p1 = proba[:, 0]
        proba = np.vstack([1 - p1, p1]).T

    pred_idx = np.argmax(proba, axis=1)
    return proba, pred_idx


def decode_labels(indices: np.ndarray, class_names, label_encoder=None):
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform(indices.astype(int)).tolist()
        except Exception:
            pass
    return [class_names[int(i)] for i in indices]


def build_probability_columns(proba: np.ndarray, class_names, prefix: str):
    cols = {}
    n_classes = min(proba.shape[1], len(class_names))
    for i in range(n_classes):
        cols[f"{prefix}_proba_{class_names[i]}"] = np.round(proba[:, i], 6)
    return pd.DataFrame(cols)


def render_status_card(title: str, ok: bool, path_text: str):
    if ok:
        st.success(f"**{title}**\\n\\n`{path_text}`")
    else:
        st.error(f"**{title}**\\n\\n`{path_text}`")


assets, load_errors = load_all_assets()

with st.sidebar:
    st.header("Model & Artifacts Status")

    rf_ok, rf_path_text = path_status(RF_MODEL_PATH)
    cnn_ok, cnn_path_text = path_status(CNN_MODEL_PATH)
    pre_ok, pre_path_text = path_status(PREPROCESSOR_PATH)
    le_ok, le_path_text = path_status(LABEL_ENCODER_PATH)

    render_status_card("RF Model", rf_ok, rf_path_text)
    render_status_card("CNN Model", cnn_ok, cnn_path_text)
    render_status_card("Preprocessor", pre_ok, pre_path_text)
    render_status_card("Label Encoder", le_ok, le_path_text)

    if CLASS_NAMES_PATH.exists():
        st.success(f"**Class Names**\\n\\n`{CLASS_NAMES_PATH.relative_to(BASE_DIR)}`")
    else:
        st.warning("**Class Names** tidak ditemukan. Akan fallback ke label encoder jika tersedia.")

    if FEATURE_COLUMNS_PATH.exists():
        st.success(f"**Feature Columns**\\n\\n`{FEATURE_COLUMNS_PATH.relative_to(BASE_DIR)}`")
    else:
        st.warning("**Feature Columns** tidak ditemukan. Mode encoded input mungkin gagal.")

st.title("🛡️ NIDS Inference App")
st.caption("Inferensi multiclass NSL-KDD menggunakan Random Forest dan 1D-CNN.")

with st.expander("Struktur repo yang diharapkan", expanded=False):
    st.code(
        ".\\n├── app.py\\n├── requirements.txt\\n├── models/\\n│   ├── rf_model.joblib\\n│   └── cnn_model.keras\\n└── artifacts/\\n    ├── preprocessor.joblib\\n    ├── label_encoder.joblib\\n    ├── class_names.json\\n    ├── feature_columns.json\\n    └── metrics_summary.json\\n",
        language="text",
    )
    st.markdown(
        \"\"\"
- Upload yang paling aman adalah **file raw NSL-KDD** (`KDDTest+.txt` atau CSV dengan 41/42/43 kolom).
- App juga bisa menerima **CSV encoded** jika kolomnya sama dengan `feature_columns.json`.
- File `preprocessor.joblib` harus berasal dari pipeline training yang sama dengan model.
\"\"\"
    )

if load_errors:
    st.error("Ada aset yang gagal dimuat.")
    for err in load_errors:
        st.write(f"- {err}")

required_ready = all(
    key in assets for key in ["rf_model", "cnn_model", "preprocessor", "label_encoder"]
) and len(assets.get("class_names", [])) > 0

if not required_ready:
    st.stop()

metrics = assets.get("metrics", {})
if metrics:
    with st.expander("Ringkasan metrik training", expanded=False):
        st.json(metrics)

uploaded_file = st.file_uploader(
    "Upload file uji",
    type=["txt", "csv"],
    help="Gunakan KDDTest+.txt atau CSV dengan format yang sama seperti data training.",
)

if uploaded_file is None:
    st.info("Silakan upload file untuk mulai inferensi.")
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"Gagal membaca file upload: {e}")
    st.stop()

st.subheader("Preview input")
st.write(f"Shape file upload: **{raw_df.shape[0]} baris x {raw_df.shape[1]} kolom**")
st.dataframe(raw_df.head(20), use_container_width=True)

try:
    X, display_features_df, input_mode = preprocess_for_models(
        raw_df,
        assets["preprocessor"],
        assets.get("feature_columns", []),
    )
except Exception as e:
    st.error(f"Gagal preprocessing input: {e}")
    st.stop()

st.success(f"Mode input terdeteksi: **{input_mode}**")
st.write(f"Bentuk fitur untuk model: **{X.shape}**")

class_names = assets["class_names"]
label_encoder = assets["label_encoder"]

try:
    rf_proba, rf_idx = predict_with_rf(assets["rf_model"], X, class_names)
    rf_labels = decode_labels(rf_idx, class_names, label_encoder)
    rf_conf = np.max(rf_proba, axis=1)
except Exception as e:
    st.error(f"Gagal inferensi Random Forest: {e}")
    st.stop()

try:
    cnn_proba, cnn_idx = predict_with_cnn(assets["cnn_model"], X)
    cnn_labels = decode_labels(cnn_idx, class_names, label_encoder)
    cnn_conf = np.max(cnn_proba, axis=1)
except Exception as e:
    st.error(f"Gagal inferensi CNN: {e}")
    st.stop()

ensemble_proba = (rf_proba + cnn_proba) / 2.0
ensemble_idx = np.argmax(ensemble_proba, axis=1)
ensemble_labels = decode_labels(ensemble_idx, class_names, label_encoder)
ensemble_conf = np.max(ensemble_proba, axis=1)

result_df = display_features_df.copy()
result_df["rf_prediction"] = rf_labels
result_df["rf_confidence"] = np.round(rf_conf, 6)
result_df["cnn_prediction"] = cnn_labels
result_df["cnn_confidence"] = np.round(cnn_conf, 6)
result_df["ensemble_prediction"] = ensemble_labels
result_df["ensemble_confidence"] = np.round(ensemble_conf, 6)

rf_prob_df = build_probability_columns(rf_proba, class_names, "rf")
cnn_prob_df = build_probability_columns(cnn_proba, class_names, "cnn")
ens_prob_df = build_probability_columns(ensemble_proba, class_names, "ensemble")

full_result_df = pd.concat([result_df.reset_index(drop=True), rf_prob_df, cnn_prob_df, ens_prob_df], axis=1)

st.subheader("Hasil prediksi")
st.dataframe(full_result_df, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah sampel", len(full_result_df))
with col2:
    st.metric("Prediksi unik RF", full_result_df["rf_prediction"].nunique())
with col3:
    st.metric("Prediksi unik Ensemble", full_result_df["ensemble_prediction"].nunique())

st.subheader("Distribusi prediksi ensemble")
ensemble_counts = full_result_df["ensemble_prediction"].value_counts().rename_axis("class").reset_index(name="count")
st.dataframe(ensemble_counts, use_container_width=True)

csv_bytes = full_result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download hasil prediksi CSV",
    data=csv_bytes,
    file_name="nids_inference_results.csv",
    mime="text/csv",
)
"""

path = Path("/mnt/data/app_updated.py")
path.write_text(code, encoding="utf-8")
print(f"Saved to {path}")
