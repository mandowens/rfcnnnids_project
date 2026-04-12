import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

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

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "apache2", "processtable", "udpstorm"
}

PROBE_ATTACKS = {
    "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"
}

R2L_ATTACKS = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
    "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
    "httptunnel", "sendmail", "named"
}

U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm"
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


def normalize_true_label(label_value):
    value = str(label_value).strip().lower()

    if value in {"normal", "dos", "probe", "r2l", "u2r"}:
        return value
    if value in DOS_ATTACKS:
        return "dos"
    if value in PROBE_ATTACKS:
        return "probe"
    if value in R2L_ATTACKS:
        return "r2l"
    if value in U2R_ATTACKS:
        return "u2r"

    return value


def normalize_uploaded_dataframe(df: pd.DataFrame):
    original_cols = list(df.columns)

    if all(isinstance(c, int) for c in original_cols):
        if df.shape[1] == 41:
            df = df.copy()
            df.columns = RAW_FEATURE_COLUMNS
            return df, None, None

        if df.shape[1] == 42:
            df = df.copy()
            df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL]
            y_true = df[LABEL_COL].astype(str).map(normalize_true_label).tolist()
            return df.drop(columns=[LABEL_COL]), y_true, LABEL_COL

        if df.shape[1] == 43:
            df = df.copy()
            df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL, DIFFICULTY_COL]
            y_true = df[LABEL_COL].astype(str).map(normalize_true_label).tolist()
            return df.drop(columns=[LABEL_COL, DIFFICULTY_COL]), y_true, LABEL_COL

        raise ValueError(
            f"Jumlah kolom {df.shape[1]} tidak cocok untuk format NSL-KDD. Gunakan 41, 42, atau 43 kolom."
        )

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    if all(col in lower_map for col in [c.lower() for c in RAW_FEATURE_COLUMNS]):
        renamed = df.rename(columns={lower_map[c.lower()]: c for c in RAW_FEATURE_COLUMNS})

        y_true = None
        y_source = None
        for candidate in ["label", "attack_class", "target", "ground_truth", "y_true"]:
            if candidate in lower_map:
                original_label_col = lower_map[candidate]
                y_true = df[original_label_col].astype(str).map(normalize_true_label).tolist()
                y_source = original_label_col
                break

        return renamed[RAW_FEATURE_COLUMNS], y_true, y_source

    if df.shape[1] == 41:
        df = df.copy()
        df.columns = RAW_FEATURE_COLUMNS
        return df, None, None

    if df.shape[1] == 42:
        df = df.copy()
        df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL]
        y_true = df[LABEL_COL].astype(str).map(normalize_true_label).tolist()
        return df.drop(columns=[LABEL_COL]), y_true, LABEL_COL

    if df.shape[1] == 43:
        df = df.copy()
        df.columns = RAW_FEATURE_COLUMNS + [LABEL_COL, DIFFICULTY_COL]
        y_true = df[LABEL_COL].astype(str).map(normalize_true_label).tolist()
        return df.drop(columns=[LABEL_COL, DIFFICULTY_COL]), y_true, LABEL_COL

    raise ValueError(
        "Format file belum dikenali. Upload CSV dengan 41 fitur mentah NSL-KDD atau file KDDTest+/KDDTrain+ asli (42/43 kolom)."
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


def compute_metrics(y_true, y_pred, proba, class_names):
    if y_true is None:
        return None

    y_true = [normalize_true_label(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]

    valid_mask = [label in class_names for label in y_true]
    if sum(valid_mask) == 0:
        return None

    y_true_f = np.array([y_true[i] for i, keep in enumerate(valid_mask) if keep])
    y_pred_f = np.array([y_pred[i] for i, keep in enumerate(valid_mask) if keep])
    proba_f = np.asarray([proba[i] for i, keep in enumerate(valid_mask) if keep])

    accuracy = accuracy_score(y_true_f, y_pred_f)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=class_names, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_f, y_pred_f, labels=class_names, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true_f, y_pred_f, labels=class_names)
    report_df = pd.DataFrame(
        classification_report(
            y_true_f,
            y_pred_f,
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    roc_auc_macro = None
    roc_auc_weighted = None
    pr_auc_macro = None
    pr_auc_weighted = None
    roc_curves = {}
    pr_curves = {}
    auc_error = None

    try:
        if len(class_names) == 2:
            positive_class = class_names[1]
            y_bin = (y_true_f == positive_class).astype(int)
            y_score = proba_f[:, 1]

            roc_auc_macro = roc_auc_score(y_bin, y_score)
            pr_auc_macro = average_precision_score(y_bin, y_score)

            fpr, tpr, _ = roc_curve(y_bin, y_score)
            precision_vals, recall_vals, _ = precision_recall_curve(y_bin, y_score)

            roc_curves[positive_class] = (fpr, tpr)
            pr_curves[positive_class] = (recall_vals, precision_vals)
        else:
            y_true_bin = label_binarize(y_true_f, classes=class_names)

            roc_auc_macro = roc_auc_score(
                y_true_bin, proba_f, multi_class="ovr", average="macro"
            )
            roc_auc_weighted = roc_auc_score(
                y_true_bin, proba_f, multi_class="ovr", average="weighted"
            )
            pr_auc_macro = average_precision_score(
                y_true_bin, proba_f, average="macro"
            )
            pr_auc_weighted = average_precision_score(
                y_true_bin, proba_f, average="weighted"
            )

            for idx, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, idx], proba_f[:, idx])
                precision_vals, recall_vals, _ = precision_recall_curve(
                    y_true_bin[:, idx], proba_f[:, idx]
                )
                roc_curves[class_name] = (fpr, tpr)
                pr_curves[class_name] = (recall_vals, precision_vals)
    except Exception as e:
        auc_error = str(e)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_weighted": pr_auc_weighted,
        "confusion_matrix": cm,
        "report_df": report_df,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "auc_error": auc_error,
        "n_samples": len(y_true_f),
    }


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    return fig


def plot_roc_curves(curves, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=class_name)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr_curves(curves, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_name, (recall_vals, precision_vals) in curves.items():
        ax.plot(recall_vals, precision_vals, label=class_name)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    return fig


def render_evaluation_block(title, evaluation, class_names):
    st.subheader(title)

    if evaluation is None:
        st.warning("Ground truth label tidak tersedia atau tidak cocok, sehingga evaluasi tidak dapat dihitung.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{evaluation['accuracy']:.4f}")
    c2.metric("Macro F1", f"{evaluation['f1_macro']:.4f}")
    c3.metric("Weighted F1", f"{evaluation['f1_weighted']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Macro Precision", f"{evaluation['precision_macro']:.4f}")
    c5.metric("Macro Recall", f"{evaluation['recall_macro']:.4f}")
    c6.metric("Samples Evaluated", evaluation["n_samples"])

    c7, c8 = st.columns(2)
    c7.metric(
        "ROC-AUC Macro",
        "-" if evaluation["roc_auc_macro"] is None else f"{evaluation['roc_auc_macro']:.4f}",
    )
    c8.metric(
        "PR-AUC Macro",
        "-" if evaluation["pr_auc_macro"] is None else f"{evaluation['pr_auc_macro']:.4f}",
    )

    c9, c10 = st.columns(2)
    c9.metric(
        "ROC-AUC Weighted",
        "-" if evaluation["roc_auc_weighted"] is None else f"{evaluation['roc_auc_weighted']:.4f}",
    )
    c10.metric(
        "PR-AUC Weighted",
        "-" if evaluation["pr_auc_weighted"] is None else f"{evaluation['pr_auc_weighted']:.4f}",
    )

    if evaluation["auc_error"]:
        st.warning(f"AUC tidak dapat dihitung penuh: {evaluation['auc_error']}")

    st.markdown("#### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(evaluation["confusion_matrix"], class_names, f"Confusion Matrix - {title}"))

    if evaluation["roc_curves"]:
        st.markdown("#### ROC Curve")
        st.pyplot(plot_roc_curves(evaluation["roc_curves"], f"ROC Curve - {title}"))

    if evaluation["pr_curves"]:
        st.markdown("#### Precision-Recall Curve")
        st.pyplot(plot_pr_curves(evaluation["pr_curves"], f"PR Curve - {title}"))

    st.markdown("#### Classification Report")
    st.dataframe(evaluation["report_df"], use_container_width=True)


def main():
    st.title("NSL-KDD Multiclass NIDS")
    st.caption("Deploy uji-coba untuk klasifikasi multiclass dengan Random Forest dan 1D-CNN.")

    try:
        rf_model, cnn_model, preprocessor, label_encoder, class_names, metrics, feature_columns = load_assets()
    except Exception as e:
        st.error(
            "Gagal memuat model/artifact. Pastikan Anda sudah menjalankan script training dan menaruh file hasilnya di folder `models/` dan `artifacts/`.\n\n"
            f"Detail error: {e}"
        )
        st.stop()

    model_choice = st.sidebar.selectbox(
        "Pilih mode inferensi",
        ["Compare: RF vs CNN", "Random Forest", "CNN"],
    )

    tab1, tab2, tab3 = st.tabs(["Prediksi", "Info Model", "Skema Input"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload CSV untuk prediksi",
            type=["csv", "txt"],
            help="Bisa berupa file KDDTest+/KDDTrain+ asli, atau CSV 41 fitur mentah NSL-KDD."
        )

        st.info(
            "Untuk uji cepat, upload file `KDDTest+.txt` atau `KDDTrain+.txt`. App akan otomatis membaca 41 fitur mentah NSL-KDD."
        )

        if uploaded is not None:
            try:
                try:
                    df_input = pd.read_csv(uploaded, header=None)
                except Exception:
                    uploaded.seek(0)
                    df_input = pd.read_csv(uploaded)

                df_norm, y_true, y_source = normalize_uploaded_dataframe(df_input)
                st.success(f"Berhasil membaca file: {df_norm.shape[0]} baris, {df_norm.shape[1]} fitur.")
                st.dataframe(df_norm.head(20), use_container_width=True)

                if y_true is not None:
                    st.info(f"Ground truth terdeteksi dari kolom: {y_source}")
                else:
                    st.warning("Ground truth label tidak terdeteksi. Confusion matrix / ROC-AUC / PR-AUC tidak akan tampil.")

                if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
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

                    if model_choice == "Random Forest":
                        rf_cols = [
                            *RAW_FEATURE_COLUMNS,
                            "rf_prediction",
                            "rf_confidence",
                            *[f"rf_proba_{name}" for name in class_names],
                        ]
                        st.dataframe(out_df[rf_cols], use_container_width=True)
                    elif model_choice == "CNN":
                        cnn_cols = [
                            *RAW_FEATURE_COLUMNS,
                            "cnn_prediction",
                            "cnn_confidence",
                            *[f"cnn_proba_{name}" for name in class_names],
                        ]
                        st.dataframe(out_df[cnn_cols], use_container_width=True)
                    else:
                        st.dataframe(out_df, use_container_width=True)

                    if model_choice == "Random Forest":
                        st.subheader("Distribusi Prediksi RF")
                        dist = out_df["rf_prediction"].value_counts().rename_axis("class").reset_index(name="count")
                        st.bar_chart(dist.set_index("class"))
                    elif model_choice == "CNN":
                        st.subheader("Distribusi Prediksi CNN")
                        dist = out_df["cnn_prediction"].value_counts().rename_axis("class").reset_index(name="count")
                        st.bar_chart(dist.set_index("class"))
                    else:
                        st.subheader("Distribusi Prediksi Ensemble")
                        dist = out_df["ensemble_prediction"].value_counts().rename_axis("class").reset_index(name="count")
                        st.bar_chart(dist.set_index("class"))

                    if y_true is not None:
                        st.markdown("## Evaluasi")

                        if model_choice == "Random Forest":
                            evaluation = compute_metrics(y_true, out_df["rf_prediction"].tolist(), rf_proba, class_names)
                            render_evaluation_block("Random Forest", evaluation, class_names)

                        elif model_choice == "CNN":
                            evaluation = compute_metrics(y_true, out_df["cnn_prediction"].tolist(), cnn_proba, class_names)
                            render_evaluation_block("CNN", evaluation, class_names)

                        else:
                            eval_tabs = st.tabs(["Random Forest", "CNN", "Ensemble"])

                            with eval_tabs[0]:
                                evaluation = compute_metrics(y_true, out_df["rf_prediction"].tolist(), rf_proba, class_names)
                                render_evaluation_block("Random Forest", evaluation, class_names)

                            with eval_tabs[1]:
                                evaluation = compute_metrics(y_true, out_df["cnn_prediction"].tolist(), cnn_proba, class_names)
                                render_evaluation_block("CNN", evaluation, class_names)

                            with eval_tabs[2]:
                                evaluation = compute_metrics(y_true, out_df["ensemble_prediction"].tolist(), ens_proba, class_names)
                                render_evaluation_block("Ensemble", evaluation, class_names)

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download hasil prediksi CSV",
                        data=csv_bytes,
                        file_name="nslkdd_multiclass_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Gagal memproses file: {e}")
                st.exception(e)

    with tab2:
        show_metrics(metrics)
        st.write("**Kelas target:**", ", ".join(class_names))

        if feature_columns:
            with st.expander("Jumlah fitur setelah preprocessing"):
                st.write(f"{len(feature_columns)} fitur hasil one-hot encoding + scaling.")

        st.markdown(
            "- Random Forest multiclass\n"
            "- 1D-CNN multiclass\n"
            "- Ensemble sederhana berbasis rata-rata probabilitas RF dan CNN"
        )

    with tab3:
        st.write("**41 fitur mentah NSL-KDD yang diharapkan app:**")
        st.code(", ".join(RAW_FEATURE_COLUMNS))
        st.write(
            "App juga bisa membaca file NSL-KDD mentah tanpa header dengan 42 atau 43 kolom, lalu otomatis menghapus kolom label dan difficulty."
        )


if __name__ == "__main__":
    main()
