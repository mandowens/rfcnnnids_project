import json
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models

RANDOM_STATE = 42
tf.keras.utils.set_random_seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RAW_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm",
    "processtable", "mailbomb"
}
PROBE_ATTACKS = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L_ATTACKS = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient",
    "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
    "named"
}
U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"
}

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def map_attack_to_class(label: str) -> str:
    label = str(label).strip().lower()
    if label == "normal":
        return "normal"
    if label in DOS_ATTACKS:
        return "dos"
    if label in PROBE_ATTACKS:
        return "probe"
    if label in R2L_ATTACKS:
        return "r2l"
    if label in U2R_ATTACKS:
        return "u2r"
    return "unknown"

def read_nslkdd_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df["attack_class"] = df["label"].map(map_attack_to_class)
    unknown_count = (df["attack_class"] == "unknown").sum()
    if unknown_count:
        print(f"[WARN] {unknown_count} baris memiliki label yang tidak dikenali dan akan dihapus.")
        df = df[df["attack_class"] != "unknown"].copy()
    return df

def cast_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in RAW_COLUMNS[:-2] if c not in categorical_cols]

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def load_data(train_path: Path, test_path: Path):
    train_df = cast_feature_types(read_nslkdd_file(train_path))
    test_df = cast_feature_types(read_nslkdd_file(test_path))

    X_train = train_df[RAW_COLUMNS[:-2]]
    X_test = test_df[RAW_COLUMNS[:-2]]
    y_train = train_df["attack_class"]
    y_test = test_df["attack_class"]
    return X_train, X_test, y_train, y_test

def build_preprocessor():
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in RAW_COLUMNS[:-2] if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_onehot_encoder(), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor

def build_cnn_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim, 1)),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),

            layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.30),

            layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_and_summarize(model_name, y_true_idx, y_pred_idx, label_encoder):
    class_names = list(label_encoder.classes_)
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx).tolist()
    summary = {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "macro_f1": float(f1_score(y_true_idx, y_pred_idx, average="macro")),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    print(f"\n=== {model_name} ===")
    print(json.dumps({"accuracy": summary["accuracy"], "macro_f1": summary["macro_f1"]}, indent=2))
    return summary

def main():
    parser = argparse.ArgumentParser(description="Train Random Forest and 1D-CNN on NSL-KDD multiclass.")
    parser.add_argument("--train", type=str, default=str(DATA_DIR / "KDDTrain+.txt"))
    parser.add_argument("--test", type=str, default=str(DATA_DIR / "KDDTest+.txt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)

    if not train_path.exists():
        raise FileNotFoundError(f"File train tidak ditemukan: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"File test tidak ditemukan: {test_path}")

    ensure_dirs()

    print("[INFO] Loading NSL-KDD data ...")
    X_train_df, X_test_df, y_train, y_test = load_data(train_path, test_path)

    label_encoder = LabelEncoder()
    y_train_idx = label_encoder.fit_transform(y_train)
    y_test_idx = label_encoder.transform(y_test)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    print("[INFO] Fitting preprocessor ...")
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    try:
        feature_columns = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_columns = [f"feature_{i}" for i in range(X_train.shape[1])]

    print(f"[INFO] Shape after preprocessing: train={X_train.shape}, test={X_test.shape}")

    print("[INFO] Training Random Forest ...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_model.fit(X_train, y_train_idx)
    rf_pred_idx = rf_model.predict(X_test)

    print("[INFO] Training 1D-CNN ...")
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_idx),
        y=y_train_idx,
    )
    class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights_values)}

    cnn_model = build_cnn_model(input_dim=X_train.shape[1], num_classes=num_classes)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    cnn_model.fit(
        X_train_cnn,
        y_train_idx,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        class_weight=class_weight_dict,
        verbose=1,
    )

    cnn_proba = cnn_model.predict(X_test_cnn, verbose=0)
    cnn_pred_idx = np.argmax(cnn_proba, axis=1)

    metrics_summary = {
        "classes": class_names,
        "rf": evaluate_and_summarize("Random Forest", y_test_idx, rf_pred_idx, label_encoder),
        "cnn": evaluate_and_summarize("1D-CNN", y_test_idx, cnn_pred_idx, label_encoder),
    }

    print("[INFO] Saving artifacts ...")
    joblib.dump(rf_model, MODELS_DIR / "rf_model.joblib")
    cnn_model.save(MODELS_DIR / "cnn_model.keras")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.joblib")

    (ARTIFACTS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print("[DONE] Training selesai.")
    print(f"Saved RF model     : {MODELS_DIR / 'rf_model.joblib'}")
    print(f"Saved CNN model    : {MODELS_DIR / 'cnn_model.keras'}")
    print(f"Saved preprocessor : {ARTIFACTS_DIR / 'preprocessor.joblib'}")
    print(f"Saved label enc    : {ARTIFACTS_DIR / 'label_encoder.joblib'}")

if __name__ == "__main__":
    main()
