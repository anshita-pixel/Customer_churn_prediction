"""Train and compare churn models.  Usage:  python train.py
Expects data/churn.csv with a target column named Churn (Yes/No or 1/0)."""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED, DATA, TARGET = 42, "data/churn.csv", "Churn"
OUT = Path("models")
OUT.mkdir(exist_ok=True)

# ---------- 1. Load and clean ----------
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
y = df[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
X = df.drop(columns=[TARGET])
ok = y.notna()
X, y = X[ok].reset_index(drop=True), y[ok].astype(int).reset_index(drop=True)

# text columns that are really numbers (e.g. TotalCharges with blank strings)
for c in X.select_dtypes(exclude=["number", "bool"]):
    conv = pd.to_numeric(X[c].str.strip(), errors="coerce")
    if conv.notna().mean() > 0.9:
        X[c] = conv

# drop ID columns (they leak the answer / carry no signal)
ids = [c for c in X.columns
       if (c.lower().endswith("id") and X[c].nunique() > 0.9 * len(X))
       or (not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.95 * len(X))]
X = X.drop(columns=ids)
print("Dropped ID columns:", ids)

dup = X.assign(_y=y).duplicated()
X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)

num = X.select_dtypes("number").columns.tolist()  # bool/text columns are treated as categories
cat = [c for c in X.columns if c not in num]
X[cat] = X[cat].fillna("Unknown").astype(str)
print(f"Rows: {len(X)} | churn rate: {y.mean():.1%} | numeric: {num} | categorical: {cat}")

# ---------- 2. Split (stratified) and preprocess ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
])
Xtr = pre.fit_transform(X_train).astype("float32")   # fit on TRAIN only
Xte = pre.transform(X_test).astype("float32")
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

# ---------- 3. Models ----------
results = {}


def evaluate(name, proba, cv_auc=None):
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_roc_auc": cv_auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
    }
    print(f"{name:20s} acc={results[name]['accuracy']:.3f} recall={results[name]['recall']:.3f} "
          f"auc={results[name]['roc_auc']:.3f}")


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, model in models.items():
    cv = cross_val_score(Pipeline([("pre", clone(pre)), ("m", clone(model))]),
                         X_train, y_train, cv=skf, scoring="roc_auc").mean()
    model.fit(Xtr, y_train)
    evaluate(name, model.predict_proba(Xte)[:, 1], float(cv))
    joblib.dump(model, OUT / f"{name.lower().replace(' ', '_')}.joblib")

# ---------- 4. 1D CNN (needs tensorflow) ----------
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(SEED)
    cnn = keras.Sequential([
        layers.Input((Xtr.shape[1], 1)),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    cnn.compile("adam", "binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    w = len(y_train) / (2 * np.bincount(y_train))
    cnn.fit(Xtr[..., None], y_train.values, validation_split=0.2, epochs=60, batch_size=64,
            class_weight={0: w[0], 1: w[1]}, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                     patience=8, restore_best_weights=True)])
    evaluate("1D CNN", cnn.predict(Xte[..., None], verbose=0).ravel())
    cnn.save(OUT / "cnn.keras")
except ImportError:
    print("TensorFlow not installed -> skipping 1D CNN (pip install tensorflow)")

# ---------- 5. Save everything app needs ----------
joblib.dump(pre, OUT / "preprocessor.joblib")
np.save(OUT / "sample_X.npy", Xte[:300])
schema = {
    "columns": list(X.columns), "features": names, "target": TARGET,
    "numeric": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in num},
    "categorical": {c: sorted(X[c].unique().tolist()) for c in cat},
}
json.dump(schema, open(OUT / "schema.json", "w"), indent=2)
json.dump({"models": results,
           "info": {"rows": len(X), "churn_rate": float(y.mean()),
                    "n_train": len(X_train), "n_test": len(X_test)}},
          open(OUT / "metrics.json", "w"), indent=2)
print("\nDone. Saved models, schema.json and metrics.json in ./models")