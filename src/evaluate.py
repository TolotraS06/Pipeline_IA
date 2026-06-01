"""Évaluation des modèles enregistrés.

Calcule des métriques de clustering (silhouette, Davies-Bouldin) et
évalue le RandomForest par rapport aux labels KMeans (pseudo-ground-truth).
Les métriques sont écrites dans `models/evaluate_metrics.json`.
"""

import json
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, classification_report

BASE = Path(__file__).parents[1]
DATA = BASE / "data" / "fruits_clean.csv"
MODEL = BASE / "models" / "model.joblib"
METRICS_OUT = BASE / "models" / "evaluate_metrics.json"


def evaluate():
    """Charge l'artefact `models/model.joblib` et calcule des métriques.

    Retourne un dict contenant les métriques calculées.
    """
    # Load cleaned data and model artefacts
    df = pd.read_csv(DATA)
    artefacts = joblib.load(MODEL)
    kmeans = artefacts.get('kmeans')
    rf = artefacts.get('rf')
    scaler = artefacts.get('scaler')

    # Transform features with the stored scaler before evaluating
    X = df[["x","y"]].values
    Xs = scaler.transform(X)

    # Evaluate KMeans clustering quality on the full dataset
    k_labels = kmeans.predict(Xs)
    sil = silhouette_score(Xs, k_labels)
    db = davies_bouldin_score(Xs, k_labels)
    print(f"Silhouette score (KMeans): {sil:.4f}")
    print(f"Davies-Bouldin score (KMeans): {db:.4f}")

    # Evaluate how well the RandomForest reproduces KMeans labels
    rf_preds = rf.predict(Xs)
    acc = accuracy_score(k_labels, rf_preds)
    print(f"RandomForest accuracy vs KMeans labels: {acc:.4f}")
    print("Classification report:\n", classification_report(k_labels, rf_preds))

    # Persist metrics to JSON for later inspection or CI checks
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "rf_accuracy": float(acc),
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics



if __name__ == "__main__":
    evaluate()
