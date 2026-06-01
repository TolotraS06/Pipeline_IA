"""Module d'entraînement.

Ce fichier exécute la sélection du nombre de clusters `k` via KMeans
en se basant sur le score de silhouette puis entraîne un RandomForest
pour prédire les étiquettes de cluster (utile pour le service).

Fonction principale exposée: `select_k_and_train(df, ...)`.
Les artefacts produits sont sauvegardés dans `models/`.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn


ROOT = Path(__file__).parents[1]
BASE = ROOT
DATA_IN = BASE / "data" / "fruits_clean.csv"
MODELS_DIR = BASE / "Deploiement_ML_DL/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODELS_DIR / "model.joblib"
FIG_DIR = MODELS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def select_k_and_train(df: pd.DataFrame, k_min=2, k_max=10, random_state=42):
    """Sélectionne `k` et entraîne KMeans + RandomForest.

    Args:
        df: DataFrame contenant les colonnes `x` et `y` (numériques).
        k_min: borne minimale pour la recherche de k.
        k_max: borne maximale pour la recherche de k.
        random_state: graine pour reproductibilité.

    Retourne:
        (final_kmeans, rf, scaler) : objets entraînés.
    """
    # split data for training classifier evaluation
    from sklearn.model_selection import train_test_split

    # Extract features and split into a small train/test partition.
    # The RF classifier will be trained to reproduce KMeans labels, so we
    # need a held-out set to estimate the classifier's ability to generalize.
    X = df[["x", "y"]].values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=random_state)

    # Standardize features prior to KMeans/RF training (important for distance-based methods)
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    inertias = []
    silhouettes = []
    Ks = list(range(k_min, k_max + 1))

    # Scan candidate k values and compute inertia + silhouette to choose a good k.
    for k in Ks:
        model = KMeans(n_clusters=k, random_state=random_state)
        labels = model.fit_predict(Xs_train)
        inertias.append(model.inertia_)
        # Silhouette requires at least 2 clusters
        if k > 1:
            silhouettes.append(silhouette_score(Xs_train, labels))
        else:
            silhouettes.append(np.nan)

    # Save plots
    plt.figure()
    plt.plot(Ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow")
    plt.savefig(FIG_DIR / "elbow.png")

    plt.figure()
    plt.plot(Ks, silhouettes, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette")
    plt.savefig(FIG_DIR / "silhouette.png")

    # choose k by max silhouette
    best_idx = int(np.nanargmax(silhouettes))
    best_k = Ks[best_idx]


    # Refit KMeans on the training set with the chosen k and label both sets
    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    final_labels_train = final_kmeans.fit_predict(Xs_train)
    final_labels_test = final_kmeans.predict(Xs_test)

    # Train a RandomForest classifier that learns to map features -> KMeans label.
    # This allows fast prediction in the API without re-running KMeans.
    rf = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf.fit(Xs_train, final_labels_train)

    # Evaluate classifier on test set (compare to KMeans labels)
    rf_preds = rf.predict(Xs_test)
    acc = accuracy_score(final_labels_test, rf_preds)
    cls_report = classification_report(final_labels_test, rf_preds)

    # Save model, kmeans, rf and scaler together
    # Save both models and the scaler together as a single artifact dict.
    artefact = {"kmeans": final_kmeans, "rf": rf, "scaler": scaler}
    joblib.dump(artefact, MODEL_OUT)

    # Save clustered data (train+test concatenated)
    # Persist a CSV with the original coordinates and assigned cluster
    df_out = pd.DataFrame(np.vstack([X_train, X_test]), columns=["x", "y"])
    df_out["cluster_kmeans"] = np.concatenate([final_labels_train, final_labels_test])
    df_out.to_csv(MODELS_DIR / "fruits_clustered.csv", index=False)

    # Save scatter using original (unscaled) coordinates colored by kmeans cluster
    plt.figure(figsize=(6,6))
    allX = np.vstack([X_train, X_test])
    all_labels = np.concatenate([final_labels_train, final_labels_test])
    plt.scatter(allX[:,0], allX[:,1], c=all_labels, cmap='tab10')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Clusters k={best_k}')
    plt.savefig(FIG_DIR / "clusters.png")

    # Attempt to log parameters, metrics and artifacts to MLflow if available.
    try:
        mlflow.start_run()
        mlflow.log_param("best_k", int(best_k))
        mlflow.log_metric("silhouette", float(silhouettes[best_idx]))
        mlflow.log_metric("rf_accuracy", float(acc))
        mlflow.log_artifact(str(MODEL_OUT))
        mlflow.log_artifact(str(FIG_DIR / "elbow.png"))
        mlflow.log_artifact(str(FIG_DIR / "silhouette.png"))
        mlflow.log_artifact(str(FIG_DIR / "clusters.png"))
        mlflow.sklearn.log_model(final_kmeans, "kmeans_model")
        mlflow.sklearn.log_model(rf, "random_forest_classifier")
        mlflow.end_run()
    except Exception as e:
        print("MLflow logging skipped (error):", e)

    print(f"Best k (by silhouette): {best_k}")
    print(f"RandomForest accuracy vs KMeans on test set: {acc:.4f}")
    print(f"Model saved to: {MODEL_OUT}")
    return final_kmeans, rf, scaler


if __name__ == "__main__":
    df = pd.read_csv(DATA_IN)
    select_k_and_train(df)
