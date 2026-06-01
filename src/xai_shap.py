"""Utilities pour expliquer le classificateur via SHAP.

Ce module génère un graphique de résumé SHAP (ou un fallback barplot)
et l'enregistre dans `models/figures/shap_summary.png`.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).parents[1]
MODEL_PATH = BASE / "Deploiement_ML_DL/models" / "model.joblib"
DATA_PATH = BASE / "data" / "fruits_clean.csv"
FIG_DIR = BASE / "Deploiement_ML_DL/models" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_shap(sample_size=200):
    """Calcule et sauvegarde un résumé SHAP pour le RandomForest.

    Args:
        sample_size: nombre d'échantillons à utiliser pour l'explication.
    """
    # Ensure model artefact exists
    if not MODEL_PATH.exists():
        print("Model not found at:", MODEL_PATH)
        return
    arte = joblib.load(MODEL_PATH)
    rf = arte.get("rf")
    scaler = arte.get("scaler")
    if rf is None or scaler is None:
        print("rf or scaler missing in artefact. Found keys:", list(arte.keys()))
        return

    # Load data and optionally sample for quicker SHAP computation
    df = pd.read_csv(DATA_PATH)
    X = df[["x", "y"]].values
    if sample_size and sample_size < len(X):
        idx = np.random.choice(len(X), sample_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    Xs_scaled = scaler.transform(Xs)

    # SHAP may be optional in environments; fail gracefully if missing
    try:
        import shap
    except Exception as e:
        print("shap not installed:", e)
        return

    # Explain model predictions using a TreeExplainer (RandomForest)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xs_scaled)
    import numpy as _np

    # Normalize SHAP output for the summary plot
    shap_values_list = None
    if isinstance(shap_values, list):
        shap_values_list = shap_values
    elif isinstance(shap_values, _np.ndarray) and shap_values.ndim == 3:
        # TreeExplainer multi-class output can be (n_samples, n_features, n_classes)
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, _np.ndarray):
        shap_values_list = [shap_values]
    else:
        shap_values_list = [shap_values]

    mean_abs = None
    try:
        # Compute robust mean absolute SHAP values
        mean_abs = _np.mean(_np.abs(_np.concatenate([_np.expand_dims(sv, axis=0) for sv in shap_values_list], axis=0)), axis=(0, 1))
    except Exception:
        mean_abs = None

    plt.figure(figsize=(6, 6))
    try:
        if len(shap_values_list) > 1:
            shap.summary_plot(shap_values_list, Xs, show=False)
        else:
            shap.summary_plot(shap_values_list[0], Xs, show=False)
        out = FIG_DIR / 'shap_summary.png'
        plt.savefig(out, bbox_inches='tight')
        print('SHAP summary saved to', out)
    except Exception as e:
        print('Error creating SHAP plot:', e)
        if mean_abs is None:
            print('Could not compute mean_abs shap; exiting')
            return
        features = ["x", "y"]
        plt.clf()
        plt.bar(features, mean_abs)
        plt.ylabel('mean |SHAP value|')
        plt.title('SHAP mean abs importance')
        out = FIG_DIR / 'shap_summary.png'
        plt.savefig(out, bbox_inches='tight')
        print('SHAP summary saved to', out)


if __name__ == '__main__':
    run_shap()
