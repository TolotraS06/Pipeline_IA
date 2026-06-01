"""Génération d'un rapport Evidently (data drift demo).

Ce module crée un rapport HTML basique. Actuellement il utilise
les mêmes données comme référence et données courantes pour démonstration.
"""

import pandas as pd
from pathlib import Path
from evidently.core.report import Report
from evidently.presets import DataDriftPreset

BASE = Path(__file__).parents[1]
DATA = BASE / "data" / "fruits_clean.csv"
OUT = BASE / "Deploiement_ML_DL/models" / "evidently_report.html"


def make_report():
    """Construit et sauvegarde un rapport Evidently en HTML.

    Le rapport utilise ici les mêmes données comme référence et actuel
    pour une démonstration locale. En production, fournir référence vs actuel.
    """
    # Load cleaned data; in a real workflow the reference/current datasets differ
    df = pd.read_csv(DATA)
    # Build a DataDrift preset report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=df, current_data=df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(OUT))
    print(f"Evidently report saved to: {OUT}")


if __name__ == "__main__":
    make_report()
