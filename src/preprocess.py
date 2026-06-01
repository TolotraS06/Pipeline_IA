"""Prétraitement et nettoyage des données brutes.

Ce module charge `data/fruits.csv`, nettoie les valeurs (conversion de
virgules décimales, suppression de NaN et doublons) et écrit
`data/fruits_clean.csv` lorsqu'il est exécuté directement.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parents[1]
DATA_IN = BASE / "data" / "fruits.csv"
DATA_OUT = BASE / "data" / "fruits_clean.csv"


def load_and_clean(input_path: Path = DATA_IN) -> pd.DataFrame:
    """Charge et nettoie le CSV d'entrée.

    - Force les colonnes en chaînes pour nettoyer les espaces
    - Remplace les virgules décimales par des points et convertit en numérique
    - Supprime les lignes invalides, les doublons et valeurs non-positives

    Retourne le DataFrame nettoyé.
    """
    # Read as strings first to robustly handle different decimal separators
    df = pd.read_csv(input_path, header=None, names=["x", "y"], dtype=str)

    # Remove leading/trailing whitespace which may break numeric parsing
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    # Replace comma decimals with dots and convert to numeric, invalid values become NaN
    df["x"] = pd.to_numeric(df["x"].str.replace(",", "."), errors="coerce")
    df["y"] = pd.to_numeric(df["y"].str.replace(",", "."), errors="coerce")

    # Record size before dropping invalids for logging/debug
    before = len(df)
    # Drop rows that could not be parsed as numbers
    df = df.dropna().reset_index(drop=True)
    after = len(df)

    # Remove exact duplicate rows to avoid biasing clustering
    df = df.drop_duplicates().reset_index(drop=True)

    # Filter out non-positive coordinates (assumption: features should be positive)
    df = df[(df["x"] > 0) & (df["y"] > 0)].reset_index(drop=True)

    # Summary log for user awareness
    print(f"Loaded {before} rows, kept {after} after dropna, {len(df)} after dedup/filtering")
    return df


if __name__ == "__main__":
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    df.to_csv(DATA_OUT, index=False)
    print(f"Cleaned data written to: {DATA_OUT}")
