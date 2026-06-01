"""Orchestrateur simple pour exécuter le pipeline end-to-end.

Ce script appelle séquentiellement les étapes: preprocess -> train -> evaluate -> report.
Il est volontairement basique et sert surtout pour des démonstrations locales.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


if __name__ == "__main__":
    commands = [
        [sys.executable, str(ROOT / "preprocess.py")],
        [sys.executable, str(ROOT / "train.py")],
        [sys.executable, str(ROOT / "evaluate.py")],
        [sys.executable, str(ROOT / "generate_evidently_report.py")],
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            sys.exit(result.returncode)
    print("Pipeline terminé.")
