"""
run_evaluation.py — Script de lancement de l'évaluation HTR sur corpus.

À lancer depuis la racine du projet, une fois le corpus téléchargé :
    python run_evaluation.py

Prérequis :
    git clone https://github.com/HTR-United/cremma-medieval data/raw/cremma-medieval
"""

import sys
sys.path.insert(0, "src")  # rend le package 'htr' importable

from htr.evaluation_corpus import evaluer_modele_sur_corpus

if __name__ == "__main__":
    rapport = evaluer_modele_sur_corpus(
        "data/raw/cremma-medieval",
        modele="trocr",
        metadata={"century": 13, "source": "CREMMA"},
        mode_simulation=False,   # ← le VRAI TrOCR (mettre True pour tester sans modèle)
    )

    print("\n=== Résultat ===")
    print(f"CER : {rapport['CER']:.2%}")
    print(f"WER : {rapport['WER']:.2%}")
