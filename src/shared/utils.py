"""
utils.py — Fonctions utilitaires partagées par tout le pipeline.

Ce module contient les fonctions de base qui sont utilisées partout :
  - Fixer les graines aléatoires (reproductibilité)
  - Calculer le hash SHA-256 d'un fichier (intégrité des données)
  - Enregistrer une expérience dans le journal
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import hashlib        # Pour calculer des hash (empreintes) de fichiers
import json           # Pour lire et écrire du JSON
import os             # Pour manipuler les chemins de fichiers
import random         # Générateur de nombres aléatoires Python natif
from datetime import datetime  # Pour horodater les expériences
from pathlib import Path       # Façon moderne de manipuler les chemins

import numpy as np    # Générateur aléatoire NumPy (utilisé par OpenCV, scikit-image)
import torch          # Générateur aléatoire PyTorch (utilisé par les modèles)


# ─── Constantes du projet ────────────────────────────────────────────────────

# Chemin racine du projet (le dossier qui contient src/, data/, etc.)
# __file__ = chemin de CE fichier (utils.py)
# .parent = dossier src/
# .parent.parent = dossier racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Chemins des dossiers principaux — on les définit ici pour les réutiliser partout
DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"        # Images brutes téléchargées
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # Images après prétraitement
DATA_SPLITS_DIR    = PROJECT_ROOT / "data" / "splits"     # train / val / test
DATASET_NLP_DIR    = PROJECT_ROOT / "dataset_nlp"         # JSON final livré au module NLP
SEGMENTATIONS_DIR  = PROJECT_ROOT / "segmentations"       # Polygones PAGE XML
EXPERIMENTS_DIR    = PROJECT_ROOT / "experiments"         # Journal des expériences

# Graine par défaut — TOUJOURS utiliser la même pour la reproductibilité
DEFAULT_SEED = 42


# ─── Fonctions ───────────────────────────────────────────────────────────────

def fixer_seeds(seed: int = DEFAULT_SEED) -> None:
    """Fixe toutes les graines aléatoires pour rendre les résultats reproductibles.

    Sans cette fonction, chaque exécution du code donnerait des résultats
    légèrement différents car les initialisations aléatoires varient.
    On fixe 3 graines différentes car 3 bibliothèques gèrent leur propre
    générateur de nombres aléatoires indépendamment.

    Args:
        seed: La valeur de la graine. Par convention, on utilise 42.

    Example:
        >>> fixer_seeds(42)  # À appeler au tout début de chaque script
    """
    # 1. Graine pour le module random natif de Python
    #    Utilisé par ex. dans random.shuffle(), random.choice()
    random.seed(seed)

    # 2. Graine pour NumPy
    #    Utilisé par OpenCV, scikit-image, et beaucoup d'opérations sur les tableaux
    np.random.seed(seed)

    # 3. Graine pour PyTorch (CPU)
    #    Utilisé pendant l'initialisation et l'entraînement des modèles
    torch.manual_seed(seed)

    # 4. Graine pour PyTorch (GPU, si disponible)
    #    Si on a une carte graphique, il faut aussi fixer sa graine
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"✓ Graines aléatoires fixées à {seed}")


def sha256_fichier(chemin: str | Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier.

    Le SHA-256 est une "empreinte digitale" unique d'un fichier.
    Si le fichier change d'un seul octet, l'empreinte change complètement.
    On l'utilise pour prouver que le test set n'a pas été modifié pendant
    le développement (exigence du projet).

    Args:
        chemin: Chemin vers le fichier à analyser.

    Returns:
        Une chaîne hexadécimale de 64 caractères représentant le hash.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.

    Example:
        >>> h = sha256_fichier("data/splits/test.json")
        >>> print(h)  # ex: "a3f2c1..."
    """
    chemin = Path(chemin)

    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    # hashlib.sha256() crée un "calculateur" de hash SHA-256
    calculateur = hashlib.sha256()

    # On lit le fichier par blocs de 8192 octets (8 Ko)
    # Pourquoi par blocs ? Pour ne pas charger un gros fichier entier en RAM
    with open(chemin, "rb") as f:  # "rb" = read binary (lecture en mode binaire)
        while True:
            bloc = f.read(8192)   # Lire 8192 octets
            if not bloc:          # Si le bloc est vide, on a atteint la fin du fichier
                break
            calculateur.update(bloc)  # Ajouter ce bloc au calcul du hash

    # .hexdigest() retourne le hash sous forme de chaîne lisible (lettres + chiffres)
    return calculateur.hexdigest()


def enregistrer_experience(
    nom: str,
    parametres: dict,
    metriques: dict,
    notes: str = "",
) -> None:
    """Enregistre une expérience dans le journal JSONL.

    Chaque ligne du fichier journal.jsonl représente une expérience.
    JSONL = JSON Lines = un objet JSON par ligne. Format pratique pour
    ajouter des entrées sans relire tout le fichier.

    Args:
        nom: Nom court de l'expérience, ex: "trocr_lora_r8_lr1e-4".
        parametres: Dictionnaire des hyperparamètres utilisés.
        metriques: Dictionnaire des métriques obtenues (CER, WER...).
        notes: Texte libre pour décrire l'expérience.

    Example:
        >>> enregistrer_experience(
        ...     nom="trocr_baseline",
        ...     parametres={"lr": 1e-4, "epochs": 10},
        ...     metriques={"CER": 0.12, "WER": 0.20},
        ...     notes="Première expérience sans augmentation"
        ... )
    """
    # Construire l'entrée à enregistrer
    entree = {
        "timestamp": datetime.now().isoformat(),  # Date et heure au format ISO 8601
        "nom": nom,
        "parametres": parametres,
        "metriques": metriques,
        "notes": notes,
    }

    # Chemin du fichier journal
    journal = EXPERIMENTS_DIR / "journal.jsonl"

    # Créer le dossier s'il n'existe pas
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # "a" = append (ajouter à la fin du fichier sans effacer ce qui existe)
    # ensure_ascii=False : permet d'écrire les accents correctement
    with open(journal, "a", encoding="utf-8") as f:
        f.write(json.dumps(entree, ensure_ascii=False) + "\n")

    print(f"✓ Expérience '{nom}' enregistrée dans {journal}")


def creer_dossiers() -> None:
    """Crée tous les dossiers du projet s'ils n'existent pas encore.

    À appeler une seule fois au début d'un nouveau projet.

    Example:
        >>> creer_dossiers()
    """
    dossiers = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_SPLITS_DIR,
        DATASET_NLP_DIR,
        SEGMENTATIONS_DIR,
        EXPERIMENTS_DIR,
    ]

    for dossier in dossiers:
        # parents=True : crée aussi les dossiers parents si nécessaire
        # exist_ok=True : ne lève pas d'erreur si le dossier existe déjà
        dossier.mkdir(parents=True, exist_ok=True)

    print(f"✓ {len(dossiers)} dossiers prêts.")


# ─── Point d'entrée (test rapide) ────────────────────────────────────────────
# Ce bloc ne s'exécute QUE si on lance ce fichier directement :
#   python src/utils.py
# Il ne s'exécute PAS si on importe ce fichier depuis un autre module.

if __name__ == "__main__":
    print("=== Test de utils.py ===")

    # Test 1 : création des dossiers
    creer_dossiers()

    # Test 2 : fixation des graines
    fixer_seeds(42)

    # Test 3 : hash d'un fichier existant (le journal)
    journal_path = EXPERIMENTS_DIR / "journal.jsonl"
    if journal_path.exists():
        h = sha256_fichier(journal_path)
        print(f"✓ Hash du journal : {h}")

    # Test 4 : enregistrement d'une expérience fictive
    enregistrer_experience(
        nom="test_utils",
        parametres={"seed": 42},
        metriques={"CER": 0.99},
        notes="Vérification que utils.py fonctionne.",
    )

    print("=== Tous les tests passés ✓ ===")
