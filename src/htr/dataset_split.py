"""
dataset_split.py — Découpage train/val/test stratifié et scellé.

Le sujet exige (section bonnes pratiques) :
  - Un split train/val/test créé AVANT tout développement.
  - Une stratification par siècle et type de document (pour que chaque
    ensemble soit représentatif).
  - Le test set scellé par un hachage SHA-256 (preuve de non-contamination).

"Stratifié" signifie que la proportion de chaque catégorie (ex: XIIIe siècle)
est la même dans train, val et test. Cela évite qu'un siècle entier ne se
retrouve que dans le test, ce qui fausserait l'évaluation.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import hashlib
import json
from collections import defaultdict
from pathlib import Path


# ─── Fonction principale ─────────────────────────────────────────────────────

def split_stratifie(
    echantillons: list[dict],
    proportions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    cle_strate: str = "century",
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Découpe les échantillons en train/val/test de façon stratifiée.

    Chaque catégorie (ex: chaque siècle) est répartie séparément selon les
    proportions, garantissant que train/val/test ont la même composition.

    Args:
        echantillons: Liste de dicts. Chaque dict doit contenir la clé de
            stratification (ex: {'century': 13, 'text': '...', ...}).
        proportions: Proportions (train, val, test). Doivent sommer à 1.
        cle_strate: La clé sur laquelle stratifier (ex: 'century').
        seed: Graine aléatoire pour la reproductibilité du mélange.

    Returns:
        Un dictionnaire {'train': [...], 'val': [...], 'test': [...]}.

    Raises:
        ValueError: Si les proportions ne somment pas à 1.

    Example:
        >>> data = [{'century': 13, 'id': i} for i in range(100)]
        >>> splits = split_stratifie(data)
        >>> print(len(splits['train']), len(splits['val']), len(splits['test']))
    """
    import random

    # Vérifie que les proportions somment à 1 (avec une petite tolérance)
    if abs(sum(proportions) - 1.0) > 1e-6:
        raise ValueError(f"Les proportions doivent sommer à 1, reçu {sum(proportions)}.")

    p_train, p_val, p_test = proportions

    # ── Étape 1 : Grouper les échantillons par strate ───────────────────────
    # defaultdict(list) crée automatiquement une liste vide pour chaque clé
    strates = defaultdict(list)
    for echantillon in echantillons:
        cle = echantillon.get(cle_strate, "inconnu")
        strates[cle].append(echantillon)

    # ── Étape 2 : Découper chaque strate séparément ─────────────────────────
    rng = random.Random(seed)  # Générateur local pour ne pas affecter le global

    train, val, test = [], [], []

    # On trie les clés pour un découpage déterministe
    for cle in sorted(strates.keys(), key=str):
        groupe = strates[cle][:]  # Copie de la liste
        rng.shuffle(groupe)       # Mélange aléatoire (reproductible avec seed)

        n = len(groupe)
        n_train = int(n * p_train)
        n_val = int(n * p_val)
        # Le reste va dans test (évite les problèmes d'arrondi)

        train.extend(groupe[:n_train])
        val.extend(groupe[n_train:n_train + n_val])
        test.extend(groupe[n_train + n_val:])

    return {"train": train, "val": val, "test": test}


# ─── Scellement du test set ──────────────────────────────────────────────────

def sceller_test_set(test_set: list[dict], chemin_sortie: str | Path) -> str:
    """Sauvegarde le test set et calcule son hachage SHA-256.

    Le hash est la "preuve" que le test set n'a pas été modifié pendant le
    développement. On le note dans l'article ; si on regardait le test set et
    le modifiait, le hash changerait et la tricherie serait détectable.

    Args:
        test_set: La liste des échantillons de test.
        chemin_sortie: Chemin du fichier JSON où sauvegarder le test set.

    Returns:
        Le hash SHA-256 (chaîne hexadécimale de 64 caractères).

    Example:
        >>> h = sceller_test_set(splits['test'], "data/splits/test.json")
        >>> print(f"SHA-256 du test set : {h}")
    """
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # On sérialise en JSON de façon DÉTERMINISTE :
    # sort_keys=True garantit que les clés sont toujours dans le même ordre,
    # sinon le hash changerait à chaque exécution même sans modification.
    contenu_json = json.dumps(test_set, sort_keys=True, ensure_ascii=False, indent=2)

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(contenu_json)

    # Calcul du hash sur le contenu encodé en UTF-8
    hash_sha256 = hashlib.sha256(contenu_json.encode("utf-8")).hexdigest()

    # On sauvegarde aussi le hash dans un petit fichier à côté
    chemin_hash = chemin_sortie.with_suffix(".sha256")
    with open(chemin_hash, "w", encoding="utf-8") as f:
        f.write(hash_sha256 + "\n")

    print(f"✓ Test set scellé : {chemin_sortie}")
    print(f"  SHA-256 : {hash_sha256}")
    return hash_sha256


def verifier_integrite_test_set(chemin_test: str | Path, hash_attendu: str) -> bool:
    """Vérifie qu'un test set n'a pas été modifié, via son hash SHA-256.

    Args:
        chemin_test: Chemin du fichier JSON du test set.
        hash_attendu: Le hash SHA-256 enregistré au moment du scellement.

    Returns:
        True si le test set est intact, False s'il a été modifié.

    Example:
        >>> intact = verifier_integrite_test_set("data/splits/test.json", h)
        >>> assert intact, "Le test set a été modifié !"
    """
    chemin_test = Path(chemin_test)
    contenu = chemin_test.read_text(encoding="utf-8")
    hash_actuel = hashlib.sha256(contenu.encode("utf-8")).hexdigest()
    return hash_actuel == hash_attendu


# ─── Statistiques de distribution (pour l'EDA) ───────────────────────────────

def distribution_strates(
    splits: dict[str, list[dict]],
    cle_strate: str = "century",
) -> dict:
    """Calcule la distribution de chaque strate dans chaque split.

    Utile pour vérifier que la stratification a bien fonctionné et pour
    l'analyse exploratoire (EDA) demandée par le projet.

    Args:
        splits: Le dictionnaire {'train': [...], 'val': [...], 'test': [...]}.
        cle_strate: La clé de stratification.

    Returns:
        Un dictionnaire imbriqué {split: {valeur_strate: compte}}.

    Example:
        >>> dist = distribution_strates(splits, 'century')
        >>> print(dist['train'])  # {13: 70, 14: 35, ...}
    """
    resultat = {}
    for nom_split, echantillons in splits.items():
        comptes = defaultdict(int)
        for e in echantillons:
            comptes[e.get(cle_strate, "inconnu")] += 1
        resultat[nom_split] = dict(comptes)
    return resultat


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.utils import fixer_seeds

    print("=== Démonstration de dataset_split.py ===\n")
    fixer_seeds(42)

    # Crée 200 échantillons fictifs répartis sur 3 siècles
    echantillons = []
    for i in range(200):
        siecle = [12, 13, 14][i % 3]  # alterne les siècles
        echantillons.append({
            "id": f"line_{i}",
            "century": siecle,
            "document_type": "texte_simple",
            "text": f"transcription {i}",
        })

    # Découpe en train/val/test
    splits = split_stratifie(echantillons, proportions=(0.7, 0.15, 0.15))
    print(f"Train : {len(splits['train'])} échantillons")
    print(f"Val   : {len(splits['val'])} échantillons")
    print(f"Test  : {len(splits['test'])} échantillons")

    # Vérifie la stratification
    print("\nDistribution par siècle :")
    dist = distribution_strates(splits, "century")
    for nom_split, comptes in dist.items():
        print(f"  {nom_split} : {comptes}")

    # Scelle le test set
    print()
    hash_test = sceller_test_set(splits["test"], "data/splits/test.json")

    # Vérifie l'intégrité
    intact = verifier_integrite_test_set("data/splits/test.json", hash_test)
    print(f"\n✓ Intégrité du test set vérifiée : {intact}")

    print("\n=== Démonstration terminée ✓ ===")
