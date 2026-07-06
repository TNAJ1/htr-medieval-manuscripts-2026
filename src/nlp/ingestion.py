"""
ingestion.py — Ingestion du data contract et analyse exploratoire (EDA).

Premier maillon du volet NLP. Les consignes le disent clairement :
« Le JSON produit par votre pipeline HTR est le point de départ obligatoire
de tout le NLP. »

Ce module :
  1. Charge un (ou plusieurs) data contract(s) JSON produit(s) par le HTR.
  2. Valide le schéma avec jsonschema (un champ manquant bloque tout le NLP).
  3. Réalise une EDA : distribution des confiances, taux needs_review,
     longueur des lignes, abréviations résiduelles.

Ces statistiques justifient ensuite les choix de normalisation (ex: quel
seuil de confiance utiliser pour la correction).
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.data_contract import valider_document


# ─── Chargement ──────────────────────────────────────────────────────────────

def charger_data_contract(chemin: str | Path) -> dict:
    """Charge un fichier data contract JSON et valide son schéma.

    La validation est faite SYSTÉMATIQUEMENT, comme l'exigent les consignes :
    « Validez systématiquement le schéma avec jsonschema avant toute
    manipulation. »

    Args:
        chemin: Chemin du fichier JSON produit par le HTR.

    Returns:
        Le document chargé (dictionnaire validé).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le document ne respecte pas le schéma.

    Example:
        >>> doc = charger_data_contract("dataset_nlp/manuscrit-001.json")
        >>> print(len(doc["lines"]), "lignes")
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Data contract introuvable : {chemin}")

    with open(chemin, encoding="utf-8") as f:
        document = json.load(f)

    # Validation du schéma — ne JAMAIS sauter cette étape
    valide, message = valider_document(document)
    if not valide:
        raise ValueError(
            f"Data contract invalide ({chemin.name}) : {message}. "
            f"Un champ manquant bloquera le NLP en aval."
        )

    return document


def charger_dossier(dossier: str | Path) -> list[dict]:
    """Charge tous les data contracts JSON d'un dossier.

    Args:
        dossier: Dossier contenant les fichiers .json.

    Returns:
        Liste des documents validés.

    Example:
        >>> docs = charger_dossier("dataset_nlp/")
        >>> print(f"{len(docs)} documents chargés")
    """
    dossier = Path(dossier)
    documents = []
    for fichier in sorted(dossier.glob("*.json")):
        documents.append(charger_data_contract(fichier))
    return documents


# ─── Extraction des lignes ───────────────────────────────────────────────────

def extraire_toutes_les_lignes(documents: list[dict]) -> list[dict]:
    """Aplatit tous les documents en une seule liste de lignes.

    Pratique pour l'EDA et la normalisation, qui travaillent ligne par ligne.

    Args:
        documents: Liste de data contracts.

    Returns:
        Liste de toutes les lignes, chacune enrichie de son document_id et
        des métadonnées (siècle, type) pour la stratification.

    Example:
        >>> lignes = extraire_toutes_les_lignes(docs)
    """
    lignes = []
    for doc in documents:
        for ligne in doc["lines"]:
            # On copie la ligne en y ajoutant le contexte du document
            ligne_enrichie = dict(ligne)
            ligne_enrichie["document_id"] = doc["document_id"]
            ligne_enrichie["century"] = doc["metadata"]["century"]
            ligne_enrichie["document_type"] = doc["metadata"]["document_type"]
            lignes.append(ligne_enrichie)
    return lignes


# ─── Analyse exploratoire (EDA) ──────────────────────────────────────────────

# Caractères d'abréviation médiévale fréquents (tilde nasal, p barré, etc.)
ABREVIATIONS_MEDIEVALES = ["~", "ꝑ", "ꝗ", "ꝫ", "ꝓ", "ẜ", "⁊", "ↄ"]


def analyser_corpus(lignes: list[dict]) -> dict:
    """Calcule les statistiques exploratoires (EDA) du corpus.

    Produit les chiffres demandés par les consignes : distribution des
    confiances, taux needs_review, longueur moyenne des lignes, nombre
    d'abréviations résiduelles.

    Args:
        lignes: Liste de lignes (sortie de extraire_toutes_les_lignes).

    Returns:
        Dictionnaire de statistiques.

    Example:
        >>> stats = analyser_corpus(lignes)
        >>> print(f"Taux needs_review : {stats['taux_needs_review']:.1%}")
    """
    if not lignes:
        return {"n_lignes": 0}

    # ── Confiances ──────────────────────────────────────────────────────────
    confiances = [ligne["confidence"] for ligne in lignes]
    confiance_moyenne = sum(confiances) / len(confiances)
    confiance_min = min(confiances)
    confiance_max = max(confiances)

    # ── Taux needs_review ───────────────────────────────────────────────────
    nb_review = sum(1 for ligne in lignes if ligne["needs_review"])
    taux_review = nb_review / len(lignes)

    # ── Longueur des lignes ─────────────────────────────────────────────────
    longueurs = [len(ligne["text"]) for ligne in lignes]
    longueur_moyenne = sum(longueurs) / len(longueurs)

    # ── Abréviations résiduelles ────────────────────────────────────────────
    nb_abreviations = 0
    for ligne in lignes:
        for car in ligne["text"]:
            if car in ABREVIATIONS_MEDIEVALES:
                nb_abreviations += 1

    # ── Distribution par siècle ─────────────────────────────────────────────
    distribution_siecles = {}
    for ligne in lignes:
        siecle = ligne.get("century", "inconnu")
        distribution_siecles[siecle] = distribution_siecles.get(siecle, 0) + 1

    return {
        "n_lignes": len(lignes),
        "confiance_moyenne": round(confiance_moyenne, 4),
        "confiance_min": round(confiance_min, 4),
        "confiance_max": round(confiance_max, 4),
        "taux_needs_review": round(taux_review, 4),
        "longueur_moyenne_lignes": round(longueur_moyenne, 1),
        "nb_abreviations_residuelles": nb_abreviations,
        "distribution_siecles": distribution_siecles,
    }


def afficher_eda(stats: dict) -> None:
    """Affiche un rapport EDA lisible (pour la console ou la présentation).

    Args:
        stats: Le dictionnaire produit par analyser_corpus.

    Example:
        >>> afficher_eda(analyser_corpus(lignes))
    """
    print("─── Analyse exploratoire (EDA) ───")
    print(f"  Nombre de lignes          : {stats['n_lignes']}")
    if stats["n_lignes"] == 0:
        return
    print(f"  Confiance moyenne         : {stats['confiance_moyenne']:.1%}")
    print(f"  Confiance min / max       : {stats['confiance_min']:.1%} / "
          f"{stats['confiance_max']:.1%}")
    print(f"  Taux needs_review         : {stats['taux_needs_review']:.1%}")
    print(f"  Longueur moyenne (caract.): {stats['longueur_moyenne_lignes']}")
    print(f"  Abréviations résiduelles  : {stats['nb_abreviations_residuelles']}")
    print(f"  Distribution par siècle   : {stats['distribution_siecles']}")


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de ingestion.py ===\n")

    # On charge le data contract produit par le pipeline HTR
    chemin = Path("dataset_nlp/demo-manuscrit-001.json")
    if not chemin.exists():
        print("⚠ Aucun data contract trouvé. Lance d'abord :")
        print("    python src/htr/pipeline.py")
    else:
        documents = charger_dossier("dataset_nlp/")
        print(f"✓ {len(documents)} document(s) chargé(s) et validé(s)\n")

        lignes = extraire_toutes_les_lignes(documents)
        stats = analyser_corpus(lignes)
        afficher_eda(stats)

    print("\n=== Démonstration terminée ✓ ===")
