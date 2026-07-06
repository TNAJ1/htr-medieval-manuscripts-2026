"""
data_contract.py — Création et validation du data contract HTR → NLP.

Le data contract est le JSON qui relie les deux volets du projet :
  - Le pipeline HTR (Volet 1) PRODUIT ce JSON.
  - Le pipeline NLP (Volet 2) CONSOMME ce JSON.

Ce module est dans src/shared/ car il est utilisé par les DEUX volets.
Il garantit que le JSON respecte toujours le schéma défini dans
schemas/data_contract.schema.json.

Champs critiques (selon les consignes NLP) :
  - text             : la transcription
  - polygon          : la position de la ligne sur l'image
  - confidence       : confiance globale de la ligne
  - char_confidences : confiance par caractère (pour la correction NLP)
  - candidates       : lectures alternatives (pour le MLM CamemBERT)
  - needs_review     : drapeau de relecture humaine
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import json
from pathlib import Path

import jsonschema  # Bibliothèque qui valide un JSON contre un schéma


# ─── Constantes ──────────────────────────────────────────────────────────────

# Chemin vers le fichier de schéma JSON (la "définition du contrat")
SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "data_contract.schema.json"

# Seuil de confiance en dessous duquel une ligne est marquée needs_review.
# Le projet demande un taux needs_review < 30 % (validation) ou < 20 % (excellence).
SEUIL_NEEDS_REVIEW = 0.70


# ─── Chargement du schéma ────────────────────────────────────────────────────

def charger_schema() -> dict:
    """Charge le schéma JSON du data contract depuis le disque.

    Returns:
        Le schéma sous forme de dictionnaire Python.

    Raises:
        FileNotFoundError: Si le fichier de schéma n'existe pas.

    Example:
        >>> schema = charger_schema()
        >>> print(schema["title"])  # "Data Contract HTR → NLP"
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schéma introuvable : {SCHEMA_PATH}")

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)  # json.load lit un fichier JSON → dictionnaire Python


# ─── Création d'une ligne ────────────────────────────────────────────────────

def creer_ligne(
    line_id: str,
    text: str,
    polygon: list[list[float]],
    char_confidences: list[float],
    reading_order: int = 0,
    candidates: list[dict] | None = None,
) -> dict:
    """Crée une entrée 'ligne' conforme au data contract.

    La confiance globale de la ligne est calculée automatiquement comme
    la moyenne des confiances par caractère. Le drapeau needs_review est
    activé si cette confiance est sous le seuil.

    Args:
        line_id: Identifiant unique, ex: 'page001_line003'.
        text: La transcription textuelle de la ligne.
        polygon: Liste de points [x, y] délimitant la ligne sur l'image.
        char_confidences: Liste des confiances par caractère (entre 0 et 1).
        reading_order: Position de la ligne dans l'ordre de lecture.
        candidates: Lectures alternatives aux positions ambiguës (optionnel).

    Returns:
        Un dictionnaire représentant la ligne, conforme au schéma.

    Raises:
        ValueError: Si char_confidences contient des valeurs hors [0, 1].

    Example:
        >>> ligne = creer_ligne(
        ...     line_id="page001_line001",
        ...     text=" wace",
        ...     polygon=[[10, 20], [100, 20], [100, 40], [10, 40]],
        ...     char_confidences=[0.9, 0.8, 0.6, 0.95],
        ... )
    """
    # Vérification des bornes des confiances
    for c in char_confidences:
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"Confiance hors plage [0,1] : {c}")

    # Calcul de la confiance globale = moyenne des confiances par caractère
    # Si la ligne est vide, on met une confiance de 0
    if char_confidences:
        confidence = sum(char_confidences) / len(char_confidences)
    else:
        confidence = 0.0

    # Drapeau needs_review : True si la ligne est jugée incertaine
    needs_review = confidence < SEUIL_NEEDS_REVIEW

    # Construction du dictionnaire de la ligne
    ligne = {
        "line_id": line_id,
        "reading_order": reading_order,
        "text": text,
        "polygon": polygon,
        "confidence": round(confidence, 4),  # Arrondi à 4 décimales
        "char_confidences": char_confidences,
        "needs_review": needs_review,
    }

    # On n'ajoute 'candidates' que s'il y en a (champ optionnel)
    if candidates:
        ligne["candidates"] = candidates

    return ligne


# ─── Création d'un document complet ──────────────────────────────────────────

def creer_document(
    document_id: str,
    century: int,
    document_type: str,
    language: str,
    source: str,
    lines: list[dict],
    image_width: int,
    image_height: int,
    license: str = "CC-BY-SA-4.0",
) -> dict:
    """Assemble un document complet conforme au data contract.

    Args:
        document_id: Identifiant du manuscrit, ex: 'bnf-ms-fr-12483'.
        century: Siècle du manuscrit (entre 8 et 17).
        document_type: 'texte_simple', 'deux_colonnes' ou 'registre_tabulaire'.
        language: Langue dominante, ex: 'ancien_francais'.
        source: Corpus d'origine, ex: 'CREMMA-Medieval'.
        lines: Liste des lignes créées par creer_ligne().
        image_width: Largeur de l'image source en pixels.
        image_height: Hauteur de l'image source en pixels.
        license: Licence du corpus source.

    Returns:
        Un dictionnaire représentant le document complet.

    Example:
        >>> doc = creer_document(
        ...     document_id="test-001", century=13,
        ...     document_type="texte_simple", language="ancien_francais",
        ...     source="CREMMA", lines=[ligne1, ligne2],
        ...     image_width=2000, image_height=3000,
        ... )
    """
    return {
        "document_id": document_id,
        "metadata": {
            "century": century,
            "document_type": document_type,
            "language": language,
            "source": source,
            "license": license,
        },
        "coordinate_system": {
            "origin": "top_left",
            "unit": "pixel",
            "image_width": image_width,
            "image_height": image_height,
        },
        "lines": lines,
    }


# ─── Validation ──────────────────────────────────────────────────────────────

def valider_document(document: dict) -> tuple[bool, str]:
    """Valide un document contre le schéma JSON du data contract.

    C'est l'étape de sécurité que les consignes NLP demandent de faire
    SYSTÉMATIQUEMENT avant toute manipulation : « Un champ manquant
    vous bloquera plus tard ».

    Args:
        document: Le document à valider (dictionnaire Python).

    Returns:
        Un tuple (valide, message) :
          - valide (bool) : True si le document est conforme.
          - message (str) : "OK" si valide, sinon la description de l'erreur.

    Example:
        >>> valide, message = valider_document(doc)
        >>> if not valide:
        ...     print(f"Erreur : {message}")
    """
    schema = charger_schema()

    try:
        # jsonschema.validate lève une exception si le document est invalide
        jsonschema.validate(instance=document, schema=schema)
        return True, "OK"
    except jsonschema.ValidationError as e:
        # e.message décrit précisément ce qui ne va pas
        # e.json_path indique l'emplacement de l'erreur dans le document
        return False, f"{e.json_path} : {e.message}"


def sauvegarder_document(document: dict, chemin: str | Path) -> None:
    """Valide puis sauvegarde un document en JSON.

    La validation est faite AVANT la sauvegarde : on refuse d'écrire
    un document invalide sur le disque.

    Args:
        document: Le document à sauvegarder.
        chemin: Chemin du fichier JSON de sortie.

    Raises:
        ValueError: Si le document n'est pas conforme au schéma.

    Example:
        >>> sauvegarder_document(doc, "dataset_nlp/test-001.json")
    """
    valide, message = valider_document(document)
    if not valide:
        raise ValueError(f"Document invalide, sauvegarde annulée : {message}")

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    with open(chemin, "w", encoding="utf-8") as f:
        # indent=2 : JSON lisible (indenté)
        # ensure_ascii=False : conserve les accents et caractères médiévaux
        json.dump(document, f, indent=2, ensure_ascii=False)

    print(f"✓ Document validé et sauvegardé : {chemin}")


def calculer_taux_needs_review(document: dict) -> float:
    """Calcule la fraction de lignes marquées needs_review.

    Le projet exige un taux < 30 % (validation) ou < 20 % (excellence).

    Args:
        document: Un document conforme au data contract.

    Returns:
        Le taux entre 0.0 et 1.0.

    Example:
        >>> taux = calculer_taux_needs_review(doc)
        >>> print(f"{taux:.1%} des lignes à relire")
    """
    lignes = document["lines"]
    if not lignes:
        return 0.0

    # On compte les lignes avec needs_review=True
    nb_review = sum(1 for ligne in lignes if ligne["needs_review"])
    return nb_review / len(lignes)


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de data_contract.py ===\n")

    # 1. Créer deux lignes d'exemple
    ligne1 = creer_ligne(
        line_id="demo_line001",
        text="Ci comence li romanz",
        polygon=[[50, 100], [600, 100], [600, 140], [50, 140]],
        char_confidences=[0.95] * 20,  # 20 caractères très confiants
        reading_order=0,
    )

    ligne2 = creer_ligne(
        line_id="demo_line002",
        text="de Brut e de sa gent",
        polygon=[[50, 150], [580, 150], [580, 190], [50, 190]],
        char_confidences=[0.5] * 20,  # 20 caractères peu confiants → needs_review
        reading_order=1,
        candidates=[{"position": 3, "options": ["u", "n"]}],  # ambiguïté sur 'r' de Brut
    )

    print(f"Ligne 1 needs_review : {ligne1['needs_review']} (confiance {ligne1['confidence']})")
    print(f"Ligne 2 needs_review : {ligne2['needs_review']} (confiance {ligne2['confidence']})")

    # 2. Assembler le document
    document = creer_document(
        document_id="demo-001",
        century=12,
        document_type="texte_simple",
        language="ancien_francais",
        source="DEMO",
        lines=[ligne1, ligne2],
        image_width=700,
        image_height=400,
    )

    # 3. Valider
    valide, message = valider_document(document)
    print(f"\nValidation : {valide} ({message})")

    # 4. Taux needs_review
    taux = calculer_taux_needs_review(document)
    print(f"Taux needs_review : {taux:.1%}")

    print("\n=== Démonstration terminée ✓ ===")
