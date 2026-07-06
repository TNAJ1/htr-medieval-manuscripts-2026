"""
pipeline.py — Orchestration de bout en bout du pipeline HTR.

Ce module est le chef d'orchestre : il enchaîne tous les composants pour
transformer une image brute de manuscrit en un data contract JSON validé,
prêt pour le volet NLP.

Chaîne complète :
    image brute
      → preprocessing (nettoyage)
      → segmentation (détection des lignes + polygones)
      → HTR (transcription + char_confidences)
      → data contract (assemblage + validation)
      → JSON + PAGE XML

C'est ce module qui produit le LIVRABLE FINAL du volet 1.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path

import cv2

# Imports des modules du projet
# On utilise des imports relatifs au package src/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from htr.preprocessing import pretraiter_image
from htr.segmentation import segmenter_lignes, dessiner_segmentation
from htr.htr_model import TranscripteurHTR
from htr.page_xml import exporter_page_xml
from shared.data_contract import (
    creer_document,
    creer_ligne,
    sauvegarder_document,
    valider_document,
    calculer_taux_needs_review,
)


# ─── Pipeline principal ──────────────────────────────────────────────────────

def traiter_manuscrit(
    chemin_image: str | Path,
    document_id: str,
    century: int,
    document_type: str,
    language: str,
    source: str,
    dossier_sortie: str | Path = "dataset_nlp",
    dossier_xml: str | Path = "segmentations",
    mode_simulation: bool = False,
    transcripteur: TranscripteurHTR | None = None,
) -> dict:
    """Traite un manuscrit de bout en bout, de l'image au data contract JSON.

    C'est LA fonction principale du volet 1. Elle enchaîne toutes les étapes
    et produit le livrable final validé.

    Args:
        chemin_image: Chemin de l'image brute du manuscrit.
        document_id: Identifiant du manuscrit, ex: 'bnf-ms-fr-12483-p001'.
        century: Siècle du manuscrit (8-17).
        document_type: 'texte_simple', 'deux_colonnes' ou 'registre_tabulaire'.
        language: Langue dominante, ex: 'ancien_francais'.
        source: Corpus d'origine, ex: 'CREMMA-Medieval' ou 'BnF-Gallica'.
        dossier_sortie: Dossier où sauvegarder le JSON du data contract.
        dossier_xml: Dossier où sauvegarder le PAGE XML.
        mode_simulation: Si True, utilise le HTR en simulation (sans modèle).
        transcripteur: Transcripteur réutilisable. Si None, en crée un.
            Passer un transcripteur évite de recharger le modèle à chaque page.

    Returns:
        Le document data contract (dictionnaire validé).

    Raises:
        FileNotFoundError: Si l'image n'existe pas.
        ValueError: Si le document produit n'est pas conforme au schéma.

    Example:
        >>> doc = traiter_manuscrit(
        ...     "data/raw/page_001.jpg",
        ...     document_id="bnf-12483-p001", century=13,
        ...     document_type="texte_simple", language="ancien_francais",
        ...     source="BnF-Gallica", mode_simulation=True,
        ... )
        >>> print(f"{len(doc['lines'])} lignes transcrites")
    """
    chemin_image = Path(chemin_image)
    print(f"\n┌─ Traitement de {chemin_image.name} ─────────────")

    # ── Étape 1 : Prétraitement ─────────────────────────────────────────────
    print("│ [1/4] Prétraitement de l'image…")
    image_propre = pretraiter_image(chemin_image)
    hauteur, largeur = image_propre.shape

    # ── Étape 2 : Segmentation ──────────────────────────────────────────────
    print("│ [2/4] Segmentation des lignes…")
    lignes_segmentees = segmenter_lignes(image_propre)
    print(f"│       {len(lignes_segmentees)} lignes détectées")

    # ── Étape 3 : Transcription HTR ─────────────────────────────────────────
    print("│ [3/4] Transcription HTR…")
    # On réutilise le transcripteur fourni, ou on en crée un
    if transcripteur is None:
        transcripteur = TranscripteurHTR(mode_simulation=mode_simulation)

    # On transcrit chaque ligne et on construit les entrées du data contract
    lignes_contract = []
    transcriptions_texte = []  # Pour le PAGE XML

    for ligne_seg in lignes_segmentees:
        # Transcrit l'image de la ligne
        resultat_htr = transcripteur.transcrire(ligne_seg["image"])

        # Crée l'entrée du data contract pour cette ligne
        ligne_contract = creer_ligne(
            line_id=f"{document_id}_line{ligne_seg['reading_order']:03d}",
            text=resultat_htr["text"],
            polygon=ligne_seg["polygon"],
            char_confidences=resultat_htr["char_confidences"],
            reading_order=ligne_seg["reading_order"],
            candidates=resultat_htr["candidates"],
        )
        lignes_contract.append(ligne_contract)
        transcriptions_texte.append(resultat_htr["text"])

    # ── Étape 4 : Assemblage et validation du data contract ─────────────────
    print("│ [4/4] Assemblage du data contract…")
    document = creer_document(
        document_id=document_id,
        century=century,
        document_type=document_type,
        language=language,
        source=source,
        lines=lignes_contract,
        image_width=largeur,
        image_height=hauteur,
    )

    # Validation contre le schéma JSON (sécurité)
    valide, message = valider_document(document)
    if not valide:
        raise ValueError(f"Document non conforme : {message}")

    # ── Sauvegardes ─────────────────────────────────────────────────────────
    # 1. Le JSON du data contract
    chemin_json = Path(dossier_sortie) / f"{document_id}.json"
    sauvegarder_document(document, chemin_json)

    # 2. Le PAGE XML (polygones réutilisables)
    chemin_xml = Path(dossier_xml) / f"{document_id}.xml"
    exporter_page_xml(
        lignes_segmentees,
        nom_image=chemin_image.name,
        largeur_image=largeur,
        hauteur_image=hauteur,
        chemin_sortie=chemin_xml,
        transcriptions=transcriptions_texte,
    )

    # ── Résumé ──────────────────────────────────────────────────────────────
    taux_review = calculer_taux_needs_review(document)
    print(f"│ ✓ Terminé : {len(lignes_contract)} lignes, "
          f"{taux_review:.1%} à relire")
    print(f"└────────────────────────────────────────────")

    return document


# ─── Traitement par lot ──────────────────────────────────────────────────────

def traiter_dossier(
    dossier_images: str | Path,
    metadonnees_communes: dict,
    mode_simulation: bool = False,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> list[dict]:
    """Traite toutes les images d'un dossier en un seul appel.

    Charge le modèle HTR une seule fois et le réutilise pour toutes les
    images (gain de temps énorme).

    Args:
        dossier_images: Dossier contenant les images à traiter.
        metadonnees_communes: Métadonnées partagées par toutes les images
            (century, document_type, language, source).
        mode_simulation: Si True, HTR en simulation.
        extensions: Extensions d'images à traiter.

    Returns:
        Liste des documents data contract produits.

    Example:
        >>> docs = traiter_dossier(
        ...     "data/raw/",
        ...     {"century": 13, "document_type": "texte_simple",
        ...      "language": "ancien_francais", "source": "BnF-Gallica"},
        ...     mode_simulation=True,
        ... )
    """
    dossier_images = Path(dossier_images)

    # Liste toutes les images du dossier
    images = []
    for ext in extensions:
        images.extend(dossier_images.glob(f"*{ext}"))
    images = sorted(images)

    print(f"=== Traitement de {len(images)} image(s) ===")

    # On charge le transcripteur UNE SEULE FOIS
    transcripteur = TranscripteurHTR(mode_simulation=mode_simulation)

    documents = []
    for image in images:
        # On utilise le nom du fichier (sans extension) comme document_id
        doc_id = image.stem

        document = traiter_manuscrit(
            chemin_image=image,
            document_id=doc_id,
            transcripteur=transcripteur,  # Réutilisé !
            mode_simulation=mode_simulation,
            **metadonnees_communes,
        )
        documents.append(document)

    print(f"\n=== {len(documents)} document(s) produit(s) ===")
    return documents


# ─── Point d'entrée (démonstration de bout en bout) ──────────────────────────

if __name__ == "__main__":
    from shared.utils import fixer_seeds
    from htr.segmentation import creer_image_multi_lignes

    print("=== Démonstration du pipeline complet (mode simulation) ===")
    fixer_seeds(42)

    # 1. Prépare une image de test (simulant un manuscrit scanné)
    fixture = Path("tests/fixtures/manuscrit_demo.png")
    creer_image_multi_lignes(fixture)

    # 2. Lance le pipeline complet sur cette image
    document = traiter_manuscrit(
        chemin_image=fixture,
        document_id="demo-manuscrit-001",
        century=12,
        document_type="texte_simple",
        language="ancien_francais",
        source="DEMO",
        mode_simulation=True,
    )

    # 3. Affiche un aperçu du data contract produit
    print("\n─── Aperçu du data contract produit ───")
    print(f"Document : {document['document_id']}")
    print(f"Siècle : {document['metadata']['century']}")
    print(f"Image : {document['coordinate_system']['image_width']}×"
          f"{document['coordinate_system']['image_height']} px")
    print(f"Lignes : {len(document['lines'])}")
    print("\nPremière ligne :")
    ligne = document["lines"][0]
    print(f"  id        : {ligne['line_id']}")
    print(f"  texte     : \"{ligne['text']}\"")
    print(f"  confiance : {ligne['confidence']}")
    print(f"  review    : {ligne['needs_review']}")
    print(f"  polygone  : {ligne['polygon']}")

    print("\n=== Pipeline terminé ✓ ===")
    print("Livrables produits :")
    print("  • dataset_nlp/demo-manuscrit-001.json  (data contract)")
    print("  • segmentations/demo-manuscrit-001.xml (PAGE XML)")
