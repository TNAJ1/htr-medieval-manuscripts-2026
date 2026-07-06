"""
visualisation.py — Visualisation des manuscrits avec segmentation colorée.

Produit les images « manuscrit coloré » classiques des présentations HTR :
  1. L'image du manuscrit avec chaque ligne détectée surlignée d'une couleur
  2. La transcription de chaque ligne affichée en regard (même couleur)

C'est le livrable visuel le plus parlant pour une soutenance : on VOIT le
pipeline fonctionner (segmentation + transcription alignées).

Usage typique (sur Colab, avec le vrai corpus) :
    from htr.visualisation import visualiser_page
    visualiser_page("data/raw/cremma-medieval/.../page.jpg",
                    "data/raw/cremma-medieval/.../page.xml",
                    "resultats/page_visualisee.png")
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Palette de couleurs (BGR pour OpenCV) ───────────────────────────────────
# Couleurs vives et distinctes pour bien différencier les lignes
PALETTE = [
    (180, 119, 31),   # bleu
    (14, 127, 255),   # orange
    (44, 160, 44),    # vert
    (40, 39, 214),    # rouge
    (189, 103, 148),  # violet
    (75, 86, 140),    # brun
    (194, 119, 227),  # rose
    (127, 127, 127),  # gris
    (34, 189, 188),   # olive
    (207, 190, 23),   # cyan
]


def couleur_ligne(index: int) -> tuple[int, int, int]:
    """Retourne une couleur de la palette (cycle si plus de lignes).

    Args:
        index: Numéro de la ligne.

    Returns:
        Couleur BGR.
    """
    return PALETTE[index % len(PALETTE)]


# ─── Visualisation d'une page ────────────────────────────────────────────────

def dessiner_segmentation(
    image: np.ndarray,
    lignes: list[dict],
    epaisseur: int = 3,
    opacite_remplissage: float = 0.25,
) -> np.ndarray:
    """Dessine les polygones colorés des lignes sur l'image.

    Chaque ligne reçoit une couleur : contour du polygone + remplissage
    semi-transparent + numéro de ligne.

    Args:
        image: L'image de la page (gris ou couleur).
        lignes: Liste de dicts avec au moins 'polygon' (et éventuellement 'text').
        epaisseur: Épaisseur du contour.
        opacite_remplissage: Transparence du remplissage (0 = invisible).

    Returns:
        L'image annotée (couleur BGR).

    Example:
        >>> annotee = dessiner_segmentation(img, lignes)
        >>> cv2.imwrite("page_annotee.png", annotee)
    """
    # Convertir en couleur si l'image est en niveaux de gris
    if image.ndim == 2:
        resultat = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        resultat = image.copy()

    calque = resultat.copy()  # pour le remplissage semi-transparent

    for i, ligne in enumerate(lignes):
        couleur = couleur_ligne(i)
        points = np.array(ligne["polygon"], dtype=np.int32)

        # Remplissage sur le calque
        cv2.fillPoly(calque, [points], couleur)
        # Contour sur l'image finale
        cv2.polylines(resultat, [points], isClosed=True,
                      color=couleur, thickness=epaisseur)

        # Numéro de la ligne près du coin supérieur gauche du polygone
        x_min = int(points[:, 0].min())
        y_min = int(points[:, 1].min())
        cv2.putText(resultat, str(i + 1), (max(5, x_min - 30), y_min + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2, cv2.LINE_AA)

    # Fusion : image + remplissage semi-transparent
    resultat = cv2.addWeighted(calque, opacite_remplissage,
                               resultat, 1 - opacite_remplissage, 0)
    return resultat


def creer_panneau_transcriptions(
    lignes: list[dict],
    hauteur: int,
    largeur: int = 700,
) -> np.ndarray:
    """Crée un panneau blanc listant les transcriptions, colorées par ligne.

    Args:
        lignes: Liste de dicts avec 'text' (transcription).
        hauteur: Hauteur du panneau (= hauteur de l'image du manuscrit).
        largeur: Largeur du panneau.

    Returns:
        L'image du panneau (BGR).
    """
    panneau = np.ones((hauteur, largeur, 3), dtype=np.uint8) * 255

    # Titre
    cv2.putText(panneau, "Transcriptions", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)

    y = 90
    pas = max(35, min(60, (hauteur - 100) // max(1, len(lignes))))

    for i, ligne in enumerate(lignes):
        couleur = couleur_ligne(i)
        texte = ligne.get("text", "(non transcrite)")

        # Pastille de couleur + numéro
        cv2.circle(panneau, (30, y - 8), 10, couleur, -1)
        cv2.putText(panneau, f"{i + 1}.", (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        # La transcription (tronquée si trop longue pour le panneau)
        if len(texte) > 52:
            texte = texte[:49] + "..."
        cv2.putText(panneau, texte, (95, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 1, cv2.LINE_AA)
        y += pas
        if y > hauteur - 20:
            cv2.putText(panneau, "...", (95, y - pas + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            break

    return panneau


def visualiser_page(
    chemin_image: str | Path,
    chemin_xml: str | Path | None = None,
    chemin_sortie: str | Path = "page_visualisee.png",
    lignes: list[dict] | None = None,
    max_lignes: int | None = None,
) -> Path:
    """Produit l'image « manuscrit coloré + transcriptions » d'une page.

    Deux façons de fournir les lignes :
      - chemin_xml : lit les polygones + transcriptions d'un PAGE/ALTO XML
        (cas du corpus CREMMA : la vérité terrain sert de transcription)
      - lignes : liste déjà prête [{'polygon': ..., 'text': ...}, ...]
        (cas d'une sortie de VOTRE pipeline HTR)

    Args:
        chemin_image: L'image de la page.
        chemin_xml: Le PAGE/ALTO XML correspondant (optionnel si lignes fourni).
        chemin_sortie: Où sauvegarder l'image produite.
        lignes: Lignes déjà extraites (optionnel si chemin_xml fourni).
        max_lignes: Limite de lignes à afficher (None = toutes).

    Returns:
        Le chemin de l'image produite.

    Example:
        >>> visualiser_page("page.jpg", "page.xml", "resultat.png")
    """
    image = cv2.imread(str(chemin_image))
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {chemin_image}")

    # Récupérer les lignes depuis le XML si non fournies
    if lignes is None:
        if chemin_xml is None:
            raise ValueError("Fournir chemin_xml ou lignes.")
        from htr.corpus_loader import extraire_lignes_xml
        lignes = extraire_lignes_xml(Path(chemin_xml))

    if max_lignes is not None:
        lignes = lignes[:max_lignes]

    # 1. Le manuscrit annoté
    manuscrit_annote = dessiner_segmentation(image, lignes)

    # 2. Le panneau des transcriptions
    panneau = creer_panneau_transcriptions(lignes, manuscrit_annote.shape[0])

    # 3. Assemblage côte à côte
    resultat = np.hstack([manuscrit_annote, panneau])

    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(chemin_sortie), resultat)
    print(f"✓ Visualisation sauvegardée : {chemin_sortie}")
    return chemin_sortie


def visualiser_avec_pipeline(
    chemin_image: str | Path,
    chemin_sortie: str | Path = "page_pipeline.png",
    modele: str = "trocr",
    mode_simulation: bool = False,
) -> Path:
    """Visualise une page en la passant dans VOTRE pipeline HTR complet.

    Contrairement à visualiser_page (qui lit la vérité terrain du XML), cette
    fonction exécute le pipeline (prétraitement → segmentation → HTR) et
    affiche VOS transcriptions. C'est la démonstration du système en action.

    Args:
        chemin_image: L'image de la page à traiter.
        chemin_sortie: Où sauvegarder la visualisation.
        modele: "trocr", "kraken" ou "fusion".
        mode_simulation: Si True, HTR simulé (test sans modèle).

    Returns:
        Le chemin de l'image produite.

    Example:
        >>> visualiser_avec_pipeline("page.jpg", "resultat.png",
        ...                          mode_simulation=False)
    """
    from htr.preprocessing import pretraiter_image
    from htr.segmentation import segmenter_lignes, extraire_image_ligne
    from htr.transcripteur_factory import creer_transcripteur

    # 1. Prétraiter
    image_pretraitee = pretraiter_image(chemin_image)

    # 2. Segmenter
    lignes_seg = segmenter_lignes(image_pretraitee)

    # 3. Transcrire chaque ligne
    transcripteur = creer_transcripteur(modele, mode_simulation=mode_simulation)
    lignes = []
    for seg in lignes_seg:
        image_ligne = extraire_image_ligne(image_pretraitee, seg)
        resultat = transcripteur.transcrire(image_ligne)
        lignes.append({"polygon": seg["polygon"], "text": resultat["text"]})

    # 4. Visualiser sur l'image ORIGINALE (plus jolie que la binarisée)
    return visualiser_page(chemin_image, chemin_sortie=chemin_sortie,
                           lignes=lignes)


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    from htr.segmentation import creer_image_multi_lignes, segmenter_lignes
    from htr.page_xml import exporter_page_xml

    print("=== Démonstration de visualisation.py ===\n")

    # Fabriquer une page de démonstration avec sa vérité terrain
    demo = Path("data/raw/demo_visu")
    demo.mkdir(parents=True, exist_ok=True)
    img_path = demo / "page.png"
    creer_image_multi_lignes(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    lignes_seg = segmenter_lignes(img_bin)
    verites = ["Ci comence li romanz", "de Brut e de sa gent",
               "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    exporter_page_xml(lignes_seg, "page.png", img.shape[1], img.shape[0],
                      demo / "page.xml", transcriptions=verites)

    # Produire la visualisation colorée
    visualiser_page(demo / "page.png", demo / "page.xml",
                    "resultats/demo_visualisation.png")

    print("\n=== Chez toi (Colab), avec une vraie page CREMMA : ===")
    print('  from htr.visualisation import visualiser_page')
    print('  # Prendre une paire (image, xml) du corpus :')
    print('  from htr.corpus_loader import trouver_paires')
    print('  paires = trouver_paires("data/raw/cremma-medieval")')
    print('  img, xml = paires[0]')
    print('  visualiser_page(img, xml, "resultats/manuscrit_colore.png")')
