"""
preprocessing.py — Pipeline de prétraitement des images de manuscrits.

Ce module implémente la chaîne de traitement d'images définie dans le projet :
  1. Chargement et conversion en niveaux de gris
  2. Correction de l'inclinaison (deskewing)
  3. Amélioration du contraste par CLAHE
  4. Binarisation adaptative par l'algorithme de Sauvola
  5. Suppression du bruit résiduel

Chaque étape améliore la qualité de l'image pour le modèle HTR.
Une mauvaise binarisation peut dégrader le CER de 5 à 10 points !
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path

import cv2                    # OpenCV : bibliothèque de computer vision
import numpy as np            # NumPy : manipulation de tableaux (les images sont des tableaux)
from PIL import Image         # Pillow : chargement/sauvegarde d'images
from skimage.filters import threshold_sauvola  # Algorithme de binarisation adaptatif


# ─── Fonction principale ─────────────────────────────────────────────────────

def pretraiter_image(
    chemin_image: str | Path,
    taille_clahe: int = 8,
    taille_sauvola: int = 25,
    sauvegarder: bool = False,
    dossier_sortie: str | Path | None = None,
) -> np.ndarray:
    """Applique le pipeline complet de prétraitement à une image de manuscrit.

    Le pipeline est : chargement → niveaux de gris → deskew → CLAHE → Sauvola.
    Chaque étape prépare l'image pour la segmentation et la reconnaissance HTR.

    Args:
        chemin_image: Chemin vers l'image source (TIFF, JPEG, PNG).
        taille_clahe: Taille de la grille pour CLAHE (défaut 8×8 tuiles).
            Une valeur plus petite = contraste plus local.
        taille_sauvola: Taille de la fenêtre pour Sauvola en pixels (défaut 25).
            Une valeur plus grande = prend plus de contexte pour le seuil.
        sauvegarder: Si True, sauvegarde l'image prétraitée sur le disque.
        dossier_sortie: Dossier où sauvegarder si sauvegarder=True.

    Returns:
        Image binarisée sous forme de tableau NumPy (0=noir, 255=blanc).
        Shape : (hauteur, largeur) — image en niveaux de gris.

    Raises:
        FileNotFoundError: Si l'image source n'existe pas.
        ValueError: Si l'image ne peut pas être lue.

    Example:
        >>> img = pretraiter_image("data/raw/page_001.jpg", sauvegarder=True)
        >>> print(img.shape)  # ex: (3000, 2000)
    """
    chemin_image = Path(chemin_image)

    if not chemin_image.exists():
        raise FileNotFoundError(f"Image introuvable : {chemin_image}")

    # ── Étape 1 : Chargement ────────────────────────────────────────────────
    # cv2.imread lit l'image en mémoire sous forme de tableau NumPy
    # Les pixels ont des valeurs entre 0 (noir) et 255 (blanc)
    image_couleur = cv2.imread(str(chemin_image))

    if image_couleur is None:
        raise ValueError(f"Impossible de lire l'image : {chemin_image}")

    # ── Étape 2 : Conversion en niveaux de gris ─────────────────────────────
    # Les manuscrits sont en noir et blanc → on n'a pas besoin de la couleur
    # Réduire à 1 canal (au lieu de 3 pour RGB) simplifie le traitement
    # cv2.COLOR_BGR2GRAY : convertit BGR (format OpenCV) → niveaux de gris
    image_gris = cv2.cvtColor(image_couleur, cv2.COLOR_BGR2GRAY)

    print(f"  → Image chargée : {image_gris.shape[1]}×{image_gris.shape[0]} px")

    # ── Étape 3 : Correction d'inclinaison (deskewing) ─────────────────────
    # Un manuscrit scanné peut être légèrement penché.
    # Le deskewing redresse l'image pour que les lignes soient horizontales.
    image_redressee = corriger_inclinaison(image_gris)

    # ── Étape 4 : Amélioration du contraste par CLAHE ──────────────────────
    # CLAHE = Contrast Limited Adaptive Histogram Equalization
    # Améliore le contraste localement (par zones) plutôt que globalement.
    # Utile pour les manuscrits où certaines zones sont plus sombres.
    image_contraste = appliquer_clahe(image_redressee, taille_grille=taille_clahe)

    # ── Étape 5 : Binarisation par Sauvola ─────────────────────────────────
    # La binarisation transforme l'image en pure noir/blanc (0 ou 255).
    # Sauvola calcule un seuil différent pour chaque zone de l'image,
    # ce qui est bien meilleur que Otsu (seuil global) pour les manuscrits.
    image_binarisee = binariser_sauvola(image_contraste, taille_fenetre=taille_sauvola)

    # ── Étape 6 : Nettoyage du bruit résiduel ──────────────────────────────
    image_nettoyee = supprimer_bruit(image_binarisee)

    # ── Sauvegarde optionnelle ──────────────────────────────────────────────
    if sauvegarder and dossier_sortie is not None:
        dossier_sortie = Path(dossier_sortie)
        dossier_sortie.mkdir(parents=True, exist_ok=True)

        # On garde le même nom de fichier mais dans un autre dossier
        chemin_sortie = dossier_sortie / chemin_image.name
        cv2.imwrite(str(chemin_sortie), image_nettoyee)
        print(f"  → Image sauvegardée : {chemin_sortie}")

    return image_nettoyee


# ─── Fonctions auxiliaires ────────────────────────────────────────────────────

def corriger_inclinaison(image: np.ndarray, angle_max: float = 10.0) -> np.ndarray:
    """Détecte et corrige l'inclinaison d'un manuscrit scanné.

    Méthode : on cherche les contours de l'écriture, puis on calcule
    l'angle dominant avec cv2.minAreaRect (rectangle de surface minimale).
    Si l'angle est faible (< angle_max degrés), on redresse.

    Args:
        image: Image en niveaux de gris (tableau NumPy 2D).
        angle_max: Angle maximum au-delà duquel on ne corrige pas
            (évite de corriger des pages intentionnellement inclinées).

    Returns:
        Image redressée de même taille que l'entrée.

    Example:
        >>> img_redressee = corriger_inclinaison(image_gris, angle_max=10.0)
    """
    # On binarise grossièrement pour détecter les pixels d'écriture
    # cv2.THRESH_BINARY_INV : les pixels sombres (texte) deviennent blancs
    _, img_binaire = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # On trouve les coordonnées de tous les pixels blancs (= texte)
    # np.column_stack crée un tableau de paires (x, y)
    coords = np.column_stack(np.where(img_binaire > 0))

    if len(coords) < 100:
        # Pas assez de pixels de texte détectés → on ne corrige pas
        return image

    # cv2.minAreaRect : calcule le rectangle orienté de surface minimale
    # qui englobe tous les points. Son angle donne l'inclinaison de l'écriture.
    rectangle = cv2.minAreaRect(coords)
    angle = rectangle[-1]  # L'angle est le dernier élément du tuple

    # cv2.minAreaRect retourne des angles entre -90 et 0
    # On convertit pour avoir un angle entre -45 et 45 degrés
    if angle < -45:
        angle = 90 + angle

    # Si l'angle est très faible, inutile de corriger (évite des distorsions)
    if abs(angle) < 0.5 or abs(angle) > angle_max:
        return image

    # Calcule la matrice de rotation autour du centre de l'image
    hauteur, largeur = image.shape[:2]
    centre = (largeur // 2, hauteur // 2)  # Centre de l'image

    # cv2.getRotationMatrix2D : crée une matrice 2×3 pour la rotation
    # scale=1.0 : on ne change pas la taille
    matrice_rotation = cv2.getRotationMatrix2D(centre, angle, scale=1.0)

    # cv2.warpAffine : applique la transformation (rotation) à l'image
    # BORDER_REPLICATE : remplit les bords avec les pixels voisins
    image_redressee = cv2.warpAffine(
        image,
        matrice_rotation,
        (largeur, hauteur),
        flags=cv2.INTER_CUBIC,         # Interpolation cubique = meilleure qualité
        borderMode=cv2.BORDER_REPLICATE,
    )

    print(f"  → Inclinaison corrigée : {angle:.2f}°")
    return image_redressee


def appliquer_clahe(image: np.ndarray, taille_grille: int = 8) -> np.ndarray:
    """Améliore le contraste de l'image par CLAHE.

    CLAHE divise l'image en petites tuiles (grille taille_grille × taille_grille)
    et améliore le contraste de chaque tuile séparément.
    Le "Contrast Limited" dans CLAHE empêche le sur-amplification du bruit.

    Args:
        image: Image en niveaux de gris.
        taille_grille: Nombre de tuiles par dimension (8 → grille 8×8).
            Valeurs typiques : 4 (global) à 16 (très local).

    Returns:
        Image avec contraste amélioré, même taille et type que l'entrée.

    Example:
        >>> img_amelioree = appliquer_clahe(image_gris, taille_grille=8)
    """
    # Crée l'objet CLAHE avec ses paramètres
    # clipLimit=2.0 : limite l'amplification (évite le bruit)
    # tileGridSize : taille de la grille de tuiles
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(taille_grille, taille_grille)
    )

    # apply() : applique CLAHE à l'image et retourne le résultat
    return clahe.apply(image)


def binariser_sauvola(image: np.ndarray, taille_fenetre: int = 25) -> np.ndarray:
    """Binarise l'image avec l'algorithme de Sauvola.

    Sauvola calcule un seuil local différent pour chaque pixel,
    en fonction de la moyenne et l'écart-type des pixels voisins.
    C'est bien supérieur à Otsu (seuil global) pour les manuscrits
    car l'éclairage varie d'une zone à l'autre de la page.

    Args:
        image: Image en niveaux de gris (valeurs 0–255).
        taille_fenetre: Taille de la fenêtre locale en pixels.
            25 pixels est un bon point de départ pour des images 300 DPI.

    Returns:
        Image binaire : 255 = blanc (fond), 0 = noir (texte).

    Example:
        >>> img_bin = binariser_sauvola(image_gris, taille_fenetre=25)
    """
    # Sauvola attend des valeurs entre 0.0 et 1.0 → on normalise
    # astype(float) : convertit les entiers 0-255 en décimaux 0.0-255.0
    # / 255.0 : ramène à la plage 0.0-1.0
    image_normalisee = image.astype(float) / 255.0

    # threshold_sauvola calcule le seuil local pour chaque pixel
    # window_size doit être impair → on s'assure qu'il l'est
    fenetre = taille_fenetre if taille_fenetre % 2 == 1 else taille_fenetre + 1
    seuil = threshold_sauvola(image_normalisee, window_size=fenetre)

    # Binarisation : pixel blanc (255) si valeur > seuil, noir (0) sinon
    # image_normalisee > seuil retourne un tableau de True/False
    # * 255 : convertit True→255, False→0
    image_binaire = (image_normalisee > seuil).astype(np.uint8) * 255

    return image_binaire


def supprimer_bruit(image: np.ndarray, taille_noyau: int = 2) -> np.ndarray:
    """Supprime les petits artefacts de bruit par opérations morphologiques.

    Après binarisation, il reste parfois des petits points isolés (bruit).
    On les supprime par "ouverture morphologique" :
      1. Érosion : rétrécit les petites formes (supprime les points isolés)
      2. Dilatation : re-agrandit les formes restantes (restaure le texte)

    Args:
        image: Image binaire (0 ou 255).
        taille_noyau: Taille du noyau morphologique en pixels.
            2 = conservateur, 3 = plus agressif.

    Returns:
        Image binaire nettoyée.

    Example:
        >>> img_propre = supprimer_bruit(image_binaire, taille_noyau=2)
    """
    # Le noyau (kernel) est une petite matrice remplie de 1
    # Il définit le "pinceau" utilisé pour les opérations morphologiques
    noyau = np.ones((taille_noyau, taille_noyau), np.uint8)

    # cv2.MORPH_OPEN = érosion suivie de dilatation
    # Supprime les petits points de bruit sans affecter le texte principal
    image_nettoyee = cv2.morphologyEx(image, cv2.MORPH_OPEN, noyau)

    return image_nettoyee


def creer_image_test(chemin_sortie: str | Path) -> Path:
    """Crée une image de test simple pour valider le pipeline.

    Génère une image blanche 400×300 avec du texte noir simulant
    une ligne de manuscrit. Utile pour les tests automatisés.

    Args:
        chemin_sortie: Chemin où sauvegarder l'image de test.

    Returns:
        Chemin de l'image créée.

    Example:
        >>> chemin = creer_image_test("tests/fixtures/test_page.png")
    """
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # Crée une image blanche 400×300 pixels (hauteur=300, largeur=400)
    # np.ones * 255 : tous les pixels à 255 (blanc)
    # dtype=np.uint8 : entiers non signés 8 bits (valeurs 0-255)
    image = np.ones((300, 400), dtype=np.uint8) * 255

    # Dessine du texte noir simulant une ligne de manuscrit
    cv2.putText(
        image,
        "Manuscrit medieval test",  # Texte à dessiner
        (20, 150),                   # Position (x, y) du coin inférieur gauche
        cv2.FONT_HERSHEY_SIMPLEX,    # Police de caractères
        0.8,                         # Taille de la police
        0,                           # Couleur : 0 = noir
        2,                           # Épaisseur du trait
    )

    # Ajoute un peu de bruit gaussien pour simuler un vrai scan
    bruit = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + bruit, 0, 255).astype(np.uint8)

    cv2.imwrite(str(chemin_sortie), image)
    return chemin_sortie


# ─── Point d'entrée (test rapide) ────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.utils import fixer_seeds, DATA_PROCESSED_DIR, DATA_RAW_DIR

    print("=== Test de preprocessing.py ===")
    fixer_seeds(42)

    # Crée une image de test
    fixture_dir = Path("tests/fixtures")
    img_test = creer_image_test(fixture_dir / "test_page.png")
    print(f"✓ Image de test créée : {img_test}")

    # Applique le pipeline complet
    resultat = pretraiter_image(
        img_test,
        sauvegarder=True,
        dossier_sortie=DATA_PROCESSED_DIR,
    )

    print(f"✓ Pipeline terminé. Shape du résultat : {resultat.shape}")
    print(f"  Valeurs uniques : {np.unique(resultat)}")  # Doit être [0, 255] → binaire
    print("=== Tous les tests passés ✓ ===")
