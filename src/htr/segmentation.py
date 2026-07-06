"""
segmentation.py — Détection des lignes de texte et extraction des polygones.

Ce module prend une image prétraitée (binaire) et détecte les lignes de
texte qui s'y trouvent. Pour chaque ligne, il produit :
  - une boîte englobante (x, y, largeur, hauteur)
  - un polygone (liste de points [x, y]) au format du data contract
  - l'image découpée de la ligne (pour la donner ensuite au modèle HTR)

Deux approches sont possibles :
  1. Méthode classique (implémentée ici) : profils de projection horizontale.
     Légère, sans GPU, idéale pour comprendre le principe.
  2. Kraken BLLA (deep learning) : plus robuste sur les manuscrits complexes.
     Recommandée par le sujet pour la version finale.

L'ordre de lecture est respecté : les lignes sont triées de haut en bas.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path

import cv2
import numpy as np


# ─── Fonction principale ─────────────────────────────────────────────────────

def segmenter_lignes(
    image_binaire: np.ndarray,
    seuil_min_hauteur: int = 10,
    marge_verticale: int = 3,
) -> list[dict]:
    """Détecte les lignes de texte dans une image binaire prétraitée.

    Principe : on calcule le "profil de projection horizontale", c'est-à-dire
    le nombre de pixels noirs (texte) sur chaque rangée de l'image.
    Les rangées avec beaucoup de texte = une ligne. Les rangées vides = un
    espace entre deux lignes. On regroupe les rangées consécutives de texte.

    Args:
        image_binaire: Image binaire (0=texte noir, 255=fond blanc).
        seuil_min_hauteur: Hauteur minimale d'une ligne en pixels.
            En dessous, on ignore (probablement du bruit).
        marge_verticale: Marge ajoutée en haut/bas de chaque ligne en pixels.

    Returns:
        Liste de dictionnaires, un par ligne détectée, triés par ordre de
        lecture (haut → bas). Chaque dict contient :
          - 'reading_order' (int) : position de la ligne (0 = première)
          - 'bbox' (tuple) : (x, y, largeur, hauteur)
          - 'polygon' (list) : points [[x,y], ...] délimitant la ligne
          - 'image' (np.ndarray) : l'image découpée de la ligne

    Raises:
        ValueError: Si l'image n'est pas binaire ou est vide.

    Example:
        >>> lignes = segmenter_lignes(image_bin)
        >>> print(f"{len(lignes)} lignes détectées")
        >>> premiere_ligne = lignes[0]
        >>> print(premiere_ligne['bbox'])  # (10, 20, 580, 35)
    """
    if image_binaire is None or image_binaire.size == 0:
        raise ValueError("L'image fournie est vide.")

    # On s'assure que l'image est en 2D (niveaux de gris)
    if image_binaire.ndim != 2:
        raise ValueError(f"Image 2D attendue, reçu {image_binaire.ndim}D.")

    hauteur, largeur = image_binaire.shape

    # ── Étape 1 : Inverser l'image ──────────────────────────────────────────
    # Dans notre image, le texte est noir (0) et le fond blanc (255).
    # Pour compter les pixels de texte, c'est plus simple de les avoir à 1.
    # On inverse : texte → 255, fond → 0.
    image_inversee = cv2.bitwise_not(image_binaire)

    # ── Étape 2 : Profil de projection horizontale ──────────────────────────
    # Pour chaque rangée (axis=1 = horizontal), on somme les pixels.
    # Une rangée avec beaucoup de texte aura une grande somme.
    # np.sum avec axis=1 : additionne chaque ligne → tableau 1D de taille hauteur
    profil = np.sum(image_inversee, axis=1)

    # On normalise en divisant par 255 pour avoir un nombre de pixels
    profil = profil / 255.0

    # ── Étape 3 : Détecter les zones de texte ───────────────────────────────
    # Une rangée "contient du texte" si son profil dépasse un petit seuil.
    # Seuil = 1 % de la largeur (au moins quelques pixels de texte).
    seuil_texte = largeur * 0.01

    # Tableau de booléens : True = rangée avec texte, False = rangée vide
    rangees_avec_texte = profil > seuil_texte

    # ── Étape 4 : Regrouper les rangées consécutives en lignes ──────────────
    lignes_brutes = _grouper_rangees(rangees_avec_texte)

    # ── Étape 5 : Construire le résultat pour chaque ligne ──────────────────
    resultats = []
    ordre = 0  # Compteur pour l'ordre de lecture

    for (debut, fin) in lignes_brutes:
        # Hauteur de la ligne
        hauteur_ligne = fin - debut

        # On ignore les lignes trop petites (probablement du bruit)
        if hauteur_ligne < seuil_min_hauteur:
            continue

        # On ajoute une petite marge en haut et en bas
        y_haut = max(0, debut - marge_verticale)
        y_bas = min(hauteur, fin + marge_verticale)

        # ── Affiner les bornes horizontales (gauche/droite) ─────────────────
        # On découpe d'abord la bande horizontale de cette ligne
        bande = image_inversee[y_haut:y_bas, :]

        # Profil vertical : pixels de texte par colonne
        profil_vertical = np.sum(bande, axis=0) / 255.0
        colonnes_avec_texte = np.where(profil_vertical > 0)[0]

        if len(colonnes_avec_texte) == 0:
            continue  # Ligne vide, on ignore

        # Première et dernière colonne contenant du texte
        x_gauche = int(colonnes_avec_texte[0])
        x_droite = int(colonnes_avec_texte[-1])

        # ── Construire la boîte englobante ──────────────────────────────────
        x = x_gauche
        y = y_haut
        w = x_droite - x_gauche
        h = y_bas - y_haut

        # ── Construire le polygone ──────────────────────────────────────────
        # Un polygone rectangulaire = 4 coins, dans le sens horaire
        # depuis le coin haut-gauche. Format attendu par le data contract.
        polygon = [
            [float(x), float(y)],          # coin haut-gauche
            [float(x + w), float(y)],      # coin haut-droite
            [float(x + w), float(y + h)],  # coin bas-droite
            [float(x), float(y + h)],      # coin bas-gauche
        ]

        # ── Découper l'image de la ligne ────────────────────────────────────
        # On découpe dans l'image ORIGINALE (pas l'inversée) pour le HTR
        image_ligne = image_binaire[y:y + h, x:x + w]

        resultats.append({
            "reading_order": ordre,
            "bbox": (x, y, w, h),
            "polygon": polygon,
            "image": image_ligne,
        })
        ordre += 1

    return resultats


# ─── Fonction auxiliaire ──────────────────────────────────────────────────────

def _grouper_rangees(rangees_avec_texte: np.ndarray) -> list[tuple[int, int]]:
    """Regroupe les rangées consécutives de texte en blocs (lignes).

    Le préfixe '_' indique que cette fonction est "privée" : utilisée
    en interne par le module, pas destinée à être appelée de l'extérieur.

    Args:
        rangees_avec_texte: Tableau de booléens. True = rangée avec texte.

    Returns:
        Liste de tuples (debut, fin) délimitant chaque bloc de texte.
        'fin' est exclusif (comme en Python : range(debut, fin)).

    Example:
        >>> import numpy as np
        >>> r = np.array([False, True, True, False, True, False])
        >>> _grouper_rangees(r)
        [(1, 3), (4, 5)]
    """
    blocs = []
    debut = None  # Début du bloc courant (None = on n'est pas dans un bloc)

    for i, a_du_texte in enumerate(rangees_avec_texte):
        if a_du_texte and debut is None:
            # On entre dans une zone de texte → on note le début
            debut = i
        elif not a_du_texte and debut is not None:
            # On sort d'une zone de texte → on ferme le bloc
            blocs.append((debut, i))
            debut = None

    # Cas particulier : si le texte va jusqu'à la dernière rangée
    if debut is not None:
        blocs.append((debut, len(rangees_avec_texte)))

    return blocs


# ─── Visualisation (pour le débogage et l'article) ───────────────────────────

def dessiner_segmentation(
    image_binaire: np.ndarray,
    lignes: list[dict],
    chemin_sortie: str | Path | None = None,
) -> np.ndarray:
    """Dessine les polygones détectés sur l'image, pour vérification visuelle.

    Très utile pour vérifier que la segmentation est correcte, et pour
    illustrer l'article scientifique.

    Args:
        image_binaire: L'image binaire d'origine.
        lignes: Le résultat de segmenter_lignes().
        chemin_sortie: Si fourni, sauvegarde l'image annotée.

    Returns:
        Image couleur (BGR) avec les polygones dessinés en rouge et les
        numéros de ligne.

    Example:
        >>> img_annotee = dessiner_segmentation(img_bin, lignes, "debug.png")
    """
    # On convertit l'image en couleur pour pouvoir dessiner en rouge
    image_couleur = cv2.cvtColor(image_binaire, cv2.COLOR_GRAY2BGR)

    for ligne in lignes:
        x, y, w, h = ligne["bbox"]
        ordre = ligne["reading_order"]

        # Dessine le rectangle de la boîte englobante (rouge, épaisseur 2)
        # Couleur en BGR : (0, 0, 255) = rouge
        cv2.rectangle(image_couleur, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Écrit le numéro de ligne à gauche du rectangle
        cv2.putText(
            image_couleur,
            str(ordre),
            (max(0, x - 25), y + h // 2),  # Position du texte
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),  # Bleu
            1,
        )

    if chemin_sortie is not None:
        chemin_sortie = Path(chemin_sortie)
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(chemin_sortie), image_couleur)
        print(f"✓ Image de segmentation sauvegardée : {chemin_sortie}")

    return image_couleur


def creer_image_multi_lignes(chemin_sortie: str | Path) -> Path:
    """Crée une image de test contenant plusieurs lignes de texte.

    Utile pour tester la segmentation (l'image de preprocessing n'a qu'une ligne).

    Args:
        chemin_sortie: Chemin où sauvegarder l'image.

    Returns:
        Le chemin de l'image créée.

    Example:
        >>> chemin = creer_image_multi_lignes("tests/fixtures/multi_lignes.png")
    """
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # Image blanche 500×400
    image = np.ones((400, 500), dtype=np.uint8) * 255

    # On dessine 4 lignes de texte à des hauteurs différentes
    textes = ["Ci comence li romanz", "de Brut e de sa gent",
              "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    for i, texte in enumerate(textes):
        y = 60 + i * 80  # Espacement vertical de 80 px entre les lignes
        cv2.putText(image, texte, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, 0, 2)

    cv2.imwrite(str(chemin_sortie), image)
    return chemin_sortie


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.utils import fixer_seeds

    print("=== Démonstration de segmentation.py ===\n")
    fixer_seeds(42)

    # 1. Crée une image de test multi-lignes
    fixture = Path("tests/fixtures/multi_lignes.png")
    creer_image_multi_lignes(fixture)
    print(f"✓ Image de test créée : {fixture}")

    # 2. Charge et binarise l'image (simple seuil pour la démo)
    img = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. Segmente les lignes
    lignes = segmenter_lignes(img_bin)
    print(f"✓ {len(lignes)} lignes détectées\n")

    for ligne in lignes:
        x, y, w, h = ligne["bbox"]
        print(f"  Ligne {ligne['reading_order']} : "
              f"position ({x},{y}), taille {w}×{h} px")

    # 4. Dessine la segmentation pour vérification
    dessiner_segmentation(img_bin, lignes, "data/processed/segmentation_demo.png")

    print("\n=== Démonstration terminée ✓ ===")
