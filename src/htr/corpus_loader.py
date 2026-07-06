"""
corpus_loader.py — Chargement d'un corpus HTR réel (CREMMA, CATMuS...).

Ce module fait le pont entre un corpus téléchargé et notre pipeline.

Les corpus HTR médiévaux (CREMMA, CATMuS, GalliCorpora...) sont distribués
sous forme de PAIRES :
    - une image de page (.jpg, .png, .tiff)
    - un fichier XML (PAGE ou ALTO) qui contient, pour chaque ligne :
        * son polygone (position sur l'image)
        * sa transcription = la VÉRITÉ TERRAIN

« Brancher le corpus » consiste à :
  1. Trouver toutes les paires (image, XML) dans le dossier téléchargé.
  2. Pour chaque paire, extraire les lignes : image découpée + texte de référence.
  3. Produire une liste d'exemples {image_ligne, texte_reference, metadata}
     directement utilisable pour l'entraînement et la mesure du CER.

Ce module lit à la fois le PAGE XML et l'ALTO XML (les deux formats de CREMMA).
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Détection du format XML ─────────────────────────────────────────────────

# Les namespaces des deux formats possibles
NS_PAGE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NS_PAGE_ALT = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS_ALTO = "http://www.loc.gov/standards/alto/ns-v4#"
NS_ALTO_ALT = "http://www.loc.gov/standards/alto/ns-v2#"


def _detecter_format(chemin_xml: Path) -> str:
    """Détecte si un fichier XML est du PAGE ou de l'ALTO.

    Args:
        chemin_xml: Chemin du fichier XML.

    Returns:
        "page" ou "alto".

    Raises:
        ValueError: Si le format n'est pas reconnu.
    """
    # On lit le début du fichier pour repérer le namespace
    contenu = chemin_xml.read_text(encoding="utf-8", errors="ignore")[:2000]
    if "PAGE/gts/pagecontent" in contenu:
        return "page"
    if "standards/alto" in contenu:
        return "alto"
    raise ValueError(f"Format XML non reconnu : {chemin_xml.name}")


# ─── Recherche des paires (image, XML) ───────────────────────────────────────

def trouver_paires(
    dossier_corpus: str | Path,
    extensions_image: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> list[tuple[Path, Path]]:
    """Trouve toutes les paires (image, XML) dans un corpus téléchargé.

    Parcourt récursivement le dossier. Pour chaque fichier XML trouvé, cherche
    l'image correspondante (même nom de base, extension différente).

    Args:
        dossier_corpus: Racine du corpus téléchargé (ex: 'data/raw/cremma').
        extensions_image: Extensions d'images à chercher.

    Returns:
        Liste de tuples (chemin_image, chemin_xml).

    Example:
        >>> paires = trouver_paires("data/raw/cremma-medieval")
        >>> print(f"{len(paires)} pages trouvées")
    """
    dossier_corpus = Path(dossier_corpus)
    paires = []

    # rglob('*.xml') cherche tous les XML récursivement (dans tous les sous-dossiers)
    for chemin_xml in sorted(dossier_corpus.rglob("*.xml")):
        # On ignore les fichiers de validation de schéma (.xsd) et métadonnées
        if chemin_xml.suffix != ".xml":
            continue

        # On cherche l'image correspondante : même nom, autre extension
        nom_base = chemin_xml.stem  # nom sans extension
        image_trouvee = None

        # 1. Chercher dans le même dossier
        for ext in extensions_image:
            candidat = chemin_xml.with_suffix(ext)
            if candidat.exists():
                image_trouvee = candidat
                break

        # 2. Si pas trouvé, chercher dans tout le corpus (parfois images/ séparé)
        if image_trouvee is None:
            for ext in extensions_image:
                candidats = list(dossier_corpus.rglob(f"{nom_base}{ext}"))
                if candidats:
                    image_trouvee = candidats[0]
                    break

        if image_trouvee is not None:
            paires.append((image_trouvee, chemin_xml))

    return paires


# ─── Extraction des lignes d'un fichier XML ──────────────────────────────────

def extraire_lignes_xml(chemin_xml: Path) -> list[dict]:
    """Extrait les lignes (polygone + transcription) d'un fichier PAGE ou ALTO.

    Args:
        chemin_xml: Chemin du fichier XML.

    Returns:
        Liste de dicts {polygon, text} pour chaque ligne ayant une transcription.

    Example:
        >>> lignes = extraire_lignes_xml(Path("page_001.xml"))
        >>> print(lignes[0]["text"])  # la vérité terrain de la 1re ligne
    """
    format_xml = _detecter_format(chemin_xml)
    if format_xml == "page":
        return _extraire_lignes_page(chemin_xml)
    return _extraire_lignes_alto(chemin_xml)


def _extraire_lignes_page(chemin_xml: Path) -> list[dict]:
    """Extrait les lignes d'un fichier PAGE XML.

    Args:
        chemin_xml: Chemin du fichier PAGE XML.

    Returns:
        Liste de dicts {polygon, text}.
    """
    arbre = ET.parse(chemin_xml)
    racine = arbre.getroot()

    # Le namespace peut être l'un des deux (2019 ou 2013)
    ns_uri = NS_PAGE if NS_PAGE in racine.tag else NS_PAGE_ALT
    ns = {"p": ns_uri}

    lignes = []
    for text_line in racine.findall(".//p:TextLine", ns):
        # Récupérer le polygone
        coords = text_line.find("p:Coords", ns)
        polygon = []
        if coords is not None:
            points_str = coords.get("points", "")
            polygon = _parser_points(points_str)

        # Récupérer la transcription (vérité terrain)
        unicode_elem = text_line.find(".//p:Unicode", ns)
        texte = ""
        if unicode_elem is not None and unicode_elem.text:
            texte = unicode_elem.text.strip()

        # On ne garde que les lignes qui ont un texte ET un polygone
        if texte and polygon:
            lignes.append({"polygon": polygon, "text": texte})

    return lignes


def _extraire_lignes_alto(chemin_xml: Path) -> list[dict]:
    """Extrait les lignes d'un fichier ALTO XML.

    En ALTO, une ligne est un <TextLine> qui contient des <String> ;
    on concatène les CONTENT des String pour reconstituer le texte.
    La position est donnée par des attributs HPOS/VPOS/WIDTH/HEIGHT.

    Args:
        chemin_xml: Chemin du fichier ALTO XML.

    Returns:
        Liste de dicts {polygon, text}.
    """
    arbre = ET.parse(chemin_xml)
    racine = arbre.getroot()

    ns_uri = NS_ALTO if NS_ALTO in racine.tag else NS_ALTO_ALT
    ns = {"a": ns_uri}

    lignes = []
    for text_line in racine.findall(".//a:TextLine", ns):
        # Le texte = concaténation des <String CONTENT="...">
        mots = []
        for string in text_line.findall("a:String", ns):
            contenu = string.get("CONTENT", "")
            if contenu:
                mots.append(contenu)
        texte = " ".join(mots).strip()

        # La position : soit un polygone (Shape/Polygon), soit une boîte HPOS/VPOS
        polygon = _polygone_depuis_alto(text_line, ns)

        if texte and polygon:
            lignes.append({"polygon": polygon, "text": texte})

    return lignes


def _polygone_depuis_alto(text_line, ns: dict) -> list[list[float]]:
    """Extrait le polygone d'une ligne ALTO (Polygon ou boîte HPOS/VPOS).

    Args:
        text_line: L'élément <TextLine> ALTO.
        ns: Le namespace ALTO.

    Returns:
        Le polygone en liste de points [[x, y], ...].
    """
    # Cas 1 : un vrai polygone est présent
    polygon_elem = text_line.find(".//a:Polygon", ns)
    if polygon_elem is not None:
        points_str = polygon_elem.get("POINTS", "")
        # ALTO sépare les nombres par des espaces : "x1 y1 x2 y2 ..."
        nombres = points_str.split()
        points = []
        for i in range(0, len(nombres) - 1, 2):
            points.append([float(nombres[i]), float(nombres[i + 1])])
        if len(points) >= 3:
            return points

    # Cas 2 : une simple boîte englobante (HPOS, VPOS, WIDTH, HEIGHT)
    try:
        x = float(text_line.get("HPOS", 0))
        y = float(text_line.get("VPOS", 0))
        w = float(text_line.get("WIDTH", 0))
        h = float(text_line.get("HEIGHT", 0))
        if w > 0 and h > 0:
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    except (TypeError, ValueError):
        pass

    return []


def _parser_points(points_str: str) -> list[list[float]]:
    """Convertit une chaîne PAGE 'x1,y1 x2,y2 ...' en polygone.

    Args:
        points_str: Chaîne de coordonnées PAGE XML.

    Returns:
        Liste de points [[x, y], ...].
    """
    polygon = []
    for paire in points_str.split():
        if "," in paire:
            x, y = paire.split(",")
            polygon.append([float(x), float(y)])
    return polygon


# ─── Découpe de l'image d'une ligne ──────────────────────────────────────────

def decouper_ligne(image: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    """Découpe l'image d'une ligne à partir de son polygone.

    On calcule la boîte englobante du polygone et on découpe ce rectangle.
    (Un découpage au polygone exact est possible mais rarement nécessaire.)

    Args:
        image: L'image complète de la page.
        polygon: Le polygone de la ligne [[x, y], ...].

    Returns:
        L'image découpée de la ligne.

    Example:
        >>> ligne_img = decouper_ligne(image_page, polygon)
    """
    points = np.array(polygon)
    x_min = int(np.min(points[:, 0]))
    x_max = int(np.max(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    y_max = int(np.max(points[:, 1]))

    # Sécurité : rester dans les limites de l'image
    h, w = image.shape[:2]
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)

    return image[y_min:y_max, x_min:x_max]


# ─── Chargement complet du corpus ────────────────────────────────────────────

def charger_corpus(
    dossier_corpus: str | Path,
    metadata: dict | None = None,
    limite: int | None = None,
) -> list[dict]:
    """Charge un corpus entier en exemples d'entraînement.

    C'est LA fonction à appeler après avoir téléchargé un corpus.
    Elle produit une liste d'exemples, chacun avec l'image d'une ligne et
    sa transcription de référence (vérité terrain).

    Args:
        dossier_corpus: Racine du corpus téléchargé.
        metadata: Métadonnées communes (century, source...) à attacher.
        limite: Nombre max d'exemples à charger (utile pour tester vite).

    Returns:
        Liste d'exemples {image, text, polygon, metadata}. La clé 'text' est
        la vérité terrain, indispensable pour mesurer le CER.

    Example:
        >>> exemples = charger_corpus(
        ...     "data/raw/cremma-medieval",
        ...     metadata={"century": 13, "source": "CREMMA"},
        ... )
        >>> print(f"{len(exemples)} lignes chargées")
        >>> print(exemples[0]["text"])  # vérité terrain de la 1re ligne
    """
    dossier_corpus = Path(dossier_corpus)
    metadata = metadata or {}

    paires = trouver_paires(dossier_corpus)
    print(f"  → {len(paires)} page(s) trouvée(s) dans {dossier_corpus.name}")

    exemples = []
    for chemin_image, chemin_xml in paires:
        # Charge l'image de la page
        image_page = cv2.imread(str(chemin_image), cv2.IMREAD_GRAYSCALE)
        if image_page is None:
            print(f"  ⚠ Image illisible, ignorée : {chemin_image.name}")
            continue

        # Extrait les lignes (polygone + vérité terrain) du XML
        try:
            lignes = extraire_lignes_xml(chemin_xml)
        except (ET.ParseError, ValueError) as e:
            print(f"  ⚠ XML illisible ({chemin_xml.name}) : {e}")
            continue

        # Pour chaque ligne, découpe l'image et crée un exemple
        for ligne in lignes:
            image_ligne = decouper_ligne(image_page, ligne["polygon"])
            if image_ligne.size == 0:
                continue  # ligne vide/hors limites

            exemples.append({
                "image": image_ligne,
                "text": ligne["text"],       # vérité terrain
                "polygon": ligne["polygon"],
                "metadata": metadata,
            })

            if limite is not None and len(exemples) >= limite:
                print(f"  → limite de {limite} exemples atteinte")
                return exemples

    print(f"  ✓ {len(exemples)} ligne(s) chargée(s) avec vérité terrain")
    return exemples


# ─── Point d'entrée (démonstration avec un faux corpus au bon format) ────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.utils import fixer_seeds
    from htr.page_xml import exporter_page_xml
    from htr.segmentation import segmenter_lignes, creer_image_multi_lignes

    print("=== Démonstration de corpus_loader.py ===\n")
    fixer_seeds(42)

    # Comme on n'a pas CREMMA ici, on FABRIQUE un mini-corpus au bon format :
    # une image + un PAGE XML avec transcriptions (comme le vrai CREMMA).
    faux_corpus = Path("data/raw/faux_corpus")
    faux_corpus.mkdir(parents=True, exist_ok=True)

    # 1. Créer une image de page
    img_path = faux_corpus / "page_001.png"
    creer_image_multi_lignes(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 2. Segmenter et créer le PAGE XML AVEC transcriptions (= vérité terrain)
    lignes = segmenter_lignes(img_bin)
    verites = ["Ci comence li romanz", "de Brut e de sa gent",
               "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    exporter_page_xml(
        lignes, "page_001.png", img.shape[1], img.shape[0],
        faux_corpus / "page_001.xml", transcriptions=verites,
    )

    # 3. Maintenant, charger ce corpus comme si c'était CREMMA
    print("\n─── Chargement du corpus ───")
    exemples = charger_corpus(
        faux_corpus,
        metadata={"century": 12, "source": "FAUX-CREMMA"},
    )

    print("\n─── Exemples chargés ───")
    for ex in exemples[:4]:
        h, w = ex["image"].shape
        print(f"  Image {w}×{h} px  →  vérité terrain : \"{ex['text']}\"")

    print("\n=== Démonstration terminée ✓ ===")
    print("\nChez toi, remplace faux_corpus par le vrai chemin :")
    print('  charger_corpus("data/raw/cremma-medieval", {"century": 13, "source": "CREMMA"})')
