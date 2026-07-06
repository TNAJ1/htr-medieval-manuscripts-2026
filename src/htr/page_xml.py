"""
page_xml.py — Export des polygones au format PAGE XML.

Le sujet exige (point 6) que les polygones de segmentation soient exportés
dans un format réutilisable par la communauté des humanités numériques.
PAGE XML est LE standard pour encoder la structure d'une page de manuscrit
(utilisé par eScriptorium, Kraken, Transkribus...).

Ce module convertit la sortie de segmentation.py en fichiers PAGE XML valides.
Référence du schéma : https://github.com/PRImA-Research-Lab/PAGE-XML
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET  # Module standard pour créer du XML
from xml.dom import minidom              # Pour formater joliment le XML


# ─── Constante : namespace PAGE XML ──────────────────────────────────────────
# Un "namespace" XML identifie de manière unique le vocabulaire utilisé.
# C'est l'URL officielle du standard PAGE 2019.
PAGE_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


# ─── Fonction principale ─────────────────────────────────────────────────────

def exporter_page_xml(
    lignes: list[dict],
    nom_image: str,
    largeur_image: int,
    hauteur_image: int,
    chemin_sortie: str | Path,
    transcriptions: list[str] | None = None,
) -> Path:
    """Exporte les lignes segmentées au format PAGE XML.

    Produit un fichier XML décrivant la structure de la page : une région
    de texte contenant toutes les lignes, chacune avec son polygone et
    éventuellement sa transcription.

    Args:
        lignes: Sortie de segmenter_lignes() (liste de dicts avec 'polygon').
        nom_image: Nom du fichier image source, ex: 'page_001.jpg'.
        largeur_image: Largeur de l'image en pixels.
        hauteur_image: Hauteur de l'image en pixels.
        chemin_sortie: Chemin du fichier .xml à créer.
        transcriptions: Liste des textes transcrits (optionnel). Si fourni,
            doit avoir la même longueur que `lignes`.

    Returns:
        Le chemin du fichier XML créé.

    Raises:
        ValueError: Si transcriptions n'a pas la même longueur que lignes.

    Example:
        >>> exporter_page_xml(
        ...     lignes, "page_001.jpg", 2000, 3000,
        ...     "segmentations/page_001.xml",
        ...     transcriptions=["Ci comence", "li romanz"]
        ... )
    """
    # Vérification de cohérence
    if transcriptions is not None and len(transcriptions) != len(lignes):
        raise ValueError(
            f"Nombre de transcriptions ({len(transcriptions)}) différent "
            f"du nombre de lignes ({len(lignes)})."
        )

    # ── Élément racine : <PcGts> ────────────────────────────────────────────
    # On déclare le namespace PAGE sur l'élément racine
    racine = ET.Element("PcGts", xmlns=PAGE_NAMESPACE)

    # ── Métadonnées : <Metadata> ────────────────────────────────────────────
    metadata = ET.SubElement(racine, "Metadata")
    createur = ET.SubElement(metadata, "Creator")
    createur.text = "Projet MD5 - Pipeline HTR"
    date_creation = ET.SubElement(metadata, "Created")
    # Format ISO 8601, requis par PAGE XML
    date_creation.text = datetime.now().isoformat()
    date_modif = ET.SubElement(metadata, "LastChange")
    date_modif.text = datetime.now().isoformat()

    # ── Page : <Page> ───────────────────────────────────────────────────────
    # Décrit les dimensions de l'image
    page = ET.SubElement(
        racine,
        "Page",
        imageFilename=nom_image,
        imageWidth=str(largeur_image),
        imageHeight=str(hauteur_image),
    )

    # ── Région de texte : <TextRegion> ──────────────────────────────────────
    # Une région qui contient toutes nos lignes.
    # Dans un projet avancé, on aurait plusieurs régions (colonnes, marges...).
    region = ET.SubElement(page, "TextRegion", id="region_0")

    # Le polygone de la région = toute la zone de l'image (englobe tout)
    coords_region = ET.SubElement(region, "Coords")
    coords_region.set("points", _polygone_vers_points(_polygone_page_entiere(
        largeur_image, hauteur_image)))

    # ── Lignes de texte : <TextLine> ────────────────────────────────────────
    for i, ligne in enumerate(lignes):
        ordre = ligne["reading_order"]

        # Crée l'élément <TextLine> avec un identifiant unique
        text_line = ET.SubElement(region, "TextLine", id=f"line_{ordre}")

        # <Coords> : le polygone de la ligne
        coords = ET.SubElement(text_line, "Coords")
        # On convertit notre polygone [[x,y],...] en chaîne "x1,y1 x2,y2 ..."
        coords.set("points", _polygone_vers_points(ligne["polygon"]))

        # <TextEquiv> : la transcription, si elle est fournie
        if transcriptions is not None:
            text_equiv = ET.SubElement(text_line, "TextEquiv")
            unicode_elem = ET.SubElement(text_equiv, "Unicode")
            unicode_elem.text = transcriptions[i]

    # ── Sauvegarde avec indentation ─────────────────────────────────────────
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    # ET.tostring transforme l'arbre XML en chaîne de caractères (bytes)
    xml_brut = ET.tostring(racine, encoding="utf-8")

    # minidom reformate le XML avec une belle indentation (lisible)
    xml_joli = minidom.parseString(xml_brut).toprettyxml(indent="  ")

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(xml_joli)

    print(f"✓ PAGE XML exporté : {chemin_sortie} ({len(lignes)} lignes)")
    return chemin_sortie


# ─── Fonctions auxiliaires ────────────────────────────────────────────────────

def _polygone_vers_points(polygon: list[list[float]]) -> str:
    """Convertit un polygone [[x,y],...] en chaîne PAGE XML "x1,y1 x2,y2 ...".

    PAGE XML attend les coordonnées sous forme de chaîne, avec les points
    séparés par des espaces et x,y séparés par une virgule.

    Args:
        polygon: Liste de points [[x, y], ...].

    Returns:
        Chaîne de coordonnées, ex: "10,20 100,20 100,40 10,40".

    Example:
        >>> _polygone_vers_points([[10, 20], [100, 40]])
        '10,20 100,40'
    """
    # Pour chaque point [x, y], on crée "x,y" (en entiers), puis on joint par espaces
    # PAGE XML utilise des coordonnées entières
    return " ".join(f"{int(x)},{int(y)}" for x, y in polygon)


def _polygone_page_entiere(largeur: int, hauteur: int) -> list[list[float]]:
    """Crée un polygone rectangulaire couvrant toute l'image.

    Args:
        largeur: Largeur de l'image.
        hauteur: Hauteur de l'image.

    Returns:
        Polygone des 4 coins de l'image.
    """
    return [
        [0, 0],
        [largeur, 0],
        [largeur, hauteur],
        [0, hauteur],
    ]


def lire_page_xml(chemin_xml: str | Path) -> list[dict]:
    """Lit un fichier PAGE XML et en extrait les lignes (pour vérification).

    Permet de relire ce qu'on a écrit, ou de charger des annotations
    existantes (par ex. depuis eScriptorium).

    Args:
        chemin_xml: Chemin du fichier PAGE XML.

    Returns:
        Liste de dicts avec 'line_id', 'polygon' et 'text' (si présent).

    Example:
        >>> lignes = lire_page_xml("segmentations/page_001.xml")
        >>> print(lignes[0]['text'])
    """
    chemin_xml = Path(chemin_xml)
    arbre = ET.parse(chemin_xml)
    racine = arbre.getroot()

    # Le namespace doit être pris en compte pour trouver les éléments
    ns = {"page": PAGE_NAMESPACE}

    lignes = []
    # findall avec './/' cherche tous les <TextLine> à n'importe quelle profondeur
    for text_line in racine.findall(".//page:TextLine", ns):
        line_id = text_line.get("id")

        # Récupère le polygone
        coords = text_line.find("page:Coords", ns)
        points_str = coords.get("points") if coords is not None else ""
        polygon = _points_vers_polygone(points_str)

        # Récupère la transcription si présente
        texte = ""
        unicode_elem = text_line.find(".//page:Unicode", ns)
        if unicode_elem is not None and unicode_elem.text:
            texte = unicode_elem.text

        lignes.append({
            "line_id": line_id,
            "polygon": polygon,
            "text": texte,
        })

    return lignes


def _points_vers_polygone(points_str: str) -> list[list[float]]:
    """Convertit une chaîne PAGE XML "x1,y1 x2,y2 ..." en polygone [[x,y],...].

    Opération inverse de _polygone_vers_points.

    Args:
        points_str: Chaîne de coordonnées PAGE XML.

    Returns:
        Liste de points [[x, y], ...].
    """
    polygon = []
    # On découpe par espaces pour avoir chaque "x,y"
    for paire in points_str.split():
        x, y = paire.split(",")
        polygon.append([float(x), float(y)])
    return polygon


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import cv2
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from htr.segmentation import segmenter_lignes, creer_image_multi_lignes

    print("=== Démonstration de page_xml.py ===\n")

    # 1. Crée et segmente une image de test
    fixture = Path("tests/fixtures/multi_lignes.png")
    creer_image_multi_lignes(fixture)
    img = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    lignes = segmenter_lignes(img_bin)
    print(f"✓ {len(lignes)} lignes segmentées")

    # 2. Exporte en PAGE XML avec des transcriptions fictives
    transcriptions = ["Ci comence li romanz", "de Brut e de sa gent",
                      "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    chemin = exporter_page_xml(
        lignes,
        nom_image="multi_lignes.png",
        largeur_image=img.shape[1],
        hauteur_image=img.shape[0],
        chemin_sortie="segmentations/multi_lignes.xml",
        transcriptions=transcriptions,
    )

    # 3. Relit le fichier pour vérifier
    relues = lire_page_xml(chemin)
    print(f"\n✓ Relecture : {len(relues)} lignes")
    for ligne in relues:
        print(f"  {ligne['line_id']} : \"{ligne['text']}\"")

    print("\n=== Démonstration terminée ✓ ===")
