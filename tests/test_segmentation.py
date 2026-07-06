"""
test_segmentation.py — Tests de la segmentation et de l'export PAGE XML.

Ces tests vérifient :
  - Que la segmentation détecte le bon nombre de lignes
  - Que les lignes sont dans l'ordre de lecture (haut → bas)
  - Que les polygones sont valides (dans les limites de l'image)
  - Que l'export PAGE XML produit un fichier relisible

Lancer : pytest tests/test_segmentation.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ajoute src/ au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htr.segmentation import (
    _grouper_rangees,
    creer_image_multi_lignes,
    dessiner_segmentation,
    segmenter_lignes,
)
from htr.page_xml import (
    _points_vers_polygone,
    _polygone_vers_points,
    exporter_page_xml,
    lire_page_xml,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def image_4_lignes(tmp_path):
    """Crée une image binaire contenant 4 lignes de texte."""
    chemin = tmp_path / "multi.png"
    creer_image_multi_lignes(chemin)
    img = cv2.imread(str(chemin), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img_bin


@pytest.fixture
def lignes_segmentees(image_4_lignes):
    """Retourne les lignes segmentées de l'image de test."""
    return segmenter_lignes(image_4_lignes)


# ─── Tests de _grouper_rangees ───────────────────────────────────────────────

class TestGrouperRangees:
    """Tests de la fonction de regroupement des rangées."""

    def test_groupe_simple(self):
        """Un cas simple : deux blocs séparés."""
        rangees = np.array([False, True, True, False, True, False])
        resultat = _grouper_rangees(rangees)
        assert resultat == [(1, 3), (4, 5)]

    def test_bloc_jusqua_la_fin(self):
        """Un bloc qui va jusqu'à la dernière rangée."""
        rangees = np.array([False, True, True])
        resultat = _grouper_rangees(rangees)
        assert resultat == [(1, 3)]

    def test_aucun_texte(self):
        """Aucune rangée avec texte → aucun bloc."""
        rangees = np.array([False, False, False])
        resultat = _grouper_rangees(rangees)
        assert resultat == []


# ─── Tests de segmenter_lignes ───────────────────────────────────────────────

class TestSegmenterLignes:
    """Tests de la détection de lignes."""

    def test_detecte_4_lignes(self, lignes_segmentees):
        """L'image de test contient 4 lignes, on doit en détecter 4."""
        assert len(lignes_segmentees) == 4

    def test_ordre_de_lecture_croissant(self, lignes_segmentees):
        """Les reading_order doivent être 0, 1, 2, 3 dans l'ordre."""
        ordres = [ligne["reading_order"] for ligne in lignes_segmentees]
        assert ordres == [0, 1, 2, 3]

    def test_lignes_triees_de_haut_en_bas(self, lignes_segmentees):
        """Chaque ligne doit être plus bas que la précédente (y croissant)."""
        positions_y = [ligne["bbox"][1] for ligne in lignes_segmentees]
        # On vérifie que la liste est triée dans l'ordre croissant
        assert positions_y == sorted(positions_y)

    def test_polygones_dans_limites_image(self, lignes_segmentees, image_4_lignes):
        """Tous les points des polygones doivent être dans l'image."""
        hauteur, largeur = image_4_lignes.shape
        for ligne in lignes_segmentees:
            for (x, y) in ligne["polygon"]:
                assert 0 <= x <= largeur, f"x={x} hors limites [0,{largeur}]"
                assert 0 <= y <= hauteur, f"y={y} hors limites [0,{hauteur}]"

    def test_polygone_a_4_points(self, lignes_segmentees):
        """Chaque polygone rectangulaire doit avoir 4 coins."""
        for ligne in lignes_segmentees:
            assert len(ligne["polygon"]) == 4

    def test_chaque_ligne_a_une_image(self, lignes_segmentees):
        """Chaque ligne doit contenir son image découpée non vide."""
        for ligne in lignes_segmentees:
            assert "image" in ligne
            assert ligne["image"].size > 0

    def test_image_vide_leve_erreur(self):
        """Une image vide doit lever une ValueError."""
        with pytest.raises(ValueError):
            segmenter_lignes(np.array([]))

    def test_image_3d_leve_erreur(self):
        """Une image couleur (3D) doit lever une ValueError."""
        image_couleur = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            segmenter_lignes(image_couleur)


# ─── Tests de dessiner_segmentation ──────────────────────────────────────────

class TestDessinerSegmentation:
    """Tests de la visualisation."""

    def test_retourne_image_couleur(self, image_4_lignes, lignes_segmentees):
        """L'image annotée doit être en couleur (3 canaux)."""
        resultat = dessiner_segmentation(image_4_lignes, lignes_segmentees)
        assert resultat.ndim == 3
        assert resultat.shape[2] == 3  # 3 canaux BGR


# ─── Tests de conversion polygone ↔ points ───────────────────────────────────

class TestConversionPolygone:
    """Tests des conversions PAGE XML."""

    def test_polygone_vers_points(self):
        """Conversion polygone → chaîne PAGE XML."""
        polygon = [[10, 20], [100, 40]]
        resultat = _polygone_vers_points(polygon)
        assert resultat == "10,20 100,40"

    def test_points_vers_polygone(self):
        """Conversion chaîne PAGE XML → polygone."""
        points = "10,20 100,40"
        resultat = _points_vers_polygone(points)
        assert resultat == [[10.0, 20.0], [100.0, 40.0]]

    def test_conversion_aller_retour(self):
        """Convertir puis reconvertir doit redonner le polygone d'origine."""
        polygon = [[5, 10], [50, 10], [50, 30], [5, 30]]
        points = _polygone_vers_points(polygon)
        retour = _points_vers_polygone(points)
        assert retour == [[float(x), float(y)] for x, y in polygon]


# ─── Tests de l'export PAGE XML ──────────────────────────────────────────────

class TestExportPageXml:
    """Tests de l'export et relecture PAGE XML."""

    def test_fichier_xml_cree(self, lignes_segmentees, tmp_path):
        """L'export doit créer un fichier XML."""
        chemin = tmp_path / "sortie.xml"
        exporter_page_xml(
            lignes_segmentees, "test.png", 500, 400, chemin,
        )
        assert chemin.exists()

    def test_relecture_meme_nombre_de_lignes(self, lignes_segmentees, tmp_path):
        """Le fichier relu doit contenir autant de lignes qu'à l'écriture."""
        chemin = tmp_path / "sortie.xml"
        exporter_page_xml(lignes_segmentees, "test.png", 500, 400, chemin)

        relues = lire_page_xml(chemin)
        assert len(relues) == len(lignes_segmentees)

    def test_transcriptions_conservees(self, lignes_segmentees, tmp_path):
        """Les transcriptions écrites doivent être relues à l'identique."""
        chemin = tmp_path / "sortie.xml"
        textes = ["ligne A", "ligne B", "ligne C", "ligne D"]
        exporter_page_xml(
            lignes_segmentees, "test.png", 500, 400, chemin,
            transcriptions=textes,
        )

        relues = lire_page_xml(chemin)
        textes_relus = [ligne["text"] for ligne in relues]
        assert textes_relus == textes

    def test_erreur_si_transcriptions_mauvaise_longueur(
        self, lignes_segmentees, tmp_path
    ):
        """Trop peu de transcriptions doit lever une ValueError."""
        chemin = tmp_path / "sortie.xml"
        with pytest.raises(ValueError):
            exporter_page_xml(
                lignes_segmentees, "test.png", 500, 400, chemin,
                transcriptions=["une seule"],  # 1 transcription pour 4 lignes
            )
