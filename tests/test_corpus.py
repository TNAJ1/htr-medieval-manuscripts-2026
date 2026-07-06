"""
test_corpus.py — Tests du chargement de corpus et de l'évaluation sur corpus.

Ces tests vérifient :
  - Que le chargeur trouve les paires (image, XML)
  - Qu'il extrait correctement les lignes et la vérité terrain (PAGE et ALTO)
  - Que l'évaluation produit un rapport de métriques cohérent

On fabrique un mini-corpus au bon format dans un dossier temporaire, donc
ces tests tournent sans télécharger le vrai CREMMA.

Lancer : pytest tests/test_corpus.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.utils import fixer_seeds
from htr.segmentation import segmenter_lignes, creer_image_multi_lignes
from htr.page_xml import exporter_page_xml
from htr.corpus_loader import (
    charger_corpus,
    decouper_ligne,
    extraire_lignes_xml,
    trouver_paires,
    _parser_points,
)
from htr.evaluation_corpus import (
    evaluer_modele_sur_corpus,
    transcrire_corpus,
)
from htr.transcripteur_factory import creer_transcripteur


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def corpus_page_xml(tmp_path):
    """Fabrique un mini-corpus au format PAGE XML (comme CREMMA).

    Retourne le dossier du corpus + la liste des vérités terrain attendues.
    """
    fixer_seeds(42)
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    # Créer une image de page
    img_path = corpus / "page_001.png"
    creer_image_multi_lignes(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Segmenter et exporter le PAGE XML avec transcriptions (vérité terrain)
    lignes = segmenter_lignes(img_bin)
    verites = ["Ci comence li romanz", "de Brut e de sa gent",
               "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    exporter_page_xml(lignes, "page_001.png", img.shape[1], img.shape[0],
                      corpus / "page_001.xml", transcriptions=verites)

    return corpus, verites


# ─── Tests de trouver_paires ─────────────────────────────────────────────────

class TestTrouverPaires:
    """Tests de la détection des paires (image, XML)."""

    def test_trouve_la_paire(self, corpus_page_xml):
        """Doit trouver la paire image/XML créée."""
        corpus, _ = corpus_page_xml
        paires = trouver_paires(corpus)
        assert len(paires) == 1

    def test_paire_contient_image_et_xml(self, corpus_page_xml):
        """Chaque paire doit être (image, xml) avec les bonnes extensions."""
        corpus, _ = corpus_page_xml
        paires = trouver_paires(corpus)
        image, xml = paires[0]
        assert image.suffix == ".png"
        assert xml.suffix == ".xml"

    def test_corpus_vide(self, tmp_path):
        """Un dossier sans paire retourne une liste vide."""
        paires = trouver_paires(tmp_path)
        assert paires == []


# ─── Tests d'extraction des lignes ───────────────────────────────────────────

class TestExtractionLignes:
    """Tests de l'extraction des lignes depuis le XML."""

    def test_extrait_toutes_les_lignes(self, corpus_page_xml):
        """Doit extraire les 4 lignes du XML."""
        corpus, verites = corpus_page_xml
        _, xml = trouver_paires(corpus)[0]
        lignes = extraire_lignes_xml(xml)
        assert len(lignes) == len(verites)

    def test_verite_terrain_correcte(self, corpus_page_xml):
        """Les transcriptions extraites doivent correspondre aux vérités."""
        corpus, verites = corpus_page_xml
        _, xml = trouver_paires(corpus)[0]
        lignes = extraire_lignes_xml(xml)
        textes = [l["text"] for l in lignes]
        assert textes == verites

    def test_chaque_ligne_a_un_polygone(self, corpus_page_xml):
        """Chaque ligne extraite doit avoir un polygone non vide."""
        corpus, _ = corpus_page_xml
        _, xml = trouver_paires(corpus)[0]
        lignes = extraire_lignes_xml(xml)
        for ligne in lignes:
            assert len(ligne["polygon"]) >= 3


# ─── Tests de découpe ────────────────────────────────────────────────────────

class TestDecouperLigne:
    """Tests de la découpe d'image."""

    def test_decoupe_dans_les_limites(self):
        """La découpe reste dans les limites de l'image."""
        image = np.ones((100, 200), dtype=np.uint8) * 255
        polygon = [[10, 20], [80, 20], [80, 50], [10, 50]]
        ligne = decouper_ligne(image, polygon)
        # La ligne découpée doit avoir la taille du polygone
        assert ligne.shape[0] == 30  # hauteur : 50 - 20
        assert ligne.shape[1] == 70  # largeur : 80 - 10

    def test_polygone_hors_limites_est_clampe(self):
        """Un polygone qui dépasse est ramené dans l'image."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        polygon = [[50, 50], [200, 50], [200, 200], [50, 200]]  # dépasse
        ligne = decouper_ligne(image, polygon)
        # La découpe ne doit pas dépasser l'image
        assert ligne.shape[0] <= 100
        assert ligne.shape[1] <= 100


# ─── Tests de _parser_points ─────────────────────────────────────────────────

class TestParserPoints:
    """Tests du parsing des coordonnées PAGE."""

    def test_parse_points_simples(self):
        """Parse une chaîne PAGE en polygone."""
        resultat = _parser_points("10,20 100,20 100,40")
        assert resultat == [[10, 20], [100, 20], [100, 40]]

    def test_chaine_vide(self):
        """Une chaîne vide donne un polygone vide."""
        assert _parser_points("") == []


# ─── Tests du chargement complet ─────────────────────────────────────────────

class TestChargerCorpus:
    """Tests du chargement complet du corpus."""

    def test_charge_tous_les_exemples(self, corpus_page_xml):
        """Doit charger un exemple par ligne."""
        corpus, verites = corpus_page_xml
        exemples = charger_corpus(corpus)
        assert len(exemples) == len(verites)

    def test_exemple_a_image_et_texte(self, corpus_page_xml):
        """Chaque exemple doit contenir une image et sa vérité terrain."""
        corpus, _ = corpus_page_xml
        exemples = charger_corpus(corpus)
        for ex in exemples:
            assert "image" in ex
            assert "text" in ex
            assert ex["image"].size > 0

    def test_metadata_attachee(self, corpus_page_xml):
        """Les métadonnées fournies doivent être attachées aux exemples."""
        corpus, _ = corpus_page_xml
        exemples = charger_corpus(corpus, metadata={"century": 13})
        assert exemples[0]["metadata"]["century"] == 13

    def test_limite_respectee(self, corpus_page_xml):
        """L'option limite doit plafonner le nombre d'exemples."""
        corpus, _ = corpus_page_xml
        exemples = charger_corpus(corpus, limite=2)
        assert len(exemples) == 2


# ─── Tests de l'évaluation ───────────────────────────────────────────────────

class TestEvaluationCorpus:
    """Tests de l'évaluation d'un modèle sur corpus."""

    def test_transcrire_corpus_retourne_deux_listes(self, corpus_page_xml):
        """transcrire_corpus retourne références et prédictions de même taille."""
        corpus, _ = corpus_page_xml
        exemples = charger_corpus(corpus)
        transcripteur = creer_transcripteur("trocr", mode_simulation=True)
        refs, preds = transcrire_corpus(exemples, transcripteur)
        assert len(refs) == len(preds) == len(exemples)

    def test_references_sont_verite_terrain(self, corpus_page_xml):
        """Les références retournées sont bien la vérité terrain."""
        corpus, verites = corpus_page_xml
        exemples = charger_corpus(corpus)
        transcripteur = creer_transcripteur("trocr", mode_simulation=True)
        refs, _ = transcrire_corpus(exemples, transcripteur)
        assert refs == verites

    def test_rapport_contient_cer(self, corpus_page_xml):
        """L'évaluation produit un rapport avec le CER."""
        corpus, _ = corpus_page_xml
        rapport = evaluer_modele_sur_corpus(
            corpus, modele="trocr", mode_simulation=True,
        )
        assert "CER" in rapport
        assert "WER" in rapport
        assert 0.0 <= rapport["CER"]  # le CER est positif

    def test_corpus_vide_leve_erreur(self, tmp_path):
        """Un corpus sans données doit lever une erreur explicite."""
        with pytest.raises(ValueError):
            evaluer_modele_sur_corpus(tmp_path, modele="trocr", mode_simulation=True)
