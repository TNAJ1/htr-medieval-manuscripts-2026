"""
test_htr_model.py — Tests du transcripteur HTR (mode simulation).

On teste en mode simulation pour ne pas dépendre du téléchargement de TrOCR.
Ces tests valident la LOGIQUE : structure de sortie, char_confidences,
détection des candidats. Le mode réel produit le même format de sortie.

Lancer : pytest tests/test_htr_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.utils import fixer_seeds
from htr.htr_model import TranscripteurHTR


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def transcripteur():
    """Un transcripteur en mode simulation (pas de téléchargement)."""
    fixer_seeds(42)  # Pour des résultats reproductibles
    return TranscripteurHTR(mode_simulation=True)


@pytest.fixture
def image_ligne():
    """Une image factice de ligne (300×40, blanche)."""
    return np.ones((40, 300), dtype=np.uint8) * 255


# ─── Tests de la structure de sortie ─────────────────────────────────────────

class TestStructureSortie:
    """Tests du format de sortie du transcripteur."""

    def test_retourne_dict(self, transcripteur, image_ligne):
        """La sortie doit être un dictionnaire."""
        resultat = transcripteur.transcrire(image_ligne)
        assert isinstance(resultat, dict)

    def test_contient_cles_obligatoires(self, transcripteur, image_ligne):
        """La sortie doit contenir text, char_confidences, candidates."""
        resultat = transcripteur.transcrire(image_ligne)
        for cle in ["text", "char_confidences", "candidates"]:
            assert cle in resultat, f"Clé manquante : {cle}"

    def test_text_est_une_chaine(self, transcripteur, image_ligne):
        """Le champ text doit être une chaîne de caractères."""
        resultat = transcripteur.transcrire(image_ligne)
        assert isinstance(resultat["text"], str)

    def test_char_confidences_est_une_liste(self, transcripteur, image_ligne):
        """char_confidences doit être une liste."""
        resultat = transcripteur.transcrire(image_ligne)
        assert isinstance(resultat["char_confidences"], list)


# ─── Tests de cohérence des confiances ───────────────────────────────────────

class TestConfiances:
    """Tests des char_confidences."""

    def test_une_confiance_par_caractere(self, transcripteur, image_ligne):
        """Il doit y avoir autant de confiances que de caractères."""
        resultat = transcripteur.transcrire(image_ligne)
        assert len(resultat["char_confidences"]) == len(resultat["text"])

    def test_confiances_dans_plage_valide(self, transcripteur, image_ligne):
        """Toutes les confiances doivent être entre 0 et 1."""
        resultat = transcripteur.transcrire(image_ligne)
        for c in resultat["char_confidences"]:
            assert 0.0 <= c <= 1.0, f"Confiance hors plage : {c}"


# ─── Tests des candidats ─────────────────────────────────────────────────────

class TestCandidats:
    """Tests de la détection des positions incertaines."""

    def test_candidats_pointent_positions_faibles(self):
        """Les candidats doivent correspondre aux confiances < seuil."""
        # On teste directement la méthode statique avec des valeurs connues
        confidences = [0.9, 0.5, 0.95, 0.3]  # positions 1 et 3 sont < 0.7
        candidats = TranscripteurHTR._identifier_candidats(confidences, seuil=0.7)

        positions = [c["position"] for c in candidats]
        assert positions == [1, 3]

    def test_candidats_ont_des_options(self):
        """Chaque candidat doit proposer des options de correction."""
        confidences = [0.5]
        candidats = TranscripteurHTR._identifier_candidats(confidences, seuil=0.7)
        assert len(candidats) == 1
        assert "options" in candidats[0]
        assert len(candidats[0]["options"]) > 0

    def test_aucun_candidat_si_tout_confiant(self):
        """Aucun candidat si toutes les confiances sont hautes."""
        confidences = [0.9, 0.95, 0.99]
        candidats = TranscripteurHTR._identifier_candidats(confidences, seuil=0.7)
        assert candidats == []


# ─── Tests de la fonction d'ajustement de longueur ───────────────────────────

class TestAjusterLongueur:
    """Tests de la méthode utilitaire _ajuster_longueur."""

    def test_tronque_si_trop_long(self):
        """Une liste trop longue est tronquée."""
        resultat = TranscripteurHTR._ajuster_longueur([0.9, 0.8, 0.7], 2)
        assert resultat == [0.9, 0.8]

    def test_complete_si_trop_court(self):
        """Une liste trop courte est complétée."""
        resultat = TranscripteurHTR._ajuster_longueur([0.9], 3)
        assert len(resultat) == 3

    def test_liste_vide_completee(self):
        """Une liste vide est complétée avec 0.5."""
        resultat = TranscripteurHTR._ajuster_longueur([], 2)
        assert resultat == [0.5, 0.5]


# ─── Tests d'intégration avec le data contract ───────────────────────────────

class TestIntegrationDataContract:
    """Vérifie que la sortie HTR peut alimenter le data contract."""

    def test_sortie_compatible_creer_ligne(self, transcripteur, image_ligne):
        """La sortie HTR doit permettre de créer une ligne du data contract."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from shared.data_contract import creer_ligne

        resultat = transcripteur.transcrire(image_ligne)

        # On doit pouvoir créer une ligne valide à partir de la sortie HTR
        ligne = creer_ligne(
            line_id="test_001",
            text=resultat["text"],
            polygon=[[0, 0], [300, 0], [300, 40], [0, 40]],
            char_confidences=resultat["char_confidences"],
            candidates=resultat["candidates"],
        )
        assert ligne["text"] == resultat["text"]
        assert "needs_review" in ligne
