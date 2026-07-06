"""
test_kraken_comparaison.py — Tests de Kraken et de la comparaison de modèles.

Ces tests vérifient :
  - Que Kraken (mode simulation) produit le même format que TrOCR
  - Que l'alignement Needleman-Wunsch est correct
  - Que la fusion garde bien le caractère le plus confiant
  - Que la comparaison McNemar fonctionne

Lancer : pytest tests/test_kraken_comparaison.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.utils import fixer_seeds
from htr.kraken_model import TranscripteurKraken, commande_fine_tuning
from htr.comparaison import (
    _aligner_needleman_wunsch,
    comparer_modeles,
    fusionner_transcriptions,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def transcripteur_kraken():
    """Un transcripteur Kraken en mode simulation."""
    fixer_seeds(42)
    return TranscripteurKraken(mode_simulation=True)


@pytest.fixture
def image_ligne():
    """Une image factice de ligne."""
    return np.ones((40, 300), dtype=np.uint8) * 255


# ─── Tests de Kraken (même format que TrOCR) ─────────────────────────────────

class TestKrakenFormat:
    """Vérifie que Kraken produit le même format que TrOCR."""

    def test_retourne_dict(self, transcripteur_kraken, image_ligne):
        """La sortie doit être un dictionnaire."""
        resultat = transcripteur_kraken.transcrire(image_ligne)
        assert isinstance(resultat, dict)

    def test_memes_cles_que_trocr(self, transcripteur_kraken, image_ligne):
        """Kraken doit produire text, char_confidences, candidates."""
        resultat = transcripteur_kraken.transcrire(image_ligne)
        for cle in ["text", "char_confidences", "candidates"]:
            assert cle in resultat

    def test_une_confiance_par_caractere(self, transcripteur_kraken, image_ligne):
        """Autant de confiances que de caractères (comme TrOCR)."""
        resultat = transcripteur_kraken.transcrire(image_ligne)
        assert len(resultat["char_confidences"]) == len(resultat["text"])

    def test_confiances_valides(self, transcripteur_kraken, image_ligne):
        """Confiances entre 0 et 1."""
        resultat = transcripteur_kraken.transcrire(image_ligne)
        for c in resultat["char_confidences"]:
            assert 0.0 <= c <= 1.0

    def test_interchangeable_avec_trocr(self, image_ligne):
        """Kraken et TrOCR doivent produire des sorties du même type.

        On vérifie qu'on peut traiter les deux de la même façon.
        """
        from htr.htr_model import TranscripteurHTR

        fixer_seeds(42)
        kraken = TranscripteurKraken(mode_simulation=True)
        trocr = TranscripteurHTR(mode_simulation=True)

        res_kraken = kraken.transcrire(image_ligne)
        res_trocr = trocr.transcrire(image_ligne)

        # Les deux ont exactement les mêmes clés
        assert set(res_kraken.keys()) == set(res_trocr.keys())


class TestCommandeFineTuning:
    """Tests de la commande de fine-tuning."""

    def test_contient_ketos(self):
        """La commande doit utiliser ketos train."""
        cmd = commande_fine_tuning()
        assert "ketos train" in cmd

    def test_utilise_format_page(self):
        """La commande doit utiliser le format PAGE XML (notre sortie)."""
        cmd = commande_fine_tuning()
        assert "-f page" in cmd


# ─── Tests de l'alignement Needleman-Wunsch ──────────────────────────────────

class TestAlignement:
    """Tests de l'algorithme d'alignement."""

    def test_chaines_identiques(self):
        """Deux chaînes identiques s'alignent caractère par caractère."""
        alignement = _aligner_needleman_wunsch("chat", "chat")
        assert alignement == [("c", "c"), ("h", "h"), ("a", "a"), ("t", "t")]

    def test_substitution(self):
        """Une substitution garde l'alignement mais avec des caractères différents."""
        alignement = _aligner_needleman_wunsch("chat", "chot")
        # Position 2 : 'a' vs 'o'
        assert alignement[2] == ("a", "o")

    def test_insertion(self):
        """Un caractère en plus dans B crée un gap dans A (None)."""
        alignement = _aligner_needleman_wunsch("cat", "chat")
        # Il doit y avoir un gap (None, 'h') quelque part
        gaps = [paire for paire in alignement if paire[0] is None]
        assert len(gaps) == 1
        assert gaps[0] == (None, "h")

    def test_longueur_alignement(self):
        """L'alignement couvre au moins la plus longue des deux chaînes."""
        alignement = _aligner_needleman_wunsch("abc", "abcdef")
        assert len(alignement) >= 6


# ─── Tests de la fusion ──────────────────────────────────────────────────────

class TestFusion:
    """Tests du vote pondéré par confiance."""

    def test_garde_le_plus_confiant(self):
        """À position égale, on garde le caractère le plus confiant."""
        # A dit 'o' avec confiance 0.9, B dit 'a' avec confiance 0.4
        texte, conf = fusionner_transcriptions(
            "o", [0.9], "a", [0.4]
        )
        assert texte == "o"  # On garde A (plus confiant)

    def test_fusion_corrige_erreur(self):
        """La fusion peut corriger l'erreur d'un modèle."""
        # TrOCR correct et confiant, Kraken se trompe et peu confiant
        texte, conf = fusionner_transcriptions(
            "romanz", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "ramanz", [0.95, 0.4, 0.9, 0.9, 0.9, 0.9],
        )
        assert texte == "romanz"

    def test_confiances_meme_longueur(self):
        """Le texte fusionné et ses confiances ont la même longueur."""
        texte, conf = fusionner_transcriptions(
            "chat", [0.9] * 4, "chot", [0.8] * 4
        )
        assert len(texte) == len(conf)

    def test_textes_identiques_inchanges(self):
        """Si les deux modèles sont d'accord, la fusion est identique."""
        texte, conf = fusionner_transcriptions(
            "gent", [0.9] * 4, "gent", [0.8] * 4
        )
        assert texte == "gent"


# ─── Tests de la comparaison de modèles ──────────────────────────────────────

class TestComparaisonModeles:
    """Tests de comparer_modeles."""

    def test_retourne_cer_des_deux(self):
        """Le résultat contient le CER de chaque modèle."""
        refs = ["chat", "chien"]
        preds_a = ["chat", "chien"]
        preds_b = ["chot", "chien"]
        resultat = comparer_modeles(refs, preds_a, preds_b)
        assert "cer_trocr" in resultat
        assert "cer_kraken" in resultat

    def test_identifie_meilleur_modele(self):
        """Le modèle avec le CER le plus bas est désigné meilleur."""
        refs = ["chat", "chien"]
        preds_trocr = ["chat", "chien"]   # parfait
        preds_kraken = ["chot", "chein"]  # 2 erreurs
        resultat = comparer_modeles(refs, preds_trocr, preds_kraken)
        assert resultat["meilleur_modele"] == "TrOCR"

    def test_egalite_detectee(self):
        """Deux modèles identiques → égalité."""
        refs = ["chat", "chien"]
        preds = ["chat", "chien"]
        resultat = comparer_modeles(refs, preds, preds)
        assert resultat["meilleur_modele"] == "égalité"

    def test_contient_resultat_mcnemar(self):
        """Le résultat contient la p-value de McNemar."""
        refs = ["a", "b", "c"]
        preds_a = ["a", "b", "X"]
        preds_b = ["a", "X", "c"]
        resultat = comparer_modeles(refs, preds_a, preds_b)
        assert "mcnemar_p_value" in resultat
        assert "difference_significative" in resultat
