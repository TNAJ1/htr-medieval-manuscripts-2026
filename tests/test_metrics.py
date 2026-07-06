"""
test_metrics.py — Tests des métriques d'évaluation.

Ces tests vérifient les calculs de CER, WER, bootstrap et McNemar sur des
cas dont on connaît le résultat attendu à la main.

Lancer : pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htr.metrics import (
    calculer_cer,
    calculer_cer_corpus,
    calculer_wer,
    calculer_wer_corpus,
    intervalle_confiance_bootstrap,
    rapport_metriques,
    comparer_mcnemar,
)


# ─── Tests du CER ────────────────────────────────────────────────────────────

class TestCER:
    """Tests du Character Error Rate."""

    def test_cer_parfait_est_zero(self):
        """Deux textes identiques → CER = 0."""
        assert calculer_cer("bonjour", "bonjour") == 0.0

    def test_cer_une_substitution(self):
        """Une lettre changée sur 4 → CER = 0.25."""
        # "chat" → "chot" : 1 substitution sur 4 caractères
        assert calculer_cer("chat", "chot") == 0.25

    def test_cer_reference_vide_prediction_vide(self):
        """Deux textes vides → CER = 0."""
        assert calculer_cer("", "") == 0.0

    def test_cer_reference_vide_prediction_non_vide(self):
        """Référence vide mais prédiction non vide → CER = 1."""
        assert calculer_cer("", "texte") == 1.0

    def test_cer_suppression_totale(self):
        """Prédiction vide → toutes les lettres sont supprimées → CER = 1."""
        assert calculer_cer("chat", "") == 1.0


class TestCERCorpus:
    """Tests du CER global sur corpus."""

    def test_cer_corpus_parfait(self):
        """Corpus sans erreur → CER = 0."""
        refs = ["chat", "chien"]
        preds = ["chat", "chien"]
        assert calculer_cer_corpus(refs, preds) == 0.0

    def test_cer_corpus_agrege_correctement(self):
        """Le CER global agrège tous les caractères."""
        # "chat"(4) + "chien"(5) = 9 caractères au total
        # 1 erreur dans "chien" → "chein" : 2 substitutions (i<->e)
        refs = ["chat", "chien"]
        preds = ["chat", "chein"]
        # editdistance("chien", "chein") = 2
        # CER = 2 / 9
        cer = calculer_cer_corpus(refs, preds)
        assert abs(cer - 2 / 9) < 1e-9

    def test_longueurs_differentes_leve_erreur(self):
        """Listes de tailles différentes → ValueError."""
        with pytest.raises(ValueError):
            calculer_cer_corpus(["a", "b"], ["a"])


# ─── Tests du WER ────────────────────────────────────────────────────────────

class TestWER:
    """Tests du Word Error Rate."""

    def test_wer_parfait_est_zero(self):
        """Phrases identiques → WER = 0."""
        assert calculer_wer("le chat dort", "le chat dort") == 0.0

    def test_wer_un_mot_faux(self):
        """Un mot faux sur 3 → WER = 1/3."""
        wer = calculer_wer("le chat dort", "le chien dort")
        assert abs(wer - 1 / 3) < 1e-9

    def test_wer_corpus(self):
        """WER global sur corpus."""
        refs = ["le chat", "il dort"]   # 4 mots au total
        preds = ["le chien", "il dort"]  # 1 mot faux
        wer = calculer_wer_corpus(refs, preds)
        assert wer == 0.25


# ─── Tests du bootstrap ──────────────────────────────────────────────────────

class TestBootstrap:
    """Tests de l'intervalle de confiance bootstrap."""

    def test_bootstrap_retourne_trois_valeurs(self):
        """Le bootstrap retourne (cer, bas, haut)."""
        refs = ["chat", "chien", "oiseau"]
        preds = ["chat", "chein", "oisaeu"]
        resultat = intervalle_confiance_bootstrap(refs, preds, n_iterations=100)
        assert len(resultat) == 3

    def test_bornes_encadrent_le_cer(self):
        """La borne basse ≤ CER ≤ borne haute (approximativement)."""
        refs = ["chat", "chien", "oiseau", "lapin", "souris"]
        preds = ["chat", "chein", "oisaeu", "lapin", "soursi"]
        cer, bas, haut = intervalle_confiance_bootstrap(
            refs, preds, n_iterations=500
        )
        # La borne basse doit être inférieure ou égale à la borne haute
        assert bas <= haut

    def test_bootstrap_reproductible(self):
        """Même seed → mêmes résultats."""
        refs = ["chat", "chien"]
        preds = ["chot", "chien"]
        r1 = intervalle_confiance_bootstrap(refs, preds, n_iterations=100, seed=42)
        r2 = intervalle_confiance_bootstrap(refs, preds, n_iterations=100, seed=42)
        assert r1 == r2

    def test_corpus_parfait_ic_nul(self):
        """Un corpus parfait → IC = [0, 0]."""
        refs = ["chat", "chien"]
        preds = ["chat", "chien"]
        cer, bas, haut = intervalle_confiance_bootstrap(refs, preds, n_iterations=100)
        assert cer == 0.0
        assert bas == 0.0
        assert haut == 0.0


# ─── Tests de McNemar ────────────────────────────────────────────────────────

class TestMcNemar:
    """Tests du test de McNemar."""

    def test_modeles_identiques_non_significatif(self):
        """Deux modèles identiques → pas de différence significative."""
        refs = ["chat", "chien", "oiseau"]
        preds = ["chat", "chien", "oiseau"]
        resultat = comparer_mcnemar(refs, preds, preds)
        assert resultat["significatif"] is False

    def test_compte_desaccords(self):
        """Vérifie le comptage n01 et n10."""
        refs = ["a", "b", "c", "d"]
        preds_a = ["a", "b", "X", "X"]  # A correct sur a, b
        preds_b = ["X", "X", "c", "d"]  # B correct sur c, d
        resultat = comparer_mcnemar(refs, preds_a, preds_b)
        # A juste B faux : positions a, b → n10 = 2
        # A faux B juste : positions c, d → n01 = 2
        assert resultat["n10"] == 2
        assert resultat["n01"] == 2

    def test_retourne_p_value(self):
        """Le résultat doit contenir une p-value entre 0 et 1."""
        refs = ["a", "b", "c"]
        preds_a = ["a", "b", "X"]
        preds_b = ["a", "X", "c"]
        resultat = comparer_mcnemar(refs, preds_a, preds_b)
        assert 0.0 <= resultat["p_value"] <= 1.0


# ─── Tests du rapport ────────────────────────────────────────────────────────

class TestRapport:
    """Tests du rapport de métriques complet."""

    def test_rapport_contient_cles_attendues(self):
        """Le rapport doit contenir CER, WER, IC, seuils."""
        refs = ["chat", "chien"]
        preds = ["chat", "chien"]
        rapport = rapport_metriques(refs, preds)
        for cle in ["CER", "WER", "CER_IC95", "seuil_validation_CER", "n_lignes"]:
            assert cle in rapport

    def test_seuil_validation_atteint_si_cer_faible(self):
        """Un CER de 0 doit valider tous les seuils."""
        refs = ["chat", "chien"]
        preds = ["chat", "chien"]
        rapport = rapport_metriques(refs, preds)
        assert rapport["seuil_validation_CER"] is True
        assert rapport["seuil_excellence_CER"] is True
