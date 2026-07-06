"""
test_data_contract.py — Tests du data contract HTR → NLP.

Les consignes NLP exigent explicitement un test qui valide le schéma JSON
du data contract. Ces tests vérifient :
  - Qu'un document correct est accepté
  - Qu'un document avec un champ manquant est rejeté
  - Que le calcul automatique de confidence et needs_review est correct
  - Que la validation bloque la sauvegarde d'un document invalide

Lancer : pytest tests/test_data_contract.py -v
"""

import sys
from pathlib import Path

import pytest

# Ajoute src/ au chemin Python
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.data_contract import (
    SEUIL_NEEDS_REVIEW,
    calculer_taux_needs_review,
    charger_schema,
    creer_document,
    creer_ligne,
    sauvegarder_document,
    valider_document,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def ligne_valide():
    """Une ligne de transcription correcte et confiante."""
    return creer_ligne(
        line_id="test_line001",
        text="exemple",
        polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
        char_confidences=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        reading_order=0,
    )


@pytest.fixture
def document_valide(ligne_valide):
    """Un document complet et conforme au schéma."""
    return creer_document(
        document_id="test-doc-001",
        century=13,
        document_type="texte_simple",
        language="ancien_francais",
        source="TEST",
        lines=[ligne_valide],
        image_width=1000,
        image_height=1500,
    )


# ─── Tests du chargement du schéma ───────────────────────────────────────────

class TestChargerSchema:
    """Tests du chargement du schéma JSON."""

    def test_schema_se_charge(self):
        """Le schéma doit se charger sans erreur."""
        schema = charger_schema()
        assert schema is not None

    def test_schema_a_un_titre(self):
        """Le schéma doit contenir un titre."""
        schema = charger_schema()
        assert "title" in schema

    def test_schema_exige_les_bons_champs(self):
        """Le schéma doit exiger les champs racine attendus."""
        schema = charger_schema()
        champs_requis = schema["required"]
        for champ in ["document_id", "metadata", "coordinate_system", "lines"]:
            assert champ in champs_requis


# ─── Tests de creer_ligne ────────────────────────────────────────────────────

class TestCreerLigne:
    """Tests de la création de lignes."""

    def test_confidence_est_la_moyenne(self):
        """La confiance globale doit être la moyenne des char_confidences."""
        ligne = creer_ligne(
            line_id="l1",
            text="ab",
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            char_confidences=[0.6, 0.8],  # moyenne = 0.7
        )
        assert ligne["confidence"] == 0.7

    def test_needs_review_active_si_faible_confiance(self):
        """needs_review doit être True si confiance < seuil."""
        ligne = creer_ligne(
            line_id="l2",
            text="ab",
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            char_confidences=[0.3, 0.3],  # moyenne 0.3 < seuil 0.7
        )
        assert ligne["needs_review"] is True

    def test_needs_review_inactif_si_haute_confiance(self):
        """needs_review doit être False si confiance >= seuil."""
        ligne = creer_ligne(
            line_id="l3",
            text="ab",
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            char_confidences=[0.95, 0.95],  # moyenne 0.95 >= seuil
        )
        assert ligne["needs_review"] is False

    def test_confiance_hors_plage_leve_erreur(self):
        """Une confiance > 1 doit lever une ValueError."""
        with pytest.raises(ValueError):
            creer_ligne(
                line_id="l4",
                text="a",
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
                char_confidences=[1.5],  # invalide !
            )

    def test_candidates_ajoutes_si_fournis(self):
        """Le champ candidates doit apparaître si on le fournit."""
        ligne = creer_ligne(
            line_id="l5",
            text="abc",
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            char_confidences=[0.9, 0.5, 0.9],
            candidates=[{"position": 1, "options": ["b", "h"]}],
        )
        assert "candidates" in ligne
        assert ligne["candidates"][0]["options"] == ["b", "h"]


# ─── Tests de validation ─────────────────────────────────────────────────────

class TestValiderDocument:
    """Tests de la validation contre le schéma JSON."""

    def test_document_valide_accepte(self, document_valide):
        """Un document correct doit être validé."""
        valide, message = valider_document(document_valide)
        assert valide is True, f"Document rejeté : {message}"

    def test_champ_manquant_rejete(self, document_valide):
        """Un document sans 'metadata' doit être rejeté."""
        # On retire un champ obligatoire
        del document_valide["metadata"]
        valide, message = valider_document(document_valide)
        assert valide is False

    def test_siecle_invalide_rejete(self, document_valide):
        """Un siècle hors de [8, 17] doit être rejeté."""
        document_valide["metadata"]["century"] = 25  # impossible
        valide, message = valider_document(document_valide)
        assert valide is False

    def test_type_document_invalide_rejete(self, document_valide):
        """Un document_type non listé dans l'enum doit être rejeté."""
        document_valide["metadata"]["document_type"] = "format_inconnu"
        valide, message = valider_document(document_valide)
        assert valide is False

    def test_polygon_trop_court_rejete(self, document_valide):
        """Un polygone avec moins de 3 points doit être rejeté."""
        # Un polygone valide a au moins 3 points (un triangle)
        document_valide["lines"][0]["polygon"] = [[0, 0], [10, 10]]
        valide, message = valider_document(document_valide)
        assert valide is False


# ─── Tests de la sauvegarde ──────────────────────────────────────────────────

class TestSauvegarde:
    """Tests de la sauvegarde sécurisée."""

    def test_sauvegarde_document_valide(self, document_valide, tmp_path):
        """Un document valide doit être sauvegardé sur le disque."""
        chemin = tmp_path / "doc.json"
        sauvegarder_document(document_valide, chemin)
        assert chemin.exists()

    def test_sauvegarde_document_invalide_bloquee(self, document_valide, tmp_path):
        """Un document invalide ne doit PAS être écrit sur le disque."""
        del document_valide["lines"]  # rend le document invalide
        chemin = tmp_path / "doc_invalide.json"

        with pytest.raises(ValueError):
            sauvegarder_document(document_valide, chemin)

        # Le fichier ne doit pas avoir été créé
        assert not chemin.exists()


# ─── Tests du taux needs_review ──────────────────────────────────────────────

class TestTauxNeedsReview:
    """Tests du calcul du taux needs_review."""

    def test_taux_zero_si_toutes_confiantes(self):
        """Si toutes les lignes sont confiantes, le taux doit être 0."""
        lignes = [
            creer_ligne(f"l{i}", "ab", [[0, 0], [1, 0], [1, 1], [0, 1]], [0.9, 0.9])
            for i in range(5)
        ]
        doc = creer_document(
            "d", 13, "texte_simple", "ancien_francais", "T",
            lignes, 100, 100,
        )
        assert calculer_taux_needs_review(doc) == 0.0

    def test_taux_calcule_correctement(self):
        """Le taux doit refléter la proportion de lignes incertaines."""
        # 2 lignes confiantes + 2 lignes incertaines = 50 %
        lignes_ok = [
            creer_ligne(f"ok{i}", "ab", [[0, 0], [1, 0], [1, 1], [0, 1]], [0.9, 0.9])
            for i in range(2)
        ]
        lignes_ko = [
            creer_ligne(f"ko{i}", "ab", [[0, 0], [1, 0], [1, 1], [0, 1]], [0.2, 0.2])
            for i in range(2)
        ]
        doc = creer_document(
            "d", 13, "texte_simple", "ancien_francais", "T",
            lignes_ok + lignes_ko, 100, 100,
        )
        assert calculer_taux_needs_review(doc) == 0.5
