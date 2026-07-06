"""
test_pipeline.py — Tests du pipeline complet et du split de données.

Ces tests valident :
  - Que le pipeline produit un data contract valide de bout en bout
  - Que le split est stratifié et reproductible
  - Que le scellement SHA-256 détecte les modifications

C'est aussi le TEST DE NON-RÉGRESSION exigé par le sujet : le pipeline
doit produire un document conforme sur une image de référence.

Lancer : pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.utils import fixer_seeds
from shared.data_contract import valider_document
from htr.segmentation import creer_image_multi_lignes
from htr.pipeline import traiter_manuscrit
from htr.dataset_split import (
    distribution_strates,
    sceller_test_set,
    split_stratifie,
    verifier_integrite_test_set,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def image_manuscrit(tmp_path):
    """Crée une image de manuscrit de test."""
    chemin = tmp_path / "manuscrit.png"
    creer_image_multi_lignes(chemin)
    return chemin


@pytest.fixture
def echantillons_fictifs():
    """150 échantillons répartis sur 3 siècles pour tester le split."""
    echantillons = []
    for i in range(150):
        echantillons.append({
            "id": f"line_{i}",
            "century": [12, 13, 14][i % 3],
            "text": f"texte {i}",
        })
    return echantillons


# ─── Tests du pipeline de bout en bout ───────────────────────────────────────

class TestPipelineComplet:
    """Tests d'intégration du pipeline complet."""

    def test_produit_document_valide(self, image_manuscrit, tmp_path):
        """Le pipeline doit produire un data contract conforme au schéma.

        C'est le TEST DE NON-RÉGRESSION principal : si une modification casse
        la production du data contract, ce test échoue.
        """
        fixer_seeds(42)
        document = traiter_manuscrit(
            chemin_image=image_manuscrit,
            document_id="test-pipeline-001",
            century=13,
            document_type="texte_simple",
            language="ancien_francais",
            source="TEST",
            dossier_sortie=tmp_path / "json",
            dossier_xml=tmp_path / "xml",
            mode_simulation=True,
        )

        valide, message = valider_document(document)
        assert valide is True, f"Document invalide : {message}"

    def test_produit_fichier_json(self, image_manuscrit, tmp_path):
        """Le pipeline doit créer le fichier JSON du data contract."""
        fixer_seeds(42)
        traiter_manuscrit(
            chemin_image=image_manuscrit,
            document_id="test-002",
            century=13,
            document_type="texte_simple",
            language="ancien_francais",
            source="TEST",
            dossier_sortie=tmp_path / "json",
            dossier_xml=tmp_path / "xml",
            mode_simulation=True,
        )
        assert (tmp_path / "json" / "test-002.json").exists()

    def test_produit_fichier_page_xml(self, image_manuscrit, tmp_path):
        """Le pipeline doit créer le fichier PAGE XML."""
        fixer_seeds(42)
        traiter_manuscrit(
            chemin_image=image_manuscrit,
            document_id="test-003",
            century=13,
            document_type="texte_simple",
            language="ancien_francais",
            source="TEST",
            dossier_sortie=tmp_path / "json",
            dossier_xml=tmp_path / "xml",
            mode_simulation=True,
        )
        assert (tmp_path / "xml" / "test-003.xml").exists()

    def test_toutes_les_lignes_ont_un_polygone(self, image_manuscrit, tmp_path):
        """Chaque ligne du data contract doit avoir un polygone non vide."""
        fixer_seeds(42)
        document = traiter_manuscrit(
            chemin_image=image_manuscrit,
            document_id="test-004",
            century=13,
            document_type="texte_simple",
            language="ancien_francais",
            source="TEST",
            dossier_sortie=tmp_path / "json",
            dossier_xml=tmp_path / "xml",
            mode_simulation=True,
        )
        for ligne in document["lines"]:
            assert len(ligne["polygon"]) >= 3, "Polygone invalide"

    def test_reproductible(self, image_manuscrit, tmp_path):
        """Deux exécutions avec le même seed doivent produire le même texte."""
        fixer_seeds(42)
        doc1 = traiter_manuscrit(
            chemin_image=image_manuscrit, document_id="r1", century=13,
            document_type="texte_simple", language="ancien_francais",
            source="TEST", dossier_sortie=tmp_path / "j1",
            dossier_xml=tmp_path / "x1", mode_simulation=True,
        )
        fixer_seeds(42)
        doc2 = traiter_manuscrit(
            chemin_image=image_manuscrit, document_id="r2", century=13,
            document_type="texte_simple", language="ancien_francais",
            source="TEST", dossier_sortie=tmp_path / "j2",
            dossier_xml=tmp_path / "x2", mode_simulation=True,
        )
        textes1 = [l["text"] for l in doc1["lines"]]
        textes2 = [l["text"] for l in doc2["lines"]]
        assert textes1 == textes2


# ─── Tests du split stratifié ────────────────────────────────────────────────

class TestSplitStratifie:
    """Tests du découpage train/val/test."""

    def test_proportions_respectees(self, echantillons_fictifs):
        """Les tailles des splits doivent respecter les proportions."""
        splits = split_stratifie(echantillons_fictifs, proportions=(0.7, 0.15, 0.15))
        total = len(echantillons_fictifs)
        # Train ~70 %, on tolère une marge due aux arrondis par strate
        assert abs(len(splits["train"]) - 0.7 * total) < 10

    def test_aucun_echantillon_perdu(self, echantillons_fictifs):
        """La somme des splits doit égaler le nombre d'échantillons."""
        splits = split_stratifie(echantillons_fictifs)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(echantillons_fictifs)

    def test_aucun_chevauchement(self, echantillons_fictifs):
        """Un échantillon ne doit jamais être dans deux splits à la fois."""
        splits = split_stratifie(echantillons_fictifs)
        ids_train = {e["id"] for e in splits["train"]}
        ids_val = {e["id"] for e in splits["val"]}
        ids_test = {e["id"] for e in splits["test"]}
        # Les intersections doivent être vides
        assert ids_train & ids_val == set()
        assert ids_train & ids_test == set()
        assert ids_val & ids_test == set()

    def test_stratification_equilibree(self, echantillons_fictifs):
        """Chaque siècle doit être présent dans chaque split."""
        splits = split_stratifie(echantillons_fictifs)
        dist = distribution_strates(splits, "century")
        for nom_split in ["train", "val", "test"]:
            # Les 3 siècles (12, 13, 14) doivent apparaître
            assert set(dist[nom_split].keys()) == {12, 13, 14}

    def test_reproductible(self, echantillons_fictifs):
        """Même seed → même split."""
        s1 = split_stratifie(echantillons_fictifs, seed=42)
        s2 = split_stratifie(echantillons_fictifs, seed=42)
        ids1 = [e["id"] for e in s1["train"]]
        ids2 = [e["id"] for e in s2["train"]]
        assert ids1 == ids2

    def test_proportions_invalides_levent_erreur(self, echantillons_fictifs):
        """Des proportions qui ne somment pas à 1 → ValueError."""
        with pytest.raises(ValueError):
            split_stratifie(echantillons_fictifs, proportions=(0.5, 0.3, 0.3))


# ─── Tests du scellement SHA-256 ─────────────────────────────────────────────

class TestScellement:
    """Tests du scellement et de la vérification d'intégrité."""

    def test_hash_a_la_bonne_longueur(self, echantillons_fictifs, tmp_path):
        """Un SHA-256 doit faire 64 caractères hexadécimaux."""
        splits = split_stratifie(echantillons_fictifs)
        h = sceller_test_set(splits["test"], tmp_path / "test.json")
        assert len(h) == 64

    def test_integrite_ok_si_inchange(self, echantillons_fictifs, tmp_path):
        """Un test set non modifié doit passer la vérification."""
        splits = split_stratifie(echantillons_fictifs)
        chemin = tmp_path / "test.json"
        h = sceller_test_set(splits["test"], chemin)
        assert verifier_integrite_test_set(chemin, h) is True

    def test_integrite_detecte_modification(self, echantillons_fictifs, tmp_path):
        """Un test set modifié doit être détecté."""
        splits = split_stratifie(echantillons_fictifs)
        chemin = tmp_path / "test.json"
        h = sceller_test_set(splits["test"], chemin)

        # On modifie le fichier
        chemin.write_text(chemin.read_text() + "modification frauduleuse")

        # La vérification doit échouer
        assert verifier_integrite_test_set(chemin, h) is False

    def test_hash_reproductible(self, echantillons_fictifs, tmp_path):
        """Le même test set doit toujours produire le même hash."""
        splits = split_stratifie(echantillons_fictifs, seed=42)
        h1 = sceller_test_set(splits["test"], tmp_path / "t1.json")
        h2 = sceller_test_set(splits["test"], tmp_path / "t2.json")
        assert h1 == h2
