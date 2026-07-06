"""
test_nlp.py — Tests du volet NLP.

Les consignes NLP exigent au moins deux tests :
  1. Valider le schéma JSON du data contract (ingestion).
  2. Vérifier que la normalisation par règles ne DÉGRADE pas le CER sur un
     petit échantillon de référence.

On ajoute aussi des tests unitaires sur chaque règle de normalisation.

Lancer : pytest tests/test_nlp.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shared.data_contract import creer_document, creer_ligne
from nlp.ingestion import (
    analyser_corpus,
    charger_data_contract,
    extraire_toutes_les_lignes,
)
from nlp.normalisation import (
    corriger_par_confiance,
    developper_abreviations,
    normaliser_par_regles,
    normaliser_ligne,
    normaliser_unicode,
    resoudre_tilde,
)
from nlp.evaluation_relative import (
    distance_relative,
    evaluer_avec_verite_terrain,
    evaluer_etapes,
)
from htr.metrics import calculer_cer_corpus


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def document_test(tmp_path):
    """Crée un data contract de test valide sur le disque."""
    ligne = creer_ligne(
        line_id="t_001",
        text="⁊ romanz",
        polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
        char_confidences=[0.9] * 8,
    )
    doc = creer_document(
        document_id="test-nlp-001",
        century=13,
        document_type="texte_simple",
        language="ancien_francais",
        source="TEST",
        lines=[ligne],
        image_width=200,
        image_height=300,
    )
    chemin = tmp_path / "test.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return chemin


# ─── Test exigé 1 : validation du schéma à l'ingestion ───────────────────────

class TestIngestion:
    """Tests de l'ingestion et de la validation du schéma."""

    def test_charge_document_valide(self, document_test):
        """Un data contract valide doit se charger sans erreur."""
        doc = charger_data_contract(document_test)
        assert doc["document_id"] == "test-nlp-001"

    def test_rejette_fichier_inexistant(self, tmp_path):
        """Un fichier absent doit lever une FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            charger_data_contract(tmp_path / "absent.json")

    def test_rejette_document_invalide(self, tmp_path):
        """Un JSON sans les champs requis doit être rejeté."""
        chemin = tmp_path / "invalide.json"
        with open(chemin, "w") as f:
            json.dump({"document_id": "x"}, f)  # incomplet
        with pytest.raises(ValueError):
            charger_data_contract(chemin)

    def test_extraction_lignes_enrichit_metadonnees(self, document_test):
        """Les lignes extraites doivent porter le siècle du document."""
        doc = charger_data_contract(document_test)
        lignes = extraire_toutes_les_lignes([doc])
        assert lignes[0]["century"] == 13

    def test_eda_calcule_statistiques(self, document_test):
        """L'EDA doit produire les statistiques attendues."""
        doc = charger_data_contract(document_test)
        lignes = extraire_toutes_les_lignes([doc])
        stats = analyser_corpus(lignes)
        assert stats["n_lignes"] == 1
        assert "taux_needs_review" in stats
        assert "confiance_moyenne" in stats


# ─── Test exigé 2 : la normalisation ne dégrade pas le CER ───────────────────

class TestNonDegradationCER:
    """Test de non-régression : la normalisation doit aider, pas nuire."""

    def test_normalisation_ne_degrade_pas_cer(self):
        """Sur un échantillon de référence, le CER après règles <= CER avant.

        C'est le test exigé par les consignes NLP. La normalisation par règles
        doit rapprocher le texte de la vérité terrain (ou au pire le laisser
        identique), jamais l'éloigner.
        """
        # Vérité terrain (forme développée, moderne)
        references = ["et que il vint", "mont e gent", "une dame"]
        # Transcriptions brutes du HTR (avec abréviations, tilde)
        brut = ["⁊ q~ il vint", "mõt e gent", "vne dame"]
        # Après normalisation par règles
        normalise = [normaliser_par_regles(t) for t in brut]

        cer_avant = calculer_cer_corpus(references, brut)
        cer_apres = calculer_cer_corpus(references, normalise)

        # Le CER après normalisation doit être <= au CER avant
        assert cer_apres <= cer_avant, (
            f"La normalisation a DÉGRADÉ le CER : "
            f"{cer_avant:.3f} → {cer_apres:.3f}"
        )


# ─── Tests unitaires des règles ──────────────────────────────────────────────

class TestReglesNormalisation:
    """Tests de chaque règle de normalisation."""

    def test_unicode_nfc(self):
        """La normalisation Unicode doit produire une forme NFC stable."""
        import unicodedata
        resultat = normaliser_unicode("café")
        assert unicodedata.is_normalized("NFC", resultat)

    def test_developpe_abreviations(self):
        """La nota tironienne ⁊ doit devenir 'et'."""
        assert developper_abreviations("⁊") == "et"

    def test_developpe_que(self):
        """q~ doit devenir 'que'."""
        assert developper_abreviations("q~") == "que"

    def test_resout_tilde(self):
        """õ doit devenir 'on'."""
        assert resoudre_tilde("mõt") == "mont"

    def test_pipeline_complet_regles(self):
        """Le pipeline complet enchaîne bien toutes les règles."""
        resultat = normaliser_par_regles("⁊ mõt")
        assert resultat == "et mont"

    def test_normalisation_idempotente_sur_texte_propre(self):
        """Un texte déjà propre ne doit pas être modifié."""
        texte_propre = "et mont gent"
        assert normaliser_par_regles(texte_propre) == texte_propre


class TestCorrectionConfiance:
    """Tests de la correction guidée par confiance."""

    def test_corrige_position_faible(self):
        """Une position de faible confiance est corrigée via le lexique."""
        resultat = corriger_par_confiance(
            "rxmanz",
            [0.9, 0.4, 0.9, 0.9, 0.9, 0.9],
            [{"position": 1, "options": ["o", "a"]}],
        )
        assert resultat == "romanz"

    def test_ne_corrige_pas_position_confiante(self):
        """Une position confiante n'est pas touchée même si candidat présent."""
        resultat = corriger_par_confiance(
            "romanz",
            [0.9, 0.95, 0.9, 0.9, 0.9, 0.9],  # position 1 confiante
            [{"position": 1, "options": ["a"]}],
        )
        assert resultat == "romanz"  # inchangé

    def test_normaliser_ligne_ajoute_champ(self):
        """normaliser_ligne doit ajouter text_normalise sans perdre l'original."""
        ligne = {
            "text": "⁊ romanz",
            "char_confidences": [0.9] * 8,
            "candidates": [],
        }
        resultat = normaliser_ligne(ligne)
        assert "text_normalise" in resultat
        assert resultat["text"] == "⁊ romanz"  # original préservé
        assert resultat["text_normalise"] == "et romanz"


# ─── Tests de l'évaluation relative ──────────────────────────────────────────

class TestEvaluationRelative:
    """Tests de la mesure d'impact sans vérité terrain."""

    def test_distance_identique_est_zero(self):
        """Deux textes identiques ont une distance relative nulle."""
        assert distance_relative("romanz", "romanz") == 0.0

    def test_distance_positive_si_different(self):
        """Deux textes différents ont une distance positive."""
        assert distance_relative("⁊ romanz", "et romanz") > 0

    def test_evaluer_etapes_compte_transitions(self):
        """evaluer_etapes doit produire une entrée par transition."""
        versions = {
            "brut": ["⁊ a", "õb"],
            "regles": ["et a", "onb"],
            "correction": ["et a", "onb"],
        }
        impact = evaluer_etapes(versions)
        # 3 étapes → 2 transitions
        assert len(impact) == 2
        assert "brut→regles" in impact

    def test_evaluer_etapes_longueurs_differentes_erreur(self):
        """Des listes de tailles différentes doivent lever une erreur."""
        with pytest.raises(ValueError):
            evaluer_etapes({"a": ["x", "y"], "b": ["x"]})

    def test_cer_baisse_avec_verite_terrain(self):
        """Avec vérité terrain, le CER doit baisser après normalisation."""
        references = ["et romanz"]
        versions = {
            "brut": ["⁊ romanz"],
            "regles": ["et romanz"],
        }
        cers = evaluer_avec_verite_terrain(references, versions)
        assert cers["regles"] <= cers["brut"]


# ═══════════════════════════════════════════════════════════════════
# Tests de la NER (schéma BIO)
# ═══════════════════════════════════════════════════════════════════

from nlp.ner import (
    LABELS_BIO,
    LABEL_IGNORE,
    LABEL_TO_ID,
    ReconnaisseurEntites,
    aligner_labels_sur_tokens,
)


class TestSchemaBIO:
    """Tests du schéma d'étiquetage BIO."""

    def test_schema_contient_O(self):
        """Le label 'O' (Outside) doit exister."""
        assert "O" in LABELS_BIO

    def test_schema_contient_B_et_I_pour_chaque_type(self):
        """Chaque type doit avoir un B- et un I-."""
        for type_entite in ["PER", "LOC", "DATE", "ORG", "TITLE"]:
            assert f"B-{type_entite}" in LABELS_BIO
            assert f"I-{type_entite}" in LABELS_BIO

    def test_nombre_de_labels(self):
        """5 types × 2 (B/I) + 1 (O) = 11 labels."""
        assert len(LABELS_BIO) == 11


class TestReconnaisseurEntites:
    """Tests de l'étiquetage NER en mode simulation."""

    def test_detecte_personne(self):
        """Un nom de personne connu reçoit le label B-PER."""
        ner = ReconnaisseurEntites(mode_simulation=True)
        labels = dict(ner.etiqueter("Charles"))
        assert labels["Charles"] == "B-PER"

    def test_detecte_lieu(self):
        """Un lieu connu reçoit le label B-LOC."""
        ner = ReconnaisseurEntites(mode_simulation=True)
        labels = dict(ner.etiqueter("France"))
        assert labels["France"] == "B-LOC"

    def test_detecte_titre(self):
        """Un titre reçoit le label B-TITLE."""
        ner = ReconnaisseurEntites(mode_simulation=True)
        labels = dict(ner.etiqueter("roi"))
        assert labels["roi"] == "B-TITLE"

    def test_mot_inconnu_est_O(self):
        """Un mot ordinaire reçoit le label O."""
        ner = ReconnaisseurEntites(mode_simulation=True)
        labels = dict(ner.etiqueter("tint"))
        assert labels["tint"] == "O"

    def test_extraction_entites(self):
        """L'extraction regroupe et type les entités."""
        ner = ReconnaisseurEntites(mode_simulation=True)
        entites = ner.extraire_entites("le roi Charles de France")
        types = {e["type"] for e in entites}
        assert "PER" in types
        assert "LOC" in types
        assert "TITLE" in types


class TestAlignementLabels:
    """Tests de l'alignement labels/tokens — le point critique du cours."""

    def test_sous_token_continuation_recoit_ignore(self):
        """Un sous-token de continuation reçoit -100."""
        # "Engleterre" (B-LOC) découpé en 2 sous-tokens, entre [CLS] et [SEP]
        word_ids = [None, 0, 0, None]  # CLS, Engle, ##terre, SEP
        resultat = aligner_labels_sur_tokens(["Engleterre"], ["B-LOC"], word_ids)
        # CLS=-100, premier sous-token=B-LOC, continuation=-100, SEP=-100
        assert resultat[0] == LABEL_IGNORE   # [CLS]
        assert resultat[1] == LABEL_TO_ID["B-LOC"]  # premier sous-token
        assert resultat[2] == LABEL_IGNORE   # continuation
        assert resultat[3] == LABEL_IGNORE   # [SEP]

    def test_tokens_speciaux_ignores(self):
        """Les tokens spéciaux (word_id=None) reçoivent -100."""
        word_ids = [None, 0, None]
        resultat = aligner_labels_sur_tokens(["roi"], ["B-TITLE"], word_ids)
        assert resultat[0] == LABEL_IGNORE
        assert resultat[2] == LABEL_IGNORE

    def test_premier_token_de_chaque_mot_garde_label(self):
        """Le premier sous-token de chaque mot garde le vrai label."""
        # Deux mots, chacun en un seul token
        word_ids = [None, 0, 1, None]
        resultat = aligner_labels_sur_tokens(
            ["roi", "Charles"], ["B-TITLE", "B-PER"], word_ids
        )
        assert resultat[1] == LABEL_TO_ID["B-TITLE"]
        assert resultat[2] == LABEL_TO_ID["B-PER"]


# ═══════════════════════════════════════════════════════════════════
# Tests POS / lemmes
# ═══════════════════════════════════════════════════════════════════

from nlp.pos_lemmes import AnalyseurMorphologique


class TestPosLemmes:
    """Tests de l'analyse morphologique (POS + lemmes)."""

    def test_retourne_triplets(self):
        """Chaque mot donne un triplet (mot, pos, lemme)."""
        analyseur = AnalyseurMorphologique(mode_simulation=True)
        resultat = analyseur.analyser("li rois")
        assert all(len(t) == 3 for t in resultat)

    def test_verbe_reconnu(self):
        """Un verbe médiéval connu reçoit le POS VERB et son lemme."""
        analyseur = AnalyseurMorphologique(mode_simulation=True)
        resultat = dict((m, (p, l)) for m, p, l in analyseur.analyser("tint"))
        assert resultat["tint"][0] == "VERB"
        assert resultat["tint"][1] == "tenir"

    def test_determinant_reconnu(self):
        """Un déterminant reçoit le POS DET."""
        analyseur = AnalyseurMorphologique(mode_simulation=True)
        resultat = dict((m, p) for m, p, l in analyseur.analyser("li"))
        assert resultat["li"] == "DET"

    def test_nom_propre_reconnu(self):
        """Un mot capitalisé reçoit PROPN."""
        analyseur = AnalyseurMorphologique(mode_simulation=True)
        resultat = dict((m, p) for m, p, l in analyseur.analyser("Brut"))
        assert resultat["Brut"] == "PROPN"

    def test_extraire_lemmes(self):
        """extraire_lemmes retourne la liste des lemmes."""
        analyseur = AnalyseurMorphologique(mode_simulation=True)
        lemmes = analyseur.extraire_lemmes("li rois tint")
        assert lemmes == ["li", "roi", "tenir"]


# ═══════════════════════════════════════════════════════════════════
# Tests graphe / TEI
# ═══════════════════════════════════════════════════════════════════

from nlp.graphe_tei import (
    construire_graphe,
    exporter_tei,
    extraire_relations,
)


class TestExtractionRelations:
    """Tests de l'extraction de relations."""

    def test_extrait_relation_simple(self):
        """'Brut tint Engleterre' donne une relation."""
        relations = extraire_relations("Brut tint Engleterre")
        assert len(relations) >= 1
        rel = relations[0]
        assert rel["source"] == "Brut"
        assert rel["cible"] == "Engleterre"

    def test_relation_a_un_type(self):
        """La relation extraite a un label (ex: TIENT)."""
        relations = extraire_relations("Brut tint Engleterre")
        assert relations[0]["relation"] == "TIENT"

    def test_sans_verbe_relation_aucune_relation(self):
        """Sans verbe-relation, aucune relation n'est extraite."""
        relations = extraire_relations("le beau romanz")
        assert relations == []


class TestGraphe:
    """Tests de la construction du graphe."""

    def test_graphe_a_les_bons_noeuds(self):
        """Le graphe contient les entités comme noeuds."""
        relations = extraire_relations("Brut tint Engleterre")
        graphe = construire_graphe(relations)
        assert graphe.number_of_nodes() == 2

    def test_graphe_a_les_bonnes_aretes(self):
        """Le graphe contient les relations comme arêtes."""
        relations = extraire_relations("Brut tint Engleterre")
        graphe = construire_graphe(relations)
        assert graphe.number_of_edges() == 1


class TestExportTEI:
    """Tests de l'export TEI-XML."""

    def test_fichier_cree(self, tmp_path):
        """L'export crée un fichier XML."""
        chemin = tmp_path / "sortie.xml"
        exporter_tei("le roi Charles", chemin)
        assert chemin.exists()

    def test_contient_balises_tei(self, tmp_path):
        """Le TEI contient les balises d'entités."""
        chemin = tmp_path / "sortie.xml"
        exporter_tei("Charles de France", chemin)
        contenu = chemin.read_text(encoding="utf-8")
        # Charles → persName, France → placeName
        assert "persName" in contenu
        assert "placeName" in contenu

    def test_tei_valide_xml(self, tmp_path):
        """Le fichier TEI doit être un XML valide (parsable)."""
        from xml.etree import ElementTree as ET
        chemin = tmp_path / "sortie.xml"
        exporter_tei("le roi Charles de France", chemin)
        # Si le XML est mal formé, parse() lève une exception
        ET.parse(chemin)  # ne doit pas lever d'erreur
