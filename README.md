# htr-medieval-manuscripts-2026

Pipeline complet de reconnaissance (HTR) et d'analyse (NLP) de manuscrits
médiévaux — **Projet MD5**, Master Data/IA.

De l'image numérisée au texte structuré et analysé.

```
Image brute → Prétraitement → Segmentation → HTR → Data Contract JSON → NLP
```

---

## Installation

```bash
git clone https://github.com/<votre-equipe>/htr-medieval-manuscripts-2026.git
cd htr-medieval-manuscripts-2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduire / lancer

```bash
# Suite de tests complète (194 tests)
python -m pytest tests/ -v

# Pipeline HTR de bout en bout (mode simulation, sans GPU)
python src/htr/pipeline.py

# Chaîne réelle (chez vous, avec corpus + GPU) :
#   1. Télécharger un corpus
#      git clone https://github.com/HTR-United/cremma-medieval data/raw/cremma-medieval
#   2. Charger + entraîner + évaluer (voir exemples ci-dessous)
```

### Exemple : entraîner et mesurer le CER

```python
from htr.corpus_loader import charger_corpus
from htr.dataset_split import split_stratifie
from htr.finetuning import finetuner_trocr, ConfigEntrainement
from htr.evaluation_corpus import evaluer_modele_sur_corpus

exemples = charger_corpus("data/raw/cremma-medieval",
                          {"century": 13, "source": "CREMMA"})
splits = split_stratifie(exemples)

finetuner_trocr(splits["train"], splits["val"],
                config=ConfigEntrainement(epochs=10, lora_r=8),
                mode_simulation=False)

evaluer_modele_sur_corpus("data/raw/cremma-medieval", modele="trocr",
                          metadata={"century": 13, "source": "CREMMA"},
                          mode_simulation=False)
```

### Exemple : générer un manuscrit coloré (segmentation + transcriptions)

```python
from htr.visualisation import visualiser_page
from htr.corpus_loader import trouver_paires

paires = trouver_paires("data/raw/cremma-medieval")
visualiser_page(*paires[0], "resultats/manuscrit_colore.png")
```

---

## Structure du dépôt

```
src/
├── htr/                    # Volet 1 — Vision par ordinateur
│   ├── corpus_loader.py    # Charger CREMMA/CATMuS (image + vérité terrain)
│   ├── preprocessing.py    # Nettoyage (gris, deskew, CLAHE, Sauvola)
│   ├── segmentation.py     # Détection des lignes + polygones
│   ├── page_xml.py         # Export PAGE XML (eScriptorium)
│   ├── htr_model.py        # TrOCR + confiances
│   ├── kraken_model.py     # Kraken (comparaison)
│   ├── comparaison.py      # McNemar + fusion Needleman-Wunsch
│   ├── transcripteur_factory.py  # Choix trocr/kraken/fusion
│   ├── finetuning.py       # Fine-tuning LoRA + courbe d'apprentissage
│   ├── metrics.py          # CER, WER, bootstrap, McNemar
│   ├── dataset_split.py    # Split stratifié + scellement SHA-256
│   ├── evaluation_corpus.py # Mesure du CER réel sur corpus
│   ├── visualisation.py    # Manuscrits colorés (segmentation + transcriptions)
│   └── pipeline.py         # Orchestration de bout en bout
├── nlp/                    # Volet 2 — Traitement du langage
│   ├── ingestion.py        # Lecture + validation + EDA
│   ├── normalisation.py    # Règles + correction par confiance
│   ├── evaluation_relative.py # Impact chiffré (avec/sans vérité terrain)
│   ├── ner.py              # NER schéma BIO (PER/LOC/DATE/ORG/TITLE)
│   ├── pos_lemmes.py       # POS + lemmatisation (Stanza frm)
│   └── graphe_tei.py       # Relations, graphe NetworkX, export TEI-XML
└── shared/                 # Partagé par les deux volets
    ├── utils.py            # Seeds, hash, journal
    └── data_contract.py    # Création + validation du data contract
schemas/    data_contract.schema.json
tests/      194 tests pytest (8 fichiers)
dataset_nlp/  data contracts JSON produits
segmentations/  fichiers PAGE XML
```

---

## Documentation

| Fichier | Contenu |
|---------|---------|
| `ARTICLE_SCIENTIFIQUE.md` | Article complet (8-12 pages) |
| `SCRIPT_SOUTENANCE.md` | Script oral + banque de questions |
| `MD5_soutenance.pptx` | Présentation (11 slides) |
| `CONVENTIONS_TRANSCRIPTION.md` | Choix éditoriaux HTR |
| `CONVENTIONS_NLP.md` | Règles de normalisation |
| `BIAIS_REPRESENTATION.md` | Analyse des biais du corpus |
| `DATA_SOURCES.md` | Corpus, licences, attribution |
| `MODEL_CARD.md` | Fiche de modèle |

---

## Métriques (seuils du projet)

| Métrique | Validation | Excellence |
|----------|-----------|-----------|
| CER global | < 15 % | < 8 % |
| WER global | < 25 % | < 15 % |
| Taux needs_review | < 30 % | < 20 % |

---

## État d'avancement

**Fait** : pipeline HTR + NLP complet, data contract validé, 194 tests, article,
présentation, analyse des biais.

**Reste** : entraînement réel sur corpus (infrastructure prête), mesure du CER,
fine-tuning NER effectif.

## Licences

Seuls des corpus sous licence libre (CC-BY, CC-BY-SA, domaine public) sont
utilisés. Voir `DATA_SOURCES.md`.
