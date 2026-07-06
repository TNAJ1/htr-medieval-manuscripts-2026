# Model Card — Pipeline HTR manuscrits médiévaux

## Description
Pipeline HTR + NLP. Entrée : image de page. Sortie : data contract JSON
(transcriptions + polygones + confiances), puis analyse NLP.

## Modèles
| Composant | Modèle de base | Fine-tuning |
|-----------|----------------|-------------|
| HTR principal | trocr-base-handwritten | LoRA (r=8 puis 16) |
| HTR comparaison | Kraken (HTR-United) | ketos train |
| NER | roberta-multilingual-medieval-ner | fine-tuning léger |

## Données d'entraînement
À compléter après entraînement (corpus, volume, SHA-256 du test set).

## Performances
À compléter après évaluation. Seuils : CER < 15% (validation), < 8% (excellence).

## Biais
Voir `BIAIS_REPRESENTATION.md`. Biais temporel, géographique, de type de
document et de copiste. Traités par split stratifié + CER par strate +
needs_review sur catégories rares.

## Usage prévu
Recherche en humanités numériques. Usage non commercial (licences des modèles).
