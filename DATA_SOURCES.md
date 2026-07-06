# Sources des données

Seules des licences libres (CC-BY, CC-BY-SA, domaine public) sont utilisées.
Chaque source est documentée avec son attribution.

---

## Corpus retenu : CREMMA Medieval

Vu le temps imparti, nous avons fait le choix assumé de nous concentrer sur
**un seul corpus**, exploité de bout en bout (chargement, entraînement, mesure
du CER), plutôt que d'en survoler plusieurs.

- **Contenu** : ~15 manuscrits en ancien français
- **Période** : XIIIe–XVe siècle
- **Volume** : ~21 600 lignes, ~580 000 caractères
- **Production** : eScriptorium + Kraken
- **Licence** : CC-BY
- **Lien** : https://github.com/HTR-United/cremma-medieval
- **Rôle** : corpus à vérité terrain, sert à l'entraînement ET à la mesure du CER

> **Choix pragmatique.** Les consignes recommandent 3 à 5 corpus complémentaires.
> Nous avons privilégié la profondeur (un corpus réellement exploité) à la
> couverture (plusieurs corpus effleurés). L'extension à d'autres corpus
> (CATMuS, GalliCorpora) est notre perspective naturelle.

---

## Perspective : corpus complémentaires (non exploités à ce stade)

Pour une extension future, ces corpus élargiraient la couverture :

| Corpus | Période | Langues | Licence |
|--------|---------|---------|---------|
| CATMuS Medieval | VIIIe–XVIe s. | 10 langues | CC-BY 4.0 |
| GalliCorpora | XVe–XVIe s. | français | libre |

CATMuS est un méta-corpus (200+ manuscrits, ~160 000 lignes) qui harmonise
plusieurs corpus dont CREMMA. Il serait le candidat naturel pour élargir.

---

## Manuscrits sans vérité terrain (entrée du NLP)

| Source | Période | Licence |
|--------|---------|---------|
| BnF / Gallica | *à compléter selon les documents choisis* | domaine public |

> C'est sur ces transcriptions (non annotées) que travaille le volet NLP.

---

## Modèles pré-entraînés

| Modèle | Usage | Licence | Lien |
|--------|-------|---------|------|
| microsoft/trocr-base-handwritten | HTR principal | recherche | HuggingFace |
| Kraken + modèles HTR-United | HTR comparaison | *à vérifier* | https://kraken.re |
| magistermilitum/roberta-multilingual-medieval-ner | NER | *à vérifier* | HuggingFace |

---

## Ordre de grandeur attendu du CER

D'après la littérature récente, TrOCR sur des corpus médiévaux comparables
atteint typiquement un **CER de 9–11 %** et un **WER de 21–25 %**. Cela
placerait notre système sous le seuil de validation du projet (CER < 15 %).
Ces valeurs sont indicatives ; nos chiffres réels seront mesurés sur le test
scellé après entraînement.

---

## Attribution

- Clérice, T., Pinche, A., Vlachou-Efstathiou, M. *CREMMA Medieval model*, 2022.
  https://github.com/HTR-United/cremma-medieval

> Compléter le volume exact effectivement utilisé et la licence vérifiée après
> téléchargement.
