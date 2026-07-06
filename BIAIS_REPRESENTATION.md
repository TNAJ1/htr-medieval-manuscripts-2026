# Biais de représentation du corpus

## 1. Biais temporel
XIIIe–XIVe surreprésentés ; XIe et XVIIe sous-représentés. Le modèle NER
médiéval (XIe–XVIe) est faible sur le XVIIe.

## 2. Biais géographique / dialectal
Domaine d'oïl dominant ; occitan, anglo-normand rares.

## 3. Biais de type de document
Textes littéraires surreprésentés ; registres tabulaires rares.

## 4. Biais de copiste
Nombre limité de mains vues à l'entraînement — verrou central du domaine.

## 5. Biais d'annotation
La vérité terrain reflète des choix éditoriaux ; l'IAA donne un plancher
d'ambiguïté sous lequel le CER ne peut raisonnablement descendre.

## Recommandations
1. Diversifier le corpus (siècles, dialectes sous-représentés)
2. Split stratifié par siècle ET type (implémenté : dataset_split.py)
3. Reporter le CER par strate, pas seulement global
4. Marquer needs_review sur les catégories rares
5. Documenter les limites dans la model card

## Portée
Valide pour : *[siècles, langues, types effectivement couverts]*. Ne pas
appliquer hors périmètre sans réentraînement.
