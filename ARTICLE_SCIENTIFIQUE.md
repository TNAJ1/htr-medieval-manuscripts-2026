# Reconnaissance et analyse automatiques de manuscrits médiévaux : un pipeline HTR + NLP reproductible

**Projet MD5 — Master Data/IA**
Équipe : *[TCHAPPI Romarique · BOUDJEKA Anik · NANGUI Dorcas
 · Jordan NGUEKO]*

---

## Résumé

Ce travail présente un pipeline complet de traitement automatique de manuscrits
médiévaux, de l'image numérisée au texte structuré et analysé. Le système
articule deux volets : un volet vision par ordinateur (HTR, *Handwritten Text
Recognition*) qui transcrit les images de lignes, et un volet traitement
automatique du langage (NLP) qui normalise, corrige et analyse les
transcriptions. Les deux volets sont reliés par un *data contract* JSON validé,
qui garantit l'interopérabilité. Nous détaillons nos choix de prétraitement, la
comparaison de deux moteurs HTR (TrOCR et Kraken), la stratégie de normalisation
par règles puis correction guidée par confiance, ainsi qu'une chaîne NLP
comprenant reconnaissance d'entités nommées (NER), étiquetage morphologique et
export TEI. Nous discutons enfin des biais de représentation du corpus et des
limites de l'approche. L'ensemble du code est reproductible (graines fixées,
dépendances figées, jeu de test scellé) et couvert par 194 tests automatisés.

**Mots-clés** : HTR, TrOCR, Kraken, manuscrits médiévaux, ancien français, NER,
TEI, reproductibilité.

---

## 1. Introduction

### 1.1 Contexte et motivation

Les institutions patrimoniales ont numérisé des centaines de millions de pages
de manuscrits. Ces images constituent un patrimoine considérable, mais elles ne
sont pas directement exploitables : une image n'est pas un texte cherchable. La
transcription manuelle par des paléographes est lente, coûteuse (de l'ordre de
50 € de l'heure) et ne passe pas à l'échelle des volumes concernés.

La reconnaissance automatique d'écriture manuscrite (HTR) vise à combler cet
écart en produisant des transcriptions à partir des images. Le défi central est
que chaque manuscrit possède sa propre écriture : les mains des copistes, les
abréviations, les graphies varient d'un document à l'autre et d'un siècle à
l'autre. Un modèle entraîné sur un ensemble de manuscrits ne généralise que
partiellement à des mains jamais vues.

### 1.2 Objectifs

Notre projet poursuit deux objectifs complémentaires, correspondant à deux
volets :

1. **Volet 1 (vision par ordinateur)** : construire une chaîne complète et
   reproductible qui transforme une image de manuscrit en transcriptions
   structurées, avec des scores de confiance exploitables en aval.
2. **Volet 2 (NLP)** : normaliser et corriger ces transcriptions, puis les
   enrichir (entités nommées, analyse morphologique) et les publier dans un
   format standard des humanités numériques (TEI).

Un principe directeur est que la qualité du volet 1 conditionne l'ensemble de
l'aval : une transcription bruitée dégrade toutes les analyses ultérieures.
Nous avons donc soigné l'interface entre les deux volets, matérialisée par un
*data contract*.

### 1.3 Contributions

Nos contributions principales sont :

- une architecture modulaire orientée contrat, où le HTR et le NLP communiquent
  par un JSON validé automatiquement contre un schéma ;
- une comparaison de deux moteurs HTR (TrOCR, Transformer généraliste, et
  Kraken, spécialisé humanités numériques) avec une méthode d'agrégation par
  vote pondéré ;
- une stratégie de normalisation en deux temps (règles déterministes puis
  correction guidée par confiance) et une méthode d'évaluation relative
  applicable en l'absence de vérité terrain ;
- une chaîne NLP complète (NER en schéma BIO, POS/lemmatisation, extraction de
  relations, export TEI) ;
- une infrastructure reproductible et testée (194 tests automatisés).

---

## 2. Données et corpus

### 2.1 Deux rôles distincts

Nous distinguons deux catégories de données, jouant des rôles différents.

Les **corpus à vérité terrain** fournissent des paires (image, transcription
validée). Ils servent à entraîner les modèles et surtout à *mesurer* le taux
d'erreur, puisqu'ils offrent une référence de comparaison. Nous avons retenu
**CREMMA Medieval**, qui regroupe une quinzaine de manuscrits en ancien français
des XIIIe et XIVe siècles, produits avec eScriptorium et Kraken, sous licence
libre.

Les **manuscrits sans vérité terrain** — issus de la BnF via Gallica — n'ont pas
été transcrits manuellement. On ne peut donc pas y mesurer un taux d'erreur
absolu, mais ce sont précisément ces documents que le volet NLP traite en
conditions réelles.

### 2.2 Complémentarité et couverture

Les consignes recommandent de combiner plusieurs corpus complémentaires afin
de couvrir plusieurs siècles, régions dialectales et types de documents. Compte
tenu du temps imparti, nous avons fait le choix assumé de concentrer nos efforts
sur un seul corpus (CREMMA Medieval), exploité de bout en bout, plutôt que
d'en survoler plusieurs. Ce compromis privilégie la profondeur à la couverture.
L'extension à d'autres corpus (CATMuS Medieval, GalliCorpora) constitue une
perspective naturelle : elle est nécessaire, à terme, pour limiter les biais
(section 7) et améliorer la généralisation. La période visée par CREMMA s'étend
sur les XIIIe et XIVe siècles, en ancien français.

### 2.3 Licences et éthique des données

Nous n'utilisons que des ressources sous licence libre (CC-BY, CC-BY-SA, domaine
public). Chaque source est documentée avec son attribution. Ce choix garantit la
légalité de la réutilisation et la reproductibilité par des tiers.

### 2.4 Convention de transcription

Nous adoptons une transcription **semi-diplomatique** : les abréviations sont
développées (pour la lisibilité et l'exploitation NLP) tandis que les graphies
d'époque sont conservées. Ce choix est documenté afin que les décisions
éditoriales soient explicites et reproductibles.

---

## 3. Volet 1 — Pipeline de vision par ordinateur

### 3.1 Vue d'ensemble

Le pipeline enchaîne quatre étapes : prétraitement de l'image, segmentation en
lignes, reconnaissance HTR, puis assemblage dans le *data contract*. Chaque
étape est un module indépendant et testé.

### 3.2 Prétraitement

Le prétraitement vise à normaliser l'image pour le modèle. Il comprend :

1. la conversion en niveaux de gris, la couleur étant inutile pour du texte
   noir sur parchemin ;
2. la correction d'inclinaison (*deskewing*) par détection de l'angle dominant
   de l'écriture, un scan pouvant être légèrement penché ;
3. l'amélioration du contraste par CLAHE (*Contrast Limited Adaptive Histogram
   Equalization*), qui traite le contraste localement, par tuiles, et gère les
   éclairages inégaux ;
4. la binarisation par l'algorithme de Sauvola, qui calcule un seuil adaptatif
   local pour chaque pixel. Sauvola est nettement supérieur à un seuil global
   (Otsu) sur des manuscrits où l'éclairage et l'état du parchemin varient ;
5. un débruitage morphologique (ouverture) qui supprime les points isolés.

L'impact de chaque étape sur le taux d'erreur doit être mesuré par ablation
(avec/sans l'étape) sur le jeu de validation ; notre infrastructure permet cette
mesure, dont les chiffres seront reportés après entraînement.

### 3.3 Segmentation

La segmentation détecte les lignes de texte et en extrait les polygones. Notre
implémentation de référence utilise les profils de projection horizontale :
on somme les pixels de texte par rangée, ce qui fait apparaître les lignes
(rangées denses) et les interlignes (rangées vides). Chaque ligne produit une
boîte englobante et un polygone, et l'ordre de lecture (haut vers bas) est
préservé. Cette méthode classique est légère et explicable ; pour les mises en
page complexes (multi-colonnes, registres), une bascule vers Kraken BLLA est
prévue.

Les polygones sont exportés au format PAGE XML, standard réutilisable par
eScriptorium, ce qui assure l'interopérabilité avec l'écosystème des humanités
numériques.

### 3.4 Reconnaissance HTR : deux moteurs

Nous avons implémenté deux moteurs partageant la même interface, ce qui les rend
interchangeables et comparables.

**TrOCR** (moteur principal) est un modèle encodeur-décodeur de type Transformer
(un encodeur visuel ViT et un décodeur textuel). Nous l'employons dans sa
variante `trocr-base-handwritten`. Un point technique important est la
reconstruction des confiances par caractère : TrOCR génère le texte par tokens ;
nous récupérons, à chaque étape de génération, la probabilité du token choisi
(via softmax) puis la répartissons sur les caractères.

**Kraken** (comparaison) repose sur une architecture récurrente (CNN + BiLSTM)
conçue spécifiquement pour les humanités numériques. Son avantage est de fournir
nativement les confiances par caractère.

Le fine-tuning de TrOCR est réalisé par LoRA (*Low-Rank Adaptation*), qui
n'entraîne qu'une faible fraction des poids et permet un entraînement rapide et
peu gourmand en mémoire. Kraken se fine-tune via `ketos train` à partir de nos
fichiers PAGE XML.

### 3.5 Agrégation par vote pondéré

Disposant de deux moteurs au même format, nous pouvons fusionner leurs sorties.
La fusion aligne les deux transcriptions caractère par caractère avec
l'algorithme de Needleman-Wunsch (programmation dynamique), puis retient à
chaque position le caractère du moteur le plus confiant. Cette approche produit
souvent une transcription meilleure que chacun des deux moteurs pris isolément.

### 3.6 Le data contract

L'interface entre les deux volets est un document JSON validé contre un schéma.
Chaque ligne transcrite y porte : le texte, le polygone, la confiance globale,
les confiances par caractère, les candidats de lecture aux positions ambiguës,
et un drapeau `needs_review` activé lorsque la confiance passe sous un seuil
(0,70). La validation systématique du schéma, avant toute écriture, garantit
qu'aucun document non conforme ne se propage en aval.

---

## 4. Volet 2 — Traitement automatique du langage

### 4.1 Ingestion et analyse exploratoire

Le NLP part du *data contract*. La première étape charge le JSON, valide
systématiquement son schéma, puis réalise une analyse exploratoire
(distribution des confiances, taux de lignes à relire, longueur des lignes,
abréviations résiduelles). Ces statistiques justifient les choix de
normalisation qui suivent.

### 4.2 Normalisation

La normalisation est la brique la plus rentable du volet NLP : elle réduit le
taux d'erreur pour un effort modeste. Nous appliquons d'abord des **règles
déterministes** : normalisation Unicode (NFC), développement des abréviations
médiévales (par exemple la nota tironienne « ⁊ » en « et »), résolution du
tilde nasal (« mõt » en « mont »), harmonisation des graphies u/v et i/j.

Nous appliquons ensuite une **correction guidée par confiance** : aux positions
de faible confiance signalées par le HTR, nous testons les candidats et
retenons celui qui forme un mot du lexique de référence. Cette étape simple peut
être complétée par un modèle de langue masqué (CamemBERT en *masked language
model*) pour les cas résiduels.

### 4.3 Évaluation, avec et sans vérité terrain

Sur les corpus à vérité terrain, nous mesurons le taux d'erreur au niveau
caractère (CER) et au niveau mot (WER) à chaque étape, ce qui permet de
quantifier l'apport de la normalisation. Sur les manuscrits sans vérité terrain,
où aucune référence n'existe, nous recourons à une **évaluation relative** : nous
mesurons le taux de changement entre versions successives (brut, après règles,
après correction). Cette courbe d'évolution rend l'apport de chaque étape
objectivable sans référence humaine.

### 4.4 Reconnaissance d'entités nommées

La NER suit le schéma BIO : chaque mot reçoit une étiquette « B- » (début
d'entité), « I- » (continuation) ou « O » (hors entité). Nous reconnaissons cinq
types : personnes (PER), lieux (LOC), dates (DATE), organisations (ORG) et
titres (TITLE), ce dernier étant pertinent pour les textes médiévaux riches en
rois, comtes et évêques.

Conformément aux bonnes pratiques, nous ne partons pas de zéro mais d'un modèle
de type CamemBERT/RoBERTa déjà fine-tuné sur des chartes médiévales, couvrant le
XIe au XVIe siècle. Un point technique délicat est l'alignement des étiquettes
sur la tokenisation en sous-mots : seul le premier sous-token d'un mot reçoit
l'étiquette, les sous-tokens de continuation recevant la valeur spéciale -100
pour être ignorés dans le calcul de la perte.

### 4.5 Analyse morphologique

Nous enrichissons chaque mot de sa nature grammaticale (POS) et de son lemme
(forme de base), au moyen de Stanza et de son modèle de moyen français. Ainsi
« vindrent » est ramené au lemme « venir », et « rois » à « roi ». Cette
normalisation morphologique facilite l'indexation et les recherches.

### 4.6 Relations, graphe et export TEI

À partir des entités, nous extrayons des relations par règles
lexico-syntaxiques simples (motif « entité — verbe — entité »), par exemple
« Brut tint Engleterre » donnant la relation (Brut, TIENT, Engleterre). Ces
relations sont modélisées en graphe orienté (NetworkX), permettant de visualiser
le réseau des entités.

Enfin, nous exportons un échantillon au format TEI-XML, standard d'encodage des
humanités numériques, en balisant les entités (`<persName>`, `<placeName>`,
`<date>`). Ce format ouvre la voie à l'édition savante et à l'interopérabilité.

---

## 5. Reproductibilité et ingénierie

La reproductibilité est un critère central de notre démarche. Toutes les
sources d'aléa sont fixées par une graine unique. Les dépendances sont figées
(versions précises). Le jeu de test est scellé par une empreinte SHA-256 :
toute modification ultérieure serait détectable, ce qui prouve l'absence de
contamination entre développement et évaluation.

Le découpage en train/validation/test est stratifié par siècle et type de
document, afin que chaque ensemble soit représentatif et que les biais ne soient
pas masqués à l'évaluation. Le dépôt sépare clairement les modules du volet
vision (`src/htr/`), du volet NLP (`src/nlp/`) et les composants partagés
(`src/shared/`). L'ensemble est couvert par 194 tests automatisés (pytest),
incluant des tests de non-régression sur la production du *data contract* et sur
la non-dégradation du taux d'erreur par la normalisation.

---

## 6. Métriques d'évaluation

Nous évaluons le HTR par le CER et le WER. Le CER, taux d'erreur au niveau
caractère, est la distance d'édition (Levenshtein) rapportée à la longueur de la
référence ; le WER en est l'équivalent au niveau des mots. Nous accompagnons le
CER d'un intervalle de confiance estimé par bootstrap (mille rééchantillonnages
avec remise), afin de ne pas présenter une valeur ponctuelle comme si elle était
exacte.

Pour comparer deux moteurs, nous employons le test de McNemar, qui détermine si
l'écart de performance est statistiquement significatif ou imputable au hasard.
La NER sera évaluée par le score F1 par type d'entité. Les seuils visés sont un
CER inférieur à 15 % (validation) voire 8 % (excellence), un WER inférieur à
25 %, et un taux de lignes à relire inférieur à 30 %.

À la date de rédaction, l'infrastructure de mesure est en place mais
l'entraînement réel sur corpus n'a pas encore été exécuté ; les valeurs
chiffrées seront reportées dès que les modèles auront été entraînés.

---

## 7. Biais de représentation et limites

### 7.1 Biais du corpus

Les corpus médiévaux disponibles ne sont pas uniformes. Un **biais temporel**
existe (les XIIIe et XIVe siècles sont surreprésentés, les bornes XIe et XVIIe
moins couvertes). Un **biais géographique et dialectal** favorise le domaine
d'oïl au détriment des variantes (occitan, anglo-normand). Un **biais de type de
document** privilégie les textes littéraires sur les registres tabulaires. Enfin,
un **biais de copiste** subsiste : un nombre limité de mains est vu à
l'entraînement, ce qui est le verrou central du domaine.

### 7.2 Traitement des biais

Nous atténuons ces biais par plusieurs moyens : un découpage stratifié pour ne
pas les masquer, un taux d'erreur reporté par strate (et non seulement global),
le marquage `needs_review` sur les catégories rares afin de prioriser la
relecture humaine là où le modèle est le moins fiable, et une documentation
explicite des limites dans la fiche de modèle.

### 7.3 Limites

Les principales limites à ce stade sont : l'absence d'entraînement réel exécuté
(le développement a été mené en mode simulation pour valider la logique) ;
l'absence, en conséquence, d'un taux d'erreur mesuré ; une segmentation
classique moins robuste que Kraken BLLA sur les mises en page complexes ; et une
couverture plus faible du XVIIe siècle par le modèle NER médiéval.

---

## 8. Conclusion et perspectives

Nous avons présenté un pipeline complet et reproductible pour le traitement
automatique de manuscrits médiévaux, du scan à l'analyse. Sa force principale
réside dans une conception orientée contrat qui relie proprement vision et NLP,
et dans une infrastructure d'ingénierie soignée (reproductibilité, tests,
documentation, transparence sur les limites).

Les perspectives immédiates sont l'exécution de l'entraînement réel sur les
corpus à vérité terrain et la mesure du taux d'erreur correspondant, le
fine-tuning effectif de la NER sur un échantillon annoté, et la bascule de la
segmentation vers Kraken BLLA pour les mises en page complexes. À plus long
terme, l'enrichissement du graphe de relations et l'export TEI systématique
ouvriraient la voie à des éditions savantes assistées.

---

## Références

*[À compléter selon les sources effectivement utilisées. Exemples de références
à citer :]*

- Consortium CREMMA — Reconnaissance d'Écriture Manuscrite des Matériaux
  Anciens. *[URL, année.]*
- Projet CATMuS Medieval — Consistent Approaches to Transcribing Manuscripts.
  *[Référence, année.]*
- Li, M. et al. *TrOCR: Transformer-based Optical Character Recognition with
  Pre-trained Models.* *[Année.]*
- Kiessling, B. *Kraken — an universal text recognizer for the humanities.*
  *[Année.]*
- Sauvola, J., Pietikäinen, M. *Adaptive document image binarization.*
  Pattern Recognition, 2000.
- Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* *[Année.]*
- TEI Consortium. *Guidelines for Electronic Text Encoding and Interchange.*

---

## Annexe — Organisation de l'équipe et répartition du travail

| Membre | Rôle principal | Contributions |
|--------|----------------|---------------|
| *[Jordan NGUEKO]* | Responsable technique | Architecture, revue de code, intégration |
| *[TCHAPPI Romarique]* | Responsable données | Corpus, licences, data contract |
| *[NANGUI Dorcas]* | Responsable expérimentation | Métriques, journal des essais, courbes |
| *[BOUDJEKA Anik]* | Responsable documentation | Article, README, fiche de modèle |

*Chaque membre a contribué transversalement aux deux volets et maîtrise
l'ensemble du pipeline.*
