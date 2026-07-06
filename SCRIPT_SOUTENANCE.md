# Script de soutenance — Projet MD5 (HTR + NLP)

Groupe de 4 · 15 min + questions.

> **Règle d'or** : le jury interroge tout le monde sur tout (CV ET NLP).
> Chacun relit l'ensemble, surtout la banque de questions.

## Répartition des 11 slides
| Slides | Qui | Thème |
|--------|-----|-------|
| 1·2·3 | Membre 1 | Intro, contexte, données |
| 4·5 | Membre 2 | Pipeline CV, TrOCR/Kraken |
| 6·7 | Membre 3 | Data contract, normalisation |
| 8·9·10 | Membre 4 | NER, biais, avancement |
| 11 | Membre 1 | Conclusion |

## Fil conducteur (à dire)
- **Slide 1-2** : une image n'est pas un texte ; transcription manuelle non
  scalable ; pipeline CV → NLP ; on présente l'avancement.
- **Slide 3** : corpus à vérité terrain (mesurer le CER) vs manuscrits BnF
  (cible du NLP). Corpus identifiés, pas encore exploités.
- **Slide 4** : prétraitement (Sauvola), segmentation, HTR, data contract.
  Reproductibilité + 194 tests.
- **Slide 5** : TrOCR moteur principal, Kraken pour comparer (McNemar). Rien
  encore entraîné sur corpus.
- **Slide 6** : data contract = pont ; champs char_confidences, needs_review.
- **Slide 7** : normalisation règles d'abord, IA ensuite. Méthode de mesure
  prête, chiffres à produire (NE PAS annoncer de CER inventé).
- **Slide 8** : schéma BIO (B/I/O), 5 types, point -100. Code prêt.
- **Slide 9** : biais (temporel, géographique, document, copiste) + réponses.
- **Slide 10** : ce qui est fait / ce qui reste. Honnêteté.
- **Slide 11** : base solide, reproductible, honnête. Merci.

## Banque de questions/réponses

**Pourquoi TrOCR principal, Kraken en comparaison ?**
TrOCR est un Transformer puissant, facile à fine-tuner avec LoRA. Kraken est
pensé pour les manuscrits et donne les confiances par caractère nativement —
d'où sa valeur comme comparaison (test de McNemar).

**Sauvola vs Otsu ?**
Otsu = seuil global ; Sauvola = seuil local adaptatif par pixel, robuste aux
éclairages inégaux. Peut valoir 5-10 points de CER.

**LoRA ?**
Low-Rank Adaptation : on n'entraîne que de petites matrices ajoutées (~1% des
poids). Rapide, léger. r=8 puis 16.

**Schéma BIO ?**
B- = début d'entité, I- = suite, O = hors entité. Délimite les entités
multi-mots.

**Le -100 en NER ?**
BERT découpe un mot en sous-tokens ; seul le premier reçoit l'étiquette, les
autres reçoivent -100 (ignorés dans la perte).

**CER / WER ?**
CER = distance de Levenshtein caractère / longueur référence. WER = idem au
niveau mot (plus sévère).

**Bootstrap ?**
Rééchantillonnage avec remise (N=1000) pour estimer un intervalle de confiance
du CER.

**McNemar ?**
Teste si l'écart entre deux modèles est statistiquement significatif.

**Sans vérité terrain, comment mesurer ?**
Évaluation relative : taux de changement entre versions (brut → règles →
correction).

**Biais du corpus ?**
Temporel, géographique, type de document, copiste. Traités par split stratifié,
CER par strate, needs_review.

**Ça tourne vraiment ?**
Oui, de bout en bout, produit un data contract JSON validé. HTR en mode
simulation pour l'instant (pas d'accès modèle en dev) ; le vrai code est prêt.

**Anti-triche sur le test ?**
Test set scellé par SHA-256 dès le départ ; toute modification serait
détectable.

## Règles d'or
1. Tout le monde répond sur tout.
2. Honnêteté sur les limites (valorisé).
3. Si on ne sait pas : le dire calmement, proposer une piste.
4. Se chronométrer (15 min).
5. Avoir le dépôt GitHub ouvert.
