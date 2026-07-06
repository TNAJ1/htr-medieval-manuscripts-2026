# Conventions de normalisation NLP

## Règles déterministes (dans l'ordre)
1. **Unicode NFC** — unifie les représentations avant comparaison
2. **Abréviations** — `⁊`→`et`, `q~`→`que`, `ꝑ`→`per`, `ꝗ`→`qui`
3. **Tilde nasal** — `mõt`→`mont`, `gẽt`→`gent`
4. **u/v et i/j** — `vne`→`une`, `j`→`i`

## Correction guidée par confiance
Aux positions de confiance < 0,70, on teste les candidats et on retient celui
formant un mot du lexique. Complément possible : CamemBERT en masked LM.

## Impact (à mesurer sur corpus)
| Étape | CER (à reporter) |
|-------|------------------|
| Brut | *à mesurer* |
| + règles | *à mesurer* |
| + correction | *à mesurer* |

Lancer `python src/nlp/evaluation_relative.py` sur vos données.

## Niveau de transcription
Semi-diplomatique : abréviations développées, graphies d'époque conservées.
