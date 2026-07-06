"""
comparaison.py — Comparaison et fusion de TrOCR et Kraken.

Ce module exploite le fait qu'on a DEUX modèles HTR au même format. Il permet :

  1. COMPARER les deux modèles (test de McNemar) → bonus +1 point du sujet.
  2. FUSIONNER leurs sorties par "vote pondéré" (étape 4 du sujet).

La fusion utilise l'alignement de Needleman-Wunsch : on aligne caractère par
caractère les deux transcriptions, puis à chaque position on garde le
caractère du modèle le plus confiant. Cela produit souvent une transcription
meilleure que chacun des deux modèles pris isolément.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from htr.metrics import comparer_mcnemar, calculer_cer_corpus


# ─── Fusion par vote pondéré (Needleman-Wunsch) ──────────────────────────────

def fusionner_transcriptions(
    texte_a: str,
    confiances_a: list[float],
    texte_b: str,
    confiances_b: list[float],
) -> tuple[str, list[float]]:
    """Fusionne deux transcriptions en gardant le caractère le plus confiant.

    Principe en deux temps :
      1. ALIGNEMENT (Needleman-Wunsch) : on aligne les deux textes caractère
         par caractère, en insérant des "trous" là où l'un a un caractère que
         l'autre n'a pas. C'est le même algorithme qu'en bio-informatique
         pour aligner des séquences ADN.
      2. VOTE : à chaque position alignée, on garde le caractère du modèle
         dont la confiance est la plus élevée.

    Args:
        texte_a: Transcription du modèle A (ex: TrOCR).
        confiances_a: Confiances par caractère du modèle A.
        texte_b: Transcription du modèle B (ex: Kraken).
        confiances_b: Confiances par caractère du modèle B.

    Returns:
        Un tuple (texte_fusionné, confiances_fusionnées).

    Example:
        >>> texte, conf = fusionner_transcriptions(
        ...     "romanz", [0.9]*6, "ramanz", [0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
        ... )
        >>> print(texte)  # "romanz" (A plus confiant sur la position 1)
    """
    # ── Étape 1 : Aligner les deux textes ───────────────────────────────────
    alignement = _aligner_needleman_wunsch(texte_a, texte_b)

    # ── Étape 2 : Voter caractère par caractère ─────────────────────────────
    texte_fusionne = []
    confiances_fusionnees = []

    # On parcourt l'alignement avec deux curseurs (un par texte)
    i, j = 0, 0  # i = position dans A, j = position dans B
    for (car_a, car_b) in alignement:
        if car_a is None:
            # A n'a rien ici (trou) → on prend le caractère de B
            texte_fusionne.append(car_b)
            confiances_fusionnees.append(confiances_b[j])
            j += 1
        elif car_b is None:
            # B n'a rien ici (trou) → on prend le caractère de A
            texte_fusionne.append(car_a)
            confiances_fusionnees.append(confiances_a[i])
            i += 1
        else:
            # Les deux ont un caractère → on garde le plus confiant
            conf_a = confiances_a[i] if i < len(confiances_a) else 0.0
            conf_b = confiances_b[j] if j < len(confiances_b) else 0.0
            if conf_a >= conf_b:
                texte_fusionne.append(car_a)
                confiances_fusionnees.append(conf_a)
            else:
                texte_fusionne.append(car_b)
                confiances_fusionnees.append(conf_b)
            i += 1
            j += 1

    return "".join(texte_fusionne), confiances_fusionnees


def _aligner_needleman_wunsch(
    a: str, b: str
) -> list[tuple[str | None, str | None]]:
    """Aligne deux chaînes avec l'algorithme de Needleman-Wunsch.

    Construit une matrice de scores par programmation dynamique, puis remonte
    le chemin optimal. Le résultat est une liste de paires de caractères ;
    None représente un "trou" (gap) là où une chaîne n'a pas de caractère.

    Args:
        a: Première chaîne.
        b: Seconde chaîne.

    Returns:
        Liste de paires (car_a, car_b). L'un des deux peut être None (gap).

    Example:
        >>> _aligner_needleman_wunsch("chat", "chot")
        [('c','c'), ('h','h'), ('a','o'), ('t','t')]
    """
    n, m = len(a), len(b)

    # ── Paramètres de score ─────────────────────────────────────────────────
    SCORE_MATCH = 1       # caractères identiques
    SCORE_MISMATCH = -1   # caractères différents
    SCORE_GAP = -1        # insertion/suppression (trou)

    # ── Construction de la matrice de scores ────────────────────────────────
    # matrice[i][j] = meilleur score pour aligner a[:i] avec b[:j]
    matrice = [[0] * (m + 1) for _ in range(n + 1)]

    # Première colonne et première ligne : que des gaps
    for i in range(n + 1):
        matrice[i][0] = i * SCORE_GAP
    for j in range(m + 1):
        matrice[0][j] = j * SCORE_GAP

    # Remplissage de la matrice
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Score si on aligne a[i-1] avec b[j-1]
            if a[i - 1] == b[j - 1]:
                score_diag = matrice[i - 1][j - 1] + SCORE_MATCH
            else:
                score_diag = matrice[i - 1][j - 1] + SCORE_MISMATCH
            # Score si on met un gap
            score_haut = matrice[i - 1][j] + SCORE_GAP   # gap dans b
            score_gauche = matrice[i][j - 1] + SCORE_GAP  # gap dans a
            # On garde le meilleur des trois choix
            matrice[i][j] = max(score_diag, score_haut, score_gauche)

    # ── Remontée du chemin optimal (traceback) ──────────────────────────────
    alignement = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Score qu'on aurait eu en venant de la diagonale
            if a[i - 1] == b[j - 1]:
                score_attendu = matrice[i - 1][j - 1] + SCORE_MATCH
            else:
                score_attendu = matrice[i - 1][j - 1] + SCORE_MISMATCH
            if matrice[i][j] == score_attendu:
                # On est venu de la diagonale → les deux caractères s'alignent
                alignement.append((a[i - 1], b[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and matrice[i][j] == matrice[i - 1][j] + SCORE_GAP:
            # Gap dans b (A a un caractère, B non)
            alignement.append((a[i - 1], None))
            i -= 1
        else:
            # Gap dans a (B a un caractère, A non)
            alignement.append((None, b[j - 1]))
            j -= 1

    # On a construit l'alignement à l'envers → on le remet à l'endroit
    alignement.reverse()
    return alignement


# ─── Comparaison de deux modèles ─────────────────────────────────────────────

def comparer_modeles(
    references: list[str],
    predictions_trocr: list[str],
    predictions_kraken: list[str],
) -> dict:
    """Compare TrOCR et Kraken sur un corpus avec vérité terrain.

    Calcule le CER de chaque modèle et applique le test de McNemar pour
    savoir si la différence de performance est statistiquement significative.
    C'est exactement ce que demande le bonus +1 point du sujet.

    Args:
        references: Transcriptions correctes (vérité terrain).
        predictions_trocr: Prédictions de TrOCR.
        predictions_kraken: Prédictions de Kraken.

    Returns:
        Dictionnaire avec le CER de chaque modèle et le résultat McNemar.

    Example:
        >>> resultat = comparer_modeles(refs, preds_trocr, preds_kraken)
        >>> print(resultat["cer_trocr"], resultat["cer_kraken"])
    """
    cer_trocr = calculer_cer_corpus(references, predictions_trocr)
    cer_kraken = calculer_cer_corpus(references, predictions_kraken)

    mcnemar = comparer_mcnemar(references, predictions_trocr, predictions_kraken)

    # Détermine le meilleur modèle
    if cer_trocr < cer_kraken:
        meilleur = "TrOCR"
    elif cer_kraken < cer_trocr:
        meilleur = "Kraken"
    else:
        meilleur = "égalité"

    return {
        "cer_trocr": round(cer_trocr, 4),
        "cer_kraken": round(cer_kraken, 4),
        "meilleur_modele": meilleur,
        "mcnemar_p_value": round(mcnemar["p_value"], 4),
        "difference_significative": mcnemar["significatif"],
        "n01_kraken_seul_correct": mcnemar["n01"],
        "n10_trocr_seul_correct": mcnemar["n10"],
    }


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de comparaison.py ===\n")

    # ── Démo 1 : fusion de deux transcriptions ──────────────────────────────
    print("─── Fusion par vote pondéré ───")
    texte_trocr = "romanz"
    conf_trocr = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]  # TrOCR confiant partout
    texte_kraken = "ramanz"  # Kraken se trompe : 'a' au lieu de 'o'
    conf_kraken = [0.95, 0.4, 0.9, 0.9, 0.9, 0.9]  # peu confiant sur le 'a'

    fusion, conf_fusion = fusionner_transcriptions(
        texte_trocr, conf_trocr, texte_kraken, conf_kraken
    )
    print(f"  TrOCR  : \"{texte_trocr}\"")
    print(f"  Kraken : \"{texte_kraken}\"")
    print(f"  Fusion : \"{fusion}\"  (garde le plus confiant à chaque position)")

    # ── Démo 2 : comparaison de modèles ─────────────────────────────────────
    print("\n─── Comparaison McNemar ───")
    references = ["romanz", "Engleterre", "gent", "tindrent", "Normant"]
    preds_trocr = ["romanz", "Engleterre", "gens", "tindrent", "Normant"]   # 1 erreur
    preds_kraken = ["ramanz", "Engleterre", "gent", "tindrant", "Normant"]  # 2 erreurs

    resultat = comparer_modeles(references, preds_trocr, preds_kraken)
    print(f"  CER TrOCR  : {resultat['cer_trocr']:.2%}")
    print(f"  CER Kraken : {resultat['cer_kraken']:.2%}")
    print(f"  Meilleur   : {resultat['meilleur_modele']}")
    print(f"  p-value McNemar : {resultat['mcnemar_p_value']} "
          f"(significatif : {resultat['difference_significative']})")

    print("\n=== Démonstration terminée ✓ ===")
