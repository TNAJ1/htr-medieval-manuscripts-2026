"""
metrics.py — Métriques d'évaluation du HTR.

Ce module calcule les métriques exigées par le projet :
  - CER (Character Error Rate) : la métrique principale
  - WER (Word Error Rate) : métrique complémentaire
  - Intervalle de confiance bootstrap (N=1000)
  - Test de McNemar (pour comparer deux modèles)

Le CER mesure la proportion de caractères erronés entre une transcription
prédite et la vérité terrain. C'est la distance de Levenshtein (nombre
minimum d'insertions, suppressions, substitutions) divisée par la longueur
de la référence.

  CER = (S + I + D) / N
    S = substitutions, I = insertions, D = suppressions
    N = nombre de caractères dans la référence

Seuils du projet : CER < 15 % (validation), CER < 8 % (excellence).
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import editdistance  # Calcul rapide de la distance de Levenshtein
import numpy as np


# ─── CER : Character Error Rate ──────────────────────────────────────────────

def calculer_cer(reference: str, prediction: str) -> float:
    """Calcule le CER entre une référence et une prédiction.

    Args:
        reference: La transcription correcte (vérité terrain).
        prediction: La transcription produite par le modèle.

    Returns:
        Le CER entre 0.0 (parfait) et potentiellement > 1.0 (si beaucoup
        d'insertions). Typiquement entre 0 et 1.

    Example:
        >>> calculer_cer("chat", "chien")
        0.75
    """
    # Cas particulier : référence vide
    if len(reference) == 0:
        # Si la prédiction est aussi vide, CER=0 ; sinon CER=1 (tout est faux)
        return 0.0 if len(prediction) == 0 else 1.0

    # editdistance.eval calcule la distance de Levenshtein au niveau caractère
    distance = editdistance.eval(reference, prediction)

    # CER = distance / longueur de la référence
    return distance / len(reference)


def calculer_cer_corpus(references: list[str], predictions: list[str]) -> float:
    """Calcule le CER global sur un corpus entier.

    Le CER global agrège TOUS les caractères du corpus, pas la moyenne des
    CER par ligne. C'est important : une ligne courte avec 1 erreur ne pèse
    pas autant qu'une ligne longue avec 1 erreur.

    Args:
        references: Liste des transcriptions correctes.
        predictions: Liste des transcriptions du modèle (même longueur).

    Returns:
        Le CER global du corpus.

    Raises:
        ValueError: Si les deux listes n'ont pas la même longueur.

    Example:
        >>> calculer_cer_corpus(["chat", "chien"], ["chat", "chein"])
        0.1111...
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"Nombre de références ({len(references)}) différent du nombre "
            f"de prédictions ({len(predictions)})."
        )

    # On accumule la distance totale et le nombre total de caractères
    distance_totale = 0
    caracteres_totaux = 0

    for ref, pred in zip(references, predictions):
        distance_totale += editdistance.eval(ref, pred)
        caracteres_totaux += len(ref)

    if caracteres_totaux == 0:
        return 0.0

    return distance_totale / caracteres_totaux


# ─── WER : Word Error Rate ───────────────────────────────────────────────────

def calculer_wer(reference: str, prediction: str) -> float:
    """Calcule le WER entre une référence et une prédiction.

    Le WER est comme le CER mais au niveau des MOTS au lieu des caractères.
    Plus sensible aux erreurs lexicales. Une seule lettre fausse rend tout
    le mot faux.

    Args:
        reference: La transcription correcte.
        prediction: La transcription du modèle.

    Returns:
        Le WER (entre 0.0 et potentiellement > 1.0).

    Example:
        >>> calculer_wer("le chat dort", "le chien dort")
        0.333...
    """
    # On découpe en mots (split par espaces)
    mots_ref = reference.split()
    mots_pred = prediction.split()

    if len(mots_ref) == 0:
        return 0.0 if len(mots_pred) == 0 else 1.0

    # editdistance fonctionne aussi sur des listes (de mots ici)
    distance = editdistance.eval(mots_ref, mots_pred)
    return distance / len(mots_ref)


def calculer_wer_corpus(references: list[str], predictions: list[str]) -> float:
    """Calcule le WER global sur un corpus entier.

    Args:
        references: Liste des transcriptions correctes.
        predictions: Liste des transcriptions du modèle.

    Returns:
        Le WER global du corpus.

    Raises:
        ValueError: Si les listes n'ont pas la même longueur.

    Example:
        >>> calculer_wer_corpus(["le chat", "il dort"], ["le chien", "il dort"])
        0.25
    """
    if len(references) != len(predictions):
        raise ValueError("Les listes doivent avoir la même longueur.")

    distance_totale = 0
    mots_totaux = 0

    for ref, pred in zip(references, predictions):
        mots_ref = ref.split()
        distance_totale += editdistance.eval(mots_ref, pred.split())
        mots_totaux += len(mots_ref)

    if mots_totaux == 0:
        return 0.0

    return distance_totale / mots_totaux


# ─── Intervalle de confiance bootstrap ───────────────────────────────────────

def intervalle_confiance_bootstrap(
    references: list[str],
    predictions: list[str],
    n_iterations: int = 1000,
    niveau_confiance: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calcule un intervalle de confiance du CER par ré-échantillonnage bootstrap.

    Le bootstrap estime l'incertitude d'une métrique. Principe :
      1. On tire au hasard N lignes (avec remise) parmi nos données.
      2. On calcule le CER sur cet échantillon.
      3. On répète n_iterations fois.
      4. Les percentiles 2,5 % et 97,5 % donnent l'intervalle à 95 %.

    Cela répond à l'exigence du projet : "IC bootstrap (95 %), N = 1000".

    Args:
        references: Liste des transcriptions correctes.
        predictions: Liste des transcriptions du modèle.
        n_iterations: Nombre de ré-échantillonnages (1000 recommandé).
        niveau_confiance: Niveau de l'intervalle (0.95 = 95 %).
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Un tuple (cer_moyen, borne_basse, borne_haute).

    Example:
        >>> cer, bas, haut = intervalle_confiance_bootstrap(refs, preds)
        >>> print(f"CER = {cer:.1%} [{bas:.1%}, {haut:.1%}]")
    """
    rng = np.random.default_rng(seed)  # Générateur aléatoire reproductible
    n = len(references)

    if n == 0:
        return 0.0, 0.0, 0.0

    cers = []  # On stocke le CER de chaque ré-échantillonnage

    for _ in range(n_iterations):
        # Tire n indices au hasard AVEC remise (un indice peut sortir 2 fois)
        indices = rng.integers(0, n, size=n)

        # Construit l'échantillon bootstrap
        refs_echantillon = [references[i] for i in indices]
        preds_echantillon = [predictions[i] for i in indices]

        # Calcule le CER de cet échantillon
        cer = calculer_cer_corpus(refs_echantillon, preds_echantillon)
        cers.append(cer)

    cers = np.array(cers)

    # Le CER central = CER sur les données complètes
    cer_moyen = calculer_cer_corpus(references, predictions)

    # Les bornes de l'intervalle = percentiles
    # Pour un IC à 95 %, on prend les percentiles 2,5 et 97,5
    alpha = 1 - niveau_confiance
    borne_basse = float(np.percentile(cers, 100 * alpha / 2))      # 2,5 %
    borne_haute = float(np.percentile(cers, 100 * (1 - alpha / 2)))  # 97,5 %

    return cer_moyen, borne_basse, borne_haute


# ─── Test de McNemar ─────────────────────────────────────────────────────────

def comparer_mcnemar(
    references: list[str],
    predictions_a: list[str],
    predictions_b: list[str],
) -> dict:
    """Compare deux modèles avec le test de McNemar.

    Le test de McNemar détermine si la différence de performance entre deux
    modèles (A et B) est statistiquement significative, ou juste due au hasard.

    Il compte, ligne par ligne, qui a raison :
      - n01 : A se trompe, B a raison
      - n10 : A a raison, B se trompe
    Si n01 et n10 sont très déséquilibrés, un modèle est vraiment meilleur.

    Args:
        references: Vérité terrain.
        predictions_a: Prédictions du modèle A.
        predictions_b: Prédictions du modèle B.

    Returns:
        Un dictionnaire avec :
          - 'statistique' : la statistique de McNemar
          - 'p_value' : la p-valeur (< 0.05 = différence significative)
          - 'n01', 'n10' : les comptes de désaccord
          - 'significatif' : True si p < 0.05

    Example:
        >>> resultat = comparer_mcnemar(refs, preds_trocr, preds_kraken)
        >>> if resultat["significatif"]:
        ...     print("La différence est significative !")
    """
    from scipy.stats import chi2

    # Pour chaque ligne, on regarde si chaque modèle est exact (CER == 0)
    n01 = 0  # A faux, B juste
    n10 = 0  # A juste, B faux

    for ref, pred_a, pred_b in zip(references, predictions_a, predictions_b):
        a_correct = (pred_a == ref)
        b_correct = (pred_b == ref)

        if not a_correct and b_correct:
            n01 += 1
        elif a_correct and not b_correct:
            n10 += 1

    # Statistique de McNemar (avec correction de continuité de Yates)
    # On évite la division par zéro si n01 + n10 == 0
    if (n01 + n10) == 0:
        return {
            "statistique": 0.0,
            "p_value": 1.0,
            "n01": n01,
            "n10": n10,
            "significatif": False,
        }

    statistique = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # La statistique suit une loi du chi² à 1 degré de liberté
    # sf = "survival function" = 1 - CDF = la p-valeur
    p_value = float(chi2.sf(statistique, df=1))

    return {
        "statistique": float(statistique),
        "p_value": p_value,
        "n01": n01,
        "n10": n10,
        "significatif": p_value < 0.05,
    }


# ─── Rapport complet ─────────────────────────────────────────────────────────

def rapport_metriques(references: list[str], predictions: list[str]) -> dict:
    """Produit un rapport complet de métriques pour un corpus.

    Pratique pour l'article scientifique et la présentation orale.

    Args:
        references: Vérité terrain.
        predictions: Prédictions du modèle.

    Returns:
        Dictionnaire avec CER, WER, IC bootstrap et les seuils atteints.

    Example:
        >>> rapport = rapport_metriques(refs, preds)
        >>> print(rapport)
    """
    cer = calculer_cer_corpus(references, predictions)
    wer = calculer_wer_corpus(references, predictions)
    cer_moyen, ic_bas, ic_haut = intervalle_confiance_bootstrap(
        references, predictions, n_iterations=1000
    )

    return {
        "CER": round(cer, 4),
        "WER": round(wer, 4),
        "CER_IC95": [round(ic_bas, 4), round(ic_haut, 4)],
        "seuil_validation_CER": cer < 0.15,   # objectif < 15 %
        "seuil_excellence_CER": cer < 0.08,   # objectif < 8 %
        "seuil_validation_WER": wer < 0.25,   # objectif < 25 %
        "n_lignes": len(references),
    }


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de metrics.py ===\n")

    # Données d'exemple : 5 lignes avec quelques erreurs
    references = [
        "Ci comence li romanz",
        "de Brut e de sa gent",
        "qui Engleterre tindrent",
        "ainz que Normant i vindrent",
        "Si com li livres le devise",
    ]
    predictions = [
        "Ci comence li romanz",       # parfait
        "de Brut e de sa gers",       # 2 erreurs (gent → gers)
        "qui Engleterre tindrent",    # parfait
        "ainz que Normant i vindront",  # 1 erreur (vindrent → vindront)
        "Si com li livres le devife",  # 1 erreur (devise → devife)
    ]

    # CER et WER
    cer = calculer_cer_corpus(references, predictions)
    wer = calculer_wer_corpus(references, predictions)
    print(f"CER global : {cer:.2%}")
    print(f"WER global : {wer:.2%}")

    # Intervalle de confiance bootstrap
    cer_moy, bas, haut = intervalle_confiance_bootstrap(references, predictions)
    print(f"CER avec IC 95 % : {cer_moy:.2%} [{bas:.2%}, {haut:.2%}]")

    # Rapport complet
    print("\nRapport complet :")
    rapport = rapport_metriques(references, predictions)
    for cle, valeur in rapport.items():
        print(f"  {cle} : {valeur}")

    print("\n=== Démonstration terminée ✓ ===")
