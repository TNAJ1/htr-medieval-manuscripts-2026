"""
evaluation_relative.py — Mesure de l'impact de la normalisation SANS vérité terrain.

Problème (consignes NLP) : sur des manuscrits BnF non annotés, on ne peut pas
calculer un CER absolu (pas de référence humaine à comparer).

Solution : l'ÉVALUATION RELATIVE. On compare les différentes VERSIONS d'une
même transcription entre elles :
    brut → règles → règles+correction
La distance entre deux versions successives mesure « combien » chaque étape
a modifié le texte. Cela donne une courbe d'évolution objective, qui justifie
l'apport de chaque étape, sans vérité terrain.

C'est l'approche de l'outil Evaluation-HTR mentionné dans les consignes.
Le README doit ensuite reporter ces chiffres (impact CHIFFRÉ de chaque règle).
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from htr.metrics import calculer_cer  # on réutilise notre CER caractère


# ─── Distance relative entre deux versions ───────────────────────────────────

def distance_relative(version_a: str, version_b: str) -> float:
    """Mesure la distance relative entre deux versions d'une transcription.

    On réutilise le CER (distance de Levenshtein normalisée). Ici il ne
    mesure PAS une erreur (pas de vérité terrain), mais le TAUX DE CHANGEMENT
    entre deux versions. Plus c'est élevé, plus l'étape a modifié le texte.

    Args:
        version_a: La transcription avant une étape.
        version_b: La transcription après cette étape.

    Returns:
        Le taux de changement entre 0.0 (identique) et plus.

    Example:
        >>> distance_relative("⁊ romanz", "et romanz")
        0.25
    """
    # On prend version_a comme "référence" : combien faut-il changer pour
    # passer de a à b, rapporté à la longueur de a.
    return calculer_cer(version_a, version_b)


# ─── Évaluation d'un pipeline de versions ────────────────────────────────────

def evaluer_etapes(versions: dict[str, list[str]]) -> dict:
    """Mesure l'impact de chaque étape de normalisation sur un corpus.

    On fournit, pour chaque étape nommée, la liste des transcriptions du
    corpus à cette étape. La fonction calcule le taux de changement moyen
    entre étapes successives.

    Args:
        versions: Dictionnaire ordonné {nom_etape: [transcriptions]}.
            Toutes les listes doivent avoir la même longueur (même corpus).
            Ex: {"brut": [...], "regles": [...], "correction": [...]}

    Returns:
        Dictionnaire avec, pour chaque transition, le taux de changement moyen.

    Raises:
        ValueError: Si les listes n'ont pas toutes la même longueur.

    Example:
        >>> impact = evaluer_etapes({
        ...     "brut":   ["⁊ romanz", "mõt"],
        ...     "regles": ["et romanz", "mont"],
        ... })
        >>> print(impact["brut→regles"])
    """
    noms_etapes = list(versions.keys())
    if len(noms_etapes) < 2:
        raise ValueError("Il faut au moins deux étapes à comparer.")

    # Vérifie que toutes les listes ont la même longueur
    longueurs = {len(v) for v in versions.values()}
    if len(longueurs) != 1:
        raise ValueError("Toutes les versions doivent avoir la même longueur.")

    resultat = {}

    # Pour chaque paire d'étapes successives
    for i in range(len(noms_etapes) - 1):
        nom_avant = noms_etapes[i]
        nom_apres = noms_etapes[i + 1]
        textes_avant = versions[nom_avant]
        textes_apres = versions[nom_apres]

        # Taux de changement moyen sur tout le corpus
        distances = [
            distance_relative(a, b)
            for a, b in zip(textes_avant, textes_apres)
        ]
        changement_moyen = sum(distances) / len(distances) if distances else 0.0

        # Compte combien de lignes ont été modifiées
        nb_modifiees = sum(1 for d in distances if d > 0)

        resultat[f"{nom_avant}→{nom_apres}"] = {
            "taux_changement_moyen": round(changement_moyen, 4),
            "lignes_modifiees": nb_modifiees,
            "lignes_totales": len(distances),
        }

    return resultat


# ─── Évaluation avec vérité terrain (corpus CREMMA/CATMuS) ────────────────────

def evaluer_avec_verite_terrain(
    references: list[str],
    versions: dict[str, list[str]],
) -> dict:
    """Mesure le CER ABSOLU de chaque étape, quand la vérité terrain existe.

    Sur les corpus annotés (CREMMA, CATMuS), on PEUT calculer un vrai CER.
    On l'utilise pour valider que la normalisation améliore bien les choses
    (le CER doit baisser à chaque étape).

    Args:
        references: Les transcriptions correctes (vérité terrain).
        versions: Dictionnaire {nom_etape: [transcriptions]}.

    Returns:
        Dictionnaire {nom_etape: CER} montrant l'évolution du CER.

    Example:
        >>> cers = evaluer_avec_verite_terrain(
        ...     ["et romanz"],
        ...     {"brut": ["⁊ romanz"], "regles": ["et romanz"]}
        ... )
        >>> print(cers["brut"], "→", cers["regles"])
    """
    from htr.metrics import calculer_cer_corpus

    resultat = {}
    for nom_etape, textes in versions.items():
        cer = calculer_cer_corpus(references, textes)
        resultat[nom_etape] = round(cer, 4)
    return resultat


def afficher_impact(impact: dict, avec_verite_terrain: bool = False) -> None:
    """Affiche un rapport lisible de l'impact des étapes.

    Args:
        impact: Le résultat de evaluer_etapes ou evaluer_avec_verite_terrain.
        avec_verite_terrain: True si l'impact contient des CER absolus.

    Example:
        >>> afficher_impact(evaluer_etapes(versions))
    """
    print("─── Impact des étapes de normalisation ───")
    if avec_verite_terrain:
        # impact = {nom_etape: cer}
        for nom_etape, cer in impact.items():
            print(f"  {nom_etape:25s} : CER = {cer:.2%}")
    else:
        # impact = {transition: {détails}}
        for transition, details in impact.items():
            taux = details["taux_changement_moyen"]
            modif = details["lignes_modifiees"]
            total = details["lignes_totales"]
            print(f"  {transition:25s} : {taux:.1%} de changement "
                  f"({modif}/{total} lignes modifiées)")


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de evaluation_relative.py ===\n")

    # On simule un petit corpus à 3 étapes
    corpus_brut = ["⁊ q~ il vint", "mõt e gẽt", "iadis vne dame", "rxmanz"]
    corpus_regles = ["et que il vint", "mont e gent", "iadis une dame", "rxmanz"]
    corpus_correction = ["et que il vint", "mont e gent", "iadis une dame", "romanz"]

    versions = {
        "brut": corpus_brut,
        "regles": corpus_regles,
        "regles+correction": corpus_correction,
    }

    # ── Évaluation relative (sans vérité terrain) ───────────────────────────
    print("── Sans vérité terrain (manuscrits BnF) ──")
    impact = evaluer_etapes(versions)
    afficher_impact(impact)

    # ── Évaluation absolue (avec vérité terrain CREMMA/CATMuS) ──────────────
    print("\n── Avec vérité terrain (CREMMA/CATMuS) ──")
    references = ["et que il vint", "mont e gent", "iadis une dame", "romanz"]
    cers = evaluer_avec_verite_terrain(references, versions)
    afficher_impact(cers, avec_verite_terrain=True)

    print("\n=== Démonstration terminée ✓ ===")
