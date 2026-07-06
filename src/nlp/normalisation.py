"""
normalisation.py — Normalisation des transcriptions médiévales.

C'est la BRIQUE CENTRALE du volet NLP. Les consignes l'affirment :
« La normalisation est la brique la plus facile à mettre en œuvre et celle
qui apporte le plus de gain en CER immédiat. »

Deux niveaux, dans l'ordre :
  1. RÈGLES DÉTERMINISTES (priorité absolue) :
     - Normalisation Unicode NFC
     - Substitution u/v et i/j (conventions graphiques médiévales)
     - Résolution du tilde nasal (ã → an, õ → on…)
     - Table d'abréviations (q~ → que, ꝑ → per…)
  2. CORRECTION GUIDÉE PAR CONFIANCE :
     Aux positions de faible confiance, on remplace par le candidat le plus
     plausible (ici via un lexique ; le MLM CamemBERT peut venir en plus).

Chaque règle est documentée dans CONVENTIONS_NLP.md, et son impact sur le
CER doit être mesuré (avant/après) — voir le doc de soutenance.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import unicodedata


# ─── Tables de normalisation ─────────────────────────────────────────────────

# Table d'abréviations médiévales → forme développée.
# À ADAPTER à votre corpus (les consignes insistent là-dessus).
TABLE_ABREVIATIONS = {
    "q~":  "que",    # q avec tilde = "que"
    "p~":  "pre",
    "ꝑ":   "per",    # p barré = "per"/"par"
    "ꝗ":   "qui",    # q barré = "qui"
    "ꝓ":   "pro",
    "⁊":   "et",     # nota tironienne = "et"
    "ẜ":   "ser",
    "ꝫ":   "us",     # abréviation finale -us
}

# Tilde nasal sur une voyelle → voyelle + n
RESOLUTION_TILDE = {
    "ã": "an", "ẽ": "en", "ĩ": "in", "õ": "on", "ũ": "un",
    "Ã": "An", "Ẽ": "En", "Ĩ": "In", "Õ": "On", "Ũ": "Un",
}

# Lexique de référence pour la correction guidée par confiance.
# En pratique : un grand lexique d'ancien/moyen français. Ici, un échantillon.
LEXIQUE_REFERENCE = {
    "romanz", "comence", "engleterre", "brut", "gent", "tindrent",
    "normant", "vindrent", "livres", "devise", "que", "qui", "roi",
    "comte", "ci", "li", "de", "sa", "e", "et", "la", "le", "les",
}


# ─── Règle 1 : Normalisation Unicode ─────────────────────────────────────────

def normaliser_unicode(texte: str) -> str:
    """Applique la normalisation Unicode NFC.

    Unicode permet d'écrire un même caractère accentué de plusieurs façons
    (ex: 'é' en un seul code, ou 'e' + accent combinant). NFC les unifie en
    une seule forme canonique. Indispensable avant toute comparaison de texte.

    Args:
        texte: Le texte à normaliser.

    Returns:
        Le texte en forme normale NFC.

    Example:
        >>> normaliser_unicode("café")  # unifie la représentation du 'é'
        'café'
    """
    # NFC = Normalization Form Canonical Composition
    return unicodedata.normalize("NFC", texte)


# ─── Règle 2 : Substitution u/v et i/j ───────────────────────────────────────

def normaliser_u_v_i_j(texte: str) -> str:
    """Normalise les graphies u/v et i/j médiévales.

    Au Moyen Âge, 'u' et 'v' étaient la même lettre (de même i/j). La
    convention moderne distingue selon le contexte :
      - 'v' en début de mot devient souvent 'u' (ou l'inverse selon l'usage)
      - 'j' est souvent un 'i' médiéval

    NOTE : cette règle est une SIMPLIFICATION. La vraie normalisation dépend
    de votre convention éditoriale (à documenter dans CONVENTIONS_NLP.md).
    Ici on choisit : v→u en début de mot voyelle, j→i. Adaptez à votre corpus.

    Args:
        texte: Le texte à normaliser.

    Returns:
        Le texte avec u/v et i/j harmonisés.

    Example:
        >>> normaliser_u_v_i_j("iaditz")  # j médiéval
        'iaditz'
    """
    resultat = []
    mots = texte.split(" ")
    for mot in mots:
        if not mot:
            resultat.append(mot)
            continue
        # j → i (le 'j' n'existait pas comme lettre distincte)
        mot = mot.replace("j", "i").replace("J", "I")
        # 'v' suivi d'une consonne en début de mot → 'u' (ex: "vne" → "une")
        if len(mot) >= 2 and mot[0] in "vV" and mot[1].lower() not in "aeiou":
            mot = ("u" if mot[0] == "v" else "U") + mot[1:]
        resultat.append(mot)
    return " ".join(resultat)


# ─── Règle 3 : Résolution du tilde nasal ─────────────────────────────────────

def resoudre_tilde(texte: str) -> str:
    """Développe les tildes nasaux (ã → an, õ → on, etc.).

    Dans les manuscrits, un tilde au-dessus d'une voyelle abrège un 'n' ou
    'm' nasal. On le développe en voyelle + n.

    Args:
        texte: Le texte à traiter.

    Returns:
        Le texte avec les tildes développés.

    Example:
        >>> resoudre_tilde("mõt")
        'mont'
    """
    for tilde, remplacement in RESOLUTION_TILDE.items():
        texte = texte.replace(tilde, remplacement)
    return texte


# ─── Règle 4 : Table d'abréviations ──────────────────────────────────────────

def developper_abreviations(texte: str) -> str:
    """Développe les abréviations médiévales selon la table.

    Args:
        texte: Le texte à traiter.

    Returns:
        Le texte avec les abréviations développées.

    Example:
        >>> developper_abreviations("⁊ q~")
        'et que'
    """
    for abrev, forme in TABLE_ABREVIATIONS.items():
        texte = texte.replace(abrev, forme)
    return texte


# ─── Pipeline de règles déterministes ────────────────────────────────────────

def normaliser_par_regles(texte: str) -> str:
    """Applique toutes les règles déterministes dans le bon ordre.

    L'ordre compte : Unicode d'abord (pour unifier), puis abréviations et
    tilde (qui ajoutent des caractères), puis u/v-i/j en dernier.

    Args:
        texte: La transcription brute du HTR.

    Returns:
        La transcription normalisée par règles.

    Example:
        >>> normaliser_par_regles("⁊ mõt")
        'et mont'
    """
    texte = normaliser_unicode(texte)
    texte = developper_abreviations(texte)
    texte = resoudre_tilde(texte)
    texte = normaliser_u_v_i_j(texte)
    return texte


# ─── Correction guidée par confiance ─────────────────────────────────────────

def corriger_par_confiance(
    texte: str,
    char_confidences: list[float],
    candidates: list[dict],
    seuil: float = 0.7,
) -> str:
    """Corrige les positions de faible confiance via le lexique.

    Principe (consignes NLP) : pour chaque position où la confiance est faible
    et où plusieurs candidats existent, on essaie chaque candidat et on garde
    celui qui forme un mot présent dans le lexique de référence.

    C'est une alternative simple au MLM CamemBERT (qui peut venir en plus).

    Args:
        texte: Le texte (déjà normalisé par règles, idéalement).
        char_confidences: Confiance de chaque caractère.
        candidates: Positions ambiguës avec leurs options de remplacement.
        seuil: Seuil de confiance en dessous duquel on tente une correction.

    Returns:
        Le texte corrigé.

    Example:
        >>> corriger_par_confiance(
        ...     "rxmanz", [0.9,0.4,0.9,0.9,0.9,0.9],
        ...     [{"position": 1, "options": ["o", "a", "e"]}]
        ... )
        'romanz'
    """
    # On transforme le texte en liste de caractères (modifiable)
    caracteres = list(texte)

    for candidat in candidates:
        position = candidat["position"]

        # Sécurité : la position doit être valide
        if position >= len(caracteres):
            continue
        # On ne corrige que si la confiance est bien faible
        if position < len(char_confidences) and char_confidences[position] >= seuil:
            continue

        # On essaie chaque option de remplacement
        for option in candidat["options"]:
            # On construit le mot candidat en remplaçant à cette position
            test = caracteres[:]
            test[position] = option
            mot_test = "".join(test)

            # Si un des mots formés est dans le lexique, on garde cette option
            if _mot_dans_lexique(mot_test, position):
                caracteres[position] = option
                break

    return "".join(caracteres)


def _mot_dans_lexique(texte: str, position: int) -> bool:
    """Vérifie si le mot contenant la position donnée est dans le lexique.

    Args:
        texte: Le texte complet.
        position: La position du caractère modifié.

    Returns:
        True si le mot à cette position appartient au lexique de référence.
    """
    # On retrouve le mot qui contient cette position (entre deux espaces)
    debut = texte.rfind(" ", 0, position) + 1   # début du mot
    fin = texte.find(" ", position)             # fin du mot
    if fin == -1:
        fin = len(texte)
    mot = texte[debut:fin].lower()
    return mot in LEXIQUE_REFERENCE


# ─── Pipeline complet de normalisation ───────────────────────────────────────

def normaliser_ligne(ligne: dict, avec_correction: bool = True) -> dict:
    """Normalise une ligne complète du data contract.

    Applique les règles déterministes puis, optionnellement, la correction
    guidée par confiance. Retourne une nouvelle ligne enrichie du texte
    normalisé (sans détruire le texte brut original).

    Args:
        ligne: Une ligne du data contract (avec text, char_confidences,
            candidates...).
        avec_correction: Si True, applique aussi la correction par confiance.

    Returns:
        La ligne enrichie d'un champ 'text_normalise'.

    Example:
        >>> ligne = {"text": "⁊ romanz", "char_confidences": [...],
        ...          "candidates": []}
        >>> r = normaliser_ligne(ligne)
        >>> print(r["text_normalise"])
    """
    ligne_normalisee = dict(ligne)  # copie

    # Étape 1 : règles déterministes
    texte = normaliser_par_regles(ligne["text"])

    # Étape 2 : correction guidée par confiance (optionnelle)
    if avec_correction and ligne.get("candidates"):
        texte = corriger_par_confiance(
            texte,
            ligne.get("char_confidences", []),
            ligne.get("candidates", []),
        )

    ligne_normalisee["text_normalise"] = texte
    return ligne_normalisee


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de normalisation.py ===\n")

    # ── Démo des règles déterministes ───────────────────────────────────────
    print("─── Règles déterministes ───")
    exemples = [
        "⁊ q~ il vint",        # abréviations
        "mõt e gẽt",           # tilde nasal
        "iadis vne dame",      # u/v et i/j
    ]
    for brut in exemples:
        normalise = normaliser_par_regles(brut)
        print(f"  \"{brut}\"  →  \"{normalise}\"")

    # ── Démo de la correction par confiance ─────────────────────────────────
    print("\n─── Correction guidée par confiance ───")
    texte = "rxmanz"  # le HTR a mal lu le 'o'
    confiances = [0.9, 0.4, 0.9, 0.9, 0.9, 0.9]  # position 1 incertaine
    candidats = [{"position": 1, "options": ["o", "a", "e"]}]
    corrige = corriger_par_confiance(texte, confiances, candidats)
    print(f"  \"{texte}\"  →  \"{corrige}\"  (corrigé via le lexique)")

    print("\n=== Démonstration terminée ✓ ===")
