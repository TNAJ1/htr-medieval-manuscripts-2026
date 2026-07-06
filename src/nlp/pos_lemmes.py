"""
pos_lemmes.py — Étiquetage grammatical (POS) et lemmatisation.

Quatrième brique du volet NLP. Après avoir repéré les entités (NER), on
enrichit chaque mot avec :
  - son POS (Part-Of-Speech) = sa nature grammaticale (NOUN, VERB, ADJ...)
  - son LEMME = sa forme de base ("vindrent" → "venir", "rois" → "roi")

Outil recommandé par les consignes : Stanza avec le modèle 'frm'
(français médiéval / moyen français). Rapide : quelques secondes par page.

DEUX MODES :
  - mode réel : charge le pipeline Stanza 'frm'.
  - mode simulation : POS/lemmes par petites règles, pour tester sans
    téléchargement. Produit le même format de sortie.

Les étiquettes POS suivent le standard Universal Dependencies (UPOS) :
  NOUN (nom), VERB (verbe), ADJ (adjectif), ADP (préposition),
  DET (déterminant), PRON (pronom), PROPN (nom propre), etc.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path


# ─── Petits lexiques pour le mode simulation ─────────────────────────────────
# En mode réel, Stanza décide. En simulation, on utilise ces règles simples.

MOTS_OUTILS = {
    # déterminants
    "li": ("DET", "li"), "le": ("DET", "le"), "la": ("DET", "la"),
    "les": ("DET", "le"), "un": ("DET", "un"), "une": ("DET", "un"),
    "sa": ("DET", "son"), "son": ("DET", "son"), "ses": ("DET", "son"),
    # prépositions
    "de": ("ADP", "de"), "a": ("ADP", "a"), "en": ("ADP", "en"),
    "par": ("ADP", "par"), "por": ("ADP", "por"), "sur": ("ADP", "sur"),
    # conjonctions
    "e": ("CCONJ", "et"), "et": ("CCONJ", "et"), "que": ("SCONJ", "que"),
    "qui": ("PRON", "qui"), "ne": ("ADV", "ne"),
    # pronoms
    "il": ("PRON", "il"), "ele": ("PRON", "ele"), "je": ("PRON", "je"),
    "ci": ("ADV", "ci"), "ainz": ("ADV", "ainz"),
}

# Verbes médiévaux fréquents → lemme (forme de base à l'infinitif)
VERBES = {
    "vint": "venir", "vindrent": "venir", "tint": "tenir",
    "tindrent": "tenir", "comence": "comencier", "fu": "estre",
    "furent": "estre", "ot": "avoir", "orent": "avoir", "dist": "dire",
    "fist": "faire", "prist": "prendre",
}


# ─── Classe principale ───────────────────────────────────────────────────────

class AnalyseurMorphologique:
    """Étiquette chaque mot avec son POS et son lemme.

    Attributes:
        mode_simulation: Si True, analyse par règles sans modèle.
        pipeline: Le pipeline Stanza (None en simulation).

    Example:
        >>> analyseur = AnalyseurMorphologique(mode_simulation=True)
        >>> resultat = analyseur.analyser("li rois tint la terre")
        >>> for mot, pos, lemme in resultat:
        ...     print(mot, pos, lemme)
    """

    def __init__(self, langue: str = "frm", mode_simulation: bool = False):
        """Initialise l'analyseur morphologique.

        Args:
            langue: Code langue Stanza. 'frm' = moyen français.
            mode_simulation: Si True, aucun téléchargement (analyse par règles).
        """
        self.langue = langue
        self.mode_simulation = mode_simulation
        self.pipeline = None

        if not mode_simulation:
            import stanza

            print(f"  → Chargement du pipeline Stanza '{langue}'…")
            # Télécharge le modèle si absent, puis crée le pipeline
            stanza.download(langue, verbose=False)
            self.pipeline = stanza.Pipeline(
                lang=langue,
                processors="tokenize,pos,lemma",
                verbose=False,
            )
            print("  ✓ Pipeline Stanza chargé.")

    def analyser(self, texte: str) -> list[tuple[str, str, str]]:
        """Analyse un texte : retourne (mot, POS, lemme) pour chaque mot.

        Args:
            texte: Le texte à analyser (idéalement déjà normalisé).

        Returns:
            Liste de tuples (mot, pos, lemme).

        Example:
            >>> analyseur.analyser("li rois")
            [('li', 'DET', 'li'), ('rois', 'NOUN', 'roi')]
        """
        if self.mode_simulation:
            return self._analyser_simulation(texte)
        return self._analyser_reel(texte)

    # ── Mode réel (Stanza) ──────────────────────────────────────────────────

    def _analyser_reel(self, texte: str) -> list[tuple[str, str, str]]:
        """Analyse réelle avec Stanza.

        Args:
            texte: Le texte à analyser.

        Returns:
            Liste de tuples (mot, pos, lemme).
        """
        doc = self.pipeline(texte)
        resultat = []
        for phrase in doc.sentences:
            for mot in phrase.words:
                # Stanza fournit .text, .upos (POS universel) et .lemma
                resultat.append((mot.text, mot.upos, mot.lemma or mot.text))
        return resultat

    # ── Mode simulation ─────────────────────────────────────────────────────

    def _analyser_simulation(self, texte: str) -> list[tuple[str, str, str]]:
        """Analyse par règles simples, pour tester sans modèle.

        Args:
            texte: Le texte à analyser.

        Returns:
            Liste de tuples (mot, pos, lemme).
        """
        resultat = []
        for mot in texte.split():
            mot_propre = mot.lower().strip(".,;:!?")
            pos, lemme = self._deviner_pos_lemme(mot_propre, mot)
            resultat.append((mot, pos, lemme))
        return resultat

    @staticmethod
    def _deviner_pos_lemme(mot_propre: str, mot_original: str) -> tuple[str, str]:
        """Devine le POS et le lemme d'un mot par règles.

        Args:
            mot_propre: Le mot nettoyé (minuscule, sans ponctuation).
            mot_original: Le mot d'origine (pour garder la casse du lemme).

        Returns:
            Un tuple (pos, lemme).
        """
        # 1. Mot-outil connu (déterminant, préposition...)
        if mot_propre in MOTS_OUTILS:
            return MOTS_OUTILS[mot_propre]

        # 2. Verbe connu
        if mot_propre in VERBES:
            return "VERB", VERBES[mot_propre]

        # 3. Nom propre : commence par une majuscule
        if mot_original and mot_original[0].isupper():
            return "PROPN", mot_propre

        # 4. Par défaut : nom commun, lemme = forme sans 's' final (pluriel)
        lemme = mot_propre.rstrip("s") if len(mot_propre) > 3 else mot_propre
        return "NOUN", lemme

    def extraire_lemmes(self, texte: str) -> list[str]:
        """Extrait uniquement la liste des lemmes d'un texte.

        Pratique pour construire un index ou compter les formes de base.

        Args:
            texte: Le texte à analyser.

        Returns:
            La liste des lemmes.

        Example:
            >>> analyseur.extraire_lemmes("li rois tint")
            ['li', 'roi', 'tenir']
        """
        return [lemme for _, _, lemme in self.analyser(texte)]


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de pos_lemmes.py (mode simulation) ===\n")

    analyseur = AnalyseurMorphologique(mode_simulation=True)

    phrases = [
        "li rois tint la terre",
        "Ci comence li romanz de Brut",
    ]
    for phrase in phrases:
        print(f"─── \"{phrase}\" ───")
        for mot, pos, lemme in analyseur.analyser(phrase):
            print(f"  {mot:12s} {pos:8s} lemme: {lemme}")
        print()

    print("=== Démonstration terminée ✓ ===")
