"""
ner.py — Reconnaissance d'entités nommées (NER) avec schéma BIO.

Troisième brique du volet NLP. La NER repère dans le texte les entités :
personnes, lieux, dates, organisations, titres.

SCHÉMA BIO (le standard pour étiqueter les entités) :
  - B-XXX : Begin — premier mot d'une entité de type XXX
  - I-XXX : Inside — mot suivant de la même entité
  - O     : Outside — le mot n'est pas une entité

Exemple : "le roi Charles de France"
            O   O   B-PER  O  B-LOC
  → "Charles" est une personne (PER), "France" un lieu (LOC).

Le modèle recommandé (consignes NLP) est un CamemBERT/RoBERTa déjà fine-tuné
sur des corpus médiévaux, ex: magistermilitum/roberta-multilingual-medieval-ner.
On NE PART PAS de zéro : on adapte un modèle existant.

DEUX MODES :
  - mode réel : charge le modèle HuggingFace et étiquette le texte.
  - mode simulation : étiquetage par règles/lexique, pour tester sans
    téléchargement. Produit le même format de sortie.

Point critique (consignes) : l'alignement des labels sur la tokenisation.
Un mot peut être découpé en plusieurs sous-tokens (word-pieces) ; on utilise
-100 pour ignorer les sous-tokens de continuation pendant l'entraînement.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path


# ─── Schéma d'entités ────────────────────────────────────────────────────────

# Les types d'entités reconnus. TITLE est ajouté pour les manuscrits médiévaux
# qui contiennent beaucoup de titres (roi, comte, évêque...).
TYPES_ENTITES = ["PER", "LOC", "DATE", "ORG", "TITLE"]

# La liste complète des étiquettes BIO :
# O + (B-XXX et I-XXX pour chaque type)
LABELS_BIO = ["O"]
for type_entite in TYPES_ENTITES:
    LABELS_BIO.append(f"B-{type_entite}")
    LABELS_BIO.append(f"I-{type_entite}")
# Résultat : ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-DATE", ...]

# Correspondance étiquette → identifiant numérique (pour le modèle)
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS_BIO)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# Constante spéciale : -100 = "ignore ce token dans le calcul de la perte"
# Utilisé pour les sous-tokens de continuation (word-pieces).
LABEL_IGNORE = -100


# ─── Lexiques pour le mode simulation ────────────────────────────────────────
# En mode réel, c'est le modèle CamemBERT qui décide. En simulation, on utilise
# ces petits lexiques pour produire un étiquetage plausible.

LEXIQUE_PERSONNES = {"charles", "guillaume", "brut", "arthur", "wace", "marie",
                     "philippe", "louis", "henri", "richard"}
LEXIQUE_LIEUX = {"france", "engleterre", "angleterre", "normandie", "paris",
                 "rome", "bretagne", "londres"}
LEXIQUE_TITRES = {"roi", "reine", "comte", "comtesse", "évêque", "duc",
                  "empereur", "seigneur", "dame"}
LEXIQUE_ORGS = {"église", "abbaye", "royaume", "parlement"}
# Les dates sont détectées par un motif (présence de chiffres ou de mots-clés).
MOTS_DATE = {"mil", "cent", "ans", "an", "jour", "mois"}


# ─── Classe principale ───────────────────────────────────────────────────────

class ReconnaisseurEntites:
    """Étiquette les entités nommées d'un texte selon le schéma BIO.

    Attributes:
        nom_modele: Identifiant HuggingFace du modèle NER.
        mode_simulation: Si True, étiquette par règles sans modèle.

    Example:
        >>> ner = ReconnaisseurEntites(mode_simulation=True)
        >>> resultat = ner.etiqueter("le roi Charles de France")
        >>> for mot, label in resultat:
        ...     print(mot, label)
    """

    def __init__(
        self,
        nom_modele: str = "magistermilitum/roberta-multilingual-medieval-ner",
        mode_simulation: bool = False,
    ):
        """Initialise le reconnaisseur d'entités.

        Args:
            nom_modele: Modèle NER médiéval pré-entraîné (HuggingFace).
            mode_simulation: Si True, aucun téléchargement (étiquetage par règles).
        """
        self.nom_modele = nom_modele
        self.mode_simulation = mode_simulation
        self.pipeline = None

        if not mode_simulation:
            from transformers import pipeline

            print(f"  → Chargement du modèle NER {nom_modele}…")
            # Le "pipeline" HuggingFace gère tokenisation + modèle + décodage
            self.pipeline = pipeline("token-classification", model=nom_modele)
            print("  ✓ Modèle NER chargé.")

    def etiqueter(self, texte: str) -> list[tuple[str, str]]:
        """Étiquette chaque mot d'un texte avec son label BIO.

        Args:
            texte: Le texte à analyser (idéalement déjà normalisé).

        Returns:
            Liste de paires (mot, label_BIO), une par mot.

        Example:
            >>> ner.etiqueter("le roi Charles")
            [('le', 'O'), ('roi', 'B-TITLE'), ('Charles', 'B-PER')]
        """
        if self.mode_simulation:
            return self._etiqueter_simulation(texte)
        return self._etiqueter_reel(texte)

    # ── Mode réel (CamemBERT) ───────────────────────────────────────────────

    def _etiqueter_reel(self, texte: str) -> list[tuple[str, str]]:
        """Étiquetage réel avec le modèle CamemBERT médiéval.

        Args:
            texte: Le texte à analyser.

        Returns:
            Liste de paires (mot, label_BIO).
        """
        # Le pipeline retourne une liste de prédictions par token
        predictions = self.pipeline(texte)

        # On reconstruit l'étiquetage mot par mot
        mots = texte.split()
        resultat = []
        for mot in mots:
            # On cherche la prédiction qui correspond à ce mot
            label = "O"  # par défaut
            for pred in predictions:
                if pred["word"].strip("▁ ").lower() in mot.lower():
                    label = pred["entity"]
                    break
            resultat.append((mot, label))
        return resultat

    # ── Mode simulation (par lexique) ───────────────────────────────────────

    def _etiqueter_simulation(self, texte: str) -> list[tuple[str, str]]:
        """Étiquetage par règles/lexique, pour tester sans modèle.

        Parcourt les mots et attribue un label BIO selon les lexiques.
        Gère le préfixe B-/I- : si deux entités du même type se suivent,
        la première est B-, les suivantes I-.

        Args:
            texte: Le texte à analyser.

        Returns:
            Liste de paires (mot, label_BIO).
        """
        mots = texte.split()
        resultat = []
        type_precedent = None  # type d'entité du mot précédent

        for mot in mots:
            # On nettoie le mot pour la comparaison (minuscule, sans ponctuation)
            mot_propre = mot.lower().strip(".,;:!?")

            # On détermine le type d'entité
            type_entite = self._detecter_type(mot_propre)

            if type_entite is None:
                # Pas une entité
                resultat.append((mot, "O"))
                type_precedent = None
            else:
                # C'est une entité : B- si nouvelle, I- si continuation
                if type_entite == type_precedent:
                    prefixe = "I-"  # continuation de la même entité
                else:
                    prefixe = "B-"  # début d'une nouvelle entité
                resultat.append((mot, f"{prefixe}{type_entite}"))
                type_precedent = type_entite

        return resultat

    @staticmethod
    def _detecter_type(mot: str) -> str | None:
        """Détecte le type d'entité d'un mot via les lexiques.

        Args:
            mot: Le mot nettoyé (minuscule, sans ponctuation).

        Returns:
            Le type d'entité ("PER", "LOC"...) ou None si pas une entité.
        """
        if mot in LEXIQUE_PERSONNES:
            return "PER"
        if mot in LEXIQUE_LIEUX:
            return "LOC"
        if mot in LEXIQUE_TITRES:
            return "TITLE"
        if mot in LEXIQUE_ORGS:
            return "ORG"
        if mot in MOTS_DATE or mot.isdigit():
            return "DATE"
        return None

    # ── Extraction des entités regroupées ───────────────────────────────────

    def extraire_entites(self, texte: str) -> list[dict]:
        """Extrait les entités complètes (regroupe les B- et I- consécutifs).

        Au lieu de labels par mot, retourne les entités assemblées.
        Pratique pour construire le graphe ou l'export TEI.

        Args:
            texte: Le texte à analyser.

        Returns:
            Liste de dicts {texte, type}, une par entité.

        Example:
            >>> ner.extraire_entites("Charles de France")
            [{'texte': 'Charles', 'type': 'PER'}, {'texte': 'France', 'type': 'LOC'}]
        """
        labels = self.etiqueter(texte)
        entites = []
        entite_courante = None  # accumule les mots de l'entité en cours

        for mot, label in labels:
            if label == "O":
                # Fin d'une éventuelle entité en cours
                if entite_courante:
                    entites.append(entite_courante)
                    entite_courante = None
            elif label.startswith("B-"):
                # Nouvelle entité : on ferme l'ancienne, on en ouvre une nouvelle
                if entite_courante:
                    entites.append(entite_courante)
                type_entite = label[2:]  # retire "B-"
                entite_courante = {"texte": mot, "type": type_entite}
            elif label.startswith("I-"):
                # Continuation : on ajoute le mot à l'entité en cours
                if entite_courante:
                    entite_courante["texte"] += " " + mot

        # Ne pas oublier la dernière entité
        if entite_courante:
            entites.append(entite_courante)

        return entites


# ─── Alignement labels / tokenisation (point critique du cours) ──────────────

def aligner_labels_sur_tokens(
    mots: list[str],
    labels: list[str],
    word_ids: list[int | None],
) -> list[int]:
    """Aligne les labels (un par mot) sur les sous-tokens du tokenizer.

    POINT CRITIQUE (consignes NLP) : les modèles type BERT découpent un mot en
    plusieurs sous-tokens (word-pieces). Ex: "Engleterre" → ["Engle", "##terre"].
    On doit aligner les labels :
      - Le PREMIER sous-token du mot reçoit le vrai label.
      - Les sous-tokens de CONTINUATION reçoivent -100 (ignorés par la perte).
      - Les tokens spéciaux ([CLS], [SEP]) reçoivent aussi -100.

    Args:
        mots: La liste des mots d'origine.
        labels: Le label BIO de chaque mot.
        word_ids: Pour chaque token produit par le tokenizer, l'indice du mot
            d'origine (None pour les tokens spéciaux). Fourni par
            tokenizer(..., return_offsets) → .word_ids().

    Returns:
        Liste d'identifiants de labels alignés sur les tokens (avec -100).

    Example:
        >>> # "Engleterre" (label B-LOC) découpé en 2 sous-tokens
        >>> aligner_labels_sur_tokens(
        ...     ["Engleterre"], ["B-LOC"], [None, 0, 0, None]
        ... )
        [-100, 3, -100, -100]
    """
    labels_alignes = []
    mot_precedent = None

    for word_id in word_ids:
        if word_id is None:
            # Token spécial ([CLS], [SEP], padding) → ignoré
            labels_alignes.append(LABEL_IGNORE)
        elif word_id != mot_precedent:
            # Premier sous-token d'un mot → vrai label
            labels_alignes.append(LABEL_TO_ID[labels[word_id]])
        else:
            # Sous-token de continuation → ignoré (-100)
            labels_alignes.append(LABEL_IGNORE)
        mot_precedent = word_id

    return labels_alignes


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de ner.py (mode simulation) ===\n")

    ner = ReconnaisseurEntites(mode_simulation=True)

    # ── Étiquetage BIO ──────────────────────────────────────────────────────
    print("─── Étiquetage BIO ───")
    phrases = [
        "le roi Charles de France",
        "Brut tint Engleterre",
        "la dame Marie de Bretagne",
    ]
    for phrase in phrases:
        print(f"\n  \"{phrase}\"")
        for mot, label in ner.etiqueter(phrase):
            marque = "← entité" if label != "O" else ""
            print(f"    {mot:14s} {label:10s} {marque}")

    # ── Extraction d'entités ────────────────────────────────────────────────
    print("\n─── Entités extraites ───")
    entites = ner.extraire_entites("le roi Charles de France et la dame Marie")
    for e in entites:
        print(f"  {e['texte']:12s} → {e['type']}")

    # ── Liste des labels du schéma ──────────────────────────────────────────
    print(f"\n─── Schéma BIO ({len(LABELS_BIO)} labels) ───")
    print(f"  {LABELS_BIO}")

    print("\n=== Démonstration terminée ✓ ===")
