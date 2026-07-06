"""
kraken_model.py — Transcription des lignes par Kraken.

Kraken est un moteur HTR conçu spécifiquement pour les humanités numériques.
Contrairement à TrOCR (Transformer généraliste), Kraken repose sur un réseau
récurrent (CNN + BiLSTM) et fournit NATIVEMENT les confiances par caractère.

Ce module expose la classe TranscripteurKraken, qui a EXACTEMENT la même
interface que TranscripteurHTR (TrOCR) : la méthode transcrire() retourne le
même dictionnaire {text, char_confidences, candidates}. Grâce à cela, les
deux modèles sont interchangeables et comparables (test de McNemar).

DEUX MODES :
  - mode réel : charge un modèle Kraken (.mlmodel) et l'exécute.
  - mode simulation : sorties factices pour tester sans téléchargement.

Pour le fine-tuning, Kraken s'utilise en ligne de commande :
    ketos train -i modele_base.mlmodel -o modele_finetune *.xml
(voir la fonction commande_fine_tuning() en bas pour la commande exacte).
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path

import numpy as np


# ─── Classe principale ───────────────────────────────────────────────────────

class TranscripteurKraken:
    """Encapsule un modèle Kraken pour transcrire des images de lignes.

    Même interface que TranscripteurHTR (TrOCR) : on charge le modèle une
    fois, puis on appelle transcrire() sur chaque ligne. Cela permet de
    brancher Kraken ou TrOCR de façon interchangeable dans le pipeline.

    Attributes:
        chemin_modele: Chemin du fichier modèle Kraken (.mlmodel).
        mode_simulation: Si True, génère des sorties factices sans modèle.
        model: Le modèle Kraken chargé (None en simulation).

    Example:
        >>> transcripteur = TranscripteurKraken(mode_simulation=True)
        >>> resultat = transcripteur.transcrire(image_ligne)
        >>> print(resultat["text"])
    """

    def __init__(
        self,
        chemin_modele: str = "catmus-medieval.mlmodel",
        mode_simulation: bool = False,
    ):
        """Initialise le transcripteur Kraken.

        Args:
            chemin_modele: Chemin du modèle Kraken. En pratique, on utilise
                un modèle médiéval pré-entraîné (HTR-United, CATMuS).
            mode_simulation: Si True, aucun chargement (sorties factices).
        """
        self.chemin_modele = chemin_modele
        self.mode_simulation = mode_simulation
        self.model = None

        if not mode_simulation:
            # Import local : le mode simulation marche même sans kraken installé
            from kraken.lib import models

            print(f"  → Chargement du modèle Kraken {chemin_modele}…")
            # load_any charge un modèle Kraken quel que soit son format
            self.model = models.load_any(chemin_modele)
            print("  ✓ Modèle Kraken chargé.")

    def transcrire(self, image_ligne: np.ndarray) -> dict:
        """Transcrit l'image d'une ligne en texte + confiances par caractère.

        Args:
            image_ligne: Image de la ligne (2D niveaux de gris ou 3D couleur).

        Returns:
            Dictionnaire identique à celui de TrOCR :
              - 'text' (str)
              - 'char_confidences' (list[float])
              - 'candidates' (list[dict])

        Example:
            >>> resultat = transcripteur.transcrire(image)
        """
        if self.mode_simulation:
            return self._transcrire_simulation(image_ligne)
        return self._transcrire_reel(image_ligne)

    # ── Mode réel (Kraken) ──────────────────────────────────────────────────

    def _transcrire_reel(self, image_ligne: np.ndarray) -> dict:
        """Transcription réelle avec Kraken.

        Kraken attend une image PIL et un objet de segmentation décrivant
        où se trouve la ligne. Comme notre image EST déjà une ligne découpée,
        on construit une segmentation minimale couvrant toute l'image.

        Avantage de Kraken : l'objet de prédiction (ocr_record) fournit
        directement record.prediction (texte) et record.confidences
        (confiance par caractère) — pas de reconstruction token→caractère.

        Args:
            image_ligne: Image de la ligne déjà découpée.

        Returns:
            Dictionnaire text / char_confidences / candidates.

        Note:
            L'API exacte de Kraken varie selon la version (4.x vs 5.x).
            Ce code suit l'API récente ; vérifiez la doc de votre version
            (https://kraken.re) si un appel diffère.
        """
        from PIL import Image
        from kraken import rpred
        from kraken.containers import Segmentation, BaselineLine

        # ── Préparer l'image ────────────────────────────────────────────────
        if image_ligne.ndim == 2:
            image_pil = Image.fromarray(image_ligne.astype(np.uint8)).convert("RGB")
        else:
            image_pil = Image.fromarray(image_ligne.astype(np.uint8))

        hauteur, largeur = image_ligne.shape[:2]

        # ── Construire une segmentation couvrant toute l'image ──────────────
        # La ligne de base (baseline) est une ligne horizontale au milieu.
        # Le polygone (boundary) entoure toute l'image.
        ligne = BaselineLine(
            id="ligne_0",
            baseline=[[0, hauteur // 2], [largeur, hauteur // 2]],
            boundary=[[0, 0], [largeur, 0], [largeur, hauteur], [0, hauteur]],
        )
        segmentation = Segmentation(
            type="baselines",
            imagename="ligne",
            text_direction="horizontal-lr",  # gauche → droite
            script_detection=False,
            lines=[ligne],
        )

        # ── Lancer la reconnaissance ────────────────────────────────────────
        # rpred retourne un itérateur de ocr_record (un par ligne)
        predictions = rpred.rpred(self.model, image_pil, segmentation)
        record = next(iter(predictions))

        # Kraken fournit directement le texte et les confiances par caractère
        texte = record.prediction
        char_confidences = [float(c) for c in record.confidences]

        # Identifier les positions incertaines
        candidates = self._identifier_candidats(char_confidences, seuil=0.7)

        return {
            "text": texte,
            "char_confidences": char_confidences,
            "candidates": candidates,
        }

    # ── Mode simulation ─────────────────────────────────────────────────────

    def _transcrire_simulation(self, image_ligne: np.ndarray) -> dict:
        """Génère une transcription factice pour tester le pipeline.

        IMPORTANT : on rend la sortie LÉGÈREMENT différente de celle de TrOCR.
        Les deux moteurs simulés se "trompent" sur des positions différentes,
        ce qui rend la comparaison McNemar réaliste et non triviale.

        Args:
            image_ligne: Image de la ligne (sa largeur influence le texte).

        Returns:
            Dictionnaire text / char_confidences / candidates.
        """
        largeur = image_ligne.shape[1] if image_ligne.ndim >= 2 else 100
        nb_mots = max(1, largeur // 60)
        mots_medievaux = ["romanz", "Engleterre", "Brut", "gent", "tindrent",
                          "Normant", "vindrent", "comence"]
        texte = " ".join(mots_medievaux[i % len(mots_medievaux)]
                         for i in range(nb_mots))

        # Confiances simulées : Kraken simulé est un peu PLUS prudent que TrOCR
        # (plus de basses confiances), pour qu'il diffère de TrOCR.
        char_confidences = []
        for _ in texte:
            if np.random.random() < 0.80:           # 80 % haute (vs 85 % TrOCR)
                c = np.random.uniform(0.75, 1.0)
            else:
                c = np.random.uniform(0.35, 0.70)
            char_confidences.append(round(float(c), 3))

        candidates = self._identifier_candidats(char_confidences, seuil=0.7)

        return {
            "text": texte,
            "char_confidences": char_confidences,
            "candidates": candidates,
        }

    # ── Méthode commune (identique à TrOCR) ─────────────────────────────────

    @staticmethod
    def _identifier_candidats(
        char_confidences: list[float], seuil: float = 0.7
    ) -> list[dict]:
        """Identifie les positions incertaines (candidats pour le NLP).

        Args:
            char_confidences: Liste des confiances par caractère.
            seuil: En dessous, le caractère est jugé ambigu.

        Returns:
            Liste de dicts {position, options}.
        """
        candidates = []
        for position, confiance in enumerate(char_confidences):
            if confiance < seuil:
                candidates.append({
                    "position": position,
                    "options": ["a", "e", "o"],
                })
        return candidates


# ─── Aide au fine-tuning ─────────────────────────────────────────────────────

def commande_fine_tuning(
    modele_base: str = "catmus-medieval.mlmodel",
    dossier_xml: str = "segmentations/",
    modele_sortie: str = "modele_finetune",
) -> str:
    """Retourne la commande ketos pour fine-tuner un modèle Kraken.

    Kraken ne se fine-tune pas en Python mais en ligne de commande avec
    l'outil `ketos`. Cette fonction construit la commande à exécuter.

    Args:
        modele_base: Modèle de départ (ex: un modèle médiéval HTR-United).
        dossier_xml: Dossier contenant les fichiers PAGE XML annotés
            (notre module page_xml.py produit exactement ce format !).
        modele_sortie: Nom du modèle fine-tuné à produire.

    Returns:
        La commande shell à lancer dans un terminal.

    Example:
        >>> print(commande_fine_tuning())
        ketos train -i catmus-medieval.mlmodel ...
    """
    return (
        f"ketos train "
        f"-i {modele_base} "          # modèle de départ (-i = input)
        f"-o {modele_sortie} "        # modèle de sortie (-o = output)
        f"-f page "                   # format des annotations : PAGE XML
        f"--augment "                 # augmentation de données
        f"{dossier_xml}*.xml"         # nos fichiers PAGE XML d'entraînement
    )


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import cv2
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.utils import fixer_seeds
    from htr.segmentation import segmenter_lignes, creer_image_multi_lignes

    print("=== Démonstration de kraken_model.py (mode simulation) ===\n")
    fixer_seeds(42)

    # Prépare des lignes
    fixture = Path("tests/fixtures/multi_lignes.png")
    creer_image_multi_lignes(fixture)
    img = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    lignes = segmenter_lignes(img_bin)

    # Transcrit avec Kraken
    transcripteur = TranscripteurKraken(mode_simulation=True)
    for ligne in lignes:
        resultat = transcripteur.transcrire(ligne["image"])
        conf = np.mean(resultat["char_confidences"])
        print(f"  Ligne {ligne['reading_order']} : \"{resultat['text']}\" "
              f"(confiance {conf:.2f})")

    print("\nCommande de fine-tuning Kraken :")
    print(f"  {commande_fine_tuning()}")
    print("\n=== Démonstration terminée ✓ ===")
