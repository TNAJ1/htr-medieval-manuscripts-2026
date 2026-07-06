"""
transcripteur_factory.py — Sélection du moteur HTR (TrOCR, Kraken ou fusion).

Ce petit module permet de choisir facilement quel moteur HTR utiliser dans
le pipeline, sans toucher au code du pipeline lui-même. Trois choix :

  - "trocr"  : utilise uniquement TrOCR
  - "kraken" : utilise uniquement Kraken
  - "fusion" : exécute les deux et fusionne par vote pondéré

Tous renvoient le même format de sortie {text, char_confidences, candidates},
donc le pipeline ne voit aucune différence.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from htr.htr_model import TranscripteurHTR
from htr.kraken_model import TranscripteurKraken
from htr.comparaison import fusionner_transcriptions


# ─── Transcripteur de fusion ─────────────────────────────────────────────────

class TranscripteurFusion:
    """Combine TrOCR et Kraken par vote pondéré sur chaque ligne.

    Exécute les deux moteurs sur chaque image de ligne, puis fusionne leurs
    sorties en gardant le caractère le plus confiant à chaque position.
    Produit le même format que les transcripteurs simples.

    Attributes:
        trocr: Le transcripteur TrOCR.
        kraken: Le transcripteur Kraken.

    Example:
        >>> fusion = TranscripteurFusion(mode_simulation=True)
        >>> resultat = fusion.transcrire(image_ligne)
    """

    def __init__(self, mode_simulation: bool = False):
        """Initialise les deux transcripteurs sous-jacents.

        Args:
            mode_simulation: Passé aux deux transcripteurs.
        """
        self.trocr = TranscripteurHTR(mode_simulation=mode_simulation)
        self.kraken = TranscripteurKraken(mode_simulation=mode_simulation)

    def transcrire(self, image_ligne: np.ndarray) -> dict:
        """Transcrit avec les deux moteurs et fusionne le résultat.

        Args:
            image_ligne: Image de la ligne.

        Returns:
            Dictionnaire text / char_confidences / candidates fusionné.
        """
        res_trocr = self.trocr.transcrire(image_ligne)
        res_kraken = self.kraken.transcrire(image_ligne)

        # Fusion par vote pondéré (Needleman-Wunsch)
        texte, confiances = fusionner_transcriptions(
            res_trocr["text"], res_trocr["char_confidences"],
            res_kraken["text"], res_kraken["char_confidences"],
        )

        # Recalcule les candidats sur le texte fusionné
        candidates = []
        for position, c in enumerate(confiances):
            if c < 0.7:
                candidates.append({"position": position, "options": ["a", "e", "o"]})

        return {
            "text": texte,
            "char_confidences": confiances,
            "candidates": candidates,
        }


# ─── Fonction de fabrique ────────────────────────────────────────────────────

def creer_transcripteur(modele: str = "trocr", mode_simulation: bool = False):
    """Crée le transcripteur correspondant au modèle demandé.

    Args:
        modele: "trocr", "kraken" ou "fusion".
        mode_simulation: Si True, tous les moteurs sont en simulation.

    Returns:
        Un transcripteur avec une méthode transcrire().

    Raises:
        ValueError: Si le modèle demandé est inconnu.

    Example:
        >>> t = creer_transcripteur("fusion", mode_simulation=True)
        >>> resultat = t.transcrire(image_ligne)
    """
    modele = modele.lower()

    if modele == "trocr":
        return TranscripteurHTR(mode_simulation=mode_simulation)
    elif modele == "kraken":
        return TranscripteurKraken(mode_simulation=mode_simulation)
    elif modele == "fusion":
        return TranscripteurFusion(mode_simulation=mode_simulation)
    else:
        raise ValueError(
            f"Modèle inconnu : '{modele}'. "
            f"Choix possibles : 'trocr', 'kraken', 'fusion'."
        )


# ─── Démonstration ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.utils import fixer_seeds

    print("=== Démonstration de transcripteur_factory.py ===\n")
    fixer_seeds(42)

    image = np.ones((40, 300), dtype=np.uint8) * 255

    for nom_modele in ["trocr", "kraken", "fusion"]:
        fixer_seeds(42)  # même graine pour comparer équitablement
        transcripteur = creer_transcripteur(nom_modele, mode_simulation=True)
        resultat = transcripteur.transcrire(image)
        conf = np.mean(resultat["char_confidences"])
        print(f"  {nom_modele:7s} : \"{resultat['text']}\" "
              f"(confiance moyenne {conf:.2f})")

    print("\n=== Démonstration terminée ✓ ===")
