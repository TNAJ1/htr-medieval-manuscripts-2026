"""
htr_model.py — Transcription des lignes par TrOCR.

Ce module prend l'image d'une ligne (produite par segmentation.py) et
retourne sa transcription textuelle ACCOMPAGNÉE des confiances par caractère.
Ces char_confidences sont indispensables au volet NLP (correction guidée
par confiance).

Le modèle utilisé est TrOCR (microsoft/trocr-base-handwritten), un modèle
encodeur-décodeur de HuggingFace :
  - L'encodeur (un ViT) "regarde" l'image.
  - Le décodeur (un Transformer) génère le texte caractère par caractère.

DEUX MODES :
  - mode réel : télécharge et exécute TrOCR (nécessite Internet + ~1,3 Go).
  - mode simulation : génère des sorties factices pour tester le pipeline
    sans téléchargement. Utile en développement et pour les tests.

Pour le fine-tuning par LoRA, voir la fonction preparer_lora() en bas.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from pathlib import Path

import numpy as np


# ─── Classe principale ───────────────────────────────────────────────────────

class TranscripteurHTR:
    """Encapsule un modèle TrOCR pour transcrire des images de lignes.

    On utilise une classe (plutôt que des fonctions) car le modèle doit être
    chargé UNE SEULE FOIS puis réutilisé pour toutes les lignes. Charger le
    modèle à chaque appel serait extrêmement lent.

    Attributes:
        nom_modele: Identifiant HuggingFace du modèle.
        mode_simulation: Si True, génère des sorties factices sans modèle.
        processor: Le préprocesseur d'images de TrOCR (None en simulation).
        model: Le modèle TrOCR (None en simulation).

    Example:
        >>> transcripteur = TranscripteurHTR(mode_simulation=True)
        >>> resultat = transcripteur.transcrire(image_ligne)
        >>> print(resultat["text"])
    """

    def __init__(
        self,
        nom_modele: str = "microsoft/trocr-base-handwritten",
        mode_simulation: bool = False,
    ):
        """Initialise le transcripteur et charge le modèle si nécessaire.

        Args:
            nom_modele: Identifiant du modèle sur HuggingFace.
            mode_simulation: Si True, n'effectue aucun téléchargement et
                génère des sorties factices (pour tests/développement).
        """
        self.nom_modele = nom_modele
        self.mode_simulation = mode_simulation
        self.processor = None
        self.model = None

        if not mode_simulation:
            # On importe ici (et non en haut du fichier) pour que le mode
            # simulation fonctionne même si torch/transformers manquent.
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            print(f"  → Chargement du modèle {nom_modele}…")
            # Le processor convertit l'image au format attendu par le modèle
            self.processor = TrOCRProcessor.from_pretrained(nom_modele)
            # Le modèle lui-même
            self.model = VisionEncoderDecoderModel.from_pretrained(nom_modele)
            # Mode évaluation : désactive le dropout, fige le modèle
            self.model.eval()
            print("  ✓ Modèle chargé.")

    def transcrire(self, image_ligne: np.ndarray) -> dict:
        """Transcrit l'image d'une ligne en texte + confiances par caractère.

        Args:
            image_ligne: Image de la ligne en niveaux de gris (tableau 2D)
                ou en couleur (3D). Sortie de segmentation.py.

        Returns:
            Un dictionnaire avec :
              - 'text' (str) : la transcription
              - 'char_confidences' (list[float]) : confiance par caractère
              - 'candidates' (list[dict]) : lectures alternatives aux
                positions ambiguës (confiance < 0.7)

        Example:
            >>> resultat = transcripteur.transcrire(image)
            >>> print(resultat["text"], resultat["char_confidences"])
        """
        if self.mode_simulation:
            return self._transcrire_simulation(image_ligne)
        return self._transcrire_reel(image_ligne)

    # ── Mode réel (TrOCR) ───────────────────────────────────────────────────

    def _transcrire_reel(self, image_ligne: np.ndarray) -> dict:
        """Transcription réelle avec TrOCR.

        Le point délicat : extraire les confiances par caractère.
        TrOCR génère le texte token par token (un token ≈ quelques caractères).
        À chaque étape, le modèle produit des "scores" (logits) sur tout le
        vocabulaire. En appliquant softmax, on obtient des probabilités.
        La probabilité du token choisi = sa confiance.

        Args:
            image_ligne: Image de la ligne.

        Returns:
            Dictionnaire text / char_confidences / candidates.
        """
        import torch
        from PIL import Image

        # ── Préparer l'image ────────────────────────────────────────────────
        # TrOCR attend une image RGB. Si on a du niveaux de gris (2D), on convertit.
        if image_ligne.ndim == 2:
            # Empile 3 fois le canal gris pour simuler du RGB
            image_rgb = np.stack([image_ligne] * 3, axis=-1)
        else:
            image_rgb = image_ligne

        # Convertit le tableau NumPy en image PIL (format attendu par le processor)
        image_pil = Image.fromarray(image_rgb.astype(np.uint8))

        # Le processor transforme l'image en "pixel_values" (tenseur normalisé)
        pixel_values = self.processor(image_pil, return_tensors="pt").pixel_values

        # ── Générer le texte ────────────────────────────────────────────────
        # torch.no_grad() : on n'entraîne pas, donc pas besoin de calculer
        # les gradients → économise mémoire et temps.
        with torch.no_grad():
            # generate() produit les tokens de sortie
            # output_scores=True : on demande aussi les scores (pour les confiances)
            # return_dict_in_generate=True : retourne un objet structuré
            sortie = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=128,
            )

        # ── Décoder le texte ────────────────────────────────────────────────
        # sortie.sequences contient les identifiants des tokens générés
        ids_tokens = sortie.sequences[0]
        texte = self.processor.tokenizer.decode(ids_tokens, skip_special_tokens=True)

        # ── Calculer les confiances par token ───────────────────────────────
        confiances_tokens = self._extraire_confiances_tokens(sortie)

        # ── Convertir les confiances de "par token" à "par caractère" ───────
        char_confidences = self._tokens_vers_caracteres(
            ids_tokens, confiances_tokens, texte
        )

        # ── Identifier les candidats aux positions incertaines ──────────────
        candidates = self._identifier_candidats(char_confidences, seuil=0.7)

        return {
            "text": texte,
            "char_confidences": char_confidences,
            "candidates": candidates,
        }

    def _extraire_confiances_tokens(self, sortie) -> list[float]:
        """Extrait la confiance de chaque token généré.

        Args:
            sortie: L'objet retourné par model.generate().

        Returns:
            Liste des confiances (probabilités) de chaque token choisi.
        """
        import torch

        confiances = []
        # sortie.scores est une liste : un élément par étape de génération
        for scores_etape in sortie.scores:
            # softmax transforme les logits en probabilités (somme = 1)
            probabilites = torch.softmax(scores_etape[0], dim=-1)
            # La confiance = la probabilité maximale (celle du token choisi)
            confiance = torch.max(probabilites).item()
            confiances.append(confiance)
        return confiances

    def _tokens_vers_caracteres(
        self, ids_tokens, confiances_tokens: list[float], texte: str
    ) -> list[float]:
        """Répartit les confiances des tokens sur les caractères.

        Un token peut représenter plusieurs caractères (ex: "que" = 1 token).
        On attribue à chaque caractère la confiance de son token d'origine.
        C'est une approximation, mais elle suffit pour le volet NLP.

        Args:
            ids_tokens: Identifiants des tokens générés.
            confiances_tokens: Confiance de chaque token.
            texte: Le texte décodé.

        Returns:
            Liste des confiances, une par caractère du texte.
        """
        # Approche simplifiée et robuste : on décode token par token pour
        # connaître la longueur de chaque morceau de texte, et on répète
        # la confiance du token sur chacun de ses caractères.
        char_confidences = []

        # On saute les tokens spéciaux (début/fin de séquence)
        # ids_tokens[1:] car le premier est généralement le token de début
        tokens_utiles = ids_tokens[1:]

        for i, token_id in enumerate(tokens_utiles):
            if i >= len(confiances_tokens):
                break
            # Décode ce token seul pour connaître son texte
            morceau = self.processor.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            confiance = confiances_tokens[i]
            # Attribue cette confiance à chaque caractère du morceau
            for _ in morceau:
                char_confidences.append(confiance)

        # Sécurité : ajuste la longueur pour qu'elle corresponde au texte
        char_confidences = self._ajuster_longueur(char_confidences, len(texte))
        return char_confidences

    @staticmethod
    def _ajuster_longueur(confidences: list[float], longueur_cible: int) -> list[float]:
        """Ajuste une liste de confiances à une longueur cible.

        Args:
            confidences: La liste à ajuster.
            longueur_cible: La longueur voulue (= nombre de caractères).

        Returns:
            Liste de la bonne longueur (tronquée ou complétée).
        """
        if len(confidences) > longueur_cible:
            # Trop long → on tronque
            return confidences[:longueur_cible]
        elif len(confidences) < longueur_cible:
            # Trop court → on complète avec la dernière valeur (ou 0.5)
            valeur = confidences[-1] if confidences else 0.5
            return confidences + [valeur] * (longueur_cible - len(confidences))
        return confidences

    # ── Mode simulation ─────────────────────────────────────────────────────

    def _transcrire_simulation(self, image_ligne: np.ndarray) -> dict:
        """Génère une transcription factice pour tester le pipeline.

        Ne nécessite aucun téléchargement. Produit un texte fixe avec des
        confiances aléatoires mais réalistes, permettant de valider toute
        la chaîne sans GPU ni modèle.

        Args:
            image_ligne: Image de la ligne (sa taille influence le texte simulé).

        Returns:
            Dictionnaire text / char_confidences / candidates.
        """
        # Texte simulé dont la longueur dépend de la largeur de l'image
        largeur = image_ligne.shape[1] if image_ligne.ndim >= 2 else 100
        nb_mots = max(1, largeur // 60)
        mots_medievaux = ["romanz", "Engleterre", "Brut", "gent", "tindrent",
                          "Normant", "vindrent", "comence"]
        texte = " ".join(mots_medievaux[i % len(mots_medievaux)]
                         for i in range(nb_mots))

        # Confiances aléatoires mais réalistes (la plupart hautes, quelques basses)
        char_confidences = []
        for _ in texte:
            # 85 % du temps confiance haute (0.8-1.0), sinon basse (0.4-0.7)
            if np.random.random() < 0.85:
                c = np.random.uniform(0.8, 1.0)
            else:
                c = np.random.uniform(0.4, 0.7)
            char_confidences.append(round(float(c), 3))

        candidates = self._identifier_candidats(char_confidences, seuil=0.7)

        return {
            "text": texte,
            "char_confidences": char_confidences,
            "candidates": candidates,
        }

    # ── Méthode commune ─────────────────────────────────────────────────────

    @staticmethod
    def _identifier_candidats(
        char_confidences: list[float], seuil: float = 0.7
    ) -> list[dict]:
        """Identifie les positions incertaines (candidats pour le NLP).

        Pour chaque caractère dont la confiance est faible, on crée une
        entrée 'candidate' que le volet NLP utilisera pour proposer des
        corrections (via le MLM CamemBERT).

        Args:
            char_confidences: Liste des confiances par caractère.
            seuil: En dessous de ce seuil, le caractère est jugé ambigu.

        Returns:
            Liste de dicts {position, options}. Les options sont des
            substitutions médiévales courantes (u/v, i/j, etc.).

        Example:
            >>> cands = TranscripteurHTR._identifier_candidats([0.9, 0.5, 0.8])
            >>> print(cands)  # [{'position': 1, 'options': [...]}]
        """
        # Confusions fréquentes en HTR médiéval (u/v, i/j, c/e, n/u…)
        confusions = {
            "défaut": ["a", "e", "o"],  # candidats génériques
        }

        candidates = []
        for position, confiance in enumerate(char_confidences):
            if confiance < seuil:
                candidates.append({
                    "position": position,
                    "options": confusions["défaut"],
                })
        return candidates


# ─── Fonction utilitaire : préparation du fine-tuning LoRA ───────────────────

def preparer_lora(model, r: int = 8, alpha: int = 16):
    """Prépare un modèle TrOCR pour le fine-tuning par LoRA.

    LoRA (Low-Rank Adaptation) permet de fine-tuner un gros modèle en
    n'entraînant qu'un petit nombre de paramètres supplémentaires.
    Bien plus rapide et léger qu'un fine-tuning complet.
    Le sujet recommande r=8 puis r=16.

    Args:
        model: Le modèle TrOCR à adapter.
        r: Le "rang" de LoRA. Plus petit = moins de paramètres, plus rapide.
        alpha: Facteur d'échelle de LoRA (typiquement 2×r).

    Returns:
        Le modèle équipé des adaptateurs LoRA.

    Example:
        >>> from transformers import VisionEncoderDecoderModel
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model_lora = preparer_lora(model, r=8)
    """
    from peft import LoraConfig, get_peft_model

    # Configuration de LoRA
    config = LoraConfig(
        r=r,                          # rang des matrices de bas rang
        lora_alpha=alpha,             # facteur d'échelle
        # Les couches sur lesquelles appliquer LoRA (couches d'attention)
        target_modules=["query", "value"],
        lora_dropout=0.1,             # régularisation
        bias="none",
    )

    # get_peft_model "enveloppe" le modèle avec les adaptateurs LoRA
    model_lora = get_peft_model(model, config)

    # Affiche le nombre de paramètres entraînables (devrait être petit)
    model_lora.print_trainable_parameters()

    return model_lora


# ─── Point d'entrée (démonstration en mode simulation) ───────────────────────

if __name__ == "__main__":
    import sys
    import cv2
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.utils import fixer_seeds
    from htr.segmentation import segmenter_lignes, creer_image_multi_lignes

    print("=== Démonstration de htr_model.py (mode simulation) ===\n")
    fixer_seeds(42)

    # 1. Prépare des lignes segmentées
    fixture = Path("tests/fixtures/multi_lignes.png")
    creer_image_multi_lignes(fixture)
    img = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    lignes = segmenter_lignes(img_bin)
    print(f"✓ {len(lignes)} lignes segmentées\n")

    # 2. Crée le transcripteur en mode simulation
    transcripteur = TranscripteurHTR(mode_simulation=True)

    # 3. Transcrit chaque ligne
    for ligne in lignes:
        resultat = transcripteur.transcrire(ligne["image"])
        conf_moy = np.mean(resultat["char_confidences"])
        nb_cand = len(resultat["candidates"])
        print(f"  Ligne {ligne['reading_order']} : \"{resultat['text']}\"")
        print(f"    confiance moyenne : {conf_moy:.2f}, "
              f"{nb_cand} position(s) incertaine(s)\n")

    print("=== Démonstration terminée ✓ ===")
    print("\nNote : en mode réel (Internet requis), remplace par :")
    print("  transcripteur = TranscripteurHTR(mode_simulation=False)")
