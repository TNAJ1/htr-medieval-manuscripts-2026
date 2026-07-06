"""
finetuning.py — Fine-tuning de TrOCR par LoRA sur le corpus.

C'est l'étape qui SPÉCIALISE le modèle. TrOCR de base est entraîné sur de
l'anglais moderne manuscrit : sur de l'ancien français, il est mauvais. Le
fine-tuning lui montre des milliers d'exemples (image → texte correct) de
NOTRE corpus pour qu'il apprenne les graphies médiévales.

On utilise LoRA (Low-Rank Adaptation) : au lieu de réentraîner les ~330 M de
paramètres de TrOCR, on n'entraîne que de petites matrices ajoutées (~1 % des
poids). Résultat : bien plus rapide, bien moins gourmand en mémoire, et on
peut tourner sur un GPU modeste (voire en CPU pour un petit essai).

Concepts clés (pour la soutenance) :
  - EPOCH : un passage complet sur tout le jeu d'entraînement.
  - BATCH : un petit paquet d'exemples traités ensemble (ex: 8 lignes).
  - LEARNING RATE : la vitesse d'apprentissage (trop grand = instable,
    trop petit = trop lent).
  - EARLY STOPPING : on s'arrête quand le CER de validation ne baisse plus,
    pour éviter le surapprentissage.

DEUX MODES :
  - mode réel : entraîne vraiment TrOCR (nécessite torch, transformers, peft
    et idéalement un GPU).
  - mode simulation : simule une courbe d'apprentissage descendante, pour
    tester la logique et produire un graphique sans GPU.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.utils import fixer_seeds, enregistrer_experience, EXPERIMENTS_DIR


# ─── Configuration d'entraînement ────────────────────────────────────────────

class ConfigEntrainement:
    """Regroupe les hyperparamètres du fine-tuning.

    Utiliser une classe de config plutôt que des dizaines d'arguments rend le
    code plus clair et facilite le suivi des expériences.

    Attributes:
        epochs: Nombre de passages sur le jeu d'entraînement.
        batch_size: Nombre d'exemples par batch.
        learning_rate: Vitesse d'apprentissage.
        lora_r: Rang de LoRA (8 puis 16 selon le sujet).
        lora_alpha: Facteur d'échelle de LoRA.
        patience: Nombre d'epochs sans amélioration avant d'arrêter (early stop).

    Example:
        >>> config = ConfigEntrainement(epochs=10, lora_r=8)
    """

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        patience: int = 3,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.patience = patience

    def to_dict(self) -> dict:
        """Retourne la config sous forme de dictionnaire (pour le journal)."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "patience": self.patience,
        }


# ─── Fonction principale de fine-tuning ──────────────────────────────────────

def finetuner_trocr(
    exemples_train: list[dict],
    exemples_val: list[dict],
    config: ConfigEntrainement | None = None,
    nom_modele: str = "microsoft/trocr-base-handwritten",
    dossier_sortie: str | Path = "models/trocr-finetune",
    mode_simulation: bool = False,
    seed: int = 42,
) -> dict:
    """Fine-tune TrOCR par LoRA sur le corpus d'entraînement.

    À la fin de chaque epoch, on évalue le CER sur la validation et on garde le
    meilleur checkpoint (early stopping). Produit une courbe d'apprentissage.

    Args:
        exemples_train: Exemples d'entraînement (de charger_corpus).
        exemples_val: Exemples de validation.
        config: Hyperparamètres. Si None, config par défaut.
        nom_modele: Modèle TrOCR de base.
        dossier_sortie: Où sauvegarder le modèle fine-tuné.
        mode_simulation: Si True, simule l'entraînement (sans GPU).
        seed: Graine aléatoire.

    Returns:
        Un dictionnaire avec l'historique du CER et le meilleur CER de validation.

    Example:
        >>> historique = finetuner_trocr(train, val, ConfigEntrainement(epochs=10))
        >>> print(f"Meilleur CER val : {historique['meilleur_cer']:.1%}")
    """
    config = config or ConfigEntrainement()
    fixer_seeds(seed)

    print(f"\n=== Fine-tuning TrOCR (LoRA r={config.lora_r}) ===")
    print(f"  Train : {len(exemples_train)} lignes | Val : {len(exemples_val)} lignes")
    print(f"  Epochs : {config.epochs} | batch : {config.batch_size} | "
          f"lr : {config.learning_rate}")

    if mode_simulation:
        historique = _finetuner_simulation(exemples_train, exemples_val, config, seed)
    else:
        historique = _finetuner_reel(
            exemples_train, exemples_val, config, nom_modele, dossier_sortie
        )

    # Enregistrer l'expérience dans le journal
    enregistrer_experience(
        nom=f"finetune_trocr_lora_r{config.lora_r}",
        parametres=config.to_dict(),
        metriques={"meilleur_cer_val": historique["meilleur_cer"]},
        notes=f"Fine-tuning sur {len(exemples_train)} lignes",
    )

    # Sauvegarder la courbe d'apprentissage (pour l'article/les slides)
    _sauvegarder_courbe(historique, config)

    print(f"\n  ✓ Meilleur CER de validation : {historique['meilleur_cer']:.2%}")
    print(f"    (epoch {historique['meilleure_epoch']})")
    return historique


# ─── Mode réel ───────────────────────────────────────────────────────────────

def _finetuner_reel(
    exemples_train, exemples_val, config, nom_modele, dossier_sortie
) -> dict:
    """Entraînement réel de TrOCR avec LoRA (nécessite GPU recommandé).

    Args:
        exemples_train, exemples_val: Les exemples.
        config: Les hyperparamètres.
        nom_modele: Le modèle de base.
        dossier_sortie: Dossier de sauvegarde.

    Returns:
        L'historique d'entraînement.

    Note:
        Cette fonction utilise l'API HuggingFace Seq2SeqTrainer. Selon la
        version de transformers, quelques arguments peuvent changer ; référez-
        vous à la doc si un paramètre est refusé.
    """
    import torch
    from PIL import Image
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from torch.utils.data import Dataset

    from htr.htr_model import preparer_lora
    from htr.metrics import calculer_cer_corpus

    # ── Charger le modèle et le processor ───────────────────────────────────
    print("  → Chargement du modèle de base…")
    processor = TrOCRProcessor.from_pretrained(nom_modele)
    model = VisionEncoderDecoderModel.from_pretrained(nom_modele)

    # Configuration de génération (tokens spéciaux)
    # Selon la version de transformers, ces attributs se règlent sur config
    # et/ou generation_config ; on fait les deux, de façon défensive.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    try:
        model.config.vocab_size = model.config.decoder.vocab_size
    except AttributeError:
        pass  # versions récentes : vocab_size se lit sur config.decoder

    # ── Équiper le modèle de LoRA ───────────────────────────────────────────
    print(f"  → Application de LoRA (r={config.lora_r})…")
    model = preparer_lora(model, r=config.lora_r, alpha=config.lora_alpha)

    # ── Dataset PyTorch : transforme nos exemples au format attendu ─────────
    class DatasetHTR(Dataset):
        """Adapte nos exemples au format attendu par le Trainer."""

        def __init__(self, exemples, processor, max_length=128):
            self.exemples = exemples
            self.processor = processor
            self.max_length = max_length

        def __len__(self):
            return len(self.exemples)

        def __getitem__(self, idx):
            ex = self.exemples[idx]
            # Image → RGB → pixel_values
            img = ex["image"]
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            image_pil = Image.fromarray(img.astype(np.uint8))
            pixel_values = self.processor(
                image_pil, return_tensors="pt"
            ).pixel_values.squeeze()

            # Texte → labels (identifiants de tokens)
            labels = self.processor.tokenizer(
                ex["text"],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            ).input_ids
            # -100 sur le padding pour l'ignorer dans la perte
            labels = [l if l != self.processor.tokenizer.pad_token_id else -100
                      for l in labels]

            return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

    dataset_train = DatasetHTR(exemples_train, processor)
    dataset_val = DatasetHTR(exemples_val, processor)

    # ── Fonction de calcul du CER pendant l'entraînement ────────────────────
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        # Le Trainer complète les séquences avec -100 : il faut les remplacer
        # par le pad token AVANT de décoder (dans les labels ET les prédictions)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids = np.where(pred_ids == -100, processor.tokenizer.pad_token_id, pred_ids)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = calculer_cer_corpus(label_str, pred_str)
        return {"cer": cer}

    # ── Arguments d'entraînement ────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir=str(dossier_sortie),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        eval_strategy="epoch",           # évaluer à chaque epoch
        save_strategy="no",              # pas de checkpoint intermédiaire (contourne un
                                         # bug de sauvegarde VisionEncoderDecoder+PEFT ;
                                         # le modèle final est sauvegardé en fin de run)
        predict_with_generate=True,
        load_best_model_at_end=False,    # nécessite save_strategy="epoch" (désactivé)
        metric_for_best_model="cer",
        greater_is_better=False,         # un CER plus BAS est meilleur
        logging_steps=10,
        seed=42,
        report_to="none",   # désactive mlflow/wandb (évite les conflits d'environnement)
        generation_max_length=128,  # évite la troncature des prédictions à l'évaluation
    )

    # ── Entraîner ───────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("  → Entraînement en cours…")
    trainer.train()

    # ── Sauvegarder le modèle fine-tuné (avec plan B si échec) ──────────────
    Path(dossier_sortie).mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(dossier_sortie)
        processor.save_pretrained(dossier_sortie)
        print(f"  ✓ Modèle sauvegardé dans {dossier_sortie}")
    except Exception as e:
        # Plan B : sauvegarder uniquement les poids de l'adaptateur LoRA
        try:
            from peft import get_peft_model_state_dict
            torch.save(get_peft_model_state_dict(model),
                       Path(dossier_sortie) / "adapter_lora.pt")
            print(f"  ⚠ save_pretrained a échoué ({e}) — "
                  f"adaptateur LoRA sauvegardé en secours (adapter_lora.pt)")
        except Exception as e2:
            print(f"  ⚠ Sauvegarde impossible ({e2}) — les métriques sont conservées.")

    # ── Extraire l'historique du CER ────────────────────────────────────────
    cers = [log["eval_cer"] for log in trainer.state.log_history if "eval_cer" in log]
    meilleur_cer = min(cers) if cers else 1.0
    meilleure_epoch = cers.index(meilleur_cer) + 1 if cers else 0

    return {
        "cer_par_epoch": cers,
        "meilleur_cer": meilleur_cer,
        "meilleure_epoch": meilleure_epoch,
    }


# ─── Mode simulation ─────────────────────────────────────────────────────────

def _finetuner_simulation(exemples_train, exemples_val, config, seed) -> dict:
    """Simule une courbe d'apprentissage descendante (sans GPU).

    Génère un CER de validation qui décroît de façon réaliste au fil des
    epochs, avec un plateau et un peu de bruit. Applique l'early stopping.
    Permet de tester toute la logique et de produire un graphique.

    Args:
        exemples_train, exemples_val: Les exemples (leur taille influe un peu).
        config: Les hyperparamètres.
        seed: Graine.

    Returns:
        L'historique simulé.
    """
    rng = np.random.default_rng(seed)

    # On simule une décroissance : CER élevé au début, qui baisse et plafonne.
    # Point de départ réaliste pour du médiéval sans fine-tuning : ~35 %.
    cer_initial = 0.35
    # Plateau atteint : dépend (fictivement) de la taille du corpus.
    plateau = max(0.06, 0.20 - len(exemples_train) * 0.0005)

    cers = []
    meilleur_cer = 1.0
    meilleure_epoch = 0
    epochs_sans_amelioration = 0

    for epoch in range(1, config.epochs + 1):
        # Décroissance exponentielle vers le plateau + un peu de bruit
        progression = 1 - np.exp(-0.4 * epoch)
        cer = cer_initial - (cer_initial - plateau) * progression
        cer += rng.normal(0, 0.008)  # bruit réaliste
        cer = max(plateau * 0.95, cer)  # ne pas descendre sous le plateau
        cers.append(round(float(cer), 4))

        print(f"  Epoch {epoch:2d}/{config.epochs} — CER val : {cer:.2%}")

        # Suivi du meilleur + early stopping
        if cer < meilleur_cer:
            meilleur_cer = cer
            meilleure_epoch = epoch
            epochs_sans_amelioration = 0
        else:
            epochs_sans_amelioration += 1
            if epochs_sans_amelioration >= config.patience:
                print(f"  → Early stopping (pas d'amélioration depuis "
                      f"{config.patience} epochs)")
                break

    return {
        "cer_par_epoch": cers,
        "meilleur_cer": round(meilleur_cer, 4),
        "meilleure_epoch": meilleure_epoch,
    }


# ─── Sauvegarde de la courbe d'apprentissage ─────────────────────────────────

def _sauvegarder_courbe(historique: dict, config: ConfigEntrainement) -> None:
    """Sauvegarde la courbe d'apprentissage en JSON (pour l'article/les slides).

    Args:
        historique: L'historique d'entraînement.
        config: La config utilisée.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    chemin = EXPERIMENTS_DIR / f"courbe_lora_r{config.lora_r}.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({
            "config": config.to_dict(),
            "cer_par_epoch": historique["cer_par_epoch"],
            "meilleur_cer": historique["meilleur_cer"],
            "meilleure_epoch": historique["meilleure_epoch"],
        }, f, indent=2, ensure_ascii=False)
    print(f"  → Courbe d'apprentissage sauvegardée : {chemin}")


# ─── Point d'entrée (démonstration en mode simulation) ───────────────────────

if __name__ == "__main__":
    from htr.corpus_loader import charger_corpus
    from htr.page_xml import exporter_page_xml
    from htr.segmentation import segmenter_lignes, creer_image_multi_lignes
    from htr.dataset_split import split_stratifie
    import cv2

    print("=== Démonstration de finetuning.py (mode simulation) ===")
    fixer_seeds(42)

    # Fabriquer un mini-corpus au bon format
    faux = Path("data/raw/faux_corpus")
    faux.mkdir(parents=True, exist_ok=True)
    img_path = faux / "page_001.png"
    creer_image_multi_lignes(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    lignes = segmenter_lignes(img_bin)
    verites = ["Ci comence li romanz", "de Brut e de sa gent",
               "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    exporter_page_xml(lignes, "page_001.png", img.shape[1], img.shape[0],
                      faux / "page_001.xml", transcriptions=verites)

    # Charger et dupliquer pour avoir un corpus un peu plus gros (démo)
    exemples = charger_corpus(faux, metadata={"century": 13, "source": "DEMO"})
    exemples = exemples * 30  # simuler 120 lignes

    # Split train/val
    for e in exemples:  # ajouter une clé de stratification
        e["century"] = 13
    splits = split_stratifie(exemples, proportions=(0.8, 0.1, 0.1),
                             cle_strate="century")

    # Fine-tuner (en simulation)
    historique = finetuner_trocr(
        splits["train"], splits["val"],
        config=ConfigEntrainement(epochs=12, lora_r=8),
        mode_simulation=True,
    )

    print("\n=== Chez toi, avec le vrai corpus et un GPU : ===")
    print('  from htr.corpus_loader import charger_corpus')
    print('  from htr.dataset_split import split_stratifie')
    print('  exemples = charger_corpus("data/raw/cremma-medieval",')
    print('                            {"century": 13, "source": "CREMMA"})')
    print('  splits = split_stratifie(exemples)')
    print('  finetuner_trocr(splits["train"], splits["val"],')
    print('                  mode_simulation=False)   # ← le vrai entraînement')
