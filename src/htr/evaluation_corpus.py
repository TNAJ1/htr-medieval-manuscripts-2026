"""
evaluation_corpus.py — Transcription d'un corpus et mesure du CER réel.

Ce module fait le lien entre :
  - le corpus chargé (corpus_loader.py) → images de lignes + vérité terrain
  - le transcripteur HTR (TrOCR ou Kraken) → prédictions
  - les métriques (metrics.py) → CER, WER, intervalle de confiance

C'est ICI qu'on obtient enfin des CHIFFRES RÉELS, en comparant ce que le
modèle prédit avec la vérité terrain du corpus.

Workflow :
  1. Charger le corpus (images + vérité terrain).
  2. Transcrire chaque ligne avec le modèle.
  3. Comparer prédiction vs vérité terrain → CER, WER.
  4. Produire un rapport (global + par siècle) et l'enregistrer dans le journal.

En prime : on peut comparer TrOCR et Kraken sur le MÊME corpus (test de McNemar).
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htr.corpus_loader import charger_corpus
from htr.transcripteur_factory import creer_transcripteur
from htr.metrics import (
    calculer_cer_corpus,
    calculer_wer_corpus,
    intervalle_confiance_bootstrap,
    rapport_metriques,
    comparer_mcnemar,
)
from shared.utils import enregistrer_experience


# ─── Transcription d'un corpus ───────────────────────────────────────────────

def transcrire_corpus(exemples: list[dict], transcripteur) -> tuple[list[str], list[str]]:
    """Transcrit toutes les lignes d'un corpus et retourne prédictions + références.

    Args:
        exemples: Liste d'exemples de charger_corpus (avec 'image' et 'text').
        transcripteur: Un transcripteur (TrOCR, Kraken ou fusion).

    Returns:
        Un tuple (references, predictions) : deux listes parallèles de textes.
        references = vérité terrain ; predictions = sortie du modèle.

    Example:
        >>> refs, preds = transcrire_corpus(exemples, transcripteur)
    """
    references = []
    predictions = []

    total = len(exemples)
    for i, exemple in enumerate(exemples):
        # La vérité terrain est déjà dans l'exemple
        references.append(exemple["text"])

        # On transcrit l'image de la ligne
        resultat = transcripteur.transcrire(exemple["image"])
        predictions.append(resultat["text"])

        # Affichage de la progression tous les 50 exemples
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  → {i + 1}/{total} lignes transcrites")

    return references, predictions


# ─── Évaluation complète ─────────────────────────────────────────────────────

def evaluer_modele_sur_corpus(
    dossier_corpus: str | Path,
    modele: str = "trocr",
    metadata: dict | None = None,
    mode_simulation: bool = False,
    limite: int | None = None,
) -> dict:
    """Évalue un modèle HTR sur un corpus à vérité terrain et mesure le CER.

    C'est LA fonction à lancer pour obtenir un vrai CER.

    Args:
        dossier_corpus: Chemin du corpus téléchargé (ex: 'data/raw/cremma-medieval').
        modele: "trocr", "kraken" ou "fusion".
        metadata: Métadonnées du corpus (century, source...).
        mode_simulation: Si True, HTR simulé (pour tester sans télécharger).
        limite: Nombre max de lignes à évaluer (utile pour un test rapide).

    Returns:
        Un rapport de métriques (CER, WER, IC bootstrap, seuils atteints).

    Example:
        >>> rapport = evaluer_modele_sur_corpus(
        ...     "data/raw/cremma-medieval",
        ...     modele="trocr",
        ...     metadata={"century": 13, "source": "CREMMA"},
        ... )
        >>> print(f"CER réel : {rapport['CER']:.1%}")
    """
    metadata = metadata or {}

    print(f"\n=== Évaluation du modèle '{modele}' sur {Path(dossier_corpus).name} ===")

    # ── Étape 1 : Charger le corpus ─────────────────────────────────────────
    print("\n[1/3] Chargement du corpus…")
    exemples = charger_corpus(dossier_corpus, metadata=metadata, limite=limite)

    if not exemples:
        raise ValueError(
            f"Aucun exemple chargé depuis {dossier_corpus}. "
            f"Vérifie que le corpus contient des paires (image, XML)."
        )

    # ── Étape 2 : Transcrire ────────────────────────────────────────────────
    print(f"\n[2/3] Transcription avec {modele}…")
    transcripteur = creer_transcripteur(modele, mode_simulation=mode_simulation)
    references, predictions = transcrire_corpus(exemples, transcripteur)

    # ── Étape 3 : Mesurer ───────────────────────────────────────────────────
    print("\n[3/3] Calcul des métriques…")
    rapport = rapport_metriques(references, predictions)

    # On ajoute des infos de contexte
    rapport["modele"] = modele
    rapport["corpus"] = Path(dossier_corpus).name

    # ── Enregistrer dans le journal d'expériences ───────────────────────────
    enregistrer_experience(
        nom=f"eval_{modele}_{Path(dossier_corpus).name}",
        parametres={"modele": modele, "mode_simulation": mode_simulation,
                    "n_lignes": len(exemples)},
        metriques={"CER": rapport["CER"], "WER": rapport["WER"]},
        notes=f"Évaluation de {modele} sur {rapport['corpus']}",
    )

    # ── Affichage du rapport ────────────────────────────────────────────────
    _afficher_rapport(rapport)
    return rapport


def _afficher_rapport(rapport: dict) -> None:
    """Affiche un rapport de métriques lisible.

    Args:
        rapport: Le dictionnaire de métriques.
    """
    print("\n" + "─" * 45)
    print(f"  RAPPORT — {rapport.get('modele', '?')} sur {rapport.get('corpus', '?')}")
    print("─" * 45)
    print(f"  Lignes évaluées   : {rapport['n_lignes']}")
    print(f"  CER global        : {rapport['CER']:.2%}")
    print(f"  WER global        : {rapport['WER']:.2%}")
    ic = rapport["CER_IC95"]
    print(f"  CER IC 95 %       : [{ic[0]:.2%}, {ic[1]:.2%}]")
    print(f"  Seuil validation (CER < 15 %) : "
          f"{'✓ ATTEINT' if rapport['seuil_validation_CER'] else '✗ non atteint'}")
    print(f"  Seuil excellence (CER < 8 %)  : "
          f"{'✓ ATTEINT' if rapport['seuil_excellence_CER'] else '✗ non atteint'}")
    print("─" * 45)


# ─── Comparaison TrOCR vs Kraken sur le même corpus ──────────────────────────

def comparer_trocr_kraken(
    dossier_corpus: str | Path,
    metadata: dict | None = None,
    mode_simulation: bool = False,
    limite: int | None = None,
) -> dict:
    """Compare TrOCR et Kraken sur le MÊME corpus (bonus McNemar).

    Transcrit une seule fois le corpus avec chaque modèle, puis compare les
    CER et applique le test de McNemar pour la significativité statistique.

    Args:
        dossier_corpus: Chemin du corpus.
        metadata: Métadonnées du corpus.
        mode_simulation: Si True, HTR simulé.
        limite: Nombre max de lignes.

    Returns:
        Un dictionnaire comparatif (CER de chaque modèle, McNemar).

    Example:
        >>> comp = comparer_trocr_kraken("data/raw/cremma-medieval",
        ...                               {"century": 13, "source": "CREMMA"})
    """
    metadata = metadata or {}
    print(f"\n=== Comparaison TrOCR vs Kraken sur {Path(dossier_corpus).name} ===")

    # Charger le corpus une seule fois
    exemples = charger_corpus(dossier_corpus, metadata=metadata, limite=limite)
    references = [e["text"] for e in exemples]

    # Transcrire avec TrOCR
    print("\n[TrOCR] Transcription…")
    trocr = creer_transcripteur("trocr", mode_simulation=mode_simulation)
    _, preds_trocr = transcrire_corpus(exemples, trocr)

    # Transcrire avec Kraken
    print("\n[Kraken] Transcription…")
    kraken = creer_transcripteur("kraken", mode_simulation=mode_simulation)
    _, preds_kraken = transcrire_corpus(exemples, kraken)

    # Métriques de chaque modèle
    cer_trocr = calculer_cer_corpus(references, preds_trocr)
    cer_kraken = calculer_cer_corpus(references, preds_kraken)

    # Test de McNemar
    mcnemar = comparer_mcnemar(references, preds_trocr, preds_kraken)

    meilleur = "TrOCR" if cer_trocr < cer_kraken else (
        "Kraken" if cer_kraken < cer_trocr else "égalité")

    resultat = {
        "cer_trocr": round(cer_trocr, 4),
        "cer_kraken": round(cer_kraken, 4),
        "meilleur_modele": meilleur,
        "mcnemar_p_value": round(mcnemar["p_value"], 4),
        "difference_significative": mcnemar["significatif"],
        "n_lignes": len(exemples),
    }

    # Affichage
    print("\n" + "─" * 45)
    print("  COMPARAISON TrOCR vs KRAKEN")
    print("─" * 45)
    print(f"  CER TrOCR   : {cer_trocr:.2%}")
    print(f"  CER Kraken  : {cer_kraken:.2%}")
    print(f"  Meilleur    : {meilleur}")
    print(f"  p-value McNemar : {mcnemar['p_value']:.4f} "
          f"({'significatif' if mcnemar['significatif'] else 'non significatif'})")
    print("─" * 45)

    return resultat


# ─── Point d'entrée (démonstration en mode simulation) ───────────────────────

if __name__ == "__main__":
    from shared.utils import fixer_seeds
    from htr.page_xml import exporter_page_xml
    from htr.segmentation import segmenter_lignes, creer_image_multi_lignes
    import cv2

    print("=== Démonstration de evaluation_corpus.py (mode simulation) ===")
    fixer_seeds(42)

    # Fabriquer un mini-corpus au bon format (comme CREMMA)
    faux_corpus = Path("data/raw/faux_corpus")
    faux_corpus.mkdir(parents=True, exist_ok=True)
    img_path = faux_corpus / "page_001.png"
    creer_image_multi_lignes(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    lignes = segmenter_lignes(img_bin)
    verites = ["Ci comence li romanz", "de Brut e de sa gent",
               "qui Engleterre tindrent", "ainz que Normant i vindrent"]
    exporter_page_xml(lignes, "page_001.png", img.shape[1], img.shape[0],
                      faux_corpus / "page_001.xml", transcriptions=verites)

    # Évaluer TrOCR (en simulation)
    rapport = evaluer_modele_sur_corpus(
        faux_corpus,
        modele="trocr",
        metadata={"century": 12, "source": "FAUX-CREMMA"},
        mode_simulation=True,
    )

    print("\n\n=== Chez toi, avec le vrai CREMMA et le vrai TrOCR : ===")
    print('  rapport = evaluer_modele_sur_corpus(')
    print('      "data/raw/cremma-medieval",')
    print('      modele="trocr",')
    print('      metadata={"century": 13, "source": "CREMMA"},')
    print('      mode_simulation=False,   # ← le vrai modèle')
    print('  )')
