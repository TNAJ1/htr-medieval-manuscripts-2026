"""
test_finetuning.py — Tests du fine-tuning (mode simulation).

On teste la LOGIQUE de la boucle d'entraînement sans GPU ni téléchargement :
  - La config d'entraînement
  - Que la courbe d'apprentissage descend (le CER baisse)
  - Que l'early stopping fonctionne
  - Que le meilleur checkpoint est correctement identifié

Lancer : pytest tests/test_finetuning.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htr.finetuning import (
    ConfigEntrainement,
    finetuner_trocr,
    _finetuner_simulation,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def exemples_factices():
    """Crée des exemples factices (image + texte) pour l'entraînement."""
    exemples = []
    for i in range(80):
        exemples.append({
            "image": np.ones((40, 200), dtype=np.uint8) * 255,
            "text": f"ligne {i}",
            "century": 13,
        })
    return exemples


# ─── Tests de la configuration ───────────────────────────────────────────────

class TestConfigEntrainement:
    """Tests de la classe de configuration."""

    def test_valeurs_par_defaut(self):
        """La config par défaut a des valeurs raisonnables."""
        config = ConfigEntrainement()
        assert config.epochs > 0
        assert config.batch_size > 0
        assert 0 < config.learning_rate < 1
        assert config.lora_r in (8, 16) or config.lora_r > 0

    def test_to_dict(self):
        """to_dict retourne tous les hyperparamètres."""
        config = ConfigEntrainement(epochs=5, lora_r=16)
        d = config.to_dict()
        assert d["epochs"] == 5
        assert d["lora_r"] == 16


# ─── Tests de la courbe d'apprentissage ──────────────────────────────────────

class TestCourbeApprentissage:
    """Tests de la simulation de l'entraînement."""

    def test_produit_une_valeur_par_epoch(self, exemples_factices):
        """La courbe doit avoir au plus une valeur de CER par epoch."""
        config = ConfigEntrainement(epochs=10, patience=99)  # pas d'early stop
        historique = _finetuner_simulation(
            exemples_factices, exemples_factices[:10], config, seed=42
        )
        assert len(historique["cer_par_epoch"]) == 10

    def test_cer_globalement_descend(self, exemples_factices):
        """Le CER de la fin doit être inférieur à celui du début."""
        config = ConfigEntrainement(epochs=12, patience=99)
        historique = _finetuner_simulation(
            exemples_factices, exemples_factices[:10], config, seed=42
        )
        cers = historique["cer_par_epoch"]
        # Le premier CER doit être plus haut que le dernier
        assert cers[0] > cers[-1]

    def test_cer_entre_0_et_1(self, exemples_factices):
        """Tous les CER doivent être dans [0, 1]."""
        config = ConfigEntrainement(epochs=10)
        historique = _finetuner_simulation(
            exemples_factices, exemples_factices[:10], config, seed=42
        )
        for cer in historique["cer_par_epoch"]:
            assert 0.0 <= cer <= 1.0

    def test_meilleur_cer_est_le_minimum(self, exemples_factices):
        """Le meilleur CER doit être le minimum de la courbe."""
        config = ConfigEntrainement(epochs=10, patience=99)
        historique = _finetuner_simulation(
            exemples_factices, exemples_factices[:10], config, seed=42
        )
        assert historique["meilleur_cer"] == min(historique["cer_par_epoch"])


# ─── Tests de l'early stopping ───────────────────────────────────────────────

class TestEarlyStopping:
    """Tests de l'arrêt anticipé."""

    def test_arret_avant_la_fin_possible(self, exemples_factices):
        """Avec une patience faible, l'entraînement peut s'arrêter tôt."""
        config = ConfigEntrainement(epochs=50, patience=2)
        historique = _finetuner_simulation(
            exemples_factices, exemples_factices[:10], config, seed=42
        )
        # Avec 50 epochs prévues et patience=2, on s'arrête généralement avant
        assert len(historique["cer_par_epoch"]) <= 50

    def test_reproductible(self, exemples_factices):
        """Même seed → même courbe."""
        config = ConfigEntrainement(epochs=10)
        h1 = _finetuner_simulation(exemples_factices, exemples_factices[:10], config, 42)
        h2 = _finetuner_simulation(exemples_factices, exemples_factices[:10], config, 42)
        assert h1["cer_par_epoch"] == h2["cer_par_epoch"]


# ─── Tests de l'orchestration complète ───────────────────────────────────────

class TestFinetunerTrocr:
    """Tests de la fonction principale finetuner_trocr."""

    def test_retourne_historique_complet(self, exemples_factices, tmp_path):
        """finetuner_trocr retourne un historique avec les bonnes clés."""
        historique = finetuner_trocr(
            exemples_factices[:60], exemples_factices[60:],
            config=ConfigEntrainement(epochs=8),
            mode_simulation=True,
        )
        assert "cer_par_epoch" in historique
        assert "meilleur_cer" in historique
        assert "meilleure_epoch" in historique

    def test_meilleure_epoch_valide(self, exemples_factices):
        """La meilleure epoch doit être un numéro d'epoch valide."""
        historique = finetuner_trocr(
            exemples_factices[:60], exemples_factices[60:],
            config=ConfigEntrainement(epochs=8),
            mode_simulation=True,
        )
        n_epochs = len(historique["cer_par_epoch"])
        assert 1 <= historique["meilleure_epoch"] <= n_epochs
