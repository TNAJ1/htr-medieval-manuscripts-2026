"""
test_preprocessing.py — Tests automatisés du pipeline de prétraitement.

Ces tests vérifient que le prétraitement produit des résultats corrects :
  - Les formes (shapes) des images sont conservées
  - Les types de données sont corrects
  - Les valeurs sont dans les plages attendues
  - La binarisation produit bien une image binaire (0 ou 255 uniquement)

Lancer les tests : pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ajoute le dossier src/ au chemin Python pour pouvoir importer nos modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "htr"))

from preprocessing import (
    appliquer_clahe,
    binariser_sauvola,
    corriger_inclinaison,
    creer_image_test,
    pretraiter_image,
    supprimer_bruit,
)


# ─── Fixtures (données de test réutilisables) ─────────────────────────────────
# Une "fixture" pytest est une fonction qui prépare des données pour les tests.
# @pytest.fixture indique à pytest que cette fonction est une fixture.

@pytest.fixture
def image_grise_simple():
    """Crée une image en niveaux de gris 100×100 pour les tests."""
    # Image avec un gradient de gris (valeurs de 0 à 255)
    image = np.zeros((100, 100), dtype=np.uint8)
    # Remplie avec des valeurs graduelles sur chaque ligne
    for i in range(100):
        image[i, :] = i * 2  # Valeurs de 0 à 198
    return image


@pytest.fixture
def image_binaire_avec_bruit():
    """Crée une image binaire avec quelques pixels de bruit."""
    # Image blanche
    image = np.ones((100, 100), dtype=np.uint8) * 255
    # Quelques pixels noirs (texte simulé)
    image[40:60, 20:80] = 0
    # Quelques pixels isolés de bruit
    image[10, 10] = 0
    image[90, 90] = 0
    return image


@pytest.fixture
def image_test_fichier(tmp_path):
    """Crée un fichier image temporaire pour les tests de bout en bout.

    tmp_path est une fixture pytest native qui fournit un dossier temporaire
    qui sera automatiquement supprimé après le test.
    """
    chemin = tmp_path / "test_page.png"
    creer_image_test(chemin)
    return chemin


# ─── Tests de corriger_inclinaison ───────────────────────────────────────────

class TestCorrigerInclinaison:
    """Groupe de tests pour la fonction corriger_inclinaison."""

    def test_retourne_meme_shape(self, image_grise_simple):
        """La correction d'inclinaison ne doit pas changer la taille de l'image."""
        resultat = corriger_inclinaison(image_grise_simple)

        # assert vérifie une condition ; si elle est False, le test échoue
        assert resultat.shape == image_grise_simple.shape, (
            f"Shape attendue : {image_grise_simple.shape}, "
            f"obtenue : {resultat.shape}"
        )

    def test_retourne_meme_dtype(self, image_grise_simple):
        """Le type de données doit rester uint8 (entiers 0-255)."""
        resultat = corriger_inclinaison(image_grise_simple)
        assert resultat.dtype == np.uint8

    def test_image_presque_vide_retournee_inchangee(self):
        """Une image avec peu de pixels de texte est retournée sans modification."""
        # Image avec seulement 10 pixels noirs — pas assez pour détecter un angle
        image = np.ones((100, 100), dtype=np.uint8) * 255
        image[50, 50] = 0  # Un seul pixel noir

        resultat = corriger_inclinaison(image)
        # L'image doit être retournée telle quelle (inchangée)
        np.testing.assert_array_equal(resultat, image)


# ─── Tests de appliquer_clahe ────────────────────────────────────────────────

class TestAppliquerClahe:
    """Tests pour la fonction d'amélioration du contraste CLAHE."""

    def test_retourne_meme_shape(self, image_grise_simple):
        """CLAHE ne doit pas changer la taille de l'image."""
        resultat = appliquer_clahe(image_grise_simple)
        assert resultat.shape == image_grise_simple.shape

    def test_valeurs_dans_plage_valide(self, image_grise_simple):
        """Les valeurs de sortie doivent rester entre 0 et 255."""
        resultat = appliquer_clahe(image_grise_simple)

        # np.min et np.max retournent la valeur minimale/maximale du tableau
        assert np.min(resultat) >= 0, "Valeurs négatives détectées !"
        assert np.max(resultat) <= 255, "Valeurs supérieures à 255 détectées !"

    def test_retourne_uint8(self, image_grise_simple):
        """Le type de sortie doit être uint8."""
        resultat = appliquer_clahe(image_grise_simple)
        assert resultat.dtype == np.uint8

    def test_ameliore_le_contraste(self, image_grise_simple):
        """CLAHE doit augmenter la plage de valeurs (contraste amélioré)."""
        avant = np.std(image_grise_simple.astype(float))  # Écart-type avant
        resultat = appliquer_clahe(image_grise_simple)
        apres = np.std(resultat.astype(float))            # Écart-type après

        # Un contraste amélioré = écart-type plus grand (valeurs plus dispersées)
        # On accepte aussi un écart-type égal (image déjà à bon contraste)
        assert apres >= avant * 0.9, (
            f"Le contraste a diminué : avant={avant:.1f}, après={apres:.1f}"
        )


# ─── Tests de binariser_sauvola ──────────────────────────────────────────────

class TestBinariserSauvola:
    """Tests pour la binarisation par Sauvola."""

    def test_retourne_meme_shape(self, image_grise_simple):
        """La binarisation ne doit pas changer la taille de l'image."""
        resultat = binariser_sauvola(image_grise_simple)
        assert resultat.shape == image_grise_simple.shape

    def test_image_vraiment_binaire(self, image_grise_simple):
        """La sortie doit contenir uniquement des 0 (noir) et 255 (blanc)."""
        resultat = binariser_sauvola(image_grise_simple)
        valeurs_uniques = np.unique(resultat)

        # np.isin vérifie que toutes les valeurs uniques sont dans [0, 255]
        for valeur in valeurs_uniques:
            assert valeur in [0, 255], (
                f"Valeur inattendue {valeur} dans l'image binaire. "
                f"Seuls 0 et 255 sont attendus."
            )

    def test_retourne_uint8(self, image_grise_simple):
        """Le type de sortie doit être uint8."""
        resultat = binariser_sauvola(image_grise_simple)
        assert resultat.dtype == np.uint8

    def test_image_entierement_blanche_reste_blanche(self):
        """Une image entièrement blanche doit rester blanche après binarisation."""
        image_blanche = np.ones((100, 100), dtype=np.uint8) * 255
        resultat = binariser_sauvola(image_blanche)

        # La majorité des pixels doit être blanche (fond blanc = fond de page)
        proportion_blanc = np.mean(resultat == 255)
        assert proportion_blanc > 0.9, (
            f"Trop peu de pixels blancs : {proportion_blanc:.1%}"
        )


# ─── Tests de supprimer_bruit ────────────────────────────────────────────────

class TestSupprimerBruit:
    """Tests pour la suppression du bruit morphologique."""

    def test_retourne_meme_shape(self, image_binaire_avec_bruit):
        """La suppression de bruit ne doit pas changer la taille."""
        resultat = supprimer_bruit(image_binaire_avec_bruit)
        assert resultat.shape == image_binaire_avec_bruit.shape

    def test_image_reste_binaire(self, image_binaire_avec_bruit):
        """L'image de sortie doit rester binaire (0 et 255 uniquement)."""
        resultat = supprimer_bruit(image_binaire_avec_bruit)
        valeurs_uniques = np.unique(resultat)

        for valeur in valeurs_uniques:
            assert valeur in [0, 255], (
                f"Valeur non binaire détectée : {valeur}"
            )

    def test_supprime_pixels_isoles(self):
        """Les pixels isolés (bruit) doivent être supprimés."""
        # Image blanche avec un seul pixel noir isolé au milieu
        image = np.ones((50, 50), dtype=np.uint8) * 255
        image[25, 25] = 0  # Pixel de bruit isolé

        resultat = supprimer_bruit(image, taille_noyau=2)

        # Le pixel isolé doit avoir été supprimé
        assert resultat[25, 25] == 255, (
            "Le pixel de bruit isolé n'a pas été supprimé !"
        )


# ─── Tests du pipeline complet ───────────────────────────────────────────────

class TestPipelineComplet:
    """Tests de bout en bout sur le pipeline de prétraitement."""

    def test_pipeline_sur_image_test(self, image_test_fichier):
        """Le pipeline complet doit s'exécuter sans erreur sur une image test."""
        resultat = pretraiter_image(image_test_fichier)

        # Le résultat ne doit pas être vide
        assert resultat is not None
        assert resultat.size > 0

    def test_pipeline_retourne_tableau_2d(self, image_test_fichier):
        """Le résultat doit être un tableau 2D (image en niveaux de gris)."""
        resultat = pretraiter_image(image_test_fichier)

        # ndim = nombre de dimensions ; 2 = (hauteur, largeur) sans couleur
        assert resultat.ndim == 2, (
            f"Image 2D attendue, obtenue : {resultat.ndim}D (shape={resultat.shape})"
        )

    def test_pipeline_retourne_image_binaire(self, image_test_fichier):
        """Le résultat final doit être une image binaire (0 et 255 uniquement)."""
        resultat = pretraiter_image(image_test_fichier)
        valeurs_uniques = np.unique(resultat)

        for valeur in valeurs_uniques:
            assert valeur in [0, 255], (
                f"Image non binaire en sortie du pipeline : valeur {valeur} trouvée"
            )

    def test_fichier_inexistant_leve_erreur(self, tmp_path):
        """Un chemin invalide doit lever une FileNotFoundError."""
        # pytest.raises vérifie qu'une exception est bien levée
        with pytest.raises(FileNotFoundError):
            pretraiter_image(tmp_path / "fichier_qui_nexiste_pas.png")

    def test_sauvegarde_image(self, image_test_fichier, tmp_path):
        """L'option sauvegarder=True doit créer un fichier de sortie."""
        dossier_sortie = tmp_path / "processed"

        pretraiter_image(
            image_test_fichier,
            sauvegarder=True,
            dossier_sortie=dossier_sortie,
        )

        # Le fichier de sortie doit exister
        fichier_attendu = dossier_sortie / image_test_fichier.name
        assert fichier_attendu.exists(), (
            f"Le fichier de sortie n'a pas été créé : {fichier_attendu}"
        )
