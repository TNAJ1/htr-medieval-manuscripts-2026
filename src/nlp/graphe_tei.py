"""
graphe_tei.py — Extraction de relations, graphe et export TEI-XML.

Cinquième et dernière brique du volet NLP. Une fois qu'on a les entités (NER),
on les met en relation et on les publie :

  1. EXTRACTION DE RELATIONS : par règles lexico-syntaxiques simples.
     Ex: motif "PER ... verbe ... LOC" → relation (personne, verbe, lieu).
     Les consignes déconseillent le LLM complexe : des règles suffisent.

  2. GRAPHE (NetworkX) : on modélise les entités comme des noeuds et les
     relations comme des arêtes. Permet de visualiser "qui est lié à quoi".

  3. EXPORT TEI-XML : le standard d'encodage en humanités numériques.
     On balise les entités : <persName>, <placeName>, <date>.

DEUX MODES pour le graphe :
  - avec NetworkX installé : vrai objet graphe.
  - sans NetworkX : structure de graphe simple maison (mode dégradé),
    pour que le module tourne partout.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import re
import sys
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.ner import ReconnaisseurEntites


# ─── Verbes-relations médiévaux fréquents ────────────────────────────────────
# Ces verbes signalent une relation entre deux entités.
VERBES_RELATION = {
    "tint": "TIENT", "tindrent": "TIENT",
    "prist": "PREND", "prindrent": "PREND",
    "conquist": "CONQUIERT", "vint": "VIENT_A",
    "regna": "REGNE_SUR", "fu": "EST",
}


# ─── 1. Extraction de relations ──────────────────────────────────────────────

def extraire_relations(texte: str, ner: ReconnaisseurEntites | None = None) -> list[dict]:
    """Extrait les relations entre entités par règles lexico-syntaxiques.

    Principe : on repère les motifs "entité ... verbe-relation ... entité".
    Exemple : "Brut tint Engleterre" → (Brut, TIENT, Engleterre).

    Args:
        texte: Le texte à analyser.
        ner: Un reconnaisseur d'entités. Si None, en crée un (simulation).

    Returns:
        Liste de relations {source, relation, cible, type_source, type_cible}.

    Example:
        >>> relations = extraire_relations("Brut tint Engleterre")
        >>> print(relations[0])
        {'source': 'Brut', 'relation': 'TIENT', 'cible': 'Engleterre', ...}
    """
    if ner is None:
        ner = ReconnaisseurEntites(mode_simulation=True)

    mots = texte.split()
    labels = ner.etiqueter(texte)  # [(mot, label_BIO), ...]

    relations = []

    # On parcourt les mots à la recherche du motif : entité, verbe, entité
    for i, mot in enumerate(mots):
        mot_propre = mot.lower().strip(".,;:!?")

        # Ce mot est-il un verbe-relation ?
        if mot_propre in VERBES_RELATION:
            relation = VERBES_RELATION[mot_propre]

            # Chercher une entité AVANT (la source) et APRÈS (la cible)
            source = _entite_avant(labels, i)
            cible = _entite_apres(labels, i)

            if source and cible:
                relations.append({
                    "source": source["texte"],
                    "type_source": source["type"],
                    "relation": relation,
                    "cible": cible["texte"],
                    "type_cible": cible["type"],
                })

    return relations


def _entite_avant(labels: list[tuple[str, str]], position: int) -> dict | None:
    """Trouve l'entité la plus proche avant une position donnée.

    Args:
        labels: Liste de (mot, label_BIO).
        position: Index à partir duquel remonter.

    Returns:
        Un dict {texte, type} ou None.
    """
    # On remonte depuis la position - 1
    for j in range(position - 1, -1, -1):
        mot, label = labels[j]
        if label != "O":
            return {"texte": mot, "type": label[2:]}  # retire "B-"/"I-"
    return None


def _entite_apres(labels: list[tuple[str, str]], position: int) -> dict | None:
    """Trouve l'entité la plus proche après une position donnée.

    Args:
        labels: Liste de (mot, label_BIO).
        position: Index à partir duquel avancer.

    Returns:
        Un dict {texte, type} ou None.
    """
    for j in range(position + 1, len(labels)):
        mot, label = labels[j]
        if label != "O":
            return {"texte": mot, "type": label[2:]}
    return None


# ─── 2. Construction du graphe ───────────────────────────────────────────────

def construire_graphe(relations: list[dict]):
    """Construit un graphe des entités et relations avec NetworkX.

    Chaque entité est un noeud (avec son type), chaque relation une arête.

    Args:
        relations: Liste de relations (de extraire_relations).

    Returns:
        Un objet graphe NetworkX (DiGraph), ou un GrapheSimple si NetworkX
        n'est pas installé.

    Example:
        >>> g = construire_graphe(relations)
        >>> print(g.number_of_nodes(), "entités")
    """
    try:
        import networkx as nx
        graphe = nx.DiGraph()  # graphe orienté (les relations ont un sens)

        for rel in relations:
            # Ajoute les noeuds avec leur type comme attribut
            graphe.add_node(rel["source"], type=rel["type_source"])
            graphe.add_node(rel["cible"], type=rel["type_cible"])
            # Ajoute l'arête avec le label de relation
            graphe.add_edge(rel["source"], rel["cible"], relation=rel["relation"])

        return graphe

    except ImportError:
        # Mode dégradé sans NetworkX
        return _GrapheSimple(relations)


class _GrapheSimple:
    """Graphe minimal maison, si NetworkX n'est pas installé.

    Fournit les mêmes méthodes de base que NetworkX pour la compatibilité.
    """

    def __init__(self, relations: list[dict]):
        self.noeuds = {}   # {nom: type}
        self.aretes = []   # [(source, cible, relation)]
        for rel in relations:
            self.noeuds[rel["source"]] = rel["type_source"]
            self.noeuds[rel["cible"]] = rel["type_cible"]
            self.aretes.append((rel["source"], rel["cible"], rel["relation"]))

    def number_of_nodes(self) -> int:
        """Nombre de noeuds."""
        return len(self.noeuds)

    def number_of_edges(self) -> int:
        """Nombre d'arêtes."""
        return len(self.aretes)


# ─── 3. Export TEI-XML ───────────────────────────────────────────────────────

# Correspondance type d'entité → balise TEI
BALISES_TEI = {
    "PER": "persName",
    "LOC": "placeName",
    "DATE": "date",
    "ORG": "orgName",
    "TITLE": "roleName",
}


def exporter_tei(
    texte: str,
    chemin_sortie: str | Path,
    ner: ReconnaisseurEntites | None = None,
    titre: str = "Transcription HTR",
) -> Path:
    """Exporte un texte annoté au format TEI-XML.

    Balise chaque entité dans le texte avec la balise TEI correspondante
    (<persName>, <placeName>, <date>...). Produit un fichier TEI valide.

    Args:
        texte: Le texte à baliser.
        chemin_sortie: Chemin du fichier .xml à créer.
        ner: Un reconnaisseur d'entités. Si None, en crée un (simulation).
        titre: Le titre du document TEI.

    Returns:
        Le chemin du fichier créé.

    Example:
        >>> exporter_tei("le roi Charles de France", "sortie.xml")
    """
    if ner is None:
        ner = ReconnaisseurEntites(mode_simulation=True)

    labels = ner.etiqueter(texte)

    # ── Racine TEI ──────────────────────────────────────────────────────────
    tei = ET.Element("TEI", xmlns="http://www.tei-c.org/ns/1.0")

    # En-tête TEI (métadonnées obligatoires)
    header = ET.SubElement(tei, "teiHeader")
    file_desc = ET.SubElement(header, "fileDesc")
    title_stmt = ET.SubElement(file_desc, "titleStmt")
    title_elem = ET.SubElement(title_stmt, "title")
    title_elem.text = titre
    pub_stmt = ET.SubElement(file_desc, "publicationStmt")
    pub_p = ET.SubElement(pub_stmt, "p")
    pub_p.text = "Produit par le pipeline HTR-NLP du projet MD5."
    source_desc = ET.SubElement(file_desc, "sourceDesc")
    source_p = ET.SubElement(source_desc, "p")
    source_p.text = "Transcription automatique de manuscrit médiéval."

    # ── Corps du texte ──────────────────────────────────────────────────────
    text_elem = ET.SubElement(tei, "text")
    body = ET.SubElement(text_elem, "body")
    p = ET.SubElement(body, "p")

    # On reconstruit le texte en balisant les entités
    # (approche simple : mot par mot ; les entités multi-mots B-/I- sont
    #  fusionnées)
    _baliser_paragraphe(p, labels)

    # ── Sauvegarde avec indentation ─────────────────────────────────────────
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    xml_brut = ET.tostring(tei, encoding="utf-8")
    xml_joli = minidom.parseString(xml_brut).toprettyxml(indent="  ")
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.write(xml_joli)

    print(f"✓ TEI-XML exporté : {chemin_sortie}")
    return chemin_sortie


def _baliser_paragraphe(p_elem, labels: list[tuple[str, str]]) -> None:
    """Remplit un élément <p> en balisant les entités.

    Gère le texte courant (hors entités) et les entités multi-mots (B- puis I-).

    Args:
        p_elem: L'élément XML <p> à remplir.
        labels: Liste de (mot, label_BIO).
    """
    # 'text' et 'tail' sont les mécanismes ElementTree pour le texte mêlé au XML
    texte_courant = ""       # texte hors entité en attente
    entite_courante = None   # élément d'entité en cours de construction
    dernier_elem = None      # dernier élément ajouté (pour attacher le 'tail')

    def flush_texte():
        """Écrit le texte courant au bon endroit (text de p ou tail du dernier)."""
        nonlocal texte_courant, dernier_elem
        if not texte_courant:
            return
        if dernier_elem is None:
            p_elem.text = (p_elem.text or "") + texte_courant
        else:
            dernier_elem.tail = (dernier_elem.tail or "") + texte_courant
        texte_courant = ""

    for mot, label in labels:
        if label == "O":
            # Mot ordinaire : on ferme une éventuelle entité, on accumule le texte
            if entite_courante is not None:
                dernier_elem = entite_courante
                entite_courante = None
            texte_courant += mot + " "
        elif label.startswith("B-"):
            # Début d'entité : on écrit le texte en attente
            flush_texte()
            if entite_courante is not None:
                dernier_elem = entite_courante
            type_entite = label[2:]
            balise = BALISES_TEI.get(type_entite, "name")
            entite_courante = ET.SubElement(p_elem, balise)
            entite_courante.text = mot
            entite_courante.tail = " "
        elif label.startswith("I-") and entite_courante is not None:
            # Continuation : on ajoute le mot à l'entité en cours
            entite_courante.text += " " + mot

    # Fin : écrire ce qui reste
    if entite_courante is not None:
        dernier_elem = entite_courante
    flush_texte()


# ─── Point d'entrée (démonstration) ──────────────────────────────────────────

if __name__ == "__main__":
    print("=== Démonstration de graphe_tei.py (mode simulation) ===\n")

    texte = "Brut tint Engleterre et Charles regna sur France"

    # 1. Extraction de relations
    print("─── Relations extraites ───")
    relations = extraire_relations(texte)
    for rel in relations:
        print(f"  ({rel['source']}) --[{rel['relation']}]--> ({rel['cible']})")

    # 2. Graphe
    print("\n─── Graphe ───")
    graphe = construire_graphe(relations)
    print(f"  {graphe.number_of_nodes()} entités, {graphe.number_of_edges()} relations")

    # 3. Export TEI
    print("\n─── Export TEI ───")
    chemin = exporter_tei(texte, "dataset_nlp/exemple_tei.xml",
                          titre="Extrait du Roman de Brut")

    # Affiche un aperçu du TEI produit
    print("\n─── Aperçu du TEI ───")
    contenu = Path(chemin).read_text(encoding="utf-8")
    for ligne in contenu.split("\n"):
        if any(b in ligne for b in ["persName", "placeName", "<p>", "date"]):
            print(f"  {ligne.strip()}")

    print("\n=== Démonstration terminée ✓ ===")
