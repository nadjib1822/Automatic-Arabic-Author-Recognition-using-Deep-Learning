import os
import pandas as pd
from collections import Counter
import warnings

# =========================
# CONFIGURATION DE BASE
# =========================

# Chemin de la base
base_path = r"C:\Users\DELL\Desktop\PFE\Base de donnees-pfe\Base de donnees"

# Taille n-gram de caractères
n = 3

# Dictionnaire pour stocker les résultats
resultats = {}

# Liste pour le tableau global
tableau_global = []

# =========================
# FONCTION N-GRAM CARACTÈRES
# =========================
def create_char_ngrams(texte, n):
    texte = texte.replace("\n", " ").strip()
    return [texte[i:i+n] for i in range(len(texte)-n+1)]

# =========================
# PARCOURS DE LA BASE DE DONNÉES
# =========================
for root, dirs, files in os.walk(base_path):
    txt_files = [f for f in files if f.endswith(".txt")]
    
    if txt_files:
        locuteur = os.path.basename(root)
        resultats[locuteur] = {}
        
        for f in sorted(txt_files):
            chemin_fichier = os.path.join(root, f)
            
            with open(chemin_fichier, "r", encoding="utf-8") as file:
                texte = file.read()
            
            ngrams_chars = create_char_ngrams(texte, n)
            
            # Occurrences
            occurrences = Counter(ngrams_chars)
            
            # Totaux
            N = len(ngrams_chars)       # total n-grams avec répétition
            V = len(set(ngrams_chars))  # total n-grams uniques
            
            # Liste temporaire pour DataFrame
            data = []
            
            for ng, f_occ in occurrences.items():
                F1 = f_occ / N
                F2 = f_occ / V
                data.append([ng, f_occ, F1, F2])
                
                # Pour tableau global
                tableau_global.append([
                    locuteur,
                    f,
                    ng,
                    f_occ,
                    F1,
                    F2
                ])
            
            # Création DataFrame pour ce texte
            df = pd.DataFrame(data, columns=["Ngram", "Occurrences", "F1", "F2"])
            
            # Stocker dans dictionnaire
            resultats[locuteur][f] = df

# =========================
# CRÉATION DU TABLEAU GLOBAL
# =========================
tableau_global = pd.DataFrame(
    tableau_global,
    columns=["Locuteur", "Texte", "Ngram", "Occurrences", "F1", "F2"]
)


# =========================
# CONSTRUCTION DES MATRICES 3D F1 ET F2
# =========================

# Vocabulaire global (tous les n-grams de tous les textes)
vocabulaire_global = sorted(tableau_global["Ngram"].unique())

# Liste des colonnes sous forme Locuteur_Texte
colonnes = []
for locuteur in resultats:
    for texte in resultats[locuteur]:
        colonnes.append(f"{locuteur}_{texte}")

# Création DataFrame vide pour F1 et F2
matrice_F1 = pd.DataFrame(0, index=vocabulaire_global, columns=colonnes)
matrice_F2 = pd.DataFrame(0, index=vocabulaire_global, columns=colonnes)

# Remplissage des matrices
for locuteur in resultats:
    for texte in resultats[locuteur]:
        nom_colonne = f"{locuteur}_{texte}"
        df = resultats[locuteur][texte]
        
        for _, row in df.iterrows():
            ngram = row["Ngram"]
            matrice_F1.loc[ngram, nom_colonne] = row["F1"]
            matrice_F2.loc[ngram, nom_colonne] = row["F2"]

# Sauvegarde CSV
matrice_F1.to_csv("matrice_3D_F1_lettres.csv", encoding="utf-8")
matrice_F2.to_csv("matrice_3D_F2_lettres.csv", encoding="utf-8")

# =============
# CONSOLE CLEAN
# =============

# Infos de base sur la base
nb_locuteurs = len(resultats)
nb_textes_total = sum(len(resultats[loc]) for loc in resultats)
nb_ngrams_total = len(vocabulaire_global)

print("\n================= Résumé Base de Données =================")
print(f"Chemin : {base_path}")
print(f"Nombre de locuteurs : {nb_locuteurs}")
print(f"Nombre total de textes : {nb_textes_total}")
print(f"Taille du vocabulaire global (n-grams uniques) : {nb_ngrams_total}")
print("===========================================================\n")

# Infos sur F1 et F2
print("Matrices 3D créées :")
print(f"- matrice_F1 : {matrice_F1.shape}")
print(f"- matrice_F2 : {matrice_F2.shape}\n")


print("\n✅ opération fini.")
