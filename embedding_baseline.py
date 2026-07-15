import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# 1. Fonction de nettoyage arabe
# -----------------------------
def nettoyer_arabe(text):
    # Normalisation des lettres arabes
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    # Supprimer tout sauf les caractères arabes et espaces
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    # Supprimer espaces multiples
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# 2. Charger tous les fichiers texte
# -----------------------------
base_path = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees textuelle_PFL24"  # <<< MODIFIER ICI

texts = []
file_names = []
labels = []

for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith(".txt"):
            path = os.path.join(root, f)
            # Auteur = dernier dossier du chemin
            auteur = os.path.basename(root)
            if auteur.startswith("L"):  # on ne prend que L1, L2...
                with open(path, "r", encoding="utf-8") as file:
                    contenu = file.read()
                    contenu = nettoyer_arabe(contenu)

                    texts.append(contenu)
                    file_names.append(f)
                    labels.append(auteur)

print("Nombre total de textes :", len(texts))
print("Nombre de locuteurs :", len(set(labels)))

# -----------------------------
# 3. Calcul des embeddings TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=300,    # nombre de features
    ngram_range=(1, 2),  # unigrams + bigrams
)

X = vectorizer.fit_transform(texts)

# -----------------------------
# 4. Sauvegarde dans un CSV
# -----------------------------
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df.insert(0, "fichier", file_names)
df.insert(1, "auteur", labels)  # ajoute la colonne des locuteurs

df.to_csv("embeddings.csv", index=False, encoding="utf-8-sig")  # UTF-8-SIG pour Excel

print("Embeddings sauvegardés dans 'embeddings.csv'")

