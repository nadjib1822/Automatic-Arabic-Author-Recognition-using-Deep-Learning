import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

base_path = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees textuelle_PFL24"

model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def extraire_vecteur(texte):
    inputs = tokenizer(
        texte,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# =============================
# 🔎 Trouver tous les fichiers txt
# =============================

locuteurs_dict = {}

for root, dirs, files in os.walk(base_path):
    txt_files = [f for f in files if f.endswith(".txt")]
    
    if txt_files:
        locuteur_name = os.path.basename(root)
        locuteurs_dict[locuteur_name] = [
            os.path.join(root, f) for f in sorted(txt_files)
        ]

print("Nombre total de locuteurs détectés :", len(locuteurs_dict))

# =============================
# Détection taille embedding
# =============================

first_locuteur = next(iter(locuteurs_dict))
first_file = locuteurs_dict[first_locuteur][0]

with open(first_file, "r", encoding="utf-8") as f:
    texte_test = f.read()

feature_size = extraire_vecteur(texte_test).shape[0]
nb_locuteurs = len(locuteurs_dict)
nb_textes = len(locuteurs_dict[first_locuteur])

X = np.zeros((feature_size, nb_locuteurs, nb_textes))

# =============================
# Lecture complète sans erreur
# =============================

for i, (locuteur, fichiers) in enumerate(locuteurs_dict.items()):
    
    print(f"\n========== {locuteur} ==========")
    
    for j, fichier_path in enumerate(fichiers):
        
        with open(fichier_path, "r", encoding="utf-8") as f:
            texte = f.read()
        
        print(f"\n--- {os.path.basename(fichier_path)} ---")
        print(texte)
        
        vecteur = extraire_vecteur(texte)
        X[:, i, j] = vecteur

print("\nShape finale :", X.shape)




