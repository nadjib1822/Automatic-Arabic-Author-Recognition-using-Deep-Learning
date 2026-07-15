# ============================================================
#          PROGRAMME DE NETTOYAGE DE BASE DE DONNÉES
# ============================================================

import os
import re
import shutil

# ============================================================
# --------------------- CHEMINS ------------------------------
# ============================================================

SOURCE_DATASET = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees bakir"

CLEAN_DATASET = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees bakir CLEAN"

# ============================================================
# ---------------- PHRASES À SUPPRIMER -----------------------
# ============================================================

PROMOTIONAL_PHRASES = [

    "(Transcrit par TurboScribe. Passez à Illimité pour supprimer ce message.)",
    "(Transcribed by TurboScribe.ai - Go Unlimited to remove this message)",
    "(This file is longer than 30 minutes. Go Unlimited at TurboScribe.ai to transcribe files up to 10 hours long.)"
]

# ============================================================
# ----------------- REGEX TASHKIL ----------------------------
# ============================================================

ARABIC_DIACRITICS = re.compile(r"""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ      # Tatwil
                         """, re.VERBOSE)

# ============================================================
# ------------------ CLEAN FUNCTION --------------------------
# ============================================================

def clean_text(text):

    # Supprimer Tashkil + Chedda + Tatwil
    text = re.sub(ARABIC_DIACRITICS, '', text)

    # Supprimer les phrases promotionnelles
    for phrase in PROMOTIONAL_PHRASES:
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)

    # Supprimer espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Nettoyer espaces début/fin
    text = text.strip()

    return text

# ============================================================
# ------------------- CLEAN DATASET --------------------------
# ============================================================

def clean_dataset(source_path, clean_path):

    # Supprimer ancienne base CLEAN si elle existe
    if os.path.exists(clean_path):
        shutil.rmtree(clean_path)

    # Recréer le dossier CLEAN
    os.makedirs(clean_path)

    total_files = 0

    # Parcours des auteurs
    for author in os.listdir(source_path):

        source_author_path = os.path.join(source_path, author)
        clean_author_path = os.path.join(clean_path, author)

        # Créer dossier auteur CLEAN
        os.makedirs(clean_author_path)

        # Parcours des fichiers texte
        for file_name in os.listdir(source_author_path):

            source_file = os.path.join(source_author_path, file_name)
            clean_file = os.path.join(clean_author_path, file_name)

            # Lire texte original
            with open(source_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Nettoyage
            cleaned_text = clean_text(text)

            # Sauvegarde texte nettoyé
            with open(clean_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            total_files += 1

    return total_files

# ============================================================
# ------------------------ MAIN ------------------------------
# ============================================================

print("\nNettoyage de la base de données en cours...\n")

total = clean_dataset(SOURCE_DATASET, CLEAN_DATASET)

print("==========================================")
print("      NETTOYAGE TERMINÉ AVEC SUCCÈS")
print("==========================================\n")

print(f"Nombre total de fichiers nettoyés : {total}")
print(f"Nouvelle base enregistrée ici :")
print(CLEAN_DATASET)

print("\nArchitecture conservée :")
print("- 16 auteurs")
print("- contenu nettoyé\n")