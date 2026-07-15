# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:38:48 2026

@author: DELL
"""

import os

base_path =  r"C:\Users\DELL\Desktop\PFE\Base de donnees textuelle_PFL24\BDtxt"
resultats = {}  # dictionnaire final
total_mots_base = 0

for root, dirs, files in os.walk(base_path):
    txt_files = [f for f in files if f.endswith(".txt")]
    if txt_files:
        locuteur_name = os.path.basename(root)
        resultats[locuteur_name] = {}
        for txt_file in sorted(txt_files):
            chemin = os.path.join(root, txt_file)
            with open(chemin, 'r', encoding='utf-8') as f:
                texte = f.read()
                nb_mots = len(texte.split())
                resultats[locuteur_name][txt_file] = nb_mots
                total_mots_base += nb_mots

# Total pour toute la base
resultats['total_mots_base'] = total_mots_base

# Exemple d'affichage
from pprint import pprint
pprint(resultats)