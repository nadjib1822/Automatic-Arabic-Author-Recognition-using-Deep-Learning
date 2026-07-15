

import os  # Pour parcourir les fichiers dans un dossier


# Indique le chemin du dossier contenant les fichiers .txt

folder_path = r'C:\Users\NADJIB\Desktop\LOCUTEUR1'  


# Parcours tous les fichiers du dossier

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):  # Vérifie que c'est bien un fichier .txt
        full_path = os.path.join(folder_path, file_name)  # Chemin complet du fichier

        # Lecture du fichier en UTF-8
        with open(full_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Affichage du contenu
        print("\n=============================================")
        print(f"Contenu du fichier : {file_name}")
        print("--------------------")
        print(text)

        # Affichage des informations
        print("\nInformations :")
        print(f"Nombre de caractères : {len(text)}")
        print(f"Type d'encodage détecté : UTF-8")
        print("=============================================\n")


