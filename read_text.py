

# Indique le chemin complet de ton fichier
file_name = r'C:\Users\NADJIB\Desktop\pfe\F1.txt'  

# Lecture du fichier en UTF-8
with open(file_name, 'r', encoding='utf-8') as file:  # Ouvre le fichier en mode lecture ('r') avec encodage UTF-8
    text = file.read()  # Lit tout le contenu du fichier et le stocke dans la variable 'text'

# Affichage
print("Contenu du fichier:")  # Affiche un titre avant le contenu
print(text)  # Affiche le texte complet du fichier

print("\nInformations:")  # Affiche un titre avant les informations sur le fichier
print(f"Nombre de caractères : {len(text)}")  # Affiche le nombre total de caractères du texte
print(f"Type d'encodage détecté : UTF-8")  # Affiche le type d'encodage utilisé pour lire le fichier

