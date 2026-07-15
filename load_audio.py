

import librosa               # Librairie pour le traitement audio (lecture, analyse, etc.)
import librosa.display       # Module de librosa pour l’affichage graphique des signaux audio
import matplotlib.pyplot as plt  # Librairie pour tracer des graphiques
import numpy as np           # Librairie pour le calcul numérique (utile pour les signaux audio)
import sounddevice as sd     # Pour la lecture audio


# Indique le chemin complet du fichier audio
file_name = r'C:\Users\NADJIB\Desktop\pfe\audio.mp3'  # Chemin du fichier audio à lire

# Lecture du fichier audio
y, sr = librosa.load(file_name, sr=None)  

# Affichage du signal audio (forme d’onde)
plt.figure(figsize=(12, 4))  # Définit la taille du graphique
librosa.display.waveshow(y, sr=sr, color='b')  # Trace la forme d’onde en bleu
plt.title("Signal audio (forme d’onde)")  # Titre du graphique
plt.xlabel("Temps (secondes)")           # Label de l’axe x
plt.ylabel("Amplitude")                  # Label de l’axe y
plt.grid(True)                           # Active la grille
plt.show()                               # Affiche le graphique

# Lecture du son avec sounddevice
print("\nLecture de l'audio en cours...")
sd.play(y, sr)   # Joue le signal audio
sd.wait()        # Attend la fin de la lecture
print("Lecture terminée.")

# Informations sur l’audio
duration = librosa.get_duration(y=y, sr=sr)  # Calcul de la durée en secondes
print("\n--- Informations sur l’audio ---")   # Titre pour les infos
print(f"Durée : {duration:.2f} secondes")    # Durée avec 2 décimales
print(f"Fréquence d’échantillonnage : {sr} Hz")  # Fréquence d’échantillonnage
print(f"Nombre d’échantillons : {len(y)}")       # Nombre total d’échantillons audio
print(f"Amplitude maximale : {np.max(y):.3f}")   # Valeur maximale d’amplitude dans le signal


