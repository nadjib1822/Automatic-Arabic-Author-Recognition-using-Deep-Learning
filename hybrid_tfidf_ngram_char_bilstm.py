import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Bidirectional, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==========================================
# 0. CONFIGURATION & REPRODUCTIBILITÉ
# ==========================================
def fix_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

fix_seeds(42)

# ==========================================
# 1. CHARGEMENT ET TRI
# ==========================================
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

base_path =r"C:\Users\DELL\Desktop\PFE\Base de donnees bakir CLEAN\Base de donnees bakir CLEAN"
texts, labels = [], []
auteurs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

for auteur in auteurs:
    auteur_path = os.path.join(base_path, auteur)
    for texte_file in [f for f in os.listdir(auteur_path) if f.endswith(".txt")]:
        with open(os.path.join(auteur_path, texte_file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(auteur)

# Encodage des auteurs
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
author_names = list(le.classes_)
sorted_authors = natural_sort(author_names)
sort_idx = [author_names.index(name) for name in sorted_authors]
y_categorical = to_categorical(y_encoded, num_classes=16)

# Split 60/20/20 (Standard en Deep Learning)
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(texts, y_categorical, test_size=0.4, random_state=42, stratify=y_encoded)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))

# ==========================================
# 2. PRÉTRAITEMENT DYNAMIQUE
# ==========================================

# TF-IDF : Paramètres logiques pour Stylométrie (N-grams de caractères)
tfidf_vec = TfidfVectorizer(
    max_features=10000,   #
    ngram_range=(9,9),  # Capturer les racines et la ponctuation
    analyzer='char',     # Plus robuste pour l'arabe
    sublinear_tf=True
)
X_train_tfidf = tfidf_vec.fit_transform(X_train_raw).toarray()
X_val_tfidf = tfidf_vec.transform(X_val_raw).toarray()
X_test_tfidf = tfidf_vec.transform(X_test_raw).toarray()
tfidf_shape = X_train_tfidf.shape[1] # On récupère la taille exacte (ex: 2000)

# LSTM : Séquences de mots
max_words = 10000 
max_len = 1000 # Fenêtre de lecture assez large pour capter le style
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_raw)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_raw), maxlen=max_len,truncating='post' )
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_raw), maxlen=max_len,truncating='post' )
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_raw), maxlen=max_len,truncating='post')

# ==========================================
# 3. MODÈLE HYBRIDE (Architecture Finale)
# ==========================================

# Branche LSTM (Séquentielle)
input_lstm = Input(shape=(max_len,), name="input_sequences")
emb = Embedding(input_dim=max_words, output_dim=32)(input_lstm)
lstm_layer = Bidirectional(LSTM(32, return_sequences=True))(emb) # 64 unités pour plus de capacité
lstm_out = GlobalMaxPooling1D(name="lstm_output_layer")(lstm_layer)

# Branche TF-IDF (Statistique)
input_tfidf = Input(shape=(tfidf_shape,), name="input_tfidf")
tfidf_dense = Dense(512, activation='relu')(input_tfidf)
tfidf_out = Dropout(0.3, name="tfidf_output_layer")(tfidf_dense)

# Fusion Tardive (Late Fusion)
combined = concatenate([lstm_out, tfidf_out])
x = Dense(256, activation='relu')(combined)
x = Dropout(0.5)(x)
output = Dense(16, activation='softmax', name="final_output")(x)

model = Model(inputs=[input_lstm, input_tfidf], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary() # 

# ==========================================
# 4. ENTRAÎNEMENT & MÉTRIQUES
# ==========================================
history = model.fit([X_train_seq, X_train_tfidf], y_train, 
                    validation_data=([X_val_seq, X_val_tfidf], y_val), 
                    epochs=50, batch_size=16, verbose=1)

# ==========================================
# 4. CALCUL DES MÉTRIQUES ET TABLEAU
# ==========================================
def get_metrics(x_input, y_true):
    preds = model.predict(x_input)
    y_p = np.argmax(preds, axis=1)
    y_t = np.argmax(y_true, axis=1)
    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average='macro')
    loss = model.evaluate(x_input, y_true, verbose=0)[0]
    return loss, acc, f1

# Collecte des données
m_train = get_metrics([X_train_seq, X_train_tfidf], y_train)
m_val = get_metrics([X_val_seq, X_val_tfidf], y_val)
m_test = get_metrics([X_test_seq, X_test_tfidf], y_test)

results_df = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Loss': [m_train[0], m_val[0], m_test[0]],
    'Accuracy': [f"{m_train[1]*100:.2f}%", f"{m_val[1]*100:.2f}%", f"{m_test[1]*100:.2f}%"],
    'F1-Score': [f"{m_train[2]:.4f}", f"{m_val[2]:.4f}", f"{m_test[2]:.4f}"]
})

print("\n" + "="*50)
print("RÉSUMÉ DES PERFORMANCES")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)

# ==========================================
# 5. FIGURES (LOSS, ACC ET MATRICES)
# ==========================================

# Palette de couleurs commune (identique à l'image de référence)
COLOR_TRAIN = '#2274A5'   # bleu
COLOR_VAL   = '#E9A820'   # orange
COLOR_TEST  = '#3A9E6F'   # vert

# --- Métriques finales sur le jeu de test ---
test_loss, test_acc = model.evaluate([X_test_seq, X_test_tfidf], y_test, verbose=0)

y_pred_test = np.argmax(model.predict([X_test_seq, X_test_tfidf]), axis=1)
y_pred_val  = np.argmax(model.predict([X_val_seq,  X_val_tfidf]),  axis=1)
y_pred_train= np.argmax(model.predict([X_train_seq, X_train_tfidf]), axis=1)

test_f1  = f1_score(np.argmax(y_test,  axis=1), y_pred_test,  average='macro')
val_f1   = f1_score(np.argmax(y_val,   axis=1), y_pred_val,   average='macro')
train_f1 = f1_score(np.argmax(y_train, axis=1), y_pred_train, average='macro')

epochs_range = range(1, len(history.history['accuracy']) + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Accuracy ---
axes[0].plot(epochs_range, history.history['accuracy'],
             color=COLOR_TRAIN, label='Train', linewidth=2)
axes[0].plot(epochs_range, history.history['val_accuracy'],
             color=COLOR_VAL, label='Validation', linewidth=2)
axes[0].axhline(y=test_acc, color=COLOR_TEST, linestyle='-', linewidth=2,
                label=f'Test ({test_acc*100:.1f}%)')
axes[0].set_title('Accuracy Evolution')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- Loss ---
axes[1].plot(epochs_range, history.history['loss'],
             color=COLOR_TRAIN, label='Train', linewidth=2)
axes[1].plot(epochs_range, history.history['val_loss'],
             color=COLOR_VAL, label='Validation', linewidth=2)
axes[1].axhline(y=test_loss, color=COLOR_TEST, linestyle='-', linewidth=2,
                label=f'Test ({test_loss:.4f})')
axes[1].set_title('Loss Evolution')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- F1-Score ---
axes[2].axhline(y=train_f1, color=COLOR_TRAIN, linestyle='-',  linewidth=2,
                label=f'Train ({train_f1:.4f})')
axes[2].axhline(y=val_f1,   color=COLOR_VAL,   linestyle='-',  linewidth=2,
                label=f'Validation ({val_f1:.4f})')
axes[2].axhline(y=test_f1,  color=COLOR_TEST,  linestyle='-', linewidth=2,
                label=f'Test ({test_f1:.4f})')
axes[2].set_title('F1-Score (Train / Val / Test)')
axes[2].set_xlabel('Époque')
axes[2].set_ylabel('F1-Score (macro)')
axes[2].set_ylim(0, 1.1)
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Matrices de Confusion
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
datasets = [([X_train_seq, X_train_tfidf], y_train, 'Train'),
            ([X_val_seq, X_val_tfidf], y_val, 'Validation'),
            ([X_test_seq, X_test_tfidf], y_test, 'Test')]

for i, (data, label, title) in enumerate(datasets):
    y_p = np.argmax(model.predict(data), axis=1)
    y_t = np.argmax(label, axis=1)
    cm = confusion_matrix(y_t, y_p)[np.ix_(sort_idx, sort_idx)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=sorted_authors, yticklabels=sorted_authors)
    axes[i].set_xlabel('Auteurs Prédits', fontsize=11)
    axes[i].set_ylabel('Auteurs Réels', fontsize=11)
    axes[i].set_title(f'Confusion Matrix - {title}')
plt.tight_layout(); plt.show()

# ==========================================
# PRÉCISION ET TAUX D'ERREUR PAR AUTEUR
# ==========================================
y_pred = np.argmax(model.predict([X_test_seq, X_test_tfidf]), axis=1)
y_true = np.argmax(y_test, axis=1)

n_authors = 16
precision_per_author = []
error_per_author = []

for i in range(n_authors):
    idx = np.where(y_true == i)[0]
    acc = np.mean(y_pred[idx] == y_true[idx]) if len(idx) > 0 else 0.0
    precision_per_author.append(acc)
    error_per_author.append(1 - acc)

x = np.arange(n_authors)
width = 0.35

fig, ax = plt.subplots(figsize=(15, 9))

bars1 = ax.bar(x - width/2, precision_per_author, width,
               label='Accuracy',   color=COLOR_TRAIN)   # bleu
bars2 = ax.bar(x + width/2, error_per_author,     width,
               label='Error Rate', color=COLOR_VAL)     # orange

for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Auteurs')
ax.set_ylabel('Score')
ax.set_title("Précision et taux d'erreur par auteur")
ax.set_xticks(x)
ax.set_xticklabels([f'Auteur {i+1}' for i in range(n_authors)], rotation=45)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()