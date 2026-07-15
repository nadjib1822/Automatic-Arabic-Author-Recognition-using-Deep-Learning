# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:02:38 2026

@author: NADJIB
"""

# ====================================================================
#                Modèle LSTM pour la classification de texte 
#             utilisant des embeddings ARABERT/ELECTRA/CAMELbert
# ====================================================================

# ---------------- IMPORTS ----------------
import os
import re
import torch
import numpy as np
import pandas as pd
import random

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# -------------------- SEED (IMPORTANT) ----------------------
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# ---------------------- PARAMETERS --------------------------
# ============================================================

DATASET_PATH = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees bakir"

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 15

LR = 1.8e-5
LSTM_HIDDEN = 256
LSTM_LAYERS = 1
DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ---------------- NATURAL SORT FUNCTION ---------------------
# ============================================================

def natural_sort(l):
    return sorted(l, key=lambda x: [int(text) if text.isdigit() else text.lower()
                                   for text in re.split('([0-9]+)', x)])

# ============================================================
# ---------------- MODEL SELECTION ---------------------------
# ============================================================

print("\nChoisir le type d'embedding :")
print("1 - AraBERT")
print("2 - ELECTRA")
print("3 - CAMeLBERT")  

choice = input("Votre choix : ")

if choice == "1":
    MODEL_NAME = "aubmindlab/bert-base-arabertv02"
elif choice == "2":
    MODEL_NAME = "aubmindlab/araelectra-base-discriminator"
elif choice == "3":
    MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
else:
    raise ValueError("Choix invalide")

# ============================================================
# ---------------- DATASET LOADING ---------------------------
# ============================================================

def load_dataset(path):
    texts = []
    labels = []

    authors = natural_sort(os.listdir(path))
    label_map = {author: i for i, author in enumerate(authors)}

    for author in authors:
        author_path = os.path.join(path, author)

        for file in os.listdir(author_path):
            file_path = os.path.join(author_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            texts.append(text)
            labels.append(label_map[author])

    return texts, labels, len(authors), authors

print("\nChargement du dataset...")
texts, labels, NUM_CLASSES, authors_list = load_dataset(DATASET_PATH)

print("Nombre de textes :", len(texts))
print("Nombre d'auteurs :", NUM_CLASSES)

# ============================================================
# --------------------- TOKENIZATION -------------------------
# ============================================================

print("\nTokenization du dataset ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    texts,
    padding='max_length',
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

input_ids_all = encodings["input_ids"]
attention_mask_all = encodings["attention_mask"]
labels_all = torch.tensor(labels)

# ============================================================
# --------------------- DATASET CLASS ------------------------
# ============================================================

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx]
        }

# ============================================================
# ---------------- DATA SPLIT 60/20/20 -----------------------
# ============================================================

X_train_idx, X_temp_idx, y_train_idx, y_temp_idx = train_test_split(
    np.arange(len(labels_all)), labels_all,
    test_size=0.4, stratify=labels_all, random_state=SEED
)

X_val_idx, X_test_idx, y_val_idx, y_test_idx = train_test_split(
    X_temp_idx, y_temp_idx,
    test_size=0.5, stratify=y_temp_idx, random_state=SEED
)

train_dataset = TextDataset(input_ids_all[X_train_idx],
                            attention_mask_all[X_train_idx],
                            labels_all[X_train_idx])

val_dataset = TextDataset(input_ids_all[X_val_idx],
                          attention_mask_all[X_val_idx],
                          labels_all[X_val_idx])

test_dataset = TextDataset(input_ids_all[X_test_idx],
                           attention_mask_all[X_test_idx],
                           labels_all[X_test_idx])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ============================================================
# ------------------- MODEL ARCHITECTURE ---------------------
# ============================================================

class AraBERT_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        embedding_size = self.transformer.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(LSTM_HIDDEN * 2, NUM_CLASSES)

    def forward(self, input_ids, attention_mask):

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = outputs.last_hidden_state
        lstm_out, _ = self.lstm(embeddings)
        last_output = lstm_out[:, -1, :]
        x = self.dropout(last_output)
        logits = self.fc(x)

        return logits

model = AraBERT_LSTM().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# ---------------- TRAINING FUNCTION -------------------------
# ============================================================

def train_epoch(loader):
    model.train()

    all_preds = []
    all_labels = []
    total_loss = 0

    for batch in tqdm(loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return total_loss / len(loader), acc, f1

# ============================================================
# ---------------- EVALUATION FUNCTION -----------------------
# ============================================================

def evaluate(loader):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return total_loss / len(loader), acc, f1, all_preds, all_labels

# ============================================================
# ---------------- TRAINING LOOP -----------------------------
# ============================================================

print("\n========== TRAINING START ==========\n")

train_losses = []
val_losses = []
test_losses = []

for epoch in range(EPOCHS):

    train_loss, train_acc, train_f1 = train_epoch(train_loader)
    val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)
    test_loss_epoch, _, _, _, _ = evaluate(test_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss_epoch)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("TRAIN | Loss {:.4f} | Acc {:.4f} | F1 {:.4f}".format(
        train_loss, train_acc, train_f1))
    print("VALID | Loss {:.4f} | Acc {:.4f} | F1 {:.4f}".format(
        val_loss, val_acc, val_f1))

# ============================================================
# ---------------- TEST EVALUATION ---------------------------
# ============================================================

print("\n========== TEST RESULTS ==========\n")

test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(test_loader)

print("TEST | Loss {:.4f} | Acc {:.4f} | F1 {:.4f}".format(
    test_loss, test_acc, test_f1))

# ============================================================
# ---------------- FIGURE LOSS VS EPOCHS ---------------------
# ============================================================

plt.figure(figsize=(8,6))

plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS+1), val_losses, label="Validation Loss")
plt.plot(range(1, EPOCHS+1), test_losses, label="Test Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")

plt.legend()
plt.savefig("loss_vs_epochs.png")
plt.close()

# ============================================================
# ---------------- RAPPORT PAR AUTEUR ------------------------
# ============================================================

author_report = []

for i, author in enumerate(authors_list):

    idxs = [j for j, lbl in enumerate(test_labels) if lbl == i]

    correct = sum([1 for j in idxs if test_preds[j] == i])

    misclassified = [authors_list[test_preds[j]] for j in idxs if test_preds[j] != i]

    accuracy = correct / len(idxs) if len(idxs) > 0 else 0

    author_report.append({
        "author": author,
        "total_texts": len(idxs),
        "correct_preds": correct,
        "accuracy": accuracy,
        "misclassified_as": misclassified
    })

pd.DataFrame(author_report).to_csv("author_report.csv", index=False)

# ============================================================
# -------- FIGURE PRECISION PAR AUTEUR (COMME PROG 1) --------
# ============================================================

plt.figure(figsize=(10,6))

author_names = [r["author"] for r in author_report]
accuracies = [r["accuracy"] for r in author_report]
errors = [1 - acc for acc in accuracies]

x = np.arange(len(author_names))
width = 0.35

accuracy_color = "#1A0033"
error_color = "#4B0000"

plt.bar(x - width/2, accuracies, width, label='Accuracy', color=accuracy_color)
plt.bar(x + width/2, errors, width, label='Error Rate', color=error_color)

plt.xticks(x, author_names, rotation=45)
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Précision et taux d'erreur par auteur")

plt.legend()
plt.tight_layout()
plt.savefig("precision_per_author.png")
plt.close()

# ============================================================
# ---------------- MATRICE DE CONFUSION ----------------------
# ============================================================

cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=authors_list,
            yticklabels=authors_list,
            cmap="rocket")

plt.xlabel("Prédictions")
plt.ylabel("Labels réels")
plt.title("Matrice de confusion")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
