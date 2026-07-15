# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 11:53:11 2026

@author: NADJIB
"""

# ====================================================================
#       LSTM Text Classification avec AraBERT/ELECTRA
#       Hyperparameter Search + CSV + Figures + Rapport par auteur
#       Optimisation mémoire GPU + Early Stopping
# ====================================================================

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ---------------------- PARAMETERS -------------------------
# ============================================================

DATASET_PATH = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees bakir"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 12

LR_LIST = [2e-5, 3e-5, 4e-5, 4.5e-5, 5e-5]
LSTM_HIDDEN_LIST = [64, 256, 512]
DROPOUT_LIST = [0.2, 0.3]

LSTM_LAYERS = 1
EARLY_STOPPING_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ---------------- MODEL SELECTION --------------------------
# ============================================================

print("\nChoisir le type d'embedding :")
print("1 - AraBERT")
print("2 - ELECTRA")
choice = input("Votre choix : ")

if choice == "1":
    MODEL_NAME = "aubmindlab/bert-base-arabertv02"
elif choice == "2":
    MODEL_NAME = "aubmindlab/araelectra-base-discriminator"
else:
    raise ValueError("Choix invalide")

# ============================================================
# ---------------- DATASET LOADING --------------------------
# ============================================================

def load_dataset(path):
    texts, labels = [], []
    authors = sorted(os.listdir(path), key=lambda x: int(''.join(filter(str.isdigit, x))))
    label_map = {author: i for i, author in enumerate(authors)}
    for author in authors:
        author_path = os.path.join(path, author)
        for file in os.listdir(author_path):
            with open(os.path.join(author_path, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(label_map[author])
    return texts, labels, len(authors), authors

texts, labels, NUM_CLASSES, authors_list = load_dataset(DATASET_PATH)

# ============================================================
# --------------------- TOKENIZATION ------------------------
# ============================================================

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
# --------------------- DATASET CLASS -----------------------
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
# ---------------- DATA SPLIT 60/20/20 ----------------------
# ============================================================

indices = np.arange(len(labels_all))
X_train_idx, X_temp_idx, y_train_idx, y_temp_idx = train_test_split(
    indices, labels_all, test_size=0.4, stratify=labels_all)
X_val_idx, X_test_idx, y_val_idx, y_test_idx = train_test_split(
    X_temp_idx, y_temp_idx, test_size=0.5, stratify=y_temp_idx)

def get_dataloader(idxs):
    ds = TextDataset(input_ids_all[idxs], attention_mask_all[idxs], labels_all[idxs])
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

train_loader = get_dataloader(X_train_idx)
val_loader = get_dataloader(X_val_idx)
test_loader = get_dataloader(X_test_idx)

# ============================================================
# ---------------- MODEL ARCHITECTURE -----------------------
# ============================================================

class AraBERT_LSTM(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        embedding_size = self.transformer.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        lstm_out, _ = self.lstm(embeddings)
        last_output = lstm_out[:, -1, :]
        x = self.dropout(last_output)
        return self.fc(x)

# ============================================================
# ------------------ TRAIN / EVALUATE ----------------------
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss/len(loader), acc, f1

def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            total_loss += criterion(outputs, labels).item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss/len(loader), acc, f1, all_preds, all_labels

# ============================================================
# ------------------ HYPERPARAM SEARCH ----------------------
# ============================================================

results = []
test_counter = 0
criterion = nn.CrossEntropyLoss()

for lr in LR_LIST:
    for hidden in LSTM_HIDDEN_LIST:
        for dropout in DROPOUT_LIST:
            test_counter += 1
            model = AraBERT_LSTM(hidden_size=hidden, dropout=dropout).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            best_val_loss = float('inf')
            epochs_no_improve = 0

            # 🔴 AJOUT stockage courbes
            train_losses_epochs = []
            val_losses_epochs = []
            test_losses_epochs = []
            
            for epoch in range(EPOCHS):
                train_loss_epoch, _, _ = train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)

                # 🔴 AJOUT test à chaque epoch
                test_loss_epoch, _, _, _, _ = evaluate(model, test_loader, criterion)

                train_losses_epochs.append(train_loss_epoch)
                val_losses_epochs.append(val_loss)
                test_losses_epochs.append(test_loss_epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0

                    # 🔴 sauvegarde meilleures courbes
                    best_train_curve = train_losses_epochs.copy()
                    best_val_curve = val_losses_epochs.copy()
                    best_test_curve = test_losses_epochs.copy()
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping activated at epoch {epoch+1} for test {test_counter}")
                    break
            
            train_loss, train_acc, train_f1, _, _ = evaluate(model, train_loader, criterion)
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
            test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)
            
            results.append({
                "test_id": test_counter,
                "lr": lr,
                "hidden": hidden,
                "dropout": dropout,
                "batch": BATCH_SIZE,
                "epochs": EPOCHS,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "model": MODEL_NAME,
                "test_preds": test_preds,
                "test_labels": test_labels
            })
            
            print(f"Test {test_counter} ----- done")
            
            del model
            torch.cuda.empty_cache()

# ============================================================
# ------------------- BEST TEST IDENTIFICATION --------------
# ============================================================

df = pd.DataFrame(results)
best_idx = df["val_acc"].idxmax()
df["best_flag"] = 0
df.loc[best_idx, "best_flag"] = 1

best_params = df.loc[best_idx, ["lr", "hidden", "dropout", "batch", "epochs"]]
print(f"Meilleur test : LR={best_params.lr} | Hidden={best_params.hidden} | Dropout={best_params.dropout} | Batch={best_params.batch} | Epochs={best_params.epochs}")

# ============================================================
# ---------------------- SAVE CSV ---------------------------
# ============================================================

df.to_csv("table_results.csv", index=False)
print("Tableau CSV principal ----- done")

# ============================================================
# ------------------ REPORT PER AUTHOR ----------------------
# ============================================================

best_preds = df.loc[best_idx, "test_preds"]
best_labels = df.loc[best_idx, "test_labels"]

author_report = []
for i, author in enumerate(authors_list):
    idxs = [j for j, lbl in enumerate(best_labels) if lbl == i]
    correct = sum([1 for j in idxs if best_preds[j] == i])
    misclassified = [authors_list[best_preds[j]] for j in idxs if best_preds[j] != i]
    accuracy = correct / len(idxs) if len(idxs) > 0 else 0
    author_report.append({
        "author": author,
        "total_texts": len(idxs),
        "correct_preds": correct,
        "accuracy": accuracy,
        "misclassified_as": misclassified
    })

pd.DataFrame(author_report).to_csv("author_report.csv", index=False)
print("Rapport CSV par auteur ----- done")

# ============================================================
# ------------------ FIGURE LOSS VS EPOCHS -------------------
# ============================================================

plt.figure(figsize=(8,6))

epochs_range = range(1, len(best_train_curve)+1)

plt.plot(epochs_range, best_train_curve, label='Train Loss')
plt.plot(epochs_range, best_val_curve, label='Validation Loss')
plt.plot(epochs_range, best_test_curve, label='Test Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs - Meilleur Test")

plt.legend()
plt.grid(True)

plt.savefig("loss_vs_epochs.png")
plt.close()

print("Figure Loss vs Epochs ----- done")

# ============================================================
# ------ FIGURE PRECISION PAR AUTEUR ------(DOUBLE HISTOGRAM)
# ============================================================

plt.figure(figsize=(10,6))

author_names = [r["author"] for r in author_report]
accuracies = [r["accuracy"] for r in author_report]
errors = [1 - acc for acc in accuracies]

x = np.arange(len(author_names))
width = 0.35

# Couleurs Rocket comme la matrice

accuracy_color = "#1A0033"  # violet très foncé
error_color = "#B8860B"     

plt.bar(x - width/2, accuracies, width, label='Accuracy', color=accuracy_color)
plt.bar(x + width/2, errors, width, label='Error Rate', color=error_color)

plt.xticks(x, author_names, rotation=45)
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Précision et taux d'erreur par auteur - Meilleur Test")

plt.legend()
plt.tight_layout()
plt.savefig("precision_per_author.png")
plt.close()

print("Figure précision par auteur done")

# ============================================================
# ------------------ CONFUSION MATRIX FIGURE -----------------
# ============================================================

cm = confusion_matrix(best_labels, best_preds)
plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=authors_list,
            yticklabels=authors_list,
            cmap="rocket")

plt.xlabel("Prédictions")
plt.ylabel("Labels réels")
plt.title("Matrice de confusion - Meilleur Test")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Figure matrice de confusion done")

# ============================================================
# ------------------ FIN DU SCRIPT --------------------------
# ============================================================

print("Opération terminée")