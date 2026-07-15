# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:16:40 2026

@author: NADJIB
"""

# ==============================================================
#   TRANSFORMER — CLASSIFICATION D'AUTEUR (TEXTES ARABES)
#   Un seul fichier — tout est ici
# ==============================================================

import os
import sys
import re
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ══════════════════════════════════════════════════════════════
#   SECTION 1 — PARAMÈTRES  (modifiez ici uniquement)
# ══════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\NADJIB\Desktop\PFE-M2\Base de donnees bakir CLEAN"

# --- Découpage des textes ---
CHUNK_SIZE    = 250
CHUNK_OVERLAP = 100

# --- Architecture du Transformer ---
EMBED_DIM     = 256
NUM_HEADS     = 8
NUM_LAYERS    = 3
FF_DIM        = 512
MAX_SEQ_LEN   = 250
DROPOUT       = 0.5

# --- Entraînement ---
EPOCHS        = 70
BATCH_SIZE    = 16
LR            = 3e-4
WEIGHT_DECAY  = 2e-3

# --- Split ---
TRAIN_RATIO   = 0.60
VALID_RATIO   = 0.20
TEST_RATIO    = 0.20

# --- Reproductibilité ---
SEED          = 42


# ══════════════════════════════════════════════════════════════
#   SECTION 2 — CONSOLE PROPRE
# ══════════════════════════════════════════════════════════════

W = 70

def line(c="─"):       print(c * W)
def dline(c="═"):      print(c * W)

def title(txt):
    pad = (W - len(txt) - 2) // 2
    print("═"*pad + f" {txt} " + "═"*pad)

def section(txt):
    print()
    line()
    print(f"  ▸  {txt}")
    line()

def info(txt):   print(f"  → {txt}")
def warn(txt):   print(f"  ⚠  {txt}")
def ok(txt):     print(f"  ✔  {txt}")
def err(txt):    print(f"  ✘  ERREUR : {txt}"); sys.exit(1)


# ══════════════════════════════════════════════════════════════
#   SECTION 3 — SEED
# ══════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ══════════════════════════════════════════════════════════════
#   SECTION 4 — CHARGEMENT BRUT (aucun nettoyage)
# ══════════════════════════════════════════════════════════════

def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def chunk_text(words, size, overlap):
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        end = min(start + size, len(words))
        seg = words[start:end]
        if len(seg) >= 20:
            chunks.append(seg)
        if end == len(words):
            break
    return chunks


def load_dataset():
    if not os.path.isdir(DATASET_PATH):
        err(f"Dossier introuvable : {DATASET_PATH}")

    author_dirs = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ], key=natural_key)

    if len(author_dirs) < 2:
        err("Il faut au moins 2 sous-dossiers (auteurs) dans le dataset.")

    all_chunks, all_labels = [], []
    stats = {}

    for idx, author in enumerate(author_dirs):
        author_path   = os.path.join(DATASET_PATH, author)
        txt_files     = [f for f in os.listdir(author_path) if f.endswith(".txt")]
        author_chunks = []

        for fname in txt_files:
            fpath = os.path.join(author_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                words = raw.split()
                segs  = chunk_text(words, CHUNK_SIZE, CHUNK_OVERLAP)
                author_chunks.extend(segs)
            except Exception:
                continue

        all_chunks.extend(author_chunks)
        all_labels.extend([idx] * len(author_chunks))
        stats[author] = len(author_chunks)

    return all_chunks, all_labels, author_dirs, stats


def build_vocab(chunks):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for words in chunks:
        for w in words:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


def encode_chunk(words, vocab, max_len):
    ids = [vocab.get(w, 1) for w in words]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# ══════════════════════════════════════════════════════════════
#   SECTION 5 — SPLIT STRATIFIÉ 60 / 20 / 20
# ══════════════════════════════════════════════════════════════

def stratified_split(chunks, labels):
    by_author = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_author[lbl].append(i)

    train_idx, valid_idx, test_idx = [], [], []

    for indices in by_author.values():
        random.shuffle(indices)
        n       = len(indices)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_valid = max(1, int(n * VALID_RATIO))
        train_idx.extend(indices[:n_train])
        valid_idx.extend(indices[n_train:n_train + n_valid])
        test_idx.extend(indices[n_train + n_valid:])

    def sel(idx_list):
        return [chunks[i] for i in idx_list], [labels[i] for i in idx_list]

    return sel(train_idx), sel(valid_idx), sel(test_idx)


# ══════════════════════════════════════════════════════════════
#   SECTION 6 — DATASET PYTORCH
# ══════════════════════════════════════════════════════════════

class AuthorDataset(Dataset):
    def __init__(self, chunks, labels, vocab):
        self.X = torch.tensor(
            [encode_chunk(c, vocab, MAX_SEQ_LEN) for c in chunks],
            dtype=torch.long
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):           return len(self.y)
    def __getitem__(self, i):    return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════
#   SECTION 7 — ARCHITECTURE TRANSFORMER (FROM SCRATCH)
# ══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        pe  = torch.zeros(MAX_SEQ_LEN, EMBED_DIM)
        pos = torch.arange(0, MAX_SEQ_LEN).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, EMBED_DIM, 2).float()
            * (-np.log(10000.0) / EMBED_DIM)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn    = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS,
                                             dropout=DROPOUT,
                                             batch_first=True)
        self.norm1   = nn.LayerNorm(EMBED_DIM)
        self.ff      = nn.Sequential(
            nn.Linear(EMBED_DIM, FF_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FF_DIM, EMBED_DIM),
        )
        self.norm2   = nn.LayerNorm(EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)

    def forward(self, x, pad_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.pos     = PositionalEncoding()
        self.drop    = nn.Dropout(DROPOUT)
        self.layers  = nn.ModuleList([EncoderBlock() for _ in range(NUM_LAYERS)])
        self.head    = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM // 2, num_classes)
        )

    def forward(self, x):
        pad_mask = (x == 0)
        x = self.drop(self.pos(self.embed(x)))
        for layer in self.layers:
            x = layer(x, pad_mask)
        mask_f = (~pad_mask).unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════
#   SECTION 8 — FONCTIONS D'ÉVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_true = 0.0, [], []

    with torch.no_grad():
        for X, y in loader:
            X, y  = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds  += logits.argmax(dim=-1).cpu().tolist()
            all_true   += y.cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_true, all_preds)
    f1       = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1


# ══════════════════════════════════════════════════════════════
#   SECTION 9 — BOUCLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════

def train(model, train_loader, valid_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10
    )

    best_valid_f1 = -1.0
    best_state    = None

    # ── Listes de collecte par époque ──
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    train_acc_list,  val_acc_list,  test_acc_list  = [], [], []
    train_f1_list,   val_f1_list,   test_f1_list   = [], [], []

    section("ENTRAÎNEMENT")
    print()

    hdr = (f"  {'Ep':>4}  "
           f"{'Tr Loss':>8}  {'Tr Acc':>7}  {'Tr F1':>7}  │  "
           f"{'Va Loss':>8}  {'Va Acc':>7}  {'Va F1':>7}")
    print(hdr)
    line()

    last_tr = last_vl = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        t_loss, t_preds, t_true = 0.0, [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss  += loss.item() * len(y)
            t_preds += logits.argmax(-1).cpu().tolist()
            t_true  += y.cpu().tolist()

        scheduler.step()

        tr_loss = t_loss / len(train_loader.dataset)
        tr_acc  = accuracy_score(t_true, t_preds)
        tr_f1   = f1_score(t_true, t_preds, average="weighted", zero_division=0)

        # ── Validation ──
        vl_loss, vl_acc, vl_f1 = evaluate(model, valid_loader, criterion, device)

        # ── Test (par époque, pour les courbes) ──
        ts_loss, ts_acc, ts_f1 = evaluate(model, test_loader, criterion, device)

        # ── Collecte ──
        train_loss_list.append(tr_loss);  val_loss_list.append(vl_loss);  test_loss_list.append(ts_loss)
        train_acc_list.append(tr_acc);    val_acc_list.append(vl_acc);    test_acc_list.append(ts_acc)
        train_f1_list.append(tr_f1);      val_f1_list.append(vl_f1);      test_f1_list.append(ts_f1)

        print(f"  {epoch:>4}  "
              f"{tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_f1:>7.4f}  │  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>7.4f}")

        if vl_f1 > best_valid_f1:
            best_valid_f1 = vl_f1
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}

        last_tr = (tr_loss, tr_acc, tr_f1)
        last_vl = (vl_loss, vl_acc, vl_f1)

    # Restaure le meilleur modèle
    model.load_state_dict(best_state)
    ts_loss, ts_acc, ts_f1 = evaluate(model, test_loader, criterion, device)

    histories = {
        "train_loss": train_loss_list, "val_loss": val_loss_list, "test_loss": test_loss_list,
        "train_acc":  train_acc_list,  "val_acc":  val_acc_list,  "test_acc":  test_acc_list,
        "train_f1":   train_f1_list,   "val_f1":   val_f1_list,   "test_f1":   test_f1_list,
    }

    return last_tr, last_vl, (ts_loss, ts_acc, ts_f1), histories


# ══════════════════════════════════════════════════════════════
#   SECTION 10 — FIGURES
# ══════════════════════════════════════════════════════════════

def plot_figures(histories, model, test_loader, authors, device):

    label_map = {author: idx for idx, author in enumerate(authors)}

    # ── Collecte prédictions finales ──
    model.eval()
    test_preds  = []
    test_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            out = model(X)
            p   = torch.argmax(out, dim=1)
            test_preds.extend(p.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    authors_list = list(label_map.keys())
    epochs_range = range(1, EPOCHS + 1)

    # ── Loss vs Epochs ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs_range, histories["train_loss"], label="Train Loss")
    ax.plot(epochs_range, histories["val_loss"],   label="Validation Loss")
    ax.plot(epochs_range, histories["test_loss"],  label="Test Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epochs")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("loss_vs_epochs.png")
    plt.close()

    # ── Accuracy vs Epochs ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs_range, histories["train_acc"], label="Train Accuracy")
    ax.plot(epochs_range, histories["val_acc"],   label="Validation Accuracy")
    ax.plot(epochs_range, histories["test_acc"],  label="Test Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Epochs")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_epochs.png")
    plt.close()

    # ── F1 Score vs Epochs ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs_range, histories["train_f1"], label="Train F1-score")
    ax.plot(epochs_range, histories["val_f1"],   label="Validation F1-score")
    ax.plot(epochs_range, histories["test_f1"],  label="Test F1-score")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("F1-score (macro)")
    ax.set_title("F1-score vs Epochs")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("f1_vs_epochs.png")
    plt.close()

    # ── Précision par auteur ──
    author_report = []
    for author, idx_author in label_map.items():
        indices = [i for i, l in enumerate(test_labels) if l == idx_author]
        if len(indices) == 0:
            continue
        correct = sum(1 for i in indices if test_preds[i] == test_labels[i])
        acc     = correct / len(indices)
        author_report.append({"author": author, "accuracy": acc, "correct": correct, "total": len(indices)})

    fig, ax = plt.subplots(figsize=(10, 6))
    author_names   = [r["author"]   for r in author_report]
    accuracies     = [r["accuracy"] for r in author_report]
    errors         = [1 - acc       for acc in accuracies]
    x              = np.arange(len(author_names))
    width          = 0.35
    accuracy_color = "#1A0033"
    error_color    = "#B8860B"

    ax.bar(x - width/2, accuracies, width, label='Accuracy',   color=accuracy_color)
    ax.bar(x + width/2, errors,     width, label='Error Rate', color=error_color)
    ax.set_xticks(x)
    ax.set_xticklabels(author_names, rotation=45)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Précision et taux d'erreur par auteur")
    ax.legend()
    plt.tight_layout()
    plt.savefig("precision_per_author.png")
    plt.close()

    # ── Matrice de confusion ──
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
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

    # ── Chunks corrects vs faux par auteur ──        ← NOUVELLE FIGURE
    correct_color = "#1B3A4B"
    wrong_color   = "#6B0F1A"

    fig, ax = plt.subplots(figsize=(12, 6))
    author_names_c = [r["author"]  for r in author_report]
    corrects       = [r["correct"] for r in author_report]
    wrongs         = [r["total"] - r["correct"] for r in author_report]
    x              = np.arange(len(author_names_c))

    bars_c = ax.bar(x, corrects, label="Chunks corrects", color=correct_color)
    bars_w = ax.bar(x, wrongs,   bottom=corrects,          label="Chunks faux",     color=wrong_color)

    # Pourcentage au-dessus de chaque barre
    for i, r in enumerate(author_report):
        pct = r["accuracy"] * 100
        ax.text(x[i], r["total"] + 0.5, f"{pct:.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(author_names_c, rotation=45, ha="right")
    ax.set_ylabel("Nombre de chunks")
    ax.set_title("Chunks corrects vs faux par auteur (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("chunks_per_author.png")
    plt.close()

    print("\n================")
    print("Classification done ✅")
    print("Saving pictures done ✅")


# ══════════════════════════════════════════════════════════════
#   SECTION 11 — MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print()
    dline()
    title("CLASSIFICATION D'AUTEUR — TRANSFORMER ARABE")
    dline()

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))

    section("CHARGEMENT DES DONNÉES")
    chunks, labels, authors, stats = load_dataset()

    info(f"Device         : {device}")
    info(f"Auteurs        : {len(authors)}")
    for a in authors:
        info(f"  └─ {a:<30} {stats[a]} segments")
    info(f"Total segments : {len(chunks)}")

    vocab = build_vocab(chunks)
    info(f"Vocabulaire    : {len(vocab):,} tokens")

    (tr_c, tr_l), (vl_c, vl_l), (ts_c, ts_l) = stratified_split(chunks, labels)
    info(f"Train          : {len(tr_c)} segments ({TRAIN_RATIO*100:.0f}%)")
    info(f"Validation     : {len(vl_c)} segments ({VALID_RATIO*100:.0f}%)")
    info(f"Test           : {len(ts_c)} segments ({TEST_RATIO*100:.0f}%)")

    train_ds = AuthorDataset(tr_c, tr_l, vocab)
    valid_ds = AuthorDataset(vl_c, vl_l, vocab)
    test_ds  = AuthorDataset(ts_c, ts_l, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2)

    section("CONFIGURATION MODÈLE")
    info(f"EMBED_DIM    = {EMBED_DIM}")
    info(f"NUM_HEADS    = {NUM_HEADS}")
    info(f"NUM_LAYERS   = {NUM_LAYERS}")
    info(f"FF_DIM       = {FF_DIM}")
    info(f"MAX_SEQ_LEN  = {MAX_SEQ_LEN}")
    info(f"DROPOUT      = {DROPOUT}")
    info(f"EPOCHS       = {EPOCHS}")
    info(f"BATCH_SIZE   = {BATCH_SIZE}")
    info(f"LR           = {LR}")
    info(f"CHUNK_SIZE   = {CHUNK_SIZE}  mots")

    model = TransformerClassifier(
        vocab_size  = len(vocab),
        num_classes = len(authors)
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"Paramètres entraînables : {n_params:,}")

    last_tr, last_vl, ts, histories = train(
        model, train_loader, valid_loader, test_loader, device)

    section("RÉSULTATS FINAUX  (meilleur modèle selon Val F1)")
    print()
    print(f"  {'':12}  {'Loss':>9}  {'Accuracy':>9}  {'F1 Score':>9}")
    line()
    print(f"  {'TRAIN':<12}  {last_tr[0]:>9.4f}  {last_tr[1]:>9.4f}  {last_tr[2]:>9.4f}")
    print(f"  {'VALIDATION':<12}  {last_vl[0]:>9.4f}  {last_vl[1]:>9.4f}  {last_vl[2]:>9.4f}")
    print(f"  {'TEST':<12}  {ts[0]:>9.4f}  {ts[1]:>9.4f}  {ts[2]:>9.4f}")
    line()

    f1_test = ts[2]
    if   f1_test >= 0.85: verdict = "✔  Excellent"
    elif f1_test >= 0.70: verdict = "~  Bon — essayez plus d'époques ou EMBED_DIM plus grand"
    elif f1_test >= 0.55: verdict = "~  Moyen — augmentez NUM_LAYERS ou réduisez CHUNK_SIZE"
    else:                 verdict = "✘  Faible — vérifiez les données ou augmentez EPOCHS"

    print(f"\n  Test F1 : {f1_test:.4f}  →  {verdict}")
    print()
    dline()
    print()

    # ── Figures ──
    section("GÉNÉRATION DES FIGURES")
    plot_figures(histories, model, test_loader, authors, device)


if __name__ == "__main__":
    main()