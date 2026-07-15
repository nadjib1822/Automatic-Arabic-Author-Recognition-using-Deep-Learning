import os
import sys
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

# ── HuggingFace ──
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ══════════════════════════════════════════════════════════════
#   SECTION 1 — PARAMÈTRES  (modifiez ici uniquement)
# ══════════════════════════════════════════════════════════════

DATASET_PATH = "/kaggle/input/datasets/farahsersab/baseee/Base de donnees-pfe/Base de donnees"

# --- Modèle pré-entraîné ---
# Options disponibles :
#   "CAMeL-Lab/bert-base-arabic-camelbert-mix"   ← recommandé (MSA + dialectes)
#   "CAMeL-Lab/bert-base-arabic-camelbert-msa"   ← arabe standard uniquement
#   "CAMeL-Lab/bert-base-arabic-camelbert-ca"    ← arabe classique / ancien
CAMEL_MODEL   = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

# --- Découpage des textes ---
# ⚠ BERT accepte max 512 tokens
# Un mot arabe ≈ 1.3 token → 350 mots × 1.3 ≈ 455 tokens < 512 ✅
CHUNK_SIZE    = 350    # Nombre de mots par segment
CHUNK_OVERLAP = 50     # Chevauchement entre deux segments consécutifs

# --- Séquence ---
MAX_SEQ_LEN   = 512    # Longueur max BERT (ne pas dépasser 512)

# --- Entraînement ---
EPOCHS        = 5      # BERT fine-tune vite : 3-5 suffisent
BATCH_SIZE    = 8      # BERT est lourd (768 dim) → réduire si mémoire insuffisante
LR            = 2e-5   # ⚠ LR très petit obligatoire pour fine-tuning BERT
WEIGHT_DECAY  = 1e-2   # Régularisation L2

# --- Fine-tuning ---
FREEZE_BERT   = False  # True = seule la tête est entraînée (plus rapide, moins précis)
DROPOUT       = 0.3    # Dropout sur la tête de classification

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

def line(c="─"):    print(c * W)
def dline(c="═"):   print(c * W)

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

def chunk_text(text, size, overlap):
    """
    Découpe un texte STRING en segments STRING avec chevauchement.
    ⚠ Retourne des strings (pas des listes de mots) car le tokenizer
      HuggingFace attend des strings.
    """
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        end = min(start + size, len(words))
        seg = words[start:end]
        if len(seg) >= 20:          # ignorer les segments trop courts
            chunks.append(" ".join(seg))   # ← STRING, pas liste
        if end == len(words):
            break
    return chunks


def load_dataset():
    """
    Lit tous les .txt dans DATASET_PATH/<auteur>/<fichier>.txt
    Retourne : chunks (strings), labels (int), auteurs (noms)
    """
    if not os.path.isdir(DATASET_PATH):
        err(f"Dossier introuvable : {DATASET_PATH}")

    author_dirs = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
        and not d.startswith('.')
    ])

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
                segs = chunk_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
                author_chunks.extend(segs)
            except Exception:
                continue

        all_chunks.extend(author_chunks)
        all_labels.extend([idx] * len(author_chunks))
        stats[author] = len(author_chunks)

    return all_chunks, all_labels, author_dirs, stats


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
    """
    ✔ Utilise le tokenizer CamelBERT au lieu d'un encodage custom.
    Retourne (input_ids, attention_mask, label) par exemple.
    """
    def __init__(self, chunks, labels, tokenizer):
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Tokenisation BERT : padding + truncation automatiques
        encoded = tokenizer(
            chunks,                     # liste de strings
            padding        = "max_length",
            truncation     = True,
            max_length     = MAX_SEQ_LEN,
            return_tensors = "pt"
        )
        self.input_ids      = encoded["input_ids"]        # (N, MAX_SEQ_LEN)
        self.attention_mask = encoded["attention_mask"]   # (N, MAX_SEQ_LEN)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.input_ids[i],
                self.attention_mask[i],
                self.labels[i])


# ══════════════════════════════════════════════════════════════
#   SECTION 7 — ARCHITECTURE : CamelBERT + TÊTE DE CLASSIFICATION
# ══════════════════════════════════════════════════════════════

class CamelBertClassifier(nn.Module):
    """
    Remplace TransformerClassifier (scratch) par :
      - CamelBERT pré-entraîné  (12 couches, hidden=768)
      - Représentation via token [CLS]
      - Tête MLP de classification

    PositionalEncoding, EncoderBlock, build_vocab → supprimés.
    """
    def __init__(self, num_classes, freeze_bert=False):
        super().__init__()

        # Chargement du modèle pré-entraîné
        self.bert = AutoModel.from_pretrained(CAMEL_MODEL)

        # Option : geler les poids BERT (entraîner seulement la tête)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden = self.bert.config.hidden_size   # 768 pour BERT-base

        # Tête de classification (remplace self.head du modèle scratch)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2),     # 768 → 384
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, num_classes) # 384 → nb_auteurs
        )

    def forward(self, input_ids, attention_mask):
        # Passage dans CamelBERT
        out = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        # Token [CLS] = représentation globale du chunk
        # out.last_hidden_state : (batch, seq_len, 768)
        # [:, 0, :] → vecteur du token [CLS] : (batch, 768)
        cls_vec = out.last_hidden_state[:, 0, :]

        return self.head(cls_vec)


# ══════════════════════════════════════════════════════════════
#   SECTION 8 — FONCTIONS D'ÉVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_true = 0.0, [], []

    with torch.no_grad():
        for input_ids, attention_mask, y in loader:   # ← 3 éléments maintenant
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y              = y.to(device)

            logits = model(input_ids, attention_mask)  # ← 2 arguments
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
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10
    )

    best_valid_f1 = -1.0
    best_state    = None

    # Historiques pour les figures
    histories = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss":   [], "val_acc":   [], "val_f1":   [],
        "test_loss":  [], "test_acc":  [], "test_f1":  [],
    }

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

        for input_ids, attention_mask, y in train_loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y              = y.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)  # ← 2 arguments
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

        # ── Test par époque (pour les courbes) ──
        ts_loss_e, ts_acc_e, ts_f1_e = evaluate(model, test_loader, criterion, device)

        print(f"  {epoch:>4}  "
              f"{tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_f1:>7.4f}  │  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>7.4f}")

        # Enregistrement des historiques
        histories["train_loss"].append(tr_loss);  histories["train_acc"].append(tr_acc);  histories["train_f1"].append(tr_f1)
        histories["val_loss"].append(vl_loss);    histories["val_acc"].append(vl_acc);    histories["val_f1"].append(vl_f1)
        histories["test_loss"].append(ts_loss_e); histories["test_acc"].append(ts_acc_e); histories["test_f1"].append(ts_f1_e)

        if vl_f1 > best_valid_f1:
            best_valid_f1 = vl_f1
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}

        last_tr = (tr_loss, tr_acc, tr_f1)
        last_vl = (vl_loss, vl_acc, vl_f1)

    # Restaure le meilleur modèle puis évalue sur le test
    model.load_state_dict(best_state)
    ts_loss, ts_acc, ts_f1 = evaluate(model, test_loader, criterion, device)

    # Sauvegarde du meilleur modèle sur disque
    torch.save(best_state, "best_camelbert_model.pt")
    ok("Meilleur modèle sauvegardé → best_camelbert_model.pt")

    return last_tr, last_vl, (ts_loss, ts_acc, ts_f1), histories


# ══════════════════════════════════════════════════════════════
#   SECTION 10 — FIGURES
# ══════════════════════════════════════════════════════════════

def plot_figures(histories, model, test_loader, authors, device):

    # Labels courts : Auteur 1, Auteur 2, ... au lieu des noms de dossiers
    short_labels = [f"Auteur {i+1}" for i in range(len(authors))]
    label_map    = {author: idx for idx, author in enumerate(authors)}

    # ── Collecte prédictions finales (adapté CamelBERT : 3 éléments) ──
    model.eval()
    test_preds  = []
    test_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, y in test_loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            out = model(input_ids, attention_mask)
            p   = torch.argmax(out, dim=1)
            test_preds.extend(p.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    authors_list = short_labels
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
        author_report.append({
            "author":   f"Auteur {idx_author+1}",   # ← label court
            "accuracy": acc,
            "correct":  correct,
            "total":    len(indices)
        })

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

    # ── Chunks corrects vs faux par auteur ──
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
    title("CLASSIFICATION D'AUTEUR — CamelBERT ARABE")
    dline()

    # ── Device ──
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))

    # ── Tokenizer CamelBERT ──
    section("CHARGEMENT DU TOKENIZER CAMELBERT")
    info(f"Modèle : {CAMEL_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CAMEL_MODEL)
    ok("Tokenizer chargé")

    # ── Données ──
    section("CHARGEMENT DES DONNÉES")
    chunks, labels, authors, stats = load_dataset()

    info(f"Device         : {device}")
    info(f"Auteurs        : {len(authors)}")
    for a in authors:
        info(f"  └─ {a:<30} {stats[a]} segments")
    info(f"Total segments : {len(chunks)}")

    # ── Split ──
    (tr_c, tr_l), (vl_c, vl_l), (ts_c, ts_l) = stratified_split(chunks, labels)
    info(f"Train          : {len(tr_c)} segments ({TRAIN_RATIO*100:.0f}%)")
    info(f"Validation     : {len(vl_c)} segments ({VALID_RATIO*100:.0f}%)")
    info(f"Test           : {len(ts_c)} segments ({TEST_RATIO*100:.0f}%)")

    # ── DataLoaders ──
    section("TOKENISATION DES DONNÉES (peut prendre quelques minutes...)")
    train_ds = AuthorDataset(tr_c, tr_l, tokenizer)
    ok("Train tokenisé")
    valid_ds = AuthorDataset(vl_c, vl_l, tokenizer)
    ok("Validation tokenisée")
    test_ds  = AuthorDataset(ts_c, ts_l, tokenizer)
    ok("Test tokenisé")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE * 2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2)

    # ── Paramètres du modèle ──
    section("CONFIGURATION MODÈLE")
    info(f"CAMEL_MODEL  = {CAMEL_MODEL}")
    info(f"MAX_SEQ_LEN  = {MAX_SEQ_LEN}")
    info(f"FREEZE_BERT  = {FREEZE_BERT}")
    info(f"DROPOUT      = {DROPOUT}")
    info(f"EPOCHS       = {EPOCHS}")
    info(f"BATCH_SIZE   = {BATCH_SIZE}")
    info(f"LR           = {LR}")
    info(f"CHUNK_SIZE   = {CHUNK_SIZE}  mots")

    # ── Modèle ──
    section("CHARGEMENT DE CAMELBERT (téléchargement si première fois...)")
    model = CamelBertClassifier(
        num_classes = len(authors),
        freeze_bert = FREEZE_BERT
    ).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info(f"Paramètres totaux       : {n_total:,}")
    info(f"Paramètres entraînables : {n_trainable:,}")
    ok("CamelBERT chargé")

    # ── Entraînement ──
    last_tr, last_vl, ts, histories = train(
        model, train_loader, valid_loader, test_loader, device)

    # ── Résultats finaux ──
    section("RÉSULTATS FINAUX  (meilleur modèle selon Val F1)")
    print()
    print(f"  {'':12}  {'Loss':>9}  {'Accuracy':>9}  {'F1 Score':>9}")
    line()
    print(f"  {'TRAIN':<12}  {last_tr[0]:>9.4f}  {last_tr[1]:>9.4f}  {last_tr[2]:>9.4f}")
    print(f"  {'VALIDATION':<12}  {last_vl[0]:>9.4f}  {last_vl[1]:>9.4f}  {last_vl[2]:>9.4f}")
    print(f"  {'TEST':<12}  {ts[0]:>9.4f}  {ts[1]:>9.4f}  {ts[2]:>9.4f}")
    line()

    f1_test = ts[2]
    if   f1_test >= 0.90: verdict = "✔  Excellent — CamelBERT très efficace"
    elif f1_test >= 0.75: verdict = "~  Bon — essayez FREEZE_BERT=False ou plus d'époques"
    elif f1_test >= 0.60: verdict = "~  Moyen — vérifiez la qualité/quantité des données"
    else:                 verdict = "✘  Faible — augmentez le dataset ou réduisez CHUNK_SIZE"

    print(f"\n  Test F1 : {f1_test:.4f}  →  {verdict}")
    print()
    dline()
    print()

    # ── Figures ──
    section("GÉNÉRATION DES FIGURES")
    plot_figures(histories, model, test_loader, authors, device)


if __name__ == "__main__":
    main()
