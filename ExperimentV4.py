# ============================================================
# CrisisMMD v4 — Kaggle Notebook
# Model    : EfficientNetV2-M, ViT-B/16, ConvNeXt-Base, Swin-Small
# Training : Two-Stage Fine-Tuning
# Tasks    : Task1 (Informative), Task2 (Humanitarian), Task3 (Damage)
# Ensemble : Simple, Weighted, Best-3, Stacking (Logistic Reg)
# Metrics  : Accuracy, Macro F1, Weighted F1, Per-class F1, AUC-ROC
# ============================================================


# ============================================================
# CELL 1 — Install Library Tambahan
# ============================================================
import subprocess
subprocess.run(['pip', 'install', 'timm', 'thop', '--quiet'], check=True)
print("✅ Library installed")


# ============================================================
# CELL 2 — Import
# ============================================================
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")


# ============================================================
# CELL 3 — Konfigurasi Global
# ============================================================

# ── Path ─────────────────────────────────────────────────────
KAGGLE_INPUT   = '/kaggle/input/datasets/alieffathurrahman/crisismmd'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
RESULTS_DIR    = '/kaggle/working/results'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device  : {device}")

# ── Task Config ──────────────────────────────────────────────
TASK_CONFIG = {
    'informative': {
        'num_classes' : 2,
        'label_col'   : 'image_info',
        'split_prefix': 'task_informative_text_img',
        'class_names' : ['not_informative', 'informative'],
    },
    'humanitarian': {
        'num_classes' : 7,
        'label_col'   : 'image_human',
        'split_prefix': 'task_humanitarian_text_img',
        'class_names' : [
            'infrastructure_and_utility_damage',
            'affected_individuals',
            'injured_or_dead_people',
            'missing_or_found_people',
            'rescue_volunteering_or_donation_effort',
            'vehicle_damage',
            'other_relevant_information',
        ],
    },
    'damage': {
        'num_classes' : 3,
        'label_col'   : 'image_damage',
        'split_prefix': 'task_damage_text_img',
        'class_names' : ['little_or_no_damage', 'mild_damage', 'severe_damage'],
    },
}

# ── Model Config ─────────────────────────────────────────────
MODEL_CONFIG = {
    'efficientnetv2_m': {
        'timm_name' : 'efficientnetv2_m',
        'input_size': 224,
        'batch_size': 16,
        # Two-Stage LR
        'lr_stage1' : 5e-4,   # head only
        'lr_stage2' : 5e-5,   # full model
        'lr_backbone_stage2': 5e-6,  # backbone lr di stage 2 (10x lebih kecil dari head)
    },
    'vit': {
        'timm_name' : 'vit_base_patch16_384',
        'input_size': 384,
        'batch_size': 8,
        'lr_stage1' : 1e-3,
        'lr_stage2' : 5e-5,
        'lr_backbone_stage2': 5e-6,
    },
    'convnext': {
        'timm_name' : 'convnext_base',
        'input_size': 224,
        'batch_size': 16,
        'lr_stage1' : 5e-4,
        'lr_stage2' : 5e-5,
        'lr_backbone_stage2': 5e-6,
    },
    'swin': {
        'timm_name' : 'swin_small_patch4_window7_224',
        'input_size': 224,
        'batch_size': 16,
        'lr_stage1' : 5e-4,
        'lr_stage2' : 5e-5,
        'lr_backbone_stage2': 5e-6,
    },
}

# ── Train Config ─────────────────────────────────────────────
TRAIN_CONFIG = {
    'stage1_epochs'      : 5,      # freeze backbone
    'stage2_epochs'      : 45,     # unfreeze all
    'early_stop_patience': 5,
    'weight_decay'       : 0.01,
    'label_smoothing'    : 0.1,
    'num_workers'        : 2,
    'seed'               : 42,
    'subset_size'        : None,   # None = full dataset
}

torch.manual_seed(TRAIN_CONFIG['seed'])
np.random.seed(TRAIN_CONFIG['seed'])
print("✅ Konfigurasi selesai")


# ============================================================
# CELL 4 — Verifikasi Dataset
# ============================================================
ann_dir   = os.path.join(KAGGLE_INPUT, 'annotations')
split_dir = os.path.join(KAGGLE_INPUT, 'crisismmd_datasplit_all', 'crisismmd_datasplit_all')
img_dir   = os.path.join(KAGGLE_INPUT, 'data_image')

print("=== Verifikasi Dataset ===")
print(f"\n📁 Annotations  : {len(os.listdir(ann_dir))} file")
print(f"📁 Split files  : {len(os.listdir(split_dir))} file")
print(f"📁 Image folder : {len(os.listdir(img_dir))} event")

# Cek split file per task
print("\nSplit files tersedia:")
for f in sorted(os.listdir(split_dir)):
    print(f"  {f}")

# Cek kolom anotasi
sample = pd.read_csv(os.path.join(ann_dir, os.listdir(ann_dir)[0]),
                     sep='\t', encoding='latin-1')
print(f"\nKolom anotasi: {sample.columns.tolist()}")

# Cek label damage (Task 3)
print(f"\nSample image_damage values: {sample['image_damage'].value_counts().to_dict()}")


# ============================================================
# CELL 5 — Load Annotations & Apply Split
# ============================================================
def load_data(task: str, split: str, subset_size=None):
    """
    Baca semua TSV annotations → filter label valid →
    cocokkan dengan split IDs → kembalikan DataFrame.
    """
    cfg        = TASK_CONFIG[task]
    label_col  = cfg['label_col']
    class_names= cfg['class_names']

    # Baca semua annotation TSV
    dfs = []
    for fname in os.listdir(ann_dir):
        if not fname.endswith('.tsv'):
            continue
        try:
            df = pd.read_csv(os.path.join(ann_dir, fname),
                             sep='\t', encoding='latin-1')
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Skip {fname}: {e}")
    combined = pd.concat(dfs, ignore_index=True)

    # Filter label valid
    combined = combined[combined[label_col].isin(class_names)].copy()

    # Drop NaN di label (penting untuk Task 3)
    combined = combined.dropna(subset=[label_col])

    # Baca split IDs
    split_file = os.path.join(
        split_dir, f"{cfg['split_prefix']}_{split}.tsv"
    )
    split_df = pd.read_csv(split_file, sep='\t', encoding='latin-1')

    # Kolom ID bisa 'image_id' atau kolom pertama
    id_col = 'image_id' if 'image_id' in split_df.columns else split_df.columns[0]
    split_ids = set(split_df[id_col].astype(str).tolist())

    # Filter berdasarkan split
    result = combined[combined['image_id'].astype(str).isin(split_ids)].copy()
    result = result.reset_index(drop=True)

    if subset_size:
        result = result.sample(n=min(subset_size, len(result)),
                               random_state=42).reset_index(drop=True)

    print(f"  [{task}/{split}] {len(result)} sampel | "
          f"Labels: {result[label_col].value_counts().to_dict()}")
    return result


# Load semua task dan split
print("Loading data untuk semua task...")
data_splits = {}
for task in TASK_CONFIG:
    data_splits[task] = {}
    for split in ['train', 'dev', 'test']:
        data_splits[task][split] = load_data(
            task, split, TRAIN_CONFIG['subset_size']
        )

print("\n✅ Semua data berhasil di-load")


# ============================================================
# CELL 6 — Dataset Class
# ============================================================
class CrisisMMDDataset(Dataset):
    def __init__(self, df, task, transform=None):
        self.df        = df.reset_index(drop=True)
        self.task      = task
        self.transform = transform
        cfg            = TASK_CONFIG[task]
        self.label_col = cfg['label_col']
        self.label_map = {name: i for i, name in enumerate(cfg['class_names'])}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img_path = os.path.join(KAGGLE_INPUT, str(row['image_path']))

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback: black image
            size  = 224
            image = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        label = self.label_map[row[self.label_col]]
        return image, label


# ============================================================
# CELL 7 — Transforms
# ============================================================
from torchvision import transforms

def get_transforms(input_size: int, is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.RandomErasing(p=0.2),   # tambahan v4
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ============================================================
# CELL 8 — DataLoader Factory
# ============================================================
def create_dataloaders(task: str, model_key: str):
    cfg        = MODEL_CONFIG[model_key]
    input_size = cfg['input_size']
    batch_size = cfg['batch_size']
    nw         = TRAIN_CONFIG['num_workers']

    loaders = {}
    for split in ['train', 'dev', 'test']:
        is_train  = (split == 'train')
        transform = get_transforms(input_size, is_train)
        dataset   = CrisisMMDDataset(
            data_splits[task][split], task, transform
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = is_train,
            num_workers = nw,
            pin_memory  = True,
        )
    return loaders


# ============================================================
# CELL 9 — Model Factory + Two-Stage Helper
# ============================================================
def create_model(model_key: str, num_classes: int, pretrained=True):
    """Buat model bersih — tanpa thop hooks."""
    name  = MODEL_CONFIG[model_key]['timm_name']
    model = timm.create_model(name, pretrained=pretrained,
                              num_classes=num_classes)
    model = model.to(device)
    return model


def freeze_backbone(model):
    """Stage 1: freeze semua kecuali classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze head (nama bervariasi per arsitektur)
    for head_attr in ['classifier', 'head', 'fc']:
        if hasattr(model, head_attr):
            for param in getattr(model, head_attr).parameters():
                param.requires_grad = True
            break

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ❄️  Frozen: {frozen/1e6:.1f}M  |  🔥 Trainable: {trainable/1e6:.1f}M")


def unfreeze_all(model):
    """Stage 2: unfreeze semua layer."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  🔥 Semua layer unfreeze: {total/1e6:.1f}M params")


def get_stage2_optimizer(model, model_key, weight_decay):
    """
    Optimizer Stage 2 dengan differential LR:
    backbone → lr kecil, head → lr lebih besar
    """
    cfg     = MODEL_CONFIG[model_key]
    lr_head = cfg['lr_stage2']
    lr_bb   = cfg['lr_backbone_stage2']

    head_params     = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(h in name for h in ['classifier', 'head', 'fc'])
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return optim.AdamW([
        {'params': backbone_params, 'lr': lr_bb},
        {'params': head_params,     'lr': lr_head},
    ], weight_decay=weight_decay)


# ============================================================
# CELL 10 — Training Utilities
# ============================================================
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):    self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, val_loss):
        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc'          : val_acc,
    }, path)


def load_checkpoint(model, path):
    """Load checkpoint dengan strict=False untuk skip thop keys."""
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(
        ckpt['model_state_dict'], strict=False
    )
    non_thop = [k for k in missing
                if 'total_ops' not in k and 'total_params' not in k]
    if non_thop:
        print(f"  ⚠️  Missing non-thop keys: {non_thop}")
    return ckpt.get('val_acc', 0.0)


# ============================================================
# CELL 11 — Train / Validate One Epoch
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        acc   = (preds == labels).float().mean().item()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc   = (preds == labels).float().mean().item()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))

    return loss_m.avg, acc_m.avg


# ============================================================
# CELL 12 — Two-Stage Training Loop
# ============================================================
def train_two_stage(model, model_key, task, loaders, ckpt_name):
    """
    Stage 1: Freeze backbone, train head saja (stage1_epochs)
    Stage 2: Unfreeze all, fine-tune dengan differential LR
    """
    cfg           = MODEL_CONFIG[model_key]
    s1_epochs     = TRAIN_CONFIG['stage1_epochs']
    s2_epochs     = TRAIN_CONFIG['stage2_epochs']
    patience      = TRAIN_CONFIG['early_stop_patience']
    wd            = TRAIN_CONFIG['weight_decay']
    num_classes   = TASK_CONFIG[task]['num_classes']

    criterion  = nn.CrossEntropyLoss(
        label_smoothing=TRAIN_CONFIG['label_smoothing']
    ).to(device)
    scaler     = GradScaler()
    ckpt_path  = os.path.join(CHECKPOINT_DIR, f'{ckpt_name}_best.pth')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'stage'     : [],  # 1 atau 2 per epoch
    }

    best_val_acc = 0.0
    es = EarlyStopping(patience=patience)

    # ── STAGE 1: Freeze backbone ─────────────────────────────
    print(f"\n{'='*55}")
    print(f"  STAGE 1 — Freeze Backbone ({s1_epochs} epoch)")
    print(f"{'='*55}")
    freeze_backbone(model)

    optimizer_s1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr_stage1'], weight_decay=wd
    )
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=s1_epochs, eta_min=1e-6
    )

    for epoch in range(1, s1_epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, loaders['train'], criterion, optimizer_s1, scaler
        )
        v_loss, v_acc = validate(model, loaders['dev'], criterion)
        scheduler_s1.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['stage'].append(1)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(model, optimizer_s1, epoch, v_acc, ckpt_path)

        print(f"  S1 Ep {epoch:02d}/{s1_epochs} | "
              f"Loss {t_loss:.4f}/{v_loss:.4f} | "
              f"Acc {t_acc:.4f}/{v_acc:.4f}")

    # ── STAGE 2: Unfreeze all ─────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  STAGE 2 — Unfreeze All (max {s2_epochs} epoch, "
          f"patience={patience})")
    print(f"{'='*55}")
    unfreeze_all(model)

    optimizer_s2 = get_stage2_optimizer(model, model_key, wd)
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=s2_epochs, eta_min=1e-7
    )
    es = EarlyStopping(patience=patience)

    for epoch in range(1, s2_epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, loaders['train'], criterion, optimizer_s2, scaler
        )
        v_loss, v_acc = validate(model, loaders['dev'], criterion)
        scheduler_s2.step()
        es(v_loss)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['stage'].append(2)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint(model, optimizer_s2, epoch, v_acc, ckpt_path)
            print(f"  ✅ Best saved: {v_acc:.4f}")

        total_ep = s1_epochs + epoch
        print(f"  S2 Ep {epoch:02d}/{s2_epochs} (Total {total_ep}) | "
              f"Loss {t_loss:.4f}/{v_loss:.4f} | "
              f"Acc {t_acc:.4f}/{v_acc:.4f}")

        if es.stop:
            print(f"  ⏹  Early stopping di epoch {epoch}")
            break

    print(f"\n  Best Val Acc: {best_val_acc:.4f}")
    return history, best_val_acc


# ============================================================
# CELL 13 — Plot Training History
# ============================================================
def plot_training_history(all_histories, task, save_path):
    """Plot loss, accuracy, LR, dan best val acc untuk 4 model."""
    colors = {
        'efficientnetv2_m': '#1f77b4',
        'vit'             : '#ff7f0e',
        'convnext'        : '#2ca02c',
        'swin'            : '#d62728',
    }
    labels = {
        'efficientnetv2_m': 'EfficientNetV2-M',
        'vit'             : 'ViT-B/16',
        'convnext'        : 'ConvNeXt-Base',
        'swin'            : 'Swin-Small',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training History — Task: {task}', fontsize=14)

    for key, hist in all_histories.items():
        c     = colors[key]
        label = labels[key]
        s1    = TRAIN_CONFIG['stage1_epochs']
        eps   = range(1, len(hist['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(eps, hist['train_loss'], c=c,       label=f'{label} Train')
        axes[0, 0].plot(eps, hist['val_loss'],   c=c, ls='--', label=f'{label} Val')
        # Accuracy
        axes[0, 1].plot(eps, hist['train_acc'],  c=c,       label=f'{label} Train')
        axes[0, 1].plot(eps, hist['val_acc'],    c=c, ls='--', label=f'{label} Val')

    # Garis pemisah Stage 1/2
    for ax in axes[0]:
        ax.axvline(x=s1 + 0.5, color='gray', ls=':', alpha=0.7,
                   label='Stage 1 → 2')

    axes[0, 0].set_title('Loss');     axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(fontsize=7)

    # Best val acc bar
    best_vals = {labels[k]: max(v['val_acc']) for k, v in all_histories.items()}
    ax3 = axes[1, 1]
    bars = ax3.bar(best_vals.keys(), best_vals.values(),
                   color=[colors[k] for k in all_histories])
    for bar, val in zip(bars, best_vals.values()):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax3.set_title('Best Val Accuracy per Model')
    ax3.set_ylim(0, 1.05)

    # Stage annotation
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5,
        f"Two-Stage Fine-Tuning\n"
        f"Stage 1: {TRAIN_CONFIG['stage1_epochs']} epoch (freeze backbone)\n"
        f"Stage 2: max {TRAIN_CONFIG['stage2_epochs']} epoch (unfreeze all)\n"
        f"Early Stopping patience: {TRAIN_CONFIG['early_stop_patience']}\n"
        f"Label Smoothing: {TRAIN_CONFIG['label_smoothing']}",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0'),
        transform=axes[1, 0].transAxes
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ============================================================
# CELL 14 — Evaluation Metrics (Expanded)
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, class_names):
    """
    Evaluasi model → kembalikan dict lengkap termasuk
    probabilitas untuk AUC-ROC.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_prob  = np.array(all_probs)
    n_class = len(class_names)

    # ── Metrik utama ─────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    p_wt,  r_wt,  f1_wt,  _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # ── Per-class F1 ─────────────────────────────────────────
    f1_per_class = f1_score(y_true, y_pred, average=None,
                            zero_division=0)

    # ── AUC-ROC ──────────────────────────────────────────────
    try:
        if n_class == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_prob,
                multi_class='ovr', average='macro'
            )
    except Exception:
        auc = float('nan')

    # ── Confusion matrix ─────────────────────────────────────
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names, zero_division=0
    )

    return {
        'accuracy'      : acc,
        'macro_precision': p_mac,
        'macro_recall'  : r_mac,
        'macro_f1'      : f1_mac,
        'weighted_f1'   : f1_wt,
        'auc_roc'       : auc,
        'f1_per_class'  : f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions'   : y_pred,
        'labels'        : y_true,
        'probabilities' : y_prob,
    }


def print_metrics(name, res):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"  Accuracy    : {res['accuracy']:.4f}")
    print(f"  Macro F1    : {res['macro_f1']:.4f}")
    print(f"  Weighted F1 : {res['weighted_f1']:.4f}")
    print(f"  AUC-ROC     : {res['auc_roc']:.4f}")


# ============================================================
# CELL 15 — Confusion Matrix & Per-Class F1 Plots
# ============================================================
def plot_confusion_matrix(cm, class_names, title, save_path):
    fig_size = max(8, len(class_names) * 1.5)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_per_class_f1(results_dict, class_names, title, save_path):
    """Bar chart per-class F1 untuk semua model."""
    labels = list(results_dict.keys())
    x      = np.arange(len(class_names))
    width  = 0.8 / len(labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(max(12, len(class_names)*2), 6))
    for i, (name, res) in enumerate(results_dict.items()):
        offset = (i - len(labels)/2 + 0.5) * width
        ax.bar(x + offset, res['f1_per_class'], width,
               label=name, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace('_', '\n') for c in class_names],
        fontsize=8
    )
    ax.set_ylabel('F1-Score')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ============================================================
# CELL 16 — Ensemble Functions
# ============================================================
def ensemble_predict(probs_dict, weights=None):
    """
    probs_dict: {model_name: np.array (N, C)}
    weights   : dict {model_name: float} atau None (simple avg)
    """
    probs_list = list(probs_dict.values())
    if weights is None:
        avg = np.mean(probs_list, axis=0)
    else:
        w   = np.array([weights[k] for k in probs_dict])
        w   = w / w.sum()
        avg = sum(p * wi for p, wi in zip(probs_list, w))
    return avg


def evaluate_ensemble(probs, y_true, class_names, name, save_dir):
    y_pred  = np.argmax(probs, axis=0) if probs.ndim == 3 else np.argmax(probs, axis=1)
    n_class = len(class_names)

    acc       = accuracy_score(y_true, y_pred)
    _, _, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    _, _, f1_wt, _  = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    try:
        if n_class == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs,
                                multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  {name}")
    print(f"  Acc={acc:.4f} | Macro F1={f1_mac:.4f} | "
          f"Weighted F1={f1_wt:.4f} | AUC={auc:.4f}")

    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plot_confusion_matrix(
        cm, class_names, f'{name} Confusion Matrix',
        os.path.join(save_dir, f'{safe_name}_cm.png')
    )

    return {
        'accuracy'   : acc,
        'macro_f1'   : f1_mac,
        'weighted_f1': f1_wt,
        'auc_roc'    : auc,
        'f1_per_class': f1_per,
    }


# ============================================================
# CELL 17 — Stacking (Logistic Regression)
# ============================================================
def run_stacking_lr(val_probs_dict, test_probs_dict,
                    y_val, y_test, class_names, save_dir):
    """
    Train Logistic Regression meta-model pada val probabilities
    → evaluasi pada test probabilities.
    """
    print("\n=== Stacking — Logistic Regression ===")

    # Feature matrix
    X_val  = np.hstack(list(val_probs_dict.values()))
    X_test = np.hstack(list(test_probs_dict.values()))

    print(f"  Feature shape: val={X_val.shape}, test={X_test.shape}")

    # Train meta-model
    meta = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        ))
    ])
    meta.fit(X_val, y_val)

    # Predict
    y_pred = meta.predict(X_test)
    y_prob = meta.predict_proba(X_test)

    # Metrics
    acc     = accuracy_score(y_test, y_pred)
    _, _, f1_mac, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    _, _, f1_wt, _  = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    f1_per  = f1_score(y_test, y_pred, average=None, zero_division=0)

    n_class = len(class_names)
    try:
        if n_class == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob,
                                multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_test, y_pred)

    print(f"  Acc={acc:.4f} | Macro F1={f1_mac:.4f} | "
          f"Weighted F1={f1_wt:.4f} | AUC={auc:.4f}")

    plot_confusion_matrix(
        cm, class_names,
        'Stacking (Logistic Regression) Confusion Matrix',
        os.path.join(save_dir, 'stacking_lr_cm.png')
    )

    # Feature importance
    coef = meta.named_steps['lr'].coef_
    if coef.shape[0] == 1:
        coef = np.vstack([-coef, coef])
    mean_abs = np.abs(coef).mean(axis=0)

    feature_names = []
    for mname in val_probs_dict:
        for cn in class_names:
            feature_names.append(f'{mname}\n{cn}')

    plt.figure(figsize=(max(12, len(feature_names)*0.6), 5))
    colors = ['#d62728' if v < 0 else '#1f77b4'
              for v in mean_abs]
    plt.bar(range(len(mean_abs)), mean_abs, color=colors)
    plt.xticks(range(len(mean_abs)), feature_names,
               rotation=45, ha='right', fontsize=7)
    plt.title('Stacking LR — Feature Importance (Mean |Coef|)')
    plt.tight_layout()
    fi_path = os.path.join(save_dir, 'stacking_lr_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fi_path}")

    return {
        'accuracy'   : acc,
        'macro_f1'   : f1_mac,
        'weighted_f1': f1_wt,
        'auc_roc'    : auc,
        'f1_per_class': f1_per,
    }


# ============================================================
# CELL 18 — Final Summary & Ranking
# ============================================================
def build_summary_table(single_results, ensemble_results,
                        stacking_result, class_names, task):
    rows = []

    # Single models
    model_display = {
        'efficientnetv2_m': 'EfficientNetV2-M',
        'vit'             : 'ViT-B/16',
        'convnext'        : 'ConvNeXt-Base',
        'swin'            : 'Swin-Small',
    }
    for key, res in single_results.items():
        rows.append({
            'Model'      : model_display.get(key, key),
            'Type'       : 'Single',
            'Accuracy'   : res['accuracy'],
            'Macro F1'   : res['macro_f1'],
            'Weighted F1': res['weighted_f1'],
            'AUC-ROC'    : res['auc_roc'],
        })

    # Ensemble
    for name, res in ensemble_results.items():
        rows.append({
            'Model'      : name,
            'Type'       : 'Ensemble',
            'Accuracy'   : res['accuracy'],
            'Macro F1'   : res['macro_f1'],
            'Weighted F1': res['weighted_f1'],
            'AUC-ROC'    : res['auc_roc'],
        })

    # Stacking
    rows.append({
        'Model'      : 'Stacking (Logistic Reg)',
        'Type'       : 'Stacking',
        'Accuracy'   : stacking_result['accuracy'],
        'Macro F1'   : stacking_result['macro_f1'],
        'Weighted F1': stacking_result['weighted_f1'],
        'AUC-ROC'    : stacking_result['auc_roc'],
    })

    df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False)
    df = df.reset_index(drop=True)

    # Print
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Task: {task.upper()}")
    print(f"{'='*75}")
    print(df.to_string(index=False, float_format='{:.4f}'.format))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f'summary_{task}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Ranking bar chart
    colors_map = {'Single': '#1f77b4', 'Ensemble': '#2ca02c',
                  'Stacking': '#9467bd'}
    bar_colors = [colors_map[t] for t in df['Type']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy ranking
    bars = axes[0].barh(df['Model'], df['Accuracy'],
                        color=bar_colors, edgecolor='white')
    for bar, val in zip(bars, df['Accuracy']):
        axes[0].text(val + 0.001, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title(f'Accuracy Ranking — {task}')
    axes[0].set_xlim(0, 1.05)

    legend_patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in colors_map.items()
    ]
    axes[0].legend(handles=legend_patches, loc='lower right')

    # Macro F1 vs Weighted F1
    x    = np.arange(len(df))
    w    = 0.35
    axes[1].bar(x - w/2, df['Macro F1'],    w, label='Macro F1',
                color='#1f77b4', alpha=0.85)
    axes[1].bar(x + w/2, df['Weighted F1'], w, label='Weighted F1',
                color='#ff7f0e', alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('F1-Score')
    axes[1].set_title(f'Macro vs Weighted F1 — {task}')
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'final_comparison_{task}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fig_path}")

    return df


# ============================================================
# CELL 19 — Master Runner: Satu Task Lengkap
# ============================================================
def run_task(task: str):
    """
    Jalankan full pipeline untuk satu task:
    training → evaluasi → ensemble → stacking → summary
    """
    print(f"\n{'#'*60}")
    print(f"  TASK: {task.upper()}")
    print(f"{'#'*60}")

    class_names = TASK_CONFIG[task]['class_names']
    num_classes = TASK_CONFIG[task]['num_classes']
    save_dir    = os.path.join(RESULTS_DIR, task)
    os.makedirs(save_dir, exist_ok=True)

    model_keys    = ['efficientnetv2_m', 'vit', 'convnext', 'swin']
    all_histories = {}
    all_val_accs  = {}

    # ── Training semua model ─────────────────────────────────
    for mkey in model_keys:
        print(f"\n{'─'*55}")
        print(f"  Model: {mkey.upper()}")
        print(f"{'─'*55}")

        loaders   = create_dataloaders(task, mkey)
        model     = create_model(mkey, num_classes, pretrained=True)
        ckpt_name = f'{mkey}_{task}'

        hist, best_val = train_two_stage(
            model, mkey, task, loaders, ckpt_name
        )
        all_histories[mkey] = hist
        all_val_accs[mkey]  = best_val

        del model
        torch.cuda.empty_cache()

    # ── Plot training history ─────────────────────────────────
    plot_training_history(
        all_histories, task,
        os.path.join(save_dir, 'training_history.png')
    )

    # ── Load best checkpoint & evaluate ──────────────────────
    print(f"\n=== Evaluasi pada Test Set ===")
    single_results = {}
    test_probs     = {}
    val_probs      = {}
    y_test         = None
    y_val          = None

    for mkey in model_keys:
        model = create_model(mkey, num_classes, pretrained=False)
        load_checkpoint(
            model,
            os.path.join(CHECKPOINT_DIR, f'{mkey}_{task}_best.pth')
        )

        loaders = create_dataloaders(task, mkey)

        # Evaluasi test
        res = evaluate_model(model, loaders['test'], class_names)
        single_results[mkey] = res
        test_probs[mkey]     = res['probabilities']
        if y_test is None:
            y_test = res['labels']

        # Kumpulkan val probs untuk stacking
        res_val = evaluate_model(model, loaders['dev'], class_names)
        val_probs[mkey] = res_val['probabilities']
        if y_val is None:
            y_val = res_val['labels']

        print_metrics(mkey, res)
        plot_confusion_matrix(
            res['confusion_matrix'], class_names,
            f'{mkey} Confusion Matrix',
            os.path.join(save_dir, f'{mkey}_cm.png')
        )

        del model
        torch.cuda.empty_cache()

    # ── Per-class F1 comparison ───────────────────────────────
    display_names = {
        'efficientnetv2_m': 'EfficientNetV2-M',
        'vit'             : 'ViT-B/16',
        'convnext'        : 'ConvNeXt-Base',
        'swin'            : 'Swin-Small',
    }
    plot_per_class_f1(
        {display_names[k]: v for k, v in single_results.items()},
        class_names,
        f'Per-Class F1 Comparison — {task}',
        os.path.join(save_dir, 'per_class_f1.png')
    )

    # ── Ensemble ──────────────────────────────────────────────
    print(f"\n=== Ensemble ===")
    ensemble_results = {}

    # Simple averaging
    probs_simple = ensemble_predict(test_probs)
    ensemble_results['Ensemble Simple (4)'] = evaluate_ensemble(
        probs_simple, y_test, class_names,
        'Ensemble Simple (4)', save_dir
    )

    # Weighted (by val acc)
    probs_weighted = ensemble_predict(test_probs, weights=all_val_accs)
    ensemble_results['Ensemble Weighted (4)'] = evaluate_ensemble(
        probs_weighted, y_test, class_names,
        'Ensemble Weighted (4)', save_dir
    )

    # Best-3 (drop model dengan val_acc terendah)
    worst_model = min(all_val_accs, key=all_val_accs.get)
    best3_probs = {k: v for k, v in test_probs.items() if k != worst_model}
    best3_wts   = {k: v for k, v in all_val_accs.items() if k != worst_model}
    probs_best3 = ensemble_predict(best3_probs, weights=best3_wts)
    ensemble_results[f'Ensemble Best-3 (drop {worst_model})'] = evaluate_ensemble(
        probs_best3, y_test, class_names,
        f'Ensemble Best-3 (drop {worst_model})', save_dir
    )

    # ── Stacking LR ───────────────────────────────────────────
    stacking_result = run_stacking_lr(
        val_probs, test_probs,
        y_val, y_test,
        class_names, save_dir
    )

    # ── Final Summary ─────────────────────────────────────────
    summary_df = build_summary_table(
        single_results, ensemble_results,
        stacking_result, class_names, task
    )

    # Save JSON
    all_results = {
        'task'    : task,
        'single'  : {k: {m: float(v) for m, v in {
            'accuracy'   : r['accuracy'],
            'macro_f1'   : r['macro_f1'],
            'weighted_f1': r['weighted_f1'],
            'auc_roc'    : r['auc_roc'],
        }.items()} for k, r in single_results.items()},
        'ensemble': {k: {m: float(v) for m, v in r.items()
                         if m not in ['f1_per_class']}
                     for k, r in ensemble_results.items()},
        'stacking': {m: float(v) for m, v in stacking_result.items()
                     if m not in ['f1_per_class']},
    }
    json_path = os.path.join(save_dir, f'results_{task}.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")

    return summary_df


# ============================================================
# CELL 20 — Jalankan Task 1: Informative
# ============================================================
summary_informative = run_task('informative')


# ============================================================
# CELL 21 — Jalankan Task 2: Humanitarian
# ============================================================
summary_humanitarian = run_task('humanitarian')


# ============================================================
# CELL 22 — Jalankan Task 3: Damage Severity
# ============================================================
summary_damage = run_task('damage')


# ============================================================
# CELL 23 — Ringkasan Lintas Task
# ============================================================
def print_cross_task_summary():
    print(f"\n{'='*70}")
    print("  CROSS-TASK SUMMARY")
    print(f"{'='*70}")

    for task in ['informative', 'humanitarian', 'damage']:
        csv_path = os.path.join(RESULTS_DIR, f'summary_{task}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        best = df.iloc[0]
        print(f"\n  Task: {task.upper()}")
        print(f"  Best model : {best['Model']}")
        print(f"  Accuracy   : {best['Accuracy']:.4f}")
        print(f"  Macro F1   : {best['Macro F1']:.4f}")
        print(f"  Weighted F1: {best['Weighted F1']:.4f}")
        print(f"  AUC-ROC    : {best['AUC-ROC']:.4f}")

    # Gabungkan semua task ke satu CSV
    all_dfs = []
    for task in ['informative', 'humanitarian', 'damage']:
        csv_path = os.path.join(RESULTS_DIR, f'summary_{task}.csv')
        if os.path.exists(csv_path):
            df       = pd.read_csv(csv_path)
            df['Task'] = task
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(RESULTS_DIR, 'summary_all_tasks.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined: {combined_path}")

print_cross_task_summary()
print("\n✅ Semua eksperimen selesai!")
