# ============================================================
# CrisisMMD v5.2 — Notebook 1: Ablation Training
# Pipeline  : Two-Stage Fine-Tuning dengan flag ablation
# Perubahan : Focal Loss stage 2 humanitarian (γ=1.0)
#             MixUp khusus Task 3 (damage)
#             Weighted CE kedua stage untuk damage
#             LR backbone diperkecil (anti-spike)
# Output    : probs (.npy) + metrics (.json) + PNG
#             TIDAK menyimpan checkpoint (.pth)
# ============================================================


# ============================================================
# CELL 1 — Install Library
# ============================================================
import subprocess
subprocess.run(['pip', 'install', 'timm', '--quiet'], check=True)
print("✅ Library installed")


# ============================================================
# CELL 2 — Import
# ============================================================
import os
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")


# ============================================================
# CELL 3 — ABLATION CONFIG
# ============================================================
# ┌──────────────────────────────────────────────────────────┐
# │  UBAH HANYA BAGIAN INI UNTUK SETIAP VARIANT             │
# │                                                          │
# │  V1 Full Proposed  : default di bawah ini               │
# │  V2 w/o Two-Stage  : use_two_stage   = False            │
# │  V3 w/o Focal Loss : use_focal_loss  = False            │
# │  V4 w/o Merge Kelas: use_merge_kelas = False            │
# │  V5 w/o Weighted CE: use_weighted_ce = False            │
# │  V6 w/o MixUp      : use_mixup       = False            │
# │  V7 + Augmentasi   : use_augmentation= True             │
# └──────────────────────────────────────────────────────────┘

ABLATION_CONFIG = {
    'use_two_stage'   : True,   # True  = two-stage (frozen→unfreeze)
                                # False = full fine-tuning langsung
    'use_focal_loss'  : True,   # True  = Focal Loss humanitarian kedua stage
                                # False = Standard CE humanitarian kedua stage
    'use_merge_kelas' : True,   # True  = humanitarian 5 kelas (merged)
                                # False = humanitarian 8 kelas (original)
    'use_weighted_ce' : True,   # True  = Weighted CE damage kedua stage
                                # False = Standard CE damage kedua stage
    'use_mixup'       : True,   # True  = MixUp aktif saat training damage
                                # False = tanpa MixUp
    'use_augmentation': False,  # True  = augmentasi aktif semua task
                                # False = tanpa augmentasi
}

# ── Auto-generate nama variant ────────────────────────────
def get_variant_name(cfg):
    if (cfg['use_two_stage'] and cfg['use_focal_loss'] and
            cfg['use_merge_kelas'] and cfg['use_weighted_ce'] and
            cfg['use_mixup'] and cfg['use_augmentation']):
        return 'aug_proposed'
    if (cfg['use_two_stage'] and cfg['use_focal_loss'] and
            cfg['use_merge_kelas'] and cfg['use_weighted_ce'] and
            cfg['use_mixup'] and not cfg['use_augmentation']):
        return 'full_proposed'
    if not cfg['use_two_stage']:
        return 'wo_twostage'
    if not cfg['use_focal_loss']:
        return 'wo_focal'
    if not cfg['use_merge_kelas']:
        return 'wo_merge'
    if not cfg['use_weighted_ce']:
        return 'wo_weightedce'
    if not cfg['use_mixup']:
        return 'wo_mixup'
    return 'custom_variant'

VARIANT_NAME = get_variant_name(ABLATION_CONFIG)
print(f"\n{'='*55}")
print(f"  VARIANT AKTIF : {VARIANT_NAME.upper()}")
print(f"{'='*55}")
for k, v in ABLATION_CONFIG.items():
    print(f"  {'✅' if v else '❌'} {k}")
print(f"{'='*55}\n")


# ============================================================
# CELL 4 — Konfigurasi Global
# ============================================================
KAGGLE_INPUT   = '/kaggle/input/datasets/alieffathurrahman/crisismmd'
CHECKPOINT_DIR = f'/kaggle/working/checkpoints_{VARIANT_NAME}'
OUTPUT_DIR     = f'/kaggle/working/outputs_{VARIANT_NAME}'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device     : {device}")
print(f"Output dir : {OUTPUT_DIR}")

MODEL_CONFIG = {
    'efficientnetv2_m': {
        'timm_name'          : 'tf_efficientnetv2_m',
        'input_size'         : 224,
        'batch_size'         : 16,
        'lr_stage1'          : 5e-4,
        'lr_stage2_head'     : 5e-5,
        'lr_stage2_backbone' : 1e-6,
        'lr_uniform'         : 5e-5,
    },
    'convnext': {
        'timm_name'          : 'convnext_base',
        'input_size'         : 224,
        'batch_size'         : 16,
        'lr_stage1'          : 5e-4,
        'lr_stage2_head'     : 5e-5,
        'lr_stage2_backbone' : 1e-6,
        'lr_uniform'         : 5e-5,
    },
    'swin': {
        'timm_name'          : 'swin_small_patch4_window7_224',
        'input_size'         : 224,
        'batch_size'         : 16,
        'lr_stage1'          : 5e-4,
        'lr_stage2_head'     : 5e-5,
        'lr_stage2_backbone' : 5e-7,
        'lr_uniform'         : 5e-5,
    },
    'vit': {
        'timm_name'          : 'vit_base_patch16_384',
        'input_size'         : 384,
        'batch_size'         : 8,
        'lr_stage1'          : 1e-3,
        'lr_stage2_head'     : 5e-5,
        'lr_stage2_backbone' : 1e-6,
        'lr_uniform'         : 5e-5,
    },
}

TRAIN_CONFIG = {
    'stage1_epochs'       : 10,
    'stage2_epochs'       : 40,
    'total_epochs'        : 50,
    'early_stop_patience' : 5,
    'weight_decay'        : 0.01,
    'label_smoothing'     : 0.1,
    'focal_gamma_stage1'  : 2.0,   # humanitarian stage 1
    'focal_gamma_stage2'  : 1.0,   # humanitarian stage 2 (lebih halus)
    'mixup_alpha'         : 0.4,   # damage MixUp alpha
    'num_workers'         : 2,
    'seed'                : 42,
}

MODEL_DISPLAY = {
    'efficientnetv2_m': 'EfficientNetV2-M',
    'convnext'        : 'ConvNeXt-Base',
    'swin'            : 'Swin-Small',
    'vit'             : 'ViT-B/16',
}

torch.manual_seed(TRAIN_CONFIG['seed'])
np.random.seed(TRAIN_CONFIG['seed'])
print("✅ Konfigurasi selesai")


# ============================================================
# CELL 5 — Task Config (Conditional Merge Kelas)
# ============================================================
if ABLATION_CONFIG['use_merge_kelas']:
    HUMANITARIAN_CONFIG = {
        'num_classes': 5,
        'class_names': [
            'not_humanitarian',
            'infrastructure_and_utility_damage',
            'other_relevant_information',
            'rescue_volunteering_or_donation_effort',
            'direct_human_impact',
        ],
        'merge_map': {
            'not_humanitarian'                       : 'not_humanitarian',
            'infrastructure_and_utility_damage'      : 'infrastructure_and_utility_damage',
            'other_relevant_information'             : 'other_relevant_information',
            'rescue_volunteering_or_donation_effort' : 'rescue_volunteering_or_donation_effort',
            'affected_individuals'                   : 'direct_human_impact',
            'vehicle_damage'                         : 'direct_human_impact',
            'injured_or_dead_people'                 : 'direct_human_impact',
            'missing_or_found_people'                : 'direct_human_impact',
        },
    }
    print("  Humanitarian: 5 kelas (merge aktif)")
else:
    HUMANITARIAN_CONFIG = {
        'num_classes': 8,
        'class_names': [
            'not_humanitarian',
            'infrastructure_and_utility_damage',
            'other_relevant_information',
            'rescue_volunteering_or_donation_effort',
            'affected_individuals',
            'vehicle_damage',
            'injured_or_dead_people',
            'missing_or_found_people',
        ],
        'merge_map': None,
    }
    print("  Humanitarian: 8 kelas (original)")

TASK_CONFIG = {
    'informative': {
        'num_classes' : 2,
        'label_col'   : 'image_info',
        'split_prefix': 'task_informative_text_img',
        'class_names' : ['not_informative', 'informative'],
        'merge_map'   : None,
    },
    'humanitarian': {
        'num_classes' : HUMANITARIAN_CONFIG['num_classes'],
        'label_col'   : 'image_human',
        'split_prefix': 'task_humanitarian_text_img',
        'class_names' : HUMANITARIAN_CONFIG['class_names'],
        'merge_map'   : HUMANITARIAN_CONFIG['merge_map'],
    },
    'damage': {
        'num_classes' : 3,
        'label_col'   : 'image_damage',
        'split_prefix': 'task_damage_text_img',
        'class_names' : ['little_or_no_damage', 'mild_damage', 'severe_damage'],
        'merge_map'   : None,
    },
}


# ============================================================
# CELL 6 — Load Data
# ============================================================
ann_dir   = os.path.join(KAGGLE_INPUT, 'annotations')
split_dir = os.path.join(KAGGLE_INPUT, 'crisismmd_datasplit_all',
                         'crisismmd_datasplit_all')

def load_data(task: str, split: str):
    cfg         = TASK_CONFIG[task]
    label_col   = cfg['label_col']
    class_names = cfg['class_names']
    merge_map   = cfg.get('merge_map', None)

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
    combined = combined.dropna(subset=[label_col])
    if merge_map:
        combined[label_col] = combined[label_col].map(merge_map)
        combined = combined.dropna(subset=[label_col])
    combined = combined[combined[label_col].isin(class_names)].copy()

    split_file = os.path.join(
        split_dir, f"{cfg['split_prefix']}_{split}.tsv"
    )
    split_df  = pd.read_csv(split_file, sep='\t', encoding='latin-1')
    id_col    = 'image_id' if 'image_id' in split_df.columns \
                else split_df.columns[0]
    split_ids = set(split_df[id_col].astype(str).tolist())
    result    = combined[
        combined['image_id'].astype(str).isin(split_ids)
    ].reset_index(drop=True)

    print(f"  [{task}/{split}] {len(result)} sampel")
    return result


print("Loading data...")
data_splits = {}
for task in TASK_CONFIG:
    data_splits[task] = {}
    for split in ['train', 'dev', 'test']:
        data_splits[task][split] = load_data(task, split)
print("✅ Data loaded")


# ============================================================
# CELL 7 — Dataset & Transforms
# ============================================================
class CrisisMMDDataset(Dataset):
    def __init__(self, df, task, transform=None):
        self.df        = df.reset_index(drop=True)
        self.task      = task
        self.transform = transform
        cfg            = TASK_CONFIG[task]
        self.label_col = cfg['label_col']
        self.label_map = {name: i for i, name
                          in enumerate(cfg['class_names'])}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(KAGGLE_INPUT, str(row['image_path']))
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        label = self.label_map[row[self.label_col]]
        return image, label


def get_transforms(input_size: int, is_train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train and ABLATION_CONFIG['use_augmentation']:
        print(f"  [Transform] Augmentasi aktif (input_size={input_size})")
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def create_dataloaders(task: str, model_key: str):
    cfg        = MODEL_CONFIG[model_key]
    input_size = cfg['input_size']
    batch_size = cfg['batch_size']
    nw         = TRAIN_CONFIG['num_workers']

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(TRAIN_CONFIG['seed'])

    loaders = {}
    for split in ['train', 'dev', 'test']:
        is_train  = (split == 'train')
        transform = get_transforms(input_size, is_train)
        dataset   = CrisisMMDDataset(
            data_splits[task][split], task, transform
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size      = batch_size,
            shuffle         = is_train,
            num_workers     = nw,
            pin_memory      = True,
            worker_init_fn  = seed_worker,
            generator       = g if is_train else None,
        )
    return loaders


# ============================================================
# CELL 8 — Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al. (2017).
    Digunakan untuk humanitarian task kedua stage dengan gamma berbeda:
    - Stage 1: gamma=2.0 (agresif, fokus pada minority)
    - Stage 2: gamma=1.0 (halus, fine-tuning)
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_cls = inputs.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(inputs,
                                         self.label_smoothing / (n_cls - 1))
                smooth.scatter_(1, targets.unsqueeze(1),
                                1.0 - self.label_smoothing)
            log_prob = F.log_softmax(inputs, dim=1)
            ce_loss  = -(smooth * log_prob).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def get_weighted_ce(task, loader_train, label_smoothing):
    """Hitung class weights dari distribusi training set."""
    all_labels  = [lbl for _, lbl in loader_train.dataset]
    all_labels  = np.array(all_labels)
    num_classes = TASK_CONFIG[task]['num_classes']
    counts      = np.bincount(all_labels, minlength=num_classes).astype(float)
    counts      = np.where(counts == 0, 1, counts)
    weights     = 1.0 / counts
    weights     = weights / weights.min()
    weights     = np.clip(weights, 1.0, 10.0)
    w_tensor    = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"  Class weights [{task}]: "
          f"{ {i: round(w,2) for i,w in enumerate(weights)} }")
    return nn.CrossEntropyLoss(weight=w_tensor,
                               label_smoothing=label_smoothing).to(device)


def get_criterion(task: str, stage: int, loader_train=None):
    """
    Loss function per task per stage:

    Task 1 (informative):
      - Stage 0/1/2 : Standard CE (distribusi seimbang)

    Task 2 (humanitarian):
      - Stage 0/1   : Focal Loss γ=2.0 (jika use_focal_loss)
      - Stage 2     : Focal Loss γ=1.0 (jika use_focal_loss, lebih halus)
      - Else        : Standard CE

    Task 3 (damage):
      - Stage 0/1/2 : Weighted CE (jika use_weighted_ce)
      - Else        : Standard CE
      Note: MixUp dihandle di train_one_epoch_mixup, bukan di sini.
            Criterion tetap Weighted CE, MixUp mengubah cara loss dihitung.

    stage=0 → wo_twostage (full fine-tuning, perlakukan seperti stage 1)
    """
    ls = TRAIN_CONFIG['label_smoothing']

    # ── Task 1: selalu Standard CE ────────────────────────────
    if task == 'informative':
        print(f"  [Stage {stage}] Standard CE — informative")
        return nn.CrossEntropyLoss(label_smoothing=ls).to(device)

    # ── Task 2: Focal Loss conditional ───────────────────────
    if task == 'humanitarian':
        if ABLATION_CONFIG['use_focal_loss']:
            if stage == 2:
                gamma = TRAIN_CONFIG['focal_gamma_stage2']
                print(f"  [Stage 2] Focal Loss (γ={gamma}) — humanitarian")
            else:
                # stage 0 (wo_twostage) dan stage 1 pakai gamma tinggi
                gamma = TRAIN_CONFIG['focal_gamma_stage1']
                print(f"  [Stage {stage}] Focal Loss (γ={gamma}) — humanitarian")
            return FocalLoss(gamma=gamma, label_smoothing=ls).to(device)
        else:
            print(f"  [Stage {stage}] Standard CE — humanitarian")
            return nn.CrossEntropyLoss(label_smoothing=ls).to(device)

    # ── Task 3: Weighted CE conditional ──────────────────────
    if task == 'damage':
        if ABLATION_CONFIG['use_weighted_ce'] and loader_train is not None:
            print(f"  [Stage {stage}] Weighted CE — damage")
            return get_weighted_ce(task, loader_train, ls)
        else:
            print(f"  [Stage {stage}] Standard CE — damage")
            return nn.CrossEntropyLoss(label_smoothing=ls).to(device)

    return nn.CrossEntropyLoss(label_smoothing=ls).to(device)


# ============================================================
# CELL 9 — MixUp (khusus Task 3 / Damage)
# ============================================================
def mixup_data(x, y, alpha=0.4):
    """
    MixUp augmentation — Zhang et al. (2018).
    Menciptakan soft boundary antar kelas melalui interpolasi linear
    di input space. Khusus Task 3 untuk mengatasi boundary ambiguity
    antara kelas Mild Damage dan tetangganya.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam     = max(lam, 1 - lam)   # pastikan lam >= 0.5
    idx     = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss untuk mixed samples — kombinasi loss kedua label."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# CELL 10 — Model Helpers
# ============================================================
def create_model(model_key: str, num_classes: int, pretrained=True):
    name  = MODEL_CONFIG[model_key]['timm_name']
    model = timm.create_model(name, pretrained=pretrained,
                              num_classes=num_classes)
    return model.to(device)


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for head_attr in ['classifier', 'head', 'fc']:
        if hasattr(model, head_attr):
            for param in getattr(model, head_attr).parameters():
                param.requires_grad = True
            break
    frozen    = sum(p.numel() for p in model.parameters()
                    if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  ❄️  Frozen: {frozen/1e6:.1f}M | 🔥 Trainable: {trainable/1e6:.1f}M")


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  🔥 Unfreeze all: {total/1e6:.1f}M params")


def get_stage2_optimizer(model, model_key):
    cfg         = MODEL_CONFIG[model_key]
    lr_head     = cfg['lr_stage2_head']
    lr_bb       = cfg['lr_stage2_backbone']
    wd          = TRAIN_CONFIG['weight_decay']
    head_params, bb_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in ['classifier', 'head', 'fc']):
            head_params.append(param)
        else:
            bb_params.append(param)
    return optim.AdamW([
        {'params': bb_params,   'lr': lr_bb},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=wd)


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc'             : val_acc,
    }, path)


def load_checkpoint(model, path):
    ckpt       = torch.load(path, map_location=device)
    missing, _ = model.load_state_dict(
        ckpt['model_state_dict'], strict=False
    )
    non_meta = [k for k in missing
                if 'total_ops' not in k and 'total_params' not in k]
    if non_meta:
        print(f"  ⚠️  Missing keys: {non_meta}")
    return ckpt.get('val_acc', 0.0)


# ============================================================
# CELL 11 — Training Utilities
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

    def __call__(self, val_acc):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score = val_acc
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    task='informative'):
    """
    Training loop dengan conditional MixUp.
    MixUp hanya aktif jika:
    - task == 'damage' (boundary ambiguity problem)
    - ABLATION_CONFIG['use_mixup'] == True
    """
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    use_mixup     = (task == 'damage' and ABLATION_CONFIG['use_mixup'])

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_mixup:
            mixed_images, y_a, y_b, lam = mixup_data(
                images, labels, alpha=TRAIN_CONFIG['mixup_alpha']
            )
            with autocast():
                outputs = model(mixed_images)
                loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            # Accuracy dihitung dari prediksi label dominan
            preds = outputs.argmax(dim=1)
            acc   = (lam * (preds == y_a).float() +
                     (1 - lam) * (preds == y_b).float()).mean().item()
        else:
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            acc   = (preds == labels).float().mean().item()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion):
    """Validasi TANPA MixUp — evaluasi selalu pada data asli."""
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
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
# CELL 12 — Training Pipeline (Conditional Two-Stage)
# ============================================================
def train_model(model, model_key, task, loaders, ckpt_name):
    """
    Conditional training berdasarkan ABLATION_CONFIG:
    - use_two_stage=True  → Stage 1 (frozen) + Stage 2 (unfreeze)
    - use_two_stage=False → Full fine-tuning langsung

    Loss per task per stage ditentukan oleh get_criterion().
    MixUp dihandle di train_one_epoch() berdasarkan task + flag.
    """
    cfg      = MODEL_CONFIG[model_key]
    wd       = TRAIN_CONFIG['weight_decay']
    scaler   = GradScaler()
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{ckpt_name}_best.pth')
    history  = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : [],
        'stage'     : [],
    }
    best_val_acc = 0.0

    if ABLATION_CONFIG['use_two_stage']:
        s1_ep = TRAIN_CONFIG['stage1_epochs']
        s2_ep = TRAIN_CONFIG['stage2_epochs']

        # ── Stage 1: Frozen backbone ──────────────────────────
        print(f"\n{'='*55}")
        print(f"  STAGE 1 — Freeze Backbone ({s1_ep} epoch)")
        print(f"{'='*55}")
        freeze_backbone(model)
        crit_s1 = get_criterion(task, stage=1,
                                loader_train=loaders['train'])
        opt_s1  = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr_stage1'], weight_decay=wd
        )
        sch_s1 = optim.lr_scheduler.CosineAnnealingLR(
            opt_s1, T_max=s1_ep, eta_min=1e-6
        )
        for epoch in range(1, s1_ep + 1):
            t_loss, t_acc = train_one_epoch(
                model, loaders['train'], crit_s1, opt_s1, scaler, task)
            v_loss, v_acc = validate(model, loaders['dev'], crit_s1)
            sch_s1.step()
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)
            history['stage'].append(1)
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                save_checkpoint(model, opt_s1, epoch, v_acc, ckpt_path)
            print(f"  S1 Ep {epoch:02d}/{s1_ep} | "
                  f"Loss {t_loss:.4f}/{v_loss:.4f} | "
                  f"Acc {t_acc:.4f}/{v_acc:.4f}")

        # ── Stage 2: Full unfreeze ─────────────────────────────
        print(f"\n{'='*55}")
        print(f"  STAGE 2 — Unfreeze All (max {s2_ep} epoch)")
        print(f"{'='*55}")
        unfreeze_all(model)
        crit_s2 = get_criterion(task, stage=2,
                                loader_train=loaders['train'])
        opt_s2  = get_stage2_optimizer(model, model_key)
        sch_s2  = optim.lr_scheduler.CosineAnnealingLR(
            opt_s2, T_max=s2_ep, eta_min=1e-7
        )
        es = EarlyStopping(patience=TRAIN_CONFIG['early_stop_patience'])
        for epoch in range(1, s2_ep + 1):
            t_loss, t_acc = train_one_epoch(
                model, loaders['train'], crit_s2, opt_s2, scaler, task)
            v_loss, v_acc = validate(model, loaders['dev'], crit_s2)
            sch_s2.step()
            es(v_acc)
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)
            history['stage'].append(2)
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                save_checkpoint(model, opt_s2, epoch, v_acc, ckpt_path)
                print(f"  ✅ Best: {v_acc:.4f}")
            total_ep = s1_ep + epoch
            print(f"  S2 Ep {epoch:02d}/{s2_ep} (Total {total_ep}) | "
                  f"Loss {t_loss:.4f}/{v_loss:.4f} | "
                  f"Acc {t_acc:.4f}/{v_acc:.4f}")
            if es.stop:
                print(f"  ⏹  Early stopping epoch {total_ep}")
                break

    else:
        # ── Full Fine-Tuning (wo_twostage) ────────────────────
        total_ep = TRAIN_CONFIG['total_epochs']
        print(f"\n{'='*55}")
        print(f"  FULL FINE-TUNING — max {total_ep} epoch")
        print(f"  [wo_twostage] Focal/Weighted CE tetap aktif")
        print(f"{'='*55}")
        # stage=0 diperlakukan seperti stage 1 di get_criterion
        crit = get_criterion(task, stage=0,
                             loader_train=loaders['train'])
        opt  = optim.AdamW(model.parameters(),
                           lr=cfg['lr_uniform'], weight_decay=wd)
        sch  = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_ep, eta_min=1e-7
        )
        es = EarlyStopping(patience=TRAIN_CONFIG['early_stop_patience'])
        for epoch in range(1, total_ep + 1):
            t_loss, t_acc = train_one_epoch(
                model, loaders['train'], crit, opt, scaler, task)
            v_loss, v_acc = validate(model, loaders['dev'], crit)
            sch.step()
            es(v_acc)
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)
            history['stage'].append(0)
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                save_checkpoint(model, opt, epoch, v_acc, ckpt_path)
                print(f"  ✅ Best: {v_acc:.4f}")
            print(f"  Ep {epoch:02d}/{total_ep} | "
                  f"Loss {t_loss:.4f}/{v_loss:.4f} | "
                  f"Acc {t_acc:.4f}/{v_acc:.4f}")
            if es.stop:
                print(f"  ⏹  Early stopping epoch {epoch}")
                break

    print(f"\n  Best Val Acc: {best_val_acc:.4f}")
    return history, best_val_acc


# ============================================================
# CELL 13 — Evaluate & Save Probabilities
# ============================================================
@torch.no_grad()
def evaluate_and_save(model, loader, class_names, save_prefix, split_name):
    """Evaluasi TANPA MixUp — selalu pada data asli."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast():
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.argmax(y_prob, axis=1)
    n_cls  = len(class_names)

    np.save(f'{save_prefix}_{split_name}_probs.npy',  y_prob)
    np.save(f'{save_prefix}_{split_name}_labels.npy', y_true)

    acc = accuracy_score(y_true, y_pred)
    _, _, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    try:
        if n_cls == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr',
                                average='macro', labels=list(range(n_cls)))
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy'        : float(acc),
        'macro_f1'        : float(f1_mac),
        'weighted_f1'     : float(f1_wt),
        'auc_roc'         : float(auc),
        'f1_per_class'    : f1_per.tolist(),
        'confusion_matrix': cm.tolist(),
    }


# ============================================================
# CELL 14 — Visualisasi
# ============================================================
def plot_training_curve(history, model_key, task, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, len(history['train_loss']) + 1))
    stages = history['stage']
    stage_colors = {0: '#3498db', 1: '#3498db', 2: '#2ecc71'}

    for ax, t_key, v_key, title in [
        (axes[0], 'train_loss', 'val_loss', 'Loss'),
        (axes[1], 'train_acc',  'val_acc',  'Accuracy'),
    ]:
        t_vals = history[t_key]
        v_vals = history[v_key]
        for i in range(len(epochs) - 1):
            ax.plot([epochs[i], epochs[i+1]],
                    [t_vals[i], t_vals[i+1]],
                    color=stage_colors[stages[i]], linewidth=1.8)
        ax.plot(epochs, t_vals, 'o', markersize=3, color='gray', alpha=0.4)
        ax.plot(epochs, v_vals, 's-', markersize=3, color='red',
                alpha=0.8, linewidth=1.5, label='Val')
        if ABLATION_CONFIG['use_two_stage']:
            ax.axvline(x=TRAIN_CONFIG['stage1_epochs'] + 0.5,
                       color='orange', linestyle='--', alpha=0.8,
                       label='Stage boundary')
        ax.set_xlabel('Epoch')
        ax.set_title(f'{title} — {MODEL_DISPLAY[model_key]} / {task}')
        ax.grid(alpha=0.3)

    handles = [
        mpatches.Patch(color='#3498db', label='Stage 1 / Full FT'),
        mpatches.Patch(color='#2ecc71', label='Stage 2'),
        plt.Line2D([0],[0], color='red', marker='s',
                   label='Val', markersize=5),
    ]
    axes[0].legend(handles=handles, fontsize=8)

    plt.suptitle(
        f'Training Curve — {VARIANT_NAME} | {MODEL_DISPLAY[model_key]} / {task}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_key}_{task}_curve.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📈 Curve saved: {path}")


def plot_confusion_and_f1(all_metrics, task, save_dir):
    class_names = TASK_CONFIG[task]['class_names']
    n_cls       = len(class_names)
    short_names = [c[:12] for c in class_names]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f'Confusion Matrix & Per-Class F1 — {task.upper()}\n'
        f'Variant: {VARIANT_NAME}',
        fontsize=12, fontweight='bold'
    )
    for i, mkey in enumerate(['efficientnetv2_m','convnext','swin','vit']):
        m  = all_metrics[mkey]
        cm = np.array(m['confusion_matrix'])

        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=short_names, yticklabels=short_names,
                    ax=axes[0, i], cbar=False)
        axes[0, i].set_title(
            f"{MODEL_DISPLAY[mkey]}\n"
            f"Acc={m['accuracy']:.4f} | MacroF1={m['macro_f1']:.4f}",
            fontsize=9
        )
        axes[0, i].set_ylabel('True' if i == 0 else '')
        axes[0, i].set_xlabel('Predicted')
        axes[0, i].tick_params(labelsize=7)

        # Per-class F1
        f1_vals = m['f1_per_class']
        colors  = ['#2ecc71' if v >= 0.7 else
                   '#f39c12' if v >= 0.5 else '#e74c3c'
                   for v in f1_vals]
        axes[1, i].bar(range(n_cls), f1_vals, color=colors)
        axes[1, i].set_xticks(range(n_cls))
        axes[1, i].set_xticklabels(short_names, fontsize=7,
                                    rotation=35, ha='right')
        axes[1, i].set_ylim(0, 1.15)
        axes[1, i].set_title(
            f'Per-Class F1 — {MODEL_DISPLAY[mkey]}', fontsize=9)
        axes[1, i].axhline(y=0.7, color='green', linestyle='--',
                            alpha=0.4, linewidth=1)
        for j, v in enumerate(f1_vals):
            axes[1, i].text(j, v + 0.03, f'{v:.2f}',
                            ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, f'cm_f1_{task}.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  📊 CM+F1 saved: {path}")


def plot_ranking(all_metrics, task, save_dir):
    model_keys = ['efficientnetv2_m', 'convnext', 'swin', 'vit']
    names = [MODEL_DISPLAY[k] for k in model_keys]
    accs  = [all_metrics[k]['accuracy']  for k in model_keys]
    f1s   = [all_metrics[k]['macro_f1']  for k in model_keys]

    sorted_idx = np.argsort(accs)[::-1]
    names_s = [names[i] for i in sorted_idx]
    accs_s  = [accs[i]  for i in sorted_idx]
    f1s_s   = [f1s[i]   for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names_s))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs_s, w, label='Accuracy',
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s_s,  w, label='Macro F1',
                   color='#e74c3c', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, fontsize=10)
    min_val = max(0, min(min(accs_s), min(f1s_s)) - 0.05)
    ax.set_ylim(min_val, 1.02)
    ax.set_title(
        f'Model Ranking — Task: {task.upper()} | Variant: {VARIANT_NAME}',
        fontsize=11, fontweight='bold'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for rank, xi in enumerate(range(len(names_s))):
        ax.text(xi, min_val + 0.005, f'#{rank+1}',
                ha='center', fontsize=11, fontweight='bold', color='#2c3e50')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{bar.get_height():.4f}', ha='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, f'ranking_{task}.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  🏆 Ranking saved: {path}")


def plot_cross_task_heatmap(all_task_metrics, save_dir):
    tasks  = ['informative', 'humanitarian', 'damage']
    models = ['efficientnetv2_m', 'convnext', 'swin', 'vit']

    acc_matrix = np.array([
        [all_task_metrics[t][m]['accuracy']  for m in models]
        for t in tasks
    ])
    f1_matrix = np.array([
        [all_task_metrics[t][m]['macro_f1']  for m in models]
        for t in tasks
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'Cross-Task Heatmap — Variant: {VARIANT_NAME}',
                 fontsize=12, fontweight='bold')
    for ax, matrix, title in [
        (axes[0], acc_matrix, 'Accuracy'),
        (axes[1], f1_matrix,  'Macro F1'),
    ]:
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=[MODEL_DISPLAY[m] for m in models],
                    yticklabels=[t.capitalize() for t in tasks],
                    ax=ax, vmin=0.4, vmax=1.0,
                    annot_kws={'size': 10, 'weight': 'bold'})
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='x', rotation=20, labelsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'cross_task_heatmap.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  🗺️  Heatmap saved: {path}")


# ============================================================
# CELL 15 — Master Runner
# ============================================================
def run_task(task: str):
    print(f"\n{'#'*60}")
    print(f"  [{VARIANT_NAME.upper()}] TASK: {task.upper()}")
    print(f"{'#'*60}")

    # Print active strategy untuk task ini
    if task == 'damage':
        mixup_status = '✅ MixUp aktif' if ABLATION_CONFIG['use_mixup'] \
                       else '❌ MixUp tidak aktif'
        wce_status   = '✅ Weighted CE' if ABLATION_CONFIG['use_weighted_ce'] \
                       else '❌ Standard CE'
        print(f"  Strategy: {wce_status} | {mixup_status}")
    elif task == 'humanitarian':
        focal_status = '✅ Focal Loss' if ABLATION_CONFIG['use_focal_loss'] \
                       else '❌ Standard CE'
        merge_status = f"✅ {TASK_CONFIG[task]['num_classes']} kelas (merged)" \
                       if ABLATION_CONFIG['use_merge_kelas'] \
                       else f"❌ {TASK_CONFIG[task]['num_classes']} kelas (original)"
        print(f"  Strategy: {focal_status} | {merge_status}")

    class_names  = TASK_CONFIG[task]['class_names']
    num_classes  = TASK_CONFIG[task]['num_classes']
    task_dir     = os.path.join(OUTPUT_DIR, task)
    os.makedirs(task_dir, exist_ok=True)

    model_keys   = ['efficientnetv2_m', 'convnext', 'swin', 'vit']
    all_metrics  = {}
    all_val_accs = {}

    for mkey in model_keys:
        ckpt_path   = os.path.join(CHECKPOINT_DIR, f'{mkey}_{task}_best.pth')
        done_marker = os.path.join(task_dir, f'{mkey}_done.json')

        if os.path.exists(done_marker):
            print(f"\n  ⏭  [{mkey}/{task}] skip (sudah selesai)")
            with open(done_marker) as f:
                saved = json.load(f)
            all_val_accs[mkey] = saved['val_acc']
            all_metrics[mkey]  = saved['metrics']
            continue

        print(f"\n{'─'*55}")
        print(f"  Model: {MODEL_DISPLAY[mkey]} | Task: {task.upper()}")
        print(f"{'─'*55}")

        loaders = create_dataloaders(task, mkey)
        model   = create_model(mkey, num_classes, pretrained=True)

        history, best_val = train_model(
            model, mkey, task, loaders, f'{mkey}_{task}'
        )
        all_val_accs[mkey] = best_val

        plot_training_curve(history, mkey, task, task_dir)

        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path)

        save_prefix  = os.path.join(task_dir, mkey)
        metrics_test = evaluate_and_save(
            model, loaders['test'], class_names, save_prefix, 'test')
        # val probs untuk Stacking LR di Notebook 2
        evaluate_and_save(
            model, loaders['dev'], class_names, save_prefix, 'val')

        all_metrics[mkey] = metrics_test

        auc_str = f"{metrics_test['auc_roc']:.4f}" \
                  if not np.isnan(metrics_test['auc_roc']) else 'NaN'
        print(f"\n  [{MODEL_DISPLAY[mkey]}/{task}] "
              f"Acc={metrics_test['accuracy']:.4f} | "
              f"MacroF1={metrics_test['macro_f1']:.4f} | "
              f"AUC={auc_str}")
        print(f"  Per-class F1:")
        for i, (cn, f1v) in enumerate(
                zip(class_names, metrics_test['f1_per_class'])):
            bar = '█' * int(f1v * 20)
            print(f"    [{i}] {cn:<42} {f1v:.4f} {bar}")

        done_data = {
            'val_acc': best_val,
            'metrics': metrics_test,
            'history': history,
        }
        with open(done_marker, 'w') as f:
            json.dump(done_data, f, indent=2)

        # Hapus checkpoint — hemat storage
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"  🗑️  Checkpoint dihapus")

        del model
        torch.cuda.empty_cache()

    # Simpan val_accs dan summary
    with open(os.path.join(task_dir, 'val_accs.json'), 'w') as f:
        json.dump(all_val_accs, f, indent=2)

    metrics_clean = {
        k: {kk: vv for kk, vv in v.items() if kk != 'confusion_matrix'}
        for k, v in all_metrics.items()
    }
    with open(os.path.join(task_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    # Visualisasi
    plot_confusion_and_f1(all_metrics, task, task_dir)
    plot_ranking(all_metrics, task, task_dir)

    # Ringkasan
    print(f"\n{'='*60}")
    print(f"  RINGKASAN — Task: {task.upper()} | {VARIANT_NAME}")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Acc':>8} {'MacroF1':>9} "
          f"{'WtF1':>8} {'AUC':>8}")
    print(f"  {'─'*58}")
    for mkey in model_keys:
        m       = all_metrics[mkey]
        auc_str = f"{m['auc_roc']:>8.4f}" \
                  if not np.isnan(m['auc_roc']) else f"{'NaN':>8}"
        print(f"  {MODEL_DISPLAY[mkey]:<22} "
              f"{m['accuracy']:>8.4f} {m['macro_f1']:>9.4f} "
              f"{m['weighted_f1']:>8.4f} {auc_str}")

    return all_metrics, all_val_accs


# ============================================================
# CELL 16 — Run Semua Task
# ============================================================
metrics_informative,  val_accs_informative  = run_task('informative')
metrics_humanitarian, val_accs_humanitarian = run_task('humanitarian')
metrics_damage,       val_accs_damage       = run_task('damage')


# ============================================================
# CELL 17 — Cross-Task Summary & Visualisasi
# ============================================================
all_task_metrics = {
    'informative'  : metrics_informative,
    'humanitarian' : metrics_humanitarian,
    'damage'       : metrics_damage,
}

plot_cross_task_heatmap(all_task_metrics, OUTPUT_DIR)

rows = []
for task, metrics in all_task_metrics.items():
    for mkey, m in metrics.items():
        rows.append({
            'Variant'    : VARIANT_NAME,
            'Task'       : task,
            'Model'      : MODEL_DISPLAY[mkey],
            'Accuracy'   : round(m['accuracy'],    4),
            'Macro_F1'   : round(m['macro_f1'],    4),
            'Weighted_F1': round(m['weighted_f1'], 4),
            'AUC_ROC'    : round(m['auc_roc'], 4)
                           if not np.isnan(m['auc_roc']) else None,
        })

df_summary = pd.DataFrame(rows)
csv_path   = os.path.join(OUTPUT_DIR, f'summary_{VARIANT_NAME}.csv')
df_summary.to_csv(csv_path, index=False)

print(f"\n{'='*70}")
print(f"  CROSS-TASK SUMMARY — {VARIANT_NAME.upper()}")
print(f"{'='*70}")
print(df_summary.to_string(index=False))
print(f"\n✅ Summary CSV: {csv_path}")


# ============================================================
# CELL 18 — Simpan Config
# ============================================================
config_out = {
    'variant_name'   : VARIANT_NAME,
    'ablation_config': ABLATION_CONFIG,
    'train_config'   : TRAIN_CONFIG,
    'model_config'   : {k: {kk: str(vv) for kk, vv in v.items()}
                        for k, v in MODEL_CONFIG.items()},
}
with open(os.path.join(OUTPUT_DIR, 'variant_config.json'), 'w') as f:
    json.dump(config_out, f, indent=2)
print(f"✅ Config saved: variant_config.json")


# ============================================================
# CELL 19 — Zip & Cleanup
# ============================================================
import zipfile, shutil

zip_path = f'/kaggle/working/outputs_{VARIANT_NAME}.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            arcname  = os.path.relpath(filepath, '/kaggle/working')
            zf.write(filepath, arcname)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"✅ {zip_path} ({size_mb:.1f} MB)")

if os.path.exists(CHECKPOINT_DIR):
    shutil.rmtree(CHECKPOINT_DIR)
    print(f"🗑️  Checkpoint dir dihapus")

print(f"\n{'='*60}")
print(f"  ✅ SELESAI — Variant: {VARIANT_NAME.upper()}")
print(f"  📦 Download: outputs_{VARIANT_NAME}.zip")
print(f"{'='*60}")
