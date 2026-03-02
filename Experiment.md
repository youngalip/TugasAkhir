# Eksperimen Tugas Akhir: Klasifikasi Gambar Bencana Menggunakan Ensemble Deep Learning

> **Judul Sementara:** Perbandingan Arsitektur Deep Learning dan Strategi Ensemble untuk Klasifikasi Gambar Bencana pada Dataset CrisisMMD  
> **Dataset:** CrisisMMD v2.0  
> **Platform:** Kaggle Notebook (GPU T4/P100)  
> **Framework:** PyTorch + timm

---

## Daftar Isi

1. [Latar Belakang](#1-latar-belakang)
2. [Rumusan Masalah](#2-rumusan-masalah)
3. [Tujuan Penelitian](#3-tujuan-penelitian)
4. [Dataset](#4-dataset)
5. [Arsitektur Model](#5-arsitektur-model)
6. [Metodologi Eksperimen](#6-metodologi-eksperimen)
7. [Strategi Ensemble & Stacking](#7-strategi-ensemble--stacking)
8. [Konfigurasi Training](#8-konfigurasi-training)
9. [Metrik Evaluasi](#9-metrik-evaluasi)
10. [Struktur Kode](#10-struktur-kode)
11. [Library & Tools](#11-library--tools)
12. [Rencana Analisis Hasil](#12-rencana-analisis-hasil)
13. [Checklist Eksperimen](#13-checklist-eksperimen)

---

## 1. Latar Belakang

Pada situasi bencana alam, media sosial (terutama Twitter) menjadi salah satu sumber informasi yang sangat cepat dan masif. Namun, tidak semua konten yang beredar bersifat informatif — banyak yang berupa meme, berita lama yang disebarkan ulang, atau gambar yang tidak relevan. Penyaringan konten secara manual tidak mungkin dilakukan dalam skala besar pada kondisi darurat.

**CrisisMMD** hadir sebagai dataset benchmark yang dikumpulkan dari tweet nyata selama 7 kejadian bencana besar, dengan anotasi yang dilakukan secara crowdsourcing. Dataset ini memungkinkan pengembangan sistem otomatis untuk:

1. Memilah gambar yang **informatif** vs **tidak informatif**
2. Mengkategorikan jenis informasi kemanusiaan yang terkandung

Penelitian ini berfokus pada **pendekatan vision-only** (hanya gambar, tanpa teks tweet) menggunakan model deep learning modern dan strategi ensemble untuk memaksimalkan akurasi klasifikasi.

---

## 2. Rumusan Masalah

1. Bagaimana performa masing-masing arsitektur deep learning (EfficientNet-B0, ViT-B/16, ConvNeXt-Base, Swin-Small) dalam mengklasifikasi gambar bencana pada dataset CrisisMMD?
2. Apakah strategi ensemble (Simple Averaging, Weighted Voting) dapat meningkatkan akurasi dibandingkan model tunggal terbaik?
3. Apakah strategi stacking (Logistic Regression, MLP, Random Forest sebagai meta-model) memberikan hasil yang lebih baik dibandingkan ensemble konvensional?
4. Bagaimana trade-off antara akurasi dan efisiensi komputasi (FLOPs, inference time, memory usage) antar model dan strategi?

---

## 3. Tujuan Penelitian

- Membandingkan performa 4 arsitektur deep learning berbeda pada task klasifikasi gambar bencana
- Menganalisis efektivitas berbagai strategi ensemble dan stacking
- Mengidentifikasi model atau kombinasi model yang memberikan trade-off terbaik antara akurasi dan efisiensi
- Memberikan rekomendasi arsitektur yang tepat untuk skenario deployment berbeda (real-time vs batch)

---

## 4. Dataset

### 4.1 Informasi Umum

| Properti | Nilai |
|---|---|
| Nama | CrisisMMD v2.0 |
| Sumber | Twitter (crowdsourced annotation) |
| Total data | ~18.000+ sampel |
| Modalitas yang digunakan | **Image only** (vision-only approach) |
| Format anotasi | TSV per event bencana |
| Platform | Kaggle Private Dataset |

### 4.2 Event Bencana (7 Event)

| No | Event | Tipe Bencana | Lokasi |
|---|---|---|---|
| 1 | California Wildfires | Kebakaran hutan | California, USA |
| 2 | Hurricane Harvey | Badai | Texas, USA |
| 3 | Hurricane Irma | Badai | Caribbean/Florida |
| 4 | Hurricane Maria | Badai | Puerto Rico |
| 5 | Sri Lanka Floods | Banjir | Sri Lanka |
| 6 | Iraq Iran Earthquake | Gempa bumi | Iraq/Iran |
| 7 | Pakistan Floods | Banjir | Pakistan |

### 4.3 Task Klasifikasi

#### Task 1 — Informative Classification (Binary)

Memilah apakah gambar mengandung informasi berguna untuk respons bencana.

| Label | Kode | Deskripsi |
|---|---|---|
| `informative` | 1 | Foto kerusakan, korban, lokasi bencana, aktivitas rescue |
| `not_informative` | 0 | Meme, selfie, grafik generic, logo, foto tidak relevan |

#### Task 2 — Humanitarian Classification (Multi-class, 7 Kelas)

Hanya untuk gambar yang sudah `informative`. Mengkategorikan jenis konten kemanusiaan.

| Kode | Label | Deskripsi |
|---|---|---|
| 0 | `infrastructure_and_utility_damage` | Kerusakan jalan, jembatan, gedung, jaringan listrik |
| 1 | `affected_individuals` | Individu terdampak dalam kondisi distress |
| 2 | `injured_or_dead_people` | Korban luka atau meninggal |
| 3 | `missing_or_found_people` | Pencarian/penemuan orang hilang |
| 4 | `rescue_volunteering_or_donation_effort` | Aktivitas rescue, relawan, donasi |
| 5 | `vehicle_damage` | Kerusakan kendaraan akibat bencana |
| 6 | `other_relevant_information` | Informasi relevan lain |

### 4.4 Data Split

| Split | Proporsi | Digunakan untuk |
|---|---|---|
| Train | ~70% | Training model backbone |
| Validation (Dev) | ~15% | Early stopping + training meta-model stacking |
| Test | ~15% | Evaluasi akhir semua model & ensemble |

> Split sudah ditentukan oleh dataset author — tidak dilakukan random split manual untuk menjaga konsistensi dengan paper referensi.

### 4.5 Struktur File

```
/kaggle/input/datasets/alieffathurrahman/crisismmd/
├── annotations/                          ← Label per event (7 file TSV)
│   ├── california_wildfires_final_data.tsv
│   ├── hurricane_harvey_final_data.tsv
│   └── ...
├── crisismmd_datasplit_all/
│   └── crisismmd_datasplit_all/          ← ID split train/dev/test
│       ├── task_informative_text_img_train.tsv
│       ├── task_informative_text_img_dev.tsv
│       ├── task_informative_text_img_test.tsv
│       ├── task_humanitarian_text_img_train.tsv
│       ├── task_humanitarian_text_img_dev.tsv
│       └── task_humanitarian_text_img_test.tsv
└── data_image/                           ← File gambar asli
    ├── california_wildfires/
    ├── hurricane_harvey/
    └── ...
```

---

## 5. Arsitektur Model

### 5.1 Perbandingan 4 Model

| Model | Arsitektur | Input | Params | FLOPs | Inference | Batch |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | CNN (MBConv) | 224×224 | ~4M | ~0.38G | ~8ms | 32 |
| ViT-B/16 | Pure Transformer | 384×384 | ~86M | ~49G | ~47ms | 16 |
| ConvNeXt-Base | Modern CNN | 224×224 | ~88M | ~15G | ~17ms | 32 |
| Swin-Small | Hierarchical Transformer | 224×224 | ~49M | ~8.5G | ~22ms | 32 |

### 5.2 Alasan Pemilihan Model

Keempat model dipilih untuk merepresentasikan **diversitas arsitektur** yang saling melengkapi:

| Model | Keunggulan | Kelemahan |
|---|---|---|
| **EfficientNet-B0** | Sangat ringan, cepat, efisien | Terbatas tangkap konteks global |
| **ViT-B/16** | Global attention terbaik, resolusi tinggi | Boros memori dan komputasi |
| **ConvNeXt-Base** | Fitur lokal excellent + global baik | Parameter besar |
| **Swin-Small** | Balance lokal+global, efisien | — |

> Diversitas arsitektur → pola error yang berbeda antar model → ensemble lebih efektif

### 5.3 Pretrained Weights

Semua model menggunakan pretrained weights dari **ImageNet-21k / ImageNet-1k** via library `timm`, kemudian di-fine-tune pada CrisisMMD (transfer learning).

---

## 6. Metodologi Eksperimen

### 6.1 Alur Umum

```
Raw Dataset
    ↓
Load & Merge Annotations (7 TSV)
    ↓
Apply Train/Val/Test Split
    ↓
CrisisMMDDataset + DataLoader
    ↓
Training 4 Model (independen)
    ↓
Inference pada Val Set & Test Set
    ↓
Probabilitas Output → Feature Matrix
    ↓
Ensemble (Simple/Weighted/Best-3)
Stacking (LR / MLP / Random Forest)
    ↓
Evaluasi Akhir pada Test Set
```

### 6.2 Data Augmentation

**Training set:**
- `RandomResizedCrop` (scale 0.8–1.0)
- `RandomHorizontalFlip` (p=0.5)
- `RandomRotation` (±15°)
- `ColorJitter` (brightness, contrast, saturation ±0.2)
- `Normalize` (ImageNet mean/std)

**Validation & Test set:**
- `Resize` (input_size × 1.14)
- `CenterCrop` (input_size)
- `Normalize` (ImageNet mean/std)

### 6.3 Training Strategy

- **Optimizer:** AdamW
- **Loss Function:** CrossEntropyLoss dengan `label_smoothing=0.1`
- **Scheduler:** CosineAnnealingLR (eta_min=1e-6)
- **Early Stopping:** patience=5 epoch (monitor val_loss)
- **Mixed Precision:** Enabled (torch.cuda.amp) untuk efisiensi memori
- **Checkpoint:** Simpan model terbaik berdasarkan val_accuracy

---

## 7. Strategi Ensemble & Stacking

### 7.1 Ensemble Konvensional

| Metode | Cara Kerja | Keterangan |
|---|---|---|
| **Simple Averaging** | `(p1 + p2 + p3 + p4) / 4` | Semua model bobotnya sama |
| **Weighted Voting** | `Σ(w_i × p_i)` | Bobot dari val_accuracy masing-masing |
| **Best-3** | Rata-rata 3 model terbaik | Drop model dengan val_acc terendah |

### 7.2 Stacking

Feature matrix dibentuk dari **concatenasi probabilitas** output 4 model:

```
X = [prob_eff | prob_vit | prob_cnx | prob_swn]
Shape: (N_samples, 4 × num_classes)
→ Binary:        (N, 8)
→ Humanitarian:  (N, 28)
```

**Meta-model yang dibandingkan:**

| Meta-Model | Alasan | Konfigurasi |
|---|---|---|
| **Logistic Regression** | Baseline, interpretable | C=1.0, balanced class weight |
| **MLP** | Non-linear, tangkap interaksi antar model | 3 layer, hidden=64, dropout=0.3 |
| **Random Forest** | Robust, tidak perlu scaling | 200 trees, max_depth=8 |

> Meta-model dilatih pada **validation set** — bukan training set — untuk menghindari data leakage.

---

## 8. Konfigurasi Training

```python
TASK_CONFIG = {
    'informative':   {'num_classes': 2,  'label_col': 'image_info'},
    'humanitarian':  {'num_classes': 7,  'label_col': 'image_human'},
}

MODEL_CONFIG = {
    'efficientnet':  {'name': 'efficientnet_b0',               'input_size': 224, 'lr': 1e-4,  'batch_size': 32},
    'vit':           {'name': 'vit_base_patch16_384',           'input_size': 384, 'lr': 5e-5,  'batch_size': 16},
    'convnext':      {'name': 'convnext_base',                  'input_size': 224, 'lr': 5e-5,  'batch_size': 32},
    'swin':          {'name': 'swin_small_patch4_window7_224',  'input_size': 224, 'lr': 5e-5,  'batch_size': 32},
}

TRAIN_CONFIG = {
    'max_epochs':               50,
    'early_stopping_patience':  5,
    'weight_decay':             0.01,
    'label_smoothing':          0.1,
    'mixed_precision':          True,
}
```

---

## 9. Metrik Evaluasi

### 9.1 Metrik Performa

| Metrik | Formula | Keterangan |
|---|---|---|
| **Accuracy** | `correct / total` | Proporsi prediksi benar |
| **Precision** | `TP / (TP + FP)` | Macro-averaged |
| **Recall** | `TP / (TP + FN)` | Macro-averaged |
| **F1-Score** | `2×P×R / (P+R)` | Macro-averaged, utama untuk imbalanced |
| **Confusion Matrix** | — | Per kelas |

> Macro-averaging dipilih karena dataset memiliki ketidakseimbangan kelas, terutama pada Task 2.

### 9.2 Metrik Efisiensi

| Metrik | Satuan | Tool |
|---|---|---|
| **Inference Time** | ms/gambar | `time.time()` + CUDA sync |
| **GPU Memory** | MB | `torch.cuda.max_memory_allocated()` |
| **FLOPs** | GFLOPs | Library `thop` |
| **Parameters** | Juta (M) | `sum(p.numel())` |

---

## 10. Struktur Kode

```
Notebook Kaggle (1 file .ipynb)
│
├── Cell 1   : Verifikasi struktur dataset
├── Cell 2   : Install library tambahan
├── Cell 3   : Import semua library
├── Cell 4   : Konfigurasi (TASK, MODEL_CONFIG, TRAIN_CONFIG)
├── Cell 5   : load_annotations() — baca TSV, apply split
├── Cell 6   : CrisisMMDDataset class
├── Cell 7   : get_transforms()
├── Cell 8   : create_dataloaders()
├── Cell 9   : create_model() — via timm
├── Cell 10  : AverageMeter, EarlyStopping, save_checkpoint
├── Cell 11  : train_one_epoch()
├── Cell 12  : validate()
├── Cell 13  : train_model() — main training loop
│
├── Cell 14  : Train EfficientNet-B0
├── Cell 15  : Train ViT-B/16
├── Cell 16  : Train ConvNeXt-Base  ← baru
├── Cell 17  : Train Swin-Small     ← baru
│
├── Cell 18  : Plot training history (4 model)
├── Cell 19  : evaluate_model() + plot_confusion_matrix()
├── Cell 20  : Evaluate semua model pada test set
│
├── Cell 21  : Ensemble (Simple, Weighted, Best-3)
├── Cell 22  : Efficiency metrics (FLOPs, Memory, Time)
├── Cell 23  : Summary table + trade-off plots
├── Cell 24  : Final summary v2
│
├── Cell 25  : Stacking — kumpulkan probabilitas (val + test)
├── Cell 26  : Meta-model 1: Logistic Regression
├── Cell 27  : Meta-model 2: MLP
├── Cell 28  : Meta-model 3: Random Forest
├── Cell 29  : Summary lengkap (single + ensemble + stacking)
├── Cell 30  : Visualisasi perbandingan final
└── Cell 31  : Kesimpulan & simpan JSON/CSV
```

**Output tersimpan di `/kaggle/working/`:**
```
checkpoints/
├── efficientnet_b0_informative_best.pth
├── vit_b16_informative_best.pth
├── convnext_base_informative_best.pth
└── swin_small_informative_best.pth

results/
├── training_history.png
├── *_confusion_matrix.png
├── tradeoff_analysis.png
├── stacking_*_importance.png
├── final_comparison.png
├── summary_results.csv
└── all_results.json
```

---

## 11. Library & Tools

| Library | Versi | Fungsi |
|---|---|---|
| `torch` | 2.x | Framework deep learning utama |
| `torchvision` | — | Transforms dan augmentasi |
| `timm` | latest | Pretrained model zoo (ViT, EfficientNet, ConvNeXt, Swin) |
| `thop` | latest | Menghitung FLOPs dan parameter |
| `scikit-learn` | — | Meta-model stacking (LR, RF), metrics |
| `pandas` | — | Manipulasi DataFrame annotations |
| `numpy` | — | Operasi array probabilitas |
| `matplotlib` | — | Visualisasi training history, trade-off |
| `seaborn` | — | Confusion matrix heatmap |
| `PIL (Pillow)` | — | Load dan proses gambar |
| `tqdm` | — | Progress bar training |

**Platform & Hardware:**
- Kaggle Notebook (GPU: NVIDIA T4 atau P100)
- CUDA dengan Mixed Precision (AMP) untuk efisiensi memori
- Storage: `/kaggle/working/` (19.5 GB output limit)

---

## 12. Rencana Analisis Hasil

### 12.1 Perbandingan yang Akan Dilakukan

1. **Single model vs single model** — Mana arsitektur terbaik?
2. **Ensemble vs model terbaik** — Seberapa besar gain dari ensemble?
3. **Stacking vs ensemble** — Apakah meta-model lebih efektif dari rata-rata?
4. **Antar meta-model** — LR vs MLP vs Random Forest
5. **Akurasi vs efisiensi** — Model mana yang paling worth it?

### 12.2 Pertanyaan Analisis

- Apakah model dengan arsitektur berbeda (CNN vs Transformer) membuat kesalahan pada gambar yang berbeda? → Analisis **disagreement rate** antar model
- Apakah kelas tertentu lebih sulit diklasifikasikan? → Analisis per-class F1 dari confusion matrix
- Berapa **minimum model** yang dibutuhkan untuk ensemble yang efektif?
- Pada confidence threshold berapa EfficientNet bisa diandalkan tanpa ensemble? → Analisis untuk strategi **gatekeeper**

### 12.3 Visualisasi yang Direncanakan

- [ ] Training curve (loss & accuracy) per model
- [ ] Confusion matrix per model dan per ensemble
- [ ] Trade-off scatter plot (accuracy vs time, accuracy vs FLOPs)
- [ ] Bar chart ranking semua metode
- [ ] Feature importance meta-model (LR coefficients, RF importance)
- [ ] Per-class F1 comparison antar model

---

## 13. Checklist Eksperimen

### Setup
- [x] Dataset diupload ke Kaggle Private Dataset
- [x] Struktur folder terverifikasi
- [x] Semua library terinstall

### Task: Informative (Binary)
- [x] EfficientNet-B0 — training selesai
- [x] ViT-B/16 — training selesai
- [x] ConvNeXt-Base — training selesai
- [x] Swin-Small — training selesai
- [x] Evaluasi semua model pada test set
- [x] Ensemble (Simple, Weighted, Best-3)
- [x] Stacking (LR, MLP, Random Forest)
- [x] Summary & visualisasi lengkap

### Task: Humanitarian (7-class)
- [ ] Re-run semua eksperimen dengan `TASK = 'humanitarian'`
- [ ] Evaluasi semua model
- [ ] Ensemble & stacking
- [ ] Summary & perbandingan dengan task informative

### Analisis Akhir
- [ ] Buat tabel perbandingan lengkap kedua task
- [ ] Tulis interpretasi hasil
- [ ] Identifikasi model/kombinasi terbaik per skenario

---

*Dokumen ini merupakan rangkuman rancangan eksperimen tugas akhir dan akan diperbarui seiring perkembangan eksperimen.*
