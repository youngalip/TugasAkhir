# Ringkasan Proposal Tugas Akhir
## Seminar Proposal — BAB I s.d. BAB III

**Judul:**
> Analisis Komparatif EfficientNet-B0, Vision Transformer (ViT-B/16), dan Ensemble Heterogen untuk Klasifikasi Citra Informasi Krisis pada Dataset CrisisMMD

**Program Studi:** Teknik Informatika
**Institusi:** Institut Teknologi Sumatera (ITERA)

---

## Daftar Isi
1. [Gambaran Umum](#1-gambaran-umum)
2. [BAB I — Pendahuluan](#2-bab-i--pendahuluan)
3. [BAB II — Tinjauan Pustaka](#3-bab-ii--tinjauan-pustaka)
4. [BAB III — Metode Penelitian](#4-bab-iii--metode-penelitian)
5. [Ringkasan Alur Eksperimen](#5-ringkasan-alur-eksperimen)
6. [Jadwal Penelitian](#6-jadwal-penelitian)

---

## 1. Gambaran Umum

Penelitian ini membangun dan mengevaluasi sistem klasifikasi otomatis citra informasi krisis dari media sosial menggunakan pendekatan _deep learning_. Tiga konfigurasi model dibandingkan secara sistematis:

| Model | Tipe | Parameter |
|---|---|---|
| EfficientNet-B0 | CNN Modern | ~5,3 juta |
| ViT-B/16 | Pure Transformer | ~86 juta |
| Ensemble Heterogen (B0 + ViT) | CNN + Transformer | ~91 juta |

Dataset yang digunakan adalah **CrisisMMD** — dataset standar komunitas riset respons bencana yang berisi lebih dari 18.000 citra dari 7 kejadian bencana tahun 2017.

Kebaruan utama penelitian ini ada pada dua hal: **(1)** komparasi langsung CNN modern vs. _pure transformer_ pada domain respons bencana, dan **(2)** analisis komprehensif _trade-off_ akurasi vs. efisiensi komputasi yang selama ini belum dieksplorasi secara mendalam di domain ini.

---

## 2. BAB I — Pendahuluan

### 2.1 Latar Belakang

**Konteks masalah:**
Bencana alam menimbulkan dampak masif dan membutuhkan respons cepat. UNOCHA menetapkan **72 jam pertama** sebagai periode kritis penyelamatan korban. Dalam rentang waktu ini, tim SAR membutuhkan informasi akurat, cepat, dan terfilter untuk mengalokasikan sumber daya secara efektif.

**Peran media sosial:**
Platform seperti Twitter dan Facebook menghasilkan ribuan foto dan video dari masyarakat terdampak hanya dalam hitungan jam. Data visual ini berpotensi menjadi sumber _situational awareness_ real-time — namun volumenya terlalu besar untuk dikurasi secara manual, sehingga dibutuhkan sistem klasifikasi otomatis berbasis _machine learning_.

**Perkembangan teknologi:**
- CNN (seperti EfficientNet) dominan selama lebih dari satu dekade — unggul dalam menangkap pola lokal dengan efisiensi tinggi.
- Vision Transformer (ViT) muncul sebagai paradigma baru — mampu menangkap dependensi jarak jauh melalui mekanisme _self-attention_.
- _Ensemble learning_ menjanjikan peningkatan performa dengan menggabungkan kekuatan komplementer keduanya.

**Gap penelitian yang diisi:**

| Gap | Penjelasan |
|---|---|
| Gap 1 | Eksplorasi ViT untuk klasifikasi informasi krisis masih sangat terbatas |
| Gap 2 | Komparasi langsung CNN modern vs. ViT pada domain respons bencana belum banyak dilakukan |
| Gap 3 | Analisis _trade-off_ akurasi vs. efisiensi komputasi belum dieksplorasi secara mendalam |

### 2.2 Rumusan Masalah

1. Bagaimana performa **EfficientNet-B0** sebagai representasi CNN modern dalam klasifikasi citra informasi krisis?
2. Bagaimana performa **ViT-B/16** sebagai representasi _pure transformer_ dalam klasifikasi citra informasi krisis?
3. Bagaimana performa **ensemble heterogen** (EfficientNet-B0 + ViT-B/16) dibandingkan model tunggal terbaik?
4. Bagaimana **pertukaran akurasi vs. biaya komputasi** (_inference time_, _memory usage_, FLOPs) pada ketiga konfigurasi model?

### 2.3 Tujuan Penelitian

1. Mengimplementasikan dan mengevaluasi EfficientNet-B0 pada dataset CrisisMMD.
2. Mengimplementasikan dan mengevaluasi ViT-B/16 pada dataset CrisisMMD.
3. Mengimplementasikan dan mengevaluasi ensemble heterogen dengan strategi _simple averaging_ dan _weighted voting_.
4. Menganalisis _trade-off_ akurasi vs. efisiensi komputasi untuk memberikan rekomendasi praktis pemilihan model.

### 2.4 Batasan Masalah

- Hanya modalitas **citra** (tidak termasuk teks atau multimodal).
- CNN dibatasi pada **EfficientNet-B0**; Transformer dibatasi pada **ViT-B/16**.
- Ensemble dibatasi pada strategi **_simple averaging_** dan **_weighted voting_**.
- Dataset: **CrisisMMD** (7 jenis bencana alam).
- Tugas klasifikasi: **Task 1** (_Informative_ vs _Not Informative_) dan **Task 2** (_Humanitarian Categories_).
- Tidak mencakup deployment ke lingkungan produksi.

### 2.5 Manfaat Penelitian

**Teoritis:**
- Kontribusi komparasi mendalam paradigma CNN vs. Vision Transformer di domain respons bencana.
- Memperkaya literatur _ensemble_ heterogen CNN–Transformer.
- Analisis _trade-off_ akurasi vs. efisiensi sebagai referensi penelitian selanjutnya.

**Praktis:**
- Panduan praktis pemilihan arsitektur model untuk sistem klasifikasi informasi krisis.
- Mendukung pengembangan sistem respons bencana berbasis AI yang lebih efisien.
- Referensi implementasi pada aplikasi real-time dengan keterbatasan komputasi di lapangan.

---

## 3. BAB II — Tinjauan Pustaka

### 3.1 Ringkasan Penelitian Terdahulu

| No. | Peneliti & Tahun | Metode | Dataset | Hasil Utama | Keterbatasan |
|---|---|---|---|---|---|
| 1 | Ofli et al. (2020) | CNN + Text Multimodal | CrisisMMD | T1: 84,4% (multimodal), 83,3% (image-only); T2: 78,4% / 76,8% | Tidak ada analisis _trade-off_ komputasi |
| 2 | Mohanty et al. (2021) | CNN (Inception V3, VGG, ResNet) + Transfer Learning | Hurricane Irma | F1-score 0,95 (Tuned Inception V3) | Tidak ada ensemble, tidak ada ViT, tidak ada analisis efisiensi |
| 3 | Firmansyah et al. (2022) | Homogeneous Ensemble CNN (VGG16, DenseNet201, dll.) | CrisisMMD | Akurasi 84,6% | Tidak ada ViT, tidak ada analisis biaya komputasi |
| 4 | Lukauskas et al. (2024) | NLP + CNN | Social media | Akurasi >80% | Fokus deteksi bencana umum, bukan klasifikasi _humanitarian_ |
| 5 | Long et al. (2024) | CrisisViT (Vision Transformer) | Crisis datasets | Performa kompetitif | Tidak ada komparasi CNN, tidak ada analisis _trade-off_ |
| 6 | **Penelitian ini** | **EfficientNet-B0 + ViT-B/16 + Ensemble Heterogen** | **CrisisMMD** | **—** | **—** |

> **Baseline image-only relevan dari literatur:** Task 1 = 83,3% dan Task 2 = 76,8% (Ofli et al., 2020).

### 3.2 Dasar Teori

#### Dataset CrisisMMD
- 18.000+ citra dari 7 bencana tahun 2017: Hurricane Harvey, Hurricane Irma, Hurricane Maria, Gempa Mexico, Kebakaran Hutan California, Gempa Iraq-Iran, Banjir Sri Lanka.
- Anotasi manual oleh beberapa anotator.
- Split: 70% train / 15% validasi / 15% test (official split).
- **Task 1** (Binary): _Informative_ vs _Not Informative_.
- **Task 2** (Multi-class, 7 kategori): kerusakan infrastruktur, individu terdampak, orang terluka/meninggal, orang hilang/ditemukan, upaya penyelamatan/donasi, kerusakan kendaraan, informasi relevan lainnya.

#### EfficientNet-B0
- Diperkenalkan Tan & Le (2019) dengan konsep **_compound scaling_** — menyeimbangkan kedalaman, lebar, dan resolusi jaringan secara bersamaan.
- Ditemukan menggunakan _Neural Architecture Search_ (NAS).
- EfficientNet-B0: **77,1% ImageNet** hanya dengan **5,3 juta parameter** (vs. ResNet-50: 76,0% dengan 26 juta parameter).
- Cocok untuk skenario dengan keterbatasan sumber daya komputasi di lapangan.

#### Vision Transformer (ViT-B/16)
- Diperkenalkan Dosovitskiy et al. (2021) — mengadaptasi arsitektur Transformer NLP ke domain _computer vision_.
- Citra dibagi menjadi **patch 16×16 piksel** → setiap patch di-flatten menjadi vektor → diproses dengan mekanisme _self-attention_.
- Mampu menangkap **dependensi jarak jauh** sejak layer pertama, tanpa batasan _receptive field_ seperti pada CNN.
- ViT-B/16: 12 Transformer encoder layers, dimensi 768, 12 attention heads → **~86 juta parameter**.
- Membutuhkan pretrain data besar (ImageNet-21k) untuk mencapai performa optimal.

#### Transfer Learning
- Dua strategi: _feature extraction_ (layer pretrained dibekukan) dan _fine-tuning_ (seluruh parameter disesuaikan).
- Sangat penting untuk domain bencana karena dataset anotasi terbatas.
- Penelitian ini menggunakan **full fine-tuning** untuk kedua model.

#### Ensemble Learning
- Menggabungkan prediksi beberapa model untuk mengurangi bias dan varians.
- **Simple Averaging**: bobot sama → $p_{ensemble}(c) = \frac{p_{B0}(c) + p_{ViT}(c)}{2}$
- **Weighted Voting**: bobot proporsional terhadap akurasi validasi → $p_{ensemble}(c) = w_{B0} \cdot p_{B0}(c) + w_{ViT} \cdot p_{ViT}(c)$
- Ensemble heterogen (CNN + Transformer) berpotensi mengeksploitasi kekuatan komplementer: **pola lokal (CNN) + konteks global (Transformer)**.

#### Metrik Evaluasi
| Jenis | Metrik |
|---|---|
| Akurasi | Accuracy, Precision, Recall, F1-Score (macro & weighted), Confusion Matrix |
| Efisiensi | Inference Time (ms/citra), Memory Usage (MB), FLOPs (GFLOPs) |

---

## 4. BAB III — Metode Penelitian

### 4.1 Alur Penelitian (7 Langkah)

```
[1] Studi Literatur
        ↓
[2] Preparasi Dataset (CrisisMMD)
        ↓
[3] Preprocessing Citra
        ↓
[4] Training EfficientNet-B0 ──┐
                               ├─→ [6] Implementasi Ensemble
[5] Training ViT-B/16 ─────────┘         (Simple Averaging & Weighted Voting)
                                               ↓
                               [7] Evaluasi & Analisis Trade-off
```

### 4.2 Konfigurasi Preprocessing

| Aspek | EfficientNet-B0 | ViT-B/16 |
|---|---|---|
| Ukuran input | 224 × 224 px | 384 × 384 px |
| Normalisasi | ImageNet mean/std | ImageNet mean/std |
| Augmentasi (train) | Flip, Rotation ±15°, Color Jitter, RandomCrop | Sama |
| Augmentasi (val/test) | Tidak ada | Tidak ada |

### 4.3 Konfigurasi Training

| Parameter | EfficientNet-B0 | ViT-B/16 |
|---|---|---|
| Optimizer | AdamW (wd=0.01) | AdamW (wd=0.01) |
| Learning Rate awal | 1e-4 | 5e-5 |
| LR Schedule | Cosine Annealing | Cosine Annealing |
| LR minimum | 1e-6 | 1e-6 |
| Batch Size | 32 | 16 |
| Max Epochs | 50 | 50 |
| Early Stopping | patience = 5 | patience = 5 |
| Loss Function | Cross-Entropy / Weighted CE | Cross-Entropy / Weighted CE |
| Pretrained dari | ImageNet-1k | ImageNet-21k → 1k |

### 4.4 Spesifikasi Hardware & Software

**Hardware:**
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- CPU: Intel Xeon / AMD Ryzen (min. 16 cores)
- RAM: 32 GB DDR4
- Storage: 500 GB SSD

**Software:**
- Python 3.9+ | PyTorch 2.0+ | CUDA 11.8
- `timm` (ViT & EfficientNet), `torchvision`, `scikit-learn`
- `thop` / `fvcore` (FLOPs), `torch.profiler` (memory), TensorBoard

### 4.5 Strategi Reprodusibilitas
- Random seed tetap: **42, 123, 456** (3 runs per konfigurasi)
- Hasil dilaporkan sebagai **rata-rata ± standar deviasi**
- Kode disimpan di Git untuk _reproducibility_

### 4.6 Rancangan Pengujian

**Pengujian Metrik Akurasi:**
- Dilakukan pada test set (tidak disentuh selama training).
- Metrik: Accuracy, Precision, Recall, F1-Score (macro & weighted), Confusion Matrix.
- Dilakukan untuk kedua task: Task 1 (binary) dan Task 2 (multi-class).

**Pengujian Efisiensi Komputasi:**
- _Inference time_: 100 citra × 10 kali pengukuran → rata-rata ms/citra.
- _Memory usage_: `torch.cuda.max_memory_allocated()`.
- FLOPs: dihitung via `thop` atau `fvcore`.

**Analisis Trade-off:**
- Scatter plot Accuracy vs. Inference Time.
- Scatter plot Accuracy vs. Model Size (jumlah parameter).
- Scatter plot Accuracy vs. FLOPs.
- **Pareto Frontier** — mengidentifikasi model optimal untuk berbagai skenario kendala.

**Hipotesis:**
1. Ensemble heterogen akan memberikan akurasi tertinggi dibanding model tunggal manapun.
2. EfficientNet-B0 akan memiliki inference time dan memory usage jauh lebih rendah dari ViT-B/16 (5,3 juta vs. 86 juta parameter).
3. Terdapat titik optimal di mana peningkatan akurasi dari ensemble tidak lagi sebanding dengan biaya komputasi tambahan.

---

## 5. Ringkasan Alur Eksperimen

```
Dataset CrisisMMD (18.000+ citra, 7 bencana)
        │
        ├─ Task 1: Informative vs Not Informative (binary)
        └─ Task 2: Humanitarian Categories (7 kelas)
                │
                ▼
    ┌───────────────────────────────────┐
    │         Preprocessing             │
    │  Resize → Normalize → Augment     │
    └───────────────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
  EfficientNet-B0    ViT-B/16
  (Fine-tuning)    (Fine-tuning)
        │                │
        └───────┬─────────┘
                ▼
    ┌───────────────────────────┐
    │    Ensemble Heterogen     │
    │  Simple Averaging         │
    │  Weighted Voting          │
    └───────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────┐
    │         Evaluasi Komprehensif         │
    │  Akurasi: Acc, P, R, F1, CM           │
    │  Efisiensi: Time, Memory, FLOPs       │
    │  Analisis: Trade-off + Pareto         │
    └───────────────────────────────────────┘
```

---

## 6. Jadwal Penelitian

| Bulan | Kegiatan |
|---|---|
| Bulan 1 | Studi literatur, preparasi dataset, _exploratory data analysis_ |
| Bulan 2 | Implementasi & training EfficientNet-B0, optimasi hyperparameter |
| Bulan 3 | Implementasi & training ViT-B/16, implementasi strategi ensemble |
| Bulan 4 | Evaluasi komprehensif semua model, analisis trade-off, visualisasi hasil |
| Bulan 5 | Penulisan laporan akhir, persiapan presentasi sidang |

---

## Referensi Kunci

| Label | Keterangan |
|---|---|
| Alam et al. (2018) | CrisisMMD dataset — sumber data utama |
| Ofli et al. (2020) | Baseline image-only: Task 1 = 83,3%, Task 2 = 76,8% |
| Firmansyah et al. (2022) | Benchmark ensemble CNN: 84,6% pada CrisisMMD |
| Tan & Le (2019) | EfficientNet — arsitektur CNN yang digunakan |
| Dosovitskiy et al. (2021) | Vision Transformer (ViT) — arsitektur Transformer yang digunakan |
| Long et al. (2024) | CrisisViT — closest related work untuk ViT di domain bencana |

---

*Dokumen ini merupakan ringkasan proposal tugas akhir untuk keperluan Seminar Proposal, mencakup BAB I (Pendahuluan), BAB II (Tinjauan Pustaka), dan BAB III (Metode Penelitian).*
