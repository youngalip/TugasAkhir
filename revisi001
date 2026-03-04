# Rangkuman Arah Penelitian Tugas Akhir
## Revisi-001 — Setelah Bimbingan Dosen

---

## Judul Sementara
**Perbandingan Arsitektur Deep Learning dan Strategi Ensemble untuk Klasifikasi Gambar Bencana pada Dataset CrisisMMD**

---

## Inti Penelitian

| Properti | Detail |
|---|---|
| Dataset | CrisisMMD v2.0 |
| Pendekatan | Vision-only (image only, tanpa teks tweet) |
| Platform | Kaggle Notebook (GPU T4/P100) |
| Framework | PyTorch + timm |
| Versi Kode | v4 |

---

## Task yang Dikerjakan

| Task | Deskripsi | Kelas | Label Col | Status |
|---|---|---|---|---|
| Task 1 | Informative vs Not Informative | 2 (binary) | `image_info` | ✅ Sudah eksperimen awal (v3), akan re-run dengan v4 |
| Task 2 | Humanitarian Categories | 7 (multi-class) | `image_human` | ⏳ Belum — akan dijalankan dengan v4 |
| Task 3 | Damage Severity Assessment | 3 (multi-class) | `image_damage` | ⏳ Belum — anotasi ✅ tersedia di dataset |

### Kelas per Task

**Task 1:**
- `not_informative`, `informative`

**Task 2:**
- `infrastructure_and_utility_damage`
- `affected_individuals`
- `injured_or_dead_people`
- `missing_or_found_people`
- `rescue_volunteering_or_donation_effort`
- `vehicle_damage`
- `other_relevant_information`

**Task 3:**
- `little_or_no_damage`
- `mild_damage`
- `severe_damage`

---

## Model yang Digunakan

### Sebelum Revisi (v3)
| Model | Params | Masalah |
|---|---|---|
| ~~EfficientNet-B0~~ | ~~4M~~ | Gap parameter terlalu besar, meracuni ensemble |
| ViT-B/16 | 86M | — |
| ConvNeXt-Base | 88M | — |
| Swin-Small | 49M | — |

### Setelah Revisi ✅ (v4)
| Model | Params | FLOPs | Posisi dalam Spektrum Arsitektur |
|---|---|---|---|
| **EfficientNetV2-M** | ~54M | ~24G | CNN modern teroptimasi (NAS-based) |
| ViT-B/16 | ~86M | ~49G | Pure Transformer |
| ConvNeXt-Base | ~88M | ~15G | CNN yang mengadopsi prinsip Transformer |
| Swin-Small | ~49M | ~8.5G | Transformer yang mengadopsi prinsip CNN |

**Justifikasi penggantian EfficientNet-B0 → EfficientNetV2-M:**
- EfficientNet-B0 (4M) terlalu jauh gap-nya dari 3 model lain (49–88M)
- Terbukti dari eksperimen: meracuni performa ensemble (ensemble 4 model lebih rendah dari single model terbaik)
- EfficientNetV2-M (~54M) comparable ke Swin-Small, estimasi akurasi di atas ViT-B/16
- Tetap merepresentasikan spektrum CNN modern dalam perbandingan arsitektur

**Posisi 4 model dalam spektrum:**
```
Pure CNN ←──────────────────────────────→ Pure Transformer

EfficientNetV2-M   ConvNeXt-Base   Swin-Small   ViT-B/16
  (CNN optimal)    (CNN+Transf)   (Transf+CNN)  (Transformer)
     ~54M params      ~88M           ~49M          ~86M
```

---

## Training Strategy

### Sebelum Revisi (v3)
- Full fine-tuning langsung — semua layer update dari epoch 1
- Terbukti overfitting: gap train-val EfficientNet ~17%, ConvNeXt/Swin ~11-12%

### Setelah Revisi ✅ (v4) — Two-Stage Fine-Tuning

**Stage 1 (5 epoch) — Freeze Backbone:**
- Seluruh backbone di-freeze (tidak bisa update)
- Hanya classifier head yang ditraining
- LR lebih besar untuk konvergensikan head yang masih random
- Mencegah catastrophic forgetting pretrained features

**Stage 2 (max 45 epoch) — Unfreeze All:**
- Semua layer di-unfreeze
- Differential learning rate: backbone LR 10x lebih kecil dari head
- Early stopping patience=5 (monitor val_loss)
- Perubahan backbone halus dan terarah

**Diterapkan konsisten ke semua 4 model** — menjaga fairness perbandingan.

### Konfigurasi Training Lengkap

| Parameter | Nilai |
|---|---|
| Stage 1 epochs | 5 |
| Stage 2 epochs | max 45 |
| Early stopping patience | 5 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Mixed precision (AMP) | ✅ |
| Gradient clipping | max_norm=1.0 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Augmentasi tambahan | RandomErasing (p=0.2) |

---

## Metode Ensemble & Stacking

### Ensemble
| Metode | Deskripsi | Status |
|---|---|---|
| Simple Averaging (4 model) | Rata-rata probabilitas 4 model | ✅ |
| Weighted Voting (4 model) | Bobot dari val_accuracy masing-masing | ✅ |
| Best-3 (drop model terendah) | Otomatis drop model val_acc terendah | ✅ (dicoba, opsional) |

> Catatan: Best-3 tetap dicoba karena dengan model yang lebih seimbang, hasilnya mungkin berbeda dari eksperimen sebelumnya.

### Stacking
| Meta-Model | Status | Alasan |
|---|---|---|
| **Logistic Regression** | ✅ Digunakan | Terbukti terbaik di v3, interpretable |
| ~~MLP~~ | ❌ Dihapus | Kalah dari LR di v3, overfitting pada fitur kecil |
| ~~Random Forest~~ | ❌ Dihapus | Tidak signifikan berbeda dari LR |

---

## Metrik Evaluasi

| Metrik | Task 1 | Task 2 | Task 3 | Keterangan |
|---|---|---|---|---|
| **Accuracy** | ✅ | ✅ | ✅ | Metrik utama |
| **Macro F1** | ✅ | ✅ | ✅ | Metrik utama, tidak terpengaruh class imbalance |
| **Weighted F1** | ✅ | ✅ | ✅ | Baru v4 — mempertimbangkan proporsi kelas |
| **AUC-ROC** | ✅ | ✅ (OvR) | ✅ (OvR) | Baru v4 — tidak tergantung threshold |
| **Per-class F1** | ✅ | ✅ | ✅ | Baru v4 — penting untuk class imbalance |
| **Confusion Matrix** | ✅ | ✅ | ✅ | Visualisasi error pattern |
| ~~Cohen's Kappa~~ | ❌ | ❌ | ❌ | Tidak digunakan — sudah cukup dengan metrik di atas |
| FLOPs / Memory / Inference Time | ✅ | — | — | Hanya Task 1 untuk trade-off analysis |

> Macro F1 tetap sebagai metrik utama untuk klaim "model terbaik" agar comparable dengan paper CrisisMMD sebelumnya.

---

## Ablation Study

### Yang Direncanakan (implementasi menyusul)
| Ablation | Yang Dibandingkan | Feasibility |
|---|---|---|
| **Training strategy** | Full fine-tuning vs Two-stage fine-tuning | ✅ Feasible — hasil v3 vs v4 |
| **Ensemble composition** | 2 model (Swin+ConvNeXt) vs 3 model vs 4 model | ✅ Feasible — dari checkpoint yang ada |

### Yang Tidak Dimasukkan
| Ablation | Alasan Dihapus |
|---|---|
| ~~Training from scratch~~ | Terlalu lama (~12-20 jam), berisiko timeout Kaggle |
| ~~Ablation komponen augmentasi~~ | Belum dikonfirmasi wajib oleh dosen |

---

## Struktur Kode v4

```
Cell 1  : Install library (timm, thop)
Cell 2  : Import semua library
Cell 3  : Konfigurasi global (path, device, TASK_CONFIG, MODEL_CONFIG, TRAIN_CONFIG)
Cell 4  : Verifikasi dataset (cek kolom, split files, label Task 3)
Cell 5  : load_data() — baca TSV, filter label, apply split
Cell 6  : CrisisMMDDataset class
Cell 7  : get_transforms() — train (+ RandomErasing) & val/test
Cell 8  : create_dataloaders()
Cell 9  : create_model() + freeze_backbone() + unfreeze_all() + get_stage2_optimizer()
Cell 10 : AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint
Cell 11 : train_one_epoch() + validate()
Cell 12 : train_two_stage() — full two-stage loop
Cell 13 : plot_training_history() — dengan garis pemisah Stage 1/2
Cell 14 : evaluate_model() — Accuracy, Macro F1, Weighted F1, AUC-ROC, Per-class F1
Cell 15 : plot_confusion_matrix() + plot_per_class_f1()
Cell 16 : ensemble_predict() + evaluate_ensemble()
Cell 17 : run_stacking_lr() — Logistic Regression meta-model
Cell 18 : build_summary_table() — ranking + Macro vs Weighted F1 chart
Cell 19 : run_task() — master function yang memanggil semua cell di atas
Cell 20 : run_task('informative')   → Task 1 lengkap
Cell 21 : run_task('humanitarian')  → Task 2 lengkap
Cell 22 : run_task('damage')        → Task 3 lengkap
Cell 23 : print_cross_task_summary() — ringkasan lintas task + gabung CSV
```

---

## Output yang Dihasilkan

```
/kaggle/working/
├── checkpoints/
│   ├── efficientnetv2_m_informative_best.pth
│   ├── vit_informative_best.pth
│   ├── convnext_informative_best.pth
│   ├── swin_informative_best.pth
│   ├── efficientnetv2_m_humanitarian_best.pth
│   └── ... (dst untuk semua task)
│
└── results/
    ├── informative/
    │   ├── training_history.png
    │   ├── *_cm.png (confusion matrix per model)
    │   ├── per_class_f1.png
    │   ├── ensemble_*_cm.png
    │   ├── stacking_lr_cm.png
    │   ├── stacking_lr_importance.png
    │   ├── final_comparison_informative.png
    │   ├── summary_informative.csv
    │   └── results_informative.json
    ├── humanitarian/   (struktur sama)
    ├── damage/         (struktur sama)
    └── summary_all_tasks.csv  ← gabungan semua task
```

---

## Urutan Pengerjaan

```
✅ Selesai:
   - Eksperimen awal Task 1 dengan v3 (EfficientNet-B0, full fine-tuning)
   - Analisis hasil: overfitting, ensemble meracuni, feature importance
   - Bimbingan dosen → revisi model & training strategy

⏳ Selanjutnya:
   1. Run Cell 1–19 (setup v4) + Cell 20 (Task 1 dengan EfficientNetV2-M)
         ↓
   2. Evaluasi hasil Task 1 v4 — bandingkan dengan v3
      (ini sekaligus menjadi ablation: full fine-tuning vs two-stage)
         ↓
   3. Run Cell 21 (Task 2 — Humanitarian, 7 kelas)
         ↓
   4. Run Cell 22 (Task 3 — Damage Severity, 3 kelas)
         ↓
   5. Run Cell 23 (Cross-task summary)
         ↓
   6. Implementasi ablation study ensemble composition
         ↓
   7. Analisis & penulisan Bab 4
```

---

## Hal yang Masih Perlu Dikonfirmasi ke Dosen

| No | Pertanyaan | Prioritas |
|---|---|---|
| 1 | Ada paper SOTA CrisisMMD yang jadi acuan angka pembanding? | 🔴 Tinggi |
| 2 | Ablation study wajib di semua task atau Task 1 saja? | 🔴 Tinggi |
| 3 | Multiple random seed diperlukan untuk validitas statistik? | 🟡 Sedang |
| 4 | Analisis efisiensi (FLOPs, inference time) cukup di Task 1 saja? | 🟡 Sedang |

---

## Perkiraan Kontribusi Penelitian

1. **Perbandingan 4 arsitektur** yang merepresentasikan spektrum CNN → Transformer pada domain klasifikasi gambar bencana
2. **Analisis two-stage vs full fine-tuning** pada domain spesifik bencana (ablation study)
3. **Analisis efektivitas ensemble & stacking** dengan berbagai komposisi model
4. **Evaluasi 3 task sekaligus** pada CrisisMMD dalam satu kerangka eksperimen yang konsisten
5. **Analisis trade-off** akurasi vs efisiensi komputasi (FLOPs, memory, inference time)

---

*Revisi-001 — Disesuaikan setelah bimbingan dosen pertama.*
