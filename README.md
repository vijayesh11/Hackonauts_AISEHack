# 🌫️ AISEHack Phase 2 — Theme 2: Urban Pollution Forecasting

---

## 🏆 Competition Overview

This repository contains our submission for **AISEHack 2026 — Theme 2: Pollution Prediction**, a national AI for Science & Engineering Hackathon organized by the **Anusandhan National Research Foundation (ANRF)**, co-organized by **IBM** and **IIT Delhi**, with the Grand Finale held at **IIIT Hyderabad (4–5 April 2026)**.

> *"Tackle urban air quality by developing physics/operator-based deep learning models to forecast pollution levels, enabling smarter city planning and public health interventions."*
> — AISEHack Official Theme Statement

| Detail | Info |
|---|---|
| **Competition** | ANRF AISEHack Phase 2 · Theme 2 |
| **Platform** | Kaggle (Notebook-only, reproducible) |
| **Task** | Spatio-Temporal PM2.5 Forecasting |
| **Evaluation Metric** | sMAPE (Symmetric Mean Absolute Percentage Error) |
| **Competition Baseline** | 0.7780 |
| **Our Score** | **0.8581** |
| **Improvement** | **+8 points over baseline** |
| **Model Checkpoints** | [Available on Kaggle Dataset](https://www.kaggle.com/datasets/mokshajnaundavalli/aisehack-phase2-top-submission-weights/) |

---

## 📌 Problem Statement

Forecast **PM2.5 concentrations** (fine particulate matter pollution) across a spatial grid of **140 × 124** pixels for **16 future time steps**, given **10 historical time steps** of multi-variable meteorological and emission data.

This is a high-impact **spatio-temporal deep learning** challenge where the model must capture:
- **Spatial patterns** — pollution dispersal across urban geography
- **Temporal dynamics** — how pollution evolves over future time steps
- **Episodic spikes** — sudden high-pollution events critical for public health alerts

---

## 🗃️ Dataset Description

The dataset is provided by **IIT Delhi** and consists of multi-channel `.npy` arrays organized by month. Each file represents a spatially gridded atmospheric variable over a `140 × 124` domain.

### Input Features (15 channels)

| Feature | Type | Description |
|---|---|---|
| `cpm25` | **Target** | PM2.5 fine particulate matter concentration |
| `q2` | Meteorological | Specific humidity at 2m height |
| `t2` | Meteorological | Temperature at 2m height |
| `u10` | Meteorological | Zonal (East–West) wind component at 10m |
| `v10` | Meteorological | Meridional (North–South) wind component at 10m |
| `swdown` | Meteorological | Downward shortwave solar radiation |
| `pblh` | Meteorological | Planetary Boundary Layer Height |
| `psfc` | Meteorological | Surface pressure |
| `rain` | Meteorological | Precipitation |
| `NH3` | Emission | Ammonia emissions |
| `SO2` | Emission | Sulfur dioxide emissions |
| `NOx` | Emission | Nitrogen oxides emissions |
| `NMVOC_e` | Emission | Non-methane VOC anthropogenic emissions |
| `NMVOC_finn` | Emission | Non-methane VOC fire/biomass emissions |
| `bio` | Emission | Biogenic emissions |

### Data Structure

```
raw/
  └── <month>/
        ├── cpm25.npy        # shape: (T, 140, 124)
        ├── q2.npy
        ├── t2.npy
        └── ... (all 15 features)

test_in/
  ├── cpm25.npy              # shape: (218, 10, 140, 124)
  └── ... (all 15 features)
```

**Prediction output:** `preds.npy` → shape `(218, 140, 124, 16)`
218 test samples × spatial grid 140×124 × 16 forecast horizons

---

## 🧠 Our Solution: Ultra-Wide 4-Seed DeepUNet Ensemble

### Architecture: `DeepUNet`

We built a heavily modified **UNet** with deep residual blocks, CBAM attention, dual prediction heads, and learned output calibration.

```
Input Tensor (170 channels × 140 × 124)
   ├── 15 features × 10 lookback  =  150 channels  (Z-score normalized)
   ├── lat/lon coordinate grids   =   20 channels  (positional encoding)
   └── month index embedding      =   10 channels  (temporal context)
              │
   ┌──────────▼──────────────────────────────────┐
   │                  ENCODER                    │
   │   ResBlock(170 → 160)  ──►  MaxPool2d        │
   │   ResBlock(160 → 320)  ──►  MaxPool2d        │
   │   ResBlock(320 → 640)  ──►  MaxPool2d        │
   └──────────────────────┬──────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────┐
   │                 BOTTLENECK                  │
   │           ResBlock(640 → 1024)              │
   └──────────────────────┬──────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────┐
   │          DECODER  (with skip connections)   │
   │  ConvTranspose2d  +  ResBlock(1280 → 640)   │
   │  ConvTranspose2d  +  ResBlock( 640 → 320)   │
   │  ConvTranspose2d  +  ResBlock( 320 → 160)   │
   └──────────────────────┬──────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────┐
   │             DUAL OUTPUT HEADS               │
   │  head_main : Conv → GELU → Conv(→ 16)       │
   │  head_ep   : Conv → GELU → Conv(→ 16)       │
   │  σ-gated blend  +  learnable horizon scale  │
   └─────────────────────────────────────────────┘
             Output: (B, 16, 140, 124)
```

### Key Technical Components

| Component | Role |
|---|---|
| **CBAM Attention** | Channel + Spatial attention inside every ResBlock for focused feature learning |
| **ResBlock** | Conv → BN → GELU → Conv → BN → GELU → CBAM + skip connection |
| **Dual Head** | `head_main` captures global distribution; `head_ep` specializes on high-pollution episodes |
| **Learned Blend** | σ-gated parameter dynamically weights contributions of both heads |
| **Horizon Scale** | Per-step learnable scaling (sigmoid-gated) calibrates each of the 16 forecast steps |
| **Mixed Precision** | FP16 training via `torch.amp.GradScaler` for GPU memory efficiency |

---

## 📉 Loss Function: Episode-Aware sMAPE

Standard sMAPE treats all pixels equally. We designed a custom loss that **upweights high-pollution episodic events** — the most critical events for public health decisions.

```
Base sMAPE:
    L(p, y)  =  2 × |p − y| / (|p| + |y| + ε)

Episode Loss  (top-85th percentile pixels):
    L_ep     =  mean( L(p,y)  where  y ≥ quantile(y, 0.85) )

Combined sMAPE:
    L_smape  =  0.55 × L_global  +  0.45 × L_episode

Dual Head Loss:
    L_total  =  0.40 × L_main  +  0.40 × L_ep_head  +  0.20 × L_blend
```

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Spatial Grid | 140 × 124 |
| Lookback Window | 10 timesteps |
| Forecast Horizon | 16 timesteps |
| Total Input Channels | 170 |
| Batch Size | 4 |
| Optimizer | AdamW (lr = 6.5e-4, weight decay = 1e-4) |
| LR Scheduler | CosineAnnealingWarmRestarts (T₀ = 5, η_min = 1e-5) |
| Max Epochs | 15 with early stopping (patience = 4) |
| Dataset Stride | 2 |
| Validation Split | 10% |
| Normalization | Z-score per feature (global statistics across all months) |
| Hardware | Kaggle GPU (CUDA) |
| Precision | Mixed FP16 |

---

## 🔁 Inference: 4-Seed Ensemble + Vertical TTA

To maximize robustness and generalization, we train **4 independent models** with different random seeds and combine them using weighted averaging. We also apply **Test-Time Augmentation (TTA)** via vertical flip.

```
Seeds trained   :  { 42, 43, 44, 45 }
Ensemble weights:  [ 0.28, 0.28, 0.24, 0.20 ]

For each test sample:
  Step 1 → Build input tensor (features + coord channels + month encoding)
  Step 2 → Forward pass on original orientation
  Step 3 → Forward pass on vertically-flipped input → flip predictions back
  Step 4 → TTA average:  (pred_orig + pred_flipped) / 2
  Step 5 → Weighted sum across 4 models
  Step 6 → Denormalize → Clip to [0, 999]

Final output: preds.npy  →  shape (218, 140, 124, 16)
```

---

## 📊 Results

| Model | sMAPE Score |
|---|---|
| Competition Baseline | 0.7780 |
| Our DeepUNet Ensemble (4-seed + Vertical TTA) | **0.8581** |
| **Improvement over baseline** | **+8.01 points** |

---

## 📥 Model Checkpoints & Pre-trained Weights

We have published our trained model checkpoints and predictions on Kaggle for reproducibility and benchmarking purposes.

### Kaggle Dataset: Pre-trained Weights & Results

**Dataset Link:** [AISEHack Phase 2 - Top Submission Weights](https://www.kaggle.com/datasets/mokshajnaundavalli/aisehack-phase2-top-submission-weights/)

This dataset contains:
- ✅ **Trained model weights** for all 4 ensemble models (seeds 42, 43, 44, 45)
- ✅ **Final predictions** (`preds.npy`) — shape (218, 140, 124, 16)
- ✅ **Training logs** and validation metrics
- ✅ **Reproducibility scripts** for inference

### How to Use Pre-trained Models

1. **Option A: Download from Kaggle Dataset**
   ```bash
   kaggle datasets download -d mokshajnaundavalli/aisehack-phase2-top-submission-weights
   unzip aisehack-phase2-top-submission-weights.zip
   ```

2. **Option B: Use in Kaggle Notebook (Recommended)**
   - Mount the dataset directly in your notebook:
   ```python
   import os
   checkpoint_path = "/kaggle/input/aisehack-phase2-top-submission-weights/"
   model_files = os.listdir(checkpoint_path)
   print("Available checkpoints:", model_files)
   
   # Load pre-trained weights
   model.load_state_dict(torch.load(f"{checkpoint_path}/model_1.pt"))
   ```

3. **Option C: Use Pre-computed Predictions**
   - Skip training entirely and use our final predictions:
   ```python
   preds = np.load("/kaggle/input/aisehack-phase2-top-submission-weights/preds.npy")
   # Shape: (218, 140, 124, 16)
   np.save("/kaggle/working/preds.npy", preds)
   ```

---

## 🗂️ Repository Structure

```
├── aise6.ipynb                    # Full training + inference pipeline (Kaggle Notebook)
├── README.md                      # This file
├── LICENSE                        # License
└── Kaggle Dataset Reference
    └── aisehack-phase2-top-submission-weights/
        ├── model_1.pt            # Seed 42 checkpoint
        ├── model_2.pt            # Seed 43 checkpoint
        ├── model_3.pt            # Seed 44 checkpoint
        ├── model_4.pt            # Seed 45 checkpoint
        ├── preds.npy             # Final predictions
        └── README.txt            # Dataset documentation
```

---

## 🚀 Reproducing Results

> All submissions are backed by a reproducible Kaggle Notebook as per AISEHack competition rules (Sections 6, 7, 8, 9). No external datasets, private artifacts, or pre-trained weights are used beyond the official competition data.

### Method 1: Train from Scratch

1. Open `aise6.ipynb` on Kaggle with **GPU accelerator** enabled
2. Attach the competition dataset:
   ```
   anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd
   ```
3. Run all cells — training 4 models takes approximately **45–60 minutes** on a T4 GPU
4. Output `preds.npy` (shape: `218 × 140 × 124 × 16`) is saved to `/kaggle/working/`
5. Submit the output file to the Kaggle leaderboard

### Method 2: Use Pre-trained Checkpoints

1. Open Kaggle Notebook
2. Attach both datasets:
   - `anrf-aise-hack-phase-2-theme-2-pollution-forecasting-iitd` (competition data)
   - `aisehack-phase2-top-submission-weights` (pre-trained models)
3. Load checkpoint and run inference:
   ```python
   checkpoint_path = "/kaggle/input/aisehack-phase2-top-submission-weights/model_1.pt"
   model.load_state_dict(torch.load(checkpoint_path))
   model.eval()
   # Run inference...
   ```
4. Submit predictions

### Method 3: Use Final Predictions (Fastest)

1. Open Kaggle Notebook
2. Attach the weights dataset:
   - `aisehack-phase2-top-submission-weights`
3. Load and submit:
   ```python
   preds = np.load("/kaggle/input/aisehack-phase2-top-submission-weights/preds.npy")
   np.save("/kaggle/working/preds.npy", preds)
   # Submit preds.npy directly
   ```

---

## 📦 Dependencies

All packages are pre-installed in the **Kaggle Python Docker environment**:

```
torch           — Deep learning framework (CUDA-enabled)
numpy           — Array and numerical operations
pandas          — Data utilities
tqdm            — Training progress tracking
scipy.ndimage   — Image filtering and morphological operations
```

**Requirements File (if running locally):**
```
torch>=2.0.0
numpy>=1.21
pandas>=1.3
tqdm>=4.50
scipy>=1.7
```

---

## 🌍 Real-World Impact

PM2.5 pollution contributes to millions of premature deaths globally each year. Accurate spatial forecasting of PM2.5 levels directly enables:

- **Early warning systems** for hospitals and vulnerable populations
- **Smarter city planning** — traffic routing, industrial regulation
- **Public health interventions** during pollution spikes and hazardous episodes
- **Climate policy-making** backed by data-driven spatial predictions

This work aligns with the **Viksit Bharat 2047** vision and the **MAHA AI for Science and Engineering Mission** — building AI-powered infrastructure for a cleaner, healthier India.

---

## 📄 License

See [LICENSE](./LICENSE) for details.

---

## 📞 Citation

If you use this work in your research or development, please cite:

```bibtex
@competition{aisehack2026,
  title={AISEHack 2026 — Theme 2: Urban Pollution Forecasting},
  organization={Anusandhan National Research Foundation (ANRF)},
  note={Co-organized by IBM India & IIT Delhi, Grand Finale at IIIT Hyderabad (April 4-5, 2026)},
  year={2026}
}

@dataset{aisehack_weights_2026,
  title={AISEHack Phase 2 - Top Submission Weights},
  author={Team Name},
  year={2026},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/mokshajnaundavalli/aisehack-phase2-top-submission-weights/}
}
```

---

<div align="center">

**Built for AISEHack 2026** · *Organized by ANRF India* · *Co-organized by IBM & IIT Delhi* · *Grand Finale at IIIT Hyderabad*

✨ **Score: 0.8581** | **Baseline: 0.7780** | **Improvement: +8.01%** ✨

</div>
