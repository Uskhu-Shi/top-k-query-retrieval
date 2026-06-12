# syNNapse PS-1 — Vision-Based Inventory Re-Identification

> **Open-set image retrieval system** | ResNet-50 → Swin-T → DINOv2 benchmark  
> Architecture: PKSampler → Triplet + ArcFace Loss → FAISS Retrieval  
> Dataset: [Stanford Online Products (SOP)](https://cvgl.stanford.edu/projects/lifted_struct/)

---

## Problem Statement

Given a query image of a product (captured in-the-wild), retrieve the **top-K most visually similar items** from a large product gallery. This is an **open-set instance-level retrieval** task — at inference time the system sees items not present during training, so it must generalise via a learned metric space rather than class memorisation.

---

## Architecture Overview

```
SOP Dataset
    └─► PKBatchSampler (P=24 classes × K=4 images)
            └─► Backbone (ResNet-50 | Swin-Tiny | DINOv2 ViT-S/14)
                    └─► GeM Pooling / Global Pool
                            └─► 512-d Projection Head (BN + Linear + BN)
                                    └─► L2 Normalise ── ArcFace Head
                                                │              │
                                        Triplet Loss    ArcFace CE Loss
                                                └──── Combined Loss ────►
                                                              │
                                              FAISS Gallery Index (IndexFlatIP)
                                                              │
                                                    Top-K Retrieval & Eval
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Synnapse-PS-1.git
cd Synnapse-PS-1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebooks/Synnapse_PS1_improved.ipynb` in Google Colab (GPU runtime recommended).

The notebook will:
- Mount Google Drive and set up directory structure
- Download the Stanford Online Products dataset via `kagglehub`
- Train all three backbone variants sequentially
- Evaluate with FAISS retrieval and report Top-1/5/10 accuracy
- Build the gallery database and persist `.npy` files

### 4. Run the API server

```bash
cd Synnapse-PS-1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Test the health endpoint:
```bash
curl http://localhost:8000/health
```

Search with a query image:
```bash
curl -X POST "http://localhost:8000/search?k=5" \
     -F "file=@path/to/query_image.jpg"
```

---
## Project Structure

```
Synnapse-PS-1/
├── notebooks/
│   └── Synnapse_PS1_improved.ipynb   # Main training notebook
├── src/
│   ├── build_model.py                # Factory: instantiate any backbone by name
│   ├── feature_extraction/
│   │   ├── encoder.py                # ImageEncoder: PIL → 512-d numpy vector
│   │   └── build_gallery.py          # Build & persist FAISS gallery .npy files
│   ├── similarity_scoring_and_retrieval/
│   │   └── retriever.py              # FaissRetriever: load gallery, search top-K
│   └── api/
│       └── main.py                   # FastAPI REST server (GET /health, POST /search)
├── scripts/
│   └── upload_assets_to_hf.py        # Upload checkpoint + gallery to HuggingFace Hub
├── docs/                             # Additional documentation
├── requirements.txt
└── README.md
```

---

##  Models

| Backbone | Pooling | Emb. Dim | Pretrain |
|---|---|---|---|
| ResNet-50 | GeM (p=3.0, learnable) | 512 | ImageNet-1K V2 |
| Swin-Tiny | Global (native) | 512 | ImageNet-1K |
| **DINOv2 ViT-S/14** | CLS token | 512 | LVD-142M (self-supervised) |

All models share the same **ArcFace head** (s=64, m=0.35) and **512-d projection head** (LayerNorm/BN → Linear → BN → L2-norm).

DINOv2 fine-tuning strategy: all backbone weights frozen except the **last 4 transformer blocks** and final LayerNorm — balancing domain adaptation with preservation of self-supervised features.

---

## Loss Function

```
L_total = 1.0 × L_triplet(batch-hard) + 0.3 × L_ArcFace(label_smooth=0.1)
```

- **Batch-Hard Triplet Loss** (margin=0.3): mines the hardest positive and hardest negative within each PK batch, focusing gradients on the most informative pairs.
- **ArcFace CE Loss**: adds angular margin to the classification objective, enforcing inter-class separability on the unit hypersphere.
- The combination provides both global class structure (ArcFace) and fine-grained intra-class compactness (Triplet).

---

##  Data Loading

**PKBatchSampler** ensures every training batch contains exactly:
- `P = 24` unique item classes
- `K = 4` images per class
- Total batch size: **96 images**

This structured sampling guarantees rich hard-pair availability for the triplet miner within every batch.

---

##  Training Configuration

| Setting | ResNet-50 | Swin-Tiny | DINOv2 ViT-S |
|---|---|---|---|
| Epochs | 10 | 10 | 12 |
| lr (backbone) | 3e-5 | 2e-5 | 1e-5 |
| lr (head) | 3e-4 | 3e-4 | 3e-4 |
| Scheduler | OneCycleLR (cosine) | OneCycleLR | OneCycleLR |
| Early stop patience | 4 | 4 | 5 |
| Gradient clip | 5.0 | 5.0 | 5.0 |

---

##  Retrieval & Evaluation

- **Gallery index:** FAISS `IndexFlatIP` with L2-normalised embeddings (cosine similarity = inner product)
- **Evaluation:** Open-set Top-K accuracy — one random image per test item as query; self-matches excluded
- **TTA:** Average of 2 augmented views (centre-crop + horizontal flip) for DINOv2 at eval time

---

##  API Reference

### `GET /health`

```json
{
  "status": "ok",
  "ready": true,
  "device": "cuda",
  "gallery_size": 60502,
  "uptime_s": 42.1
}
```

### `POST /search`

**Parameters:**
- `file` (multipart): query image
- `k` (int, default 5, range 1–50): number of results to return

**Response:**
```json
{
  "k": 5,
  "results": [
    {"item_id": "11318", "score": 0.934, "ref": "path/to/matched/image.jpg"},
    ...
  ]
}
```

The API performs lazy initialisation on first request — model weights and gallery files are downloaded from Hugging Face Hub if not found locally.

---

## Deployment (Hugging Face Hub)

1. Set your HF token: `huggingface-cli login`
2. Set your repo: `export HF_REPO_ID="YOUR_USERNAME/Synnapse-ps1-checkpoints"`
3. Run: `python scripts/upload_assets_to_hf.py`

This uploads the model checkpoint and pre-built gallery `.npy` files. The API server auto-downloads them on cold start.

---

##  Requirements

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
faiss-cpu>=1.7.4
pytorch-metric-learning>=2.3.0
kagglehub>=0.2.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
huggingface_hub>=0.22.0
numpy>=1.24.0
Pillow>=9.4.0
tqdm>=4.65.0
matplotlib>=3.7.0
pandas>=2.0.0
```

---
