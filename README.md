<div align="center">

#  Inventory Item Re-Identification
### Vision-Based Open-Set Retrieval · DINOv2 · ArcFace · FAISS

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2_ViT--S/14-7952B3?style=flat-square)](https://github.com/facebookresearch/dinov2)
[![FAISS](https://img.shields.io/badge/Retrieval-FAISS-009BDE?style=flat-square)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Top-5 Retrieval Accuracy: ~91% on Stanford Online Products (22k test classes)**

*Built for the syNNapse '26 Hackathon — Electronics Engineering Society, UDYAM '26*

[Problem Statement](#problem-statement) · [Architecture](#architecture) · [Results](#results) · [Quickstart](#quickstart) · [API](#api-reference) · [Repo Structure](#repository-structure)

</div>

---

## Problem Statement

Large-scale enterprises — warehouses, logistics networks, retailers — manage hundreds of thousands of visually similar inventory items. Physical labels degrade; barcodes go missing. Human verification doesn't scale.

**Goal:** Build a model-agnostic visual recognition system that identifies any inventory item purely from an image, without barcodes or metadata, by matching it against a pre-built embedding database.

Two core modules are required:

| Module | Task |
|--------|------|
| **A — Visual Feature Extraction** | Convert an item image into a compact, robust embedding vector |
| **B — Similarity Scoring & Retrieval** | Given a query, rank the gallery by cosine similarity and return Top-K matches |

---

## Architecture

```
Query Image
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Module A — Visual Encoder                              │
│                                                         │
│  DINOv2 ViT-S/14 (last 4 blocks unfrozen)              │
│       │                                                 │
│  LayerNorm → Linear(384→512) → BN  ← Projection Head   │
│       │                                                 │
│  L2-Normalise → 512-d Unit Embedding                   │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Module B — FAISS   │
          │  IndexFlatIP        │
          │  Cosine Similarity  │
          │  Top-K Retrieval    │
          └──────────┬──────────┘
                     │
              Top-K Item IDs
           + Similarity Scores
```

### Why these choices?

| Design Decision | Rationale |
|---|---|
| **DINOv2 ViT-S/14** | Self-supervised ViT pretrained on 142M images; generalises to novel inventory categories out-of-the-box; no ImageNet classification bias |
| **Unfreeze last 4 blocks only** | Preserves robust low/mid-level representations; adapts high-level semantics to domain without catastrophic forgetting |
| **ArcFace Head (s=64, m=0.35)** | Additive angular margin forces tight intra-class clusters and wide inter-class separation in embedding space — directly benefits retrieval |
| **GeM Pooling (ResNet)** | Generalised Mean Pooling emphasises dominant activations; outperforms AvgPool for retrieval by 3–5% |
| **Batch-Hard Triplet Loss** | Mines hardest positive/negative in each PK batch; directly optimises the metric space for Top-K retrieval |
| **FAISS IndexFlatIP** | Exact cosine similarity over L2-normalised vectors; O(1) index size for gallery up to ~60k items |
| **Test-Time Augmentation** | Average of 2 views (center-crop + h-flip) at inference; ~1–2% Top-5 gain at zero training cost |

---

## Benchmark: ResNet-50 vs Swin-T vs DINOv2

All models trained with identical loss (Batch-Hard Triplet + ArcFace CE), PK sampler (P=24, K=4), and OneCycleLR scheduler on Stanford Online Products.

| Model | Backbone | Pooling | Top-1 | **Top-5** | Top-10 |
|---|---|---|---|---|---|
| ResNet-50 | CNN | GeM | ~55% | ~75% | ~82% |
| Swin-Tiny | Transformer | Global Avg | ~63% | ~82% | ~88% |
| DINOv2 ViT-S/14 | ViT (SSL) | CLS token | ~72% | ~89% | ~93% |
| **DINOv2 + TTA** | ViT (SSL) | CLS avg | **~74%** | **~91%** | **~95%** |

> Primary metric per PS: **Top-5 Retrieval Accuracy** (open-set, instance-level evaluation).

---

## Results

### Sample Retrieval Output

```
Query Image → Top-5 Retrieved Gallery Items
┌───────┐    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Query │ →  │ 0.97 │ │ 0.96 │ │ 0.95 │ │ 0.94 │ │ 0.92 │
│  item │    │  ✅  │ │  ✅  │ │  ✅  │ │  ✅  │ │  ✅  │
└───────┘    └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
                          Cosine Similarity Scores
```

Green border = correct item ID · Red border = incorrect match

Sample output images are available in `sample_output/`.

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Synnapse-PS-1.git
cd Synnapse-PS-1
pip install -r requirements.txt
```

### 2. Download Dataset

```python
import kagglehub
path = kagglehub.dataset_download("liucong12601/stanford-online-products-dataset")
```

Or directly from [Kaggle: Stanford Online Products](https://www.kaggle.com/datasets/liucong12601/stanford-online-products-dataset).

### 3. Download Pretrained Checkpoint

```bash
python scripts/download_checkpoint.py
# Downloads dinov2_vits14_best.pth → models/
```

### 4. Build the Gallery Database

```python
from src.build_model import build_model
from src.feature_extraction import build_gallery_database
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_ctor():
    return build_model("dinov2_vits14", num_classes=11318, emb_dim=512, device=device)

gallery_emb, gallery_ids, gallery_refs = build_gallery_database(
    model_ctor  = model_ctor,
    ckpt_path   = "models/dinov2_vits14_best.pth",
    gallery_loader = test_loader,
    device      = device,
    out_dir     = "features",
    normalize   = True,
)
```

### 5. Run Retrieval (Python)

```python
from src.similarity_scoring_and_retrieval import FaissRetriever
from src.feature_extraction import ImageEncoder
from PIL import Image

encoder   = ImageEncoder(model_ctor(), ckpt_path="models/dinov2_vits14_best.pth")
retriever = FaissRetriever("features/gallery_embeddings.npy",
                           "features/gallery_item_ids.npy",
                           "features/gallery_refs.npy")

q_emb             = encoder.encode_pil(Image.open("query.jpg"))
top_ids, scores, idxs, refs = retriever.search(q_emb, k=5)

for rank, (item_id, score) in enumerate(zip(top_ids, scores), 1):
    print(f"  Rank {rank}: item_id={item_id}  similarity={score:.4f}")
```

---

## API Reference

### Start the server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### `GET /health`
Returns model readiness, device info, and gallery size.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "ready": true,
  "device": "cuda",
  "gallery_size": 60502
}
```

#### `POST /search`
Upload a query image; returns Top-K matching inventory items.

```bash
curl -X POST "http://localhost:8000/search?k=5" \
     -F "file=@query_item.jpg"
```

```json
{
  "k": 5,
  "results": [
    {"item_id": "11337", "score": 0.9712, "ref": "path/to/image.jpg"},
    {"item_id": "11337", "score": 0.9658, "ref": "path/to/image2.jpg"},
    {"item_id": "11338", "score": 0.9421, "ref": "path/to/image3.jpg"}
  ]
}
```

**Interactive docs:** `http://localhost:8000/docs`

---

## Repository Structure

```
Synnapse-PS-1/
│
├── src/
│   ├── __init__.py
│   ├── build_model.py                  # Factory: resnet50 | swin_tiny | dinov2_vits14
│   │
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── encoder.py                  # ImageEncoder: PIL → 512-d numpy vector
│   │   └── build_gallery.py            # Builds gallery_embeddings.npy from DataLoader
│   │
│   ├── similarity_scoring_and_retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py                # FaissRetriever: FAISS IndexFlatIP search
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py                     # FastAPI app: /health + /search endpoints
│
├── scripts/
│   ├── download_checkpoint.py          # Pull best.pth from Hugging Face Hub
│   └── upload_assets_to_hf.py          # Push checkpoint + gallery .npy to HF
│
├── docs/
│   └── Synnapse_PS1_notebook.ipynb     # Full training + benchmark notebook
│
├── features/                           # Auto-generated (git-ignored)
│   ├── gallery_embeddings.npy
│   ├── gallery_item_ids.npy
│   └── gallery_refs.npy
│
├── models/                             # Auto-generated (git-ignored)
│   └── dinov2_vits14_best.pth
│
├── sample_output/
│   ├── retrieval_sample_1.png
│   ├── retrieval_sample_2.png
│   └── benchmark_comparison.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Training Pipeline (Summary)

```
Stanford Online Products (120k train images, 22k test images)
         │
         ▼
  PKBatchSampler (P=24 classes × K=4 images = 96/batch)
         │
         ▼
  DINOv2 ViT-S/14
  └── last 4 transformer blocks unfrozen
  └── Projection Head: LayerNorm → Linear(384→512) → BN
  └── ArcFaceHead (s=64, m=0.35, num_classes=11318)
         │
         ▼
  Combined Loss:
  └── λ=1.0 × BatchHardTriplet(margin=0.3)
  └── λ=0.3 × ArcFace-CrossEntropy(label_smoothing=0.1)
         │
         ▼
  OneCycleLR (10% warmup, cosine anneal)
  Gradient Clipping = 5.0
  AdamW (lr_head=3e-4, lr_backbone=1e-5, wd=1e-4)
         │
         ▼
  Checkpoint saved every epoch + best model saved separately
         │
         ▼
  FAISS Gallery Build (TTA: center-crop + h-flip average)
         │
         ▼
  Top-5 Retrieval Accuracy: ~91%
```

---

## Evaluation Methodology

Following the PS specification exactly:

1. **Instance-level split** — SOP provides non-overlapping train/test item IDs (verified: 0 overlap)
2. **Query selection** — one random image per test item ID designated as query
3. **Gallery** — all remaining test images (self-match excluded from results)
4. **Metric** — Top-K Accuracy: hit if correct item_id appears in Top-K retrieved results

```
Top-1  Accuracy : ~74%
Top-5  Accuracy : ~91%   ← Primary metric
Top-10 Accuracy : ~95%
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.1, torchvision |
| Backbone | DINOv2 ViT-S/14 (facebookresearch), timm (Swin-T) |
| Metric Learning | pytorch-metric-learning |
| Vector Search | FAISS (IndexFlatIP) |
| API | FastAPI + Uvicorn |
| Model Hub | Hugging Face Hub |
| Data | Stanford Online Products (Kaggle) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
syNNapse '26 · Electronics Engineering Society · UDYAM '26
</div>
