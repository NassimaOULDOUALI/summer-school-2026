# 🧠 Build a Large Language Model — From Scratch
### Summer School 2026 · Hi! PARIS 

> *"The best way to understand how something works is to build it yourself."*

This repository contains the hands-on demo notebook for the **Hi! PARIS Summer School 2026**, covering **Chapters 3 to 6** of the book [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka.

By the end of this notebook, you will have implemented **GPT-2 (124M parameters) from scratch in PyTorch** and generated coherent text using real pretrained weights from OpenAI.

---

## 📋 What's Inside

| Chapter | Section | Topics |
|---------|---------|--------|
| **Ch. 3** | 00 – 04 | Self-attention · Trainable Q/K/V weights · Causal masking · Dropout · Multi-head attention |
| **Ch. 4** | 05 – 10 | Layer Norm · GELU activation · FeedForward · Shortcut connections · TransformerBlock · Full GPTModel |
| **Ch. 5** | 11 – 12 | Text generation (greedy) · Cross-entropy loss · Training loop structure |
| **Ch. 6** | 13 – 14 | Loading pretrained GPT-2 weights from OpenAI · Temperature & top-k sampling · Inference demo |

**74 cells** · ~2 hours to run fully · Self-contained (no prior LLM knowledge required)

---

## ⚡ Quick Start

### Option 1 — Google Colab (recommended, no install)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NassimaOULDOUALI/summer-school-2026/blob/main/summer_school_demo_complete.ipynb)

> ⚠️ In Colab: go to **Runtime → Change runtime type → T4 GPU** before running.

### Option 2 — Run locally

```bash
# 1. Clone the repo
git clone https://github.com/NassimaOULDOUALI/summer-school-2026.git
cd summer-school-2026

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken numpy

# 4. Launch Jupyter
jupyter notebook summer_school_demo_complete.ipynb
```

---

## 🗺️ Notebook Roadmap

```
Ch.3 — Attention Mechanism
  ├── 00  Simple self-attention (no trainable weights)
  ├── 01  Self-attention v1  (nn.Parameter — raw initialization)
  ├── 02  Self-attention v2  (nn.Linear — Kaiming init)
  ├── 03  Causal attention + Dropout  (autoregressive masking)
  └── 04  Multi-head attention  (efficient weight-split approach)

Ch.4 — GPT Architecture
  ├── 05  GPT-2 (124M) config + DummyGPTModel
  ├── 06  Layer Normalization  (scale + shift)
  ├── 07  GELU activation + FeedForward module
  ├── 08  Shortcut connections  (residual stream)
  ├── 09  TransformerBlock  (full pre-norm block)
  └── 10  GPTModel  (complete architecture, 163M params)

Ch.5 — Training
  ├── 11  Greedy text generation  (argmax decoding)
  └── 12  Training loop  (cross-entropy loss, AdamW)

Ch.6 — Pretrained Weights & Inference
  ├── 13  Loading OpenAI GPT-2 weights  (~500 MB download)
  └── 14  Advanced generation  (temperature + top-k sampling)
```

---

## 🔑 Key Concepts You Will Implement

- **Scaled dot-product attention**: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
- **Causal masking**: upper-triangular `-inf` mask → prevents attending to future tokens
- **Multi-head attention**: parallel attention heads in a single efficient matrix multiply
- **Layer Normalization**: `x̂ = (x − μ) / √(σ² + ε)` with learnable scale and shift
- **GELU activation**: `GELU(x) = x · Φ(x)` — smooth gradient flow vs ReLU
- **Residual stream**: `output = LayerNorm(x) → SubLayer → Dropout + x`
- **Weight tying**: token embedding matrix shared with the output projection (saves 38M params)

---

## 🖥️ Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 |
| tiktoken | ≥ 0.5 |
| numpy | ≥ 1.24 |
| CUDA (optional) | ≥ 11.8 |

The notebook runs on **CPU** (slower) or **GPU** (recommended for Section 13+ which loads GPT-2 weights).

---

## 📖 Reference

This notebook is based on:

> **Sebastian Raschka** — *Build a Large Language Model (From Scratch)*  
> Manning Publications, 2024  
> 🔗 [Book page](https://www.manning.com/books/build-a-large-language-model-from-scratch) · [GitHub repo](https://github.com/rasbt/LLMs-from-scratch)

The original code has been reorganized, annotated, and extended for pedagogical clarity as part of the Hi! PARIS Summer School 2026 curriculum.

---

## 👩‍🏫 Author

**Nassima Ould Ouali**  
Research Engineer in Speech AI · Hi! PARIS / École Polytechnique  
[GitHub](https://github.com/NassimaOULDOUALI) · [Hi! PARIS](https://hi-paris.fr)

---

## 📄 License

Code: [MIT License](LICENSE)  
Original book content © Sebastian Raschka / Manning Publications — used for educational purposes.
