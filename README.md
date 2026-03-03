# 🤖 Generative AI — Assignment #1

**Course:** Generative AI (AI4009) — Spring 2026  
**Student:** Muhammad Idrees (i230582)  
**University:** National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad  
**GitHub:** [@code-with-idrees](https://github.com/code-with-idrees)

---

## 📚 Assignment Overview

This repository contains the complete implementation and report for **Assignment #1** of the Generative AI course. The assignment focuses on building a Neural Machine Translation (NMT) system from scratch using a vanilla RNN architecture.

---

## 📂 Repository Structure

```
Generative-AI-Assignment-1/
│
├── Q1_Neural_Machine_Translation/       # Question 1: NMT System
│   ├── code/
│   │   ├── Q1_Machine_Translation.py   # Full implementation (all 9 tasks)
│   │   └── Generative_AI_Q1.ipynb      # Jupyter Notebook version
│   ├── report/
│   │   ├── Q1_Machine_Translation_Report.tex    # LaTeX report (Springer LNCS)
│   │   └── Neural_Machine_Translation_Report.pdf  # Compiled PDF
│   ├── results/
│   │   ├── training_curves.png         # Grid search loss curves
│   │   └── best_model_curves.png       # Best model training curves
│   ├── assets/
│   │   ├── prompts.txt                 # AI prompts used in development
│   │   └── Spring2026_GenAI_Assignment_1.pdf   # Assignment spec
│   └── README.md                       # Detailed Q1 documentation
│
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 🧠 Question 1: Neural Machine Translation (English → Urdu)

> **[📁 View Q1 Code & Details →](Q1_Neural_Machine_Translation/)**

A complete NMT system using a **vanilla RNN Encoder-Decoder** (strictly `nn.RNN` — no LSTM/GRU/Transformer) for English-to-Urdu translation.

### Tasks Completed

| # | Task | Description |
|---|------|-------------|
| 1 | **Data Preprocessing** | NFKD/NFKC normalization, lowercasing, punctuation handling |
| 2 | **Dataset Splitting** | 80/10/10 train-val-test split (seed=42), zero overlap verified |
| 3 | **Tokenization** | Word-level vocab (EN: 3,969 tokens, UR: 4,313 tokens) |
| 4 | **Data Batching** | Sequence encoding, padding, masking, PyTorch DataLoader |
| 5 | **Model Architecture** | Vanilla RNN encoder-decoder (6.17M params, 2-layer, hidden=512) |
| 6 | **Model Training** | Adam optimizer, gradient clipping, LR scheduling, early stopping |
| 7 | **Hyperparameter Tuning** | Grid search across 8 configurations |
| 8 | **Evaluation** | Greedy & Beam Search (k=5) decoding, BLEU-1/2/3/4 scores |
| 9 | **Error Analysis** | Analysis of 15 translations + vanilla RNN limitations |

### Key Results

| Metric | Greedy | Beam Search (k=5) |
|--------|--------|-------------------|
| BLEU-1 | 0.1158 | 0.0733 |
| BLEU-4 | 0.0003 | 0.0062 |
| Best Val Loss | — | **5.6000** |

---

## 🛠 Tech Stack

- **Python 3.10+** · **PyTorch** (`nn.RNN` only) · **NLTK** · **Google Colab (T4 GPU)**
- **pandas** · **unicodedata** · **LaTeX (Springer LNCS)**

---

## 📄 License

This project is for educational purposes as part of the Generative AI course at FAST-NUCES, Spring 2026.
