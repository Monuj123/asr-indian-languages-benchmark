# 🎙️ ASR Benchmark — Top 5 Spoken Indian Languages

> Benchmarking State-of-the-Art Automatic Speech Recognition systems across **Hindi**, **Bengali**, **Marathi**, **Telugu**, and **Tamil**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Notebooks](#running-the-notebooks)
- [Requirements](#requirements)
- [Experimental Results](#experimental-results)
- [Conclusion](#conclusion)
- [References](#references)

---

## Overview

This project provides a reproducible, end-to-end benchmark for evaluating state-of-the-art Automatic Speech Recognition (ASR) models on India's five most spoken languages. Each language notebook follows a consistent pipeline — dataset loading, model inference, and metric computation (WER & CER) — enabling fair cross-model and cross-language comparisons.

---

## Repository Structure

```
asr-indian-languages-benchmark/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── notebooks/
    ├── 01_speech_signal_processing.ipynb  # Waveforms, MFCC, spectrograms
    ├── 02_hindi.ipynb
    ├── 03_bengali.ipynb
    ├── 04_marathi.ipynb
    ├── 05_telugu.ipynb
    └── 06_tamil.ipynb
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/asr-indian-languages-benchmark.git
cd asr-indian-languages-benchmark
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter lab
```

---

## Running the Notebooks

| Notebook | Description |
|---|---|
| `01_speech_signal_processing.ipynb` | Foundations — waveforms, STFT, spectrograms, MFCCs |
| `02_hindi.ipynb` | Dataset loading, model evaluation, WER/CER for Hindi |
| `03_bengali.ipynb` | Dataset loading, model evaluation, WER/CER for Bengali |
| `04_marathi.ipynb` | Dataset loading, model evaluation, WER/CER for Marathi |
| `05_telugu.ipynb` | Dataset loading, model evaluation, WER/CER for Telugu |
| `06_tamil.ipynb` | Dataset loading, model evaluation, WER/CER for Tamil |

> ⚠️ **Start with `01_speech_signal_processing.ipynb`** before running any language notebook.

Datasets are loaded directly from [HuggingFace Datasets](https://huggingface.co/datasets) — no manual download required.

---

## Requirements

- **Python** 3.9+
- **GPU** recommended for model inference (CUDA 11.8+)

### Key dependencies

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.38.0
datasets>=2.18.0
openai-whisper>=20231117
librosa>=0.10.1
jiwer>=3.0.3        # WER/CER computation
```

See [`requirements.txt`](requirements.txt) for the full list.

---

## Experimental Results

Top-3 SOTA models evaluated per language. Metrics: **Word Error Rate (WER ↓)** and **Character Error Rate (CER ↓)**.

> Results will be updated upon completion of benchmarking runs.

### Hindi

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices | 4750 | 16.6(CTC),15.3(RNNT) | 7.3(CTC),7.2(RNNT) |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |

### Bengali

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |

### Marathi

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual | IndicVoices | 3552 | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |

### Telugu

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |

### Tamil

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |

---

## Conclusion

This project establishes a reproducible benchmark for ASR across India's five most spoken languages. Key observations from the experiments:

- **[Finding 1]** — Summarise best performing model type and why.
- **[Finding 2]** — Note performance gap across languages and likely causes.
- **[Finding 3]** — Note impact of training data size on results.

---

## References

1. Author(s). (Year). *Title*. Venue.
2. Author(s). (Year). *Title*. Venue.
3. Author(s). (Year). *Title*. Venue.
4. Author(s). (Year). *Title*. Venue.
5. Author(s). (Year). *Title*. Venue.

---

<p align="center">
  Made with ❤️ for Indian language NLP research
</p>
