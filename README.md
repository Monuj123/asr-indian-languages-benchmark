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
    ├── speech_signal_processing.ipynb
    ├── hindi/
    │   └── IndicConformerASR_IndicVoices_Hindi.ipynb
    |   └── IndicWav2Vec(Hindi)_IndicVoices_Hindi.ipynb
    ├── bengali/
    │   └── IndicWav2Vec(Bengali)_IndicVoices_Bengali.ipynb
    |   └── IndicConformerASR_IndicVoices_Bengali.ipynb
    ├── marathi/
    │   └── IndicConformerASR_IndicVoices_Marathi.ipynb
    ├── telugu/
    │   └── 
    └── tamil/
        └── 
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
| `speech_signal_processing.ipynb` | Foundations — waveforms, STFT, spectrograms, MFCCs |
| `IndicConformerASR_IndicVoices_Hindi.ipynb` | Dataset loading, model evaluation, WER/CER for Hindi |
| `IndicConformerASR_IndicVoices_Bengali.ipynb` | Dataset loading, model evaluation, WER/CER for Bengali |
| `IndicConformerASR_IndicVoices_Marathi.ipynb` | Dataset loading, model evaluation, WER/CER for Marathi |
| `IndicConformerASR_IndicVoices_Telugu.ipynb` | Dataset loading, model evaluation, WER/CER for Telugu |
| `IndicConformerASR_IndicVoices_Tamil.ipynb` | Dataset loading, model evaluation, WER/CER for Tamil |
| `IndicWav2Vec(Hindi)_IndicVoices_Hindi.ipynb` | Dataset loading, model evaluation, WER/CER for Hindi |
| `IndicWav2Vec(Bengali)_IndicVoices_Bengali.ipynb` | Dataset loading, model evaluation, WER/CER for Bengali |



> ⚠️ **Start with `speech_signal_processing.ipynb`** before running any language notebook.

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
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Hindi | 4750 | 16.6(CTC),15.3(RNNT) | 7.3(CTC),7.2(RNNT) |
| 2 | IndicWav2Vec-Hindi | IndicVoices-Hindi | 4740 | 38.6 | 22.5 |
| 3 | — | — | — | — | — |

### Bengali

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicWav2Vec(Bengali) | IndicVoices-Bengali | 3906 | 46.7 | 21.2 |
| 2 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Bengali | 3906 | 100.76(CTC),100.05(RNNT) | 87(CTC),92.6(RNNT) |
| 3 | — | — | — | — | — |

### Marathi

| Rank | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices | 3552 | 16(CTC),14.9(RNNT) | 5.69(CTC),5.61(RNNT) |
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


---

## References



---

<p align="center">
  Made with ❤️ for Indian language NLP research
</p>
