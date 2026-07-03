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
    |   └── Vaani_FastConformer(Hindi)_IndicVoices_Hindi.ipynb
    |   └── IndicWhisper_Kathbath_Hindi.ipynb
    ├── bengali/
    │   └── IndicWav2Vec(Bengali)_IndicVoices_Bengali.ipynb
    |   └── IndicConformerASR_IndicVoices_Bengali.ipynb
    |   └── IndicWhisper_Kathbath_Bengali.ipynb
    ├── marathi/
    │   └── IndicConformerASR_IndicVoices_Marathi.ipynb
    |   └── IndicWhisper_Kathbath_Marathi.ipynb
    ├── telugu/
    │   └── Vaani_FastConformer(Telugu)_IndicVoices_Telugu.ipynb
    |   └── IndicWhisper_Kathbath_Telugu.ipynb
    └── tamil/
        └── IndicConformerASR_IndicVoices_Tamil.ipynb
        └── IndicWhisper_Kathbath_Tamil.ipynb
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
| `IndicConformer_IndicVoices_Telugu.ipynb` | Dataset loading, model evaluation, WER/CER for Telugu |
| `IndicConformerASR_IndicVoices_Tamil.ipynb` | Dataset loading, model evaluation, WER/CER for Tamil |
| `IndicWav2Vec(Hindi)_IndicVoices_Hindi.ipynb` | Dataset loading, model evaluation, WER/CER for Hindi |
| `IndicWav2Vec(Bengali)_IndicVoices_Bengali.ipynb` | Dataset loading, model evaluation, WER/CER for Bengali |
| `Vaani_FastConformer(Telugu)_IndicVoices_Telugu.ipynb` | Dataset loading, model evaluation, WER/CER for Telugu |
| `Vaani_FastConformer(Hindi)_IndicVoices_Hindi.ipynb` | Dataset loading, model evaluation, WER/CER for Hindi |
| `ARTPARK-IISc/whisper-medium-vaani-telugu.ipynb` | Dataset loading, model evaluation, WER/CER for Telugu |
| `IndicWhisper_Kathbath_Tamil.ipynb` | Dataset loading, model evaluation, WER/CER for Tamil |
| `IndicWhisper_Kathbath_Marathi.ipynb` | Dataset loading, model evaluation, WER/CER for Marathi |
| `IndicWhisper_Kathbath_Bengali.ipynb` | Dataset loading, model evaluation, WER/CER for Bengali |
| `IndicWhisper_Kathbath_Hindi.ipynb` | Dataset loading, model evaluation, WER/CER for Hindi |
| `IndicWhisper_Kathbath_Telugu.ipynb` | Dataset loading, model evaluation, WER/CER for Telugu |


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

Top SOTA models evaluated per language. Metrics: **Word Error Rate (WER ↓)** and **Character Error Rate (CER ↓)**.

> Results will be updated upon completion of benchmarking runs.

### Hindi

| S. No. | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | Vaani_FastConformer(Hindi)| IndicVoices-Hindi | 5530 | 15.11 | 7.09 |
| 2 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Hindi | 5530 | 16.6(CTC) | 7.3(CTC) |
|   |                                                   |                   |      | 15.3(RNNT | 7.2(RNNT)|       
| 3 | IndicWav2Vec-Hindi | IndicVoices-Hindi | 5530 | 38.6 | 22.5 |
| 4 | IndicWhisper-Hindi | Kathbath-Hindi | 1929 | 10.23 | 3.65 |
| 5 | Sarvam Saaras v3 | Kathbath-Hindi | 1929 | 14.78 | 4.59 |
| 6 | Gnani Prisma v2.5| Kathbath-Hindi | 1929 | 8.78 | 3.26 |

### Bengali

| S. No. | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Bengali | 3906 | 14.23(CTC) | 5.41(CTC) |
|   |                                                   |                     |      | 13.49(RNNT)| 5.36(RNNT)|
| 2 | IndicWav2Vec(Bengali) | IndicVoices-Bengali | 3906 | 46.7 | 21.2 |
| 3 | IndicWhisper(Bengali) | Kathbath-Bengali| 1783 | 19.22 | 5.51 |
| 4 | Sarvam Saaras v3 | Kathbath-Bengali| 1783 | 21.21 | 5.07 |
| 5 | Gnani Prisma v2.5| Kathbath-Bengali | 1783 | 13.94 | 4.79 |

### Marathi

| S. No. | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Marathi | 3552 | 16(CTC) | 5.69(CTC) |
|  |  |  |  | 14.9(RNNT) | 5.61(RNNT) |
| 2 | IndicWhisper(Marathi) | Kathbath-Marathi | 1631 | 20.25 | 6.52 |
| 3 | Sarvam Saaras v3 |Kathbath-Marathi | 1631 | 20.46 | 5.96 |
| 4 | Gnani Prisma v2.5| Kathbath-Marathi | 1631 | 16.89 | 5.72 |

### Telugu

| S. No. | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Telugu | 3295 | 27.86(CTC), | 8.53(CTC) |
|   |  |  |  | 26.28(RNNT) | 8.42(RNNT) |
| 2 | ARTPARK-IISc/Vaani-FastConformer-Telugu | IndicVoices-Telugu | 3295 | 27.31 | 8.77 |
| 3 | ARTPARK-IISc/whisper-medium-vaani-telugu| IndicVoices-Telugu | 3295 | 61.69 | 25.69 |
| 4 | Sarvam Saaras v3| Kathbath-Telugu | 1492 | 24.8 | 3.7 |
| 5 | IndicWhisper-Telugu| Kathbath-Telugu | 1492 | 36.9 | 12.9 |
| 6 | Gnani Prisma v2.5| Kathbath-Telugu | 1492 | 25.53 | 7.55 |


### Tamil

| S. No. | Model | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual) | IndicVoices-Tamil | 5276 | 34.7(CTC) | 10.1(CTC) |
|  |  |  |  | 30.7(RNNT) | 9.1(RNNT) |
| 2 | IndicWhisper(Tamil) | Kathbath-Tamil | 1642 | 24.9 | 4.47 |
| 3 | Sarvam Saaras v3 | Kathbath-Tamil | 1642 | 28.65 | 4.74 |
| 4 | Gnani Prisma v2.5| Kathbath-Tamil | 1642 | 24.96 | 5.51 |

---

## Conclusion

This project establishes a reproducible benchmark for ASR across India's five most spoken languages. Key observations from the experiments:

- IndicConformer-600M-Multilingual with RNNT decoding is the strongest open-source model across all five languages, achieving the best WER in Bengali (13.49%), Marathi (14.9%), Hindi (15.3%), Telugu (26.28%), and Tamil (30.7%). The RNNT decoder consistently outperforms CTC across every language and metric, with the gap being most notable in Tamil (30.7% vs 34.7%) and Telugu (26.28% vs 27.86%).

- Among commercial APIs, Gnani Prisma v2.5 outperforms Sarvam Saaras v3 on Hindi (8.78% vs 14.78% WER) and Telugu (25.53% vs 24.8% WER), while Saaras v3 has a slight edge on Telugu CER (3.7% vs 7.55%). Notably, Gnani Hindi (8.78% WER) is the single best result across all models and languages in this benchmark, outperforming even IndicConformer on Kathbath-Hindi.
- Vaani-FastConformer-Hindi is the sole exception to IndicConformer's dominance, edging it out on Hindi with a WER of 15.11% vs 15.3% (RNNT). For Hindi-only deployments, it is the preferred choice.
- Telugu is the most competitive language in this benchmark, with IndicConformer RNNT (26.28%) and Vaani-FastConformer-Telugu (27.31%) separated by less than 1 percentage point — both are viable options depending on deployment constraints.
- Vaani-FastConformer-Hindi is the sole open-source model that challenges IndicConformer, edging it on Hindi (15.11% vs 15.3% RNNT), making it the preferred choice for Hindi-only deployments. In contrast, IndicWav2Vec models lag far behind, scoring 38.6% WER on Hindi and 46.7% on Bengali compared to sub-17% and sub-14% from IndicConformer , making them unsuitable for production use.
- Tamil remains the hardest language in this benchmark, with the best result at 30.7% WER (IndicConformer RNNT), nearly twice the error rate of Bengali (13.49%), suggesting limited training data coverage or greater acoustic complexity. Bengali is the easiest, with consistently strong results across all models.

Overall, for multilingual Indic ASR, IndicConformer-600M RNNT is the recommended open-source baseline, while Gnani Prisma v2.5 is the best-performing commercial option tested in this benchmark. 


---

## References



---

<p align="center">
  Made with ❤️ for Indian language NLP research
</p>
