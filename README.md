# 🎙️ ASR Benchmark — Top 5 Spoken Indian Languages

> Benchmarking State-of-the-Art Automatic Speech Recognition systems across **Hindi**, **Bengali**, **Marathi**, **Telugu**, and **Tamil**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Models Evaluated](#models-evaluated)
- [Datasets Used](#datasets-used)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Notebooks](#running-the-notebooks)
- [Getting API Keys/Tokens](#getting-api-keys--tokens)
- [Requirements](#requirements)
- [Experimental Results](#experimental-results)
- [Conclusion](#conclusion)
- [References](#references)

---

## Overview

This project provides a reproducible, end-to-end benchmark for evaluating state-of-the-art Automatic Speech Recognition (ASR) models on India's five most spoken languages. Each language notebook follows a consistent pipeline — dataset loading, model inference, and metric computation (WER & CER) — enabling fair cross-model and cross-language comparisons.

---
## Models Evaluated
 
This benchmark covers open-source and commercial ASR systems for Indian languages, spanning CTC, RNN-T, encoder-decoder.

### Open-Source Models
 
| Model | Type / Architecture | Organization | Languages | Notes |
|---|---|---|---|---|
| **IndicWhisper** | Encoder-decoder transformer (Whisper fine-tune) | AI4Bharat | Hindi + major Indic languages | Whisper backbone fine-tuned on Indic speech (Shrutilipi, MUCS, etc.); multitask token format inherited from Whisper. |
| **IndicWav2Vec** | Self-supervised CNN + transformer encoder, CTC decoding | AI4Bharat | 9+ Indian languages | Based on wav2vec 2.0 pretraining, fine-tuned with CTC head; strong low-resource performance. |
| **IndicConformer** (CTC / RNN-T) | Conformer encoder, with CTC or RNN-Transducer decoding heads | AI4Bharat | Hindi, Bengali, Tamil, Telugu, Marathi, Assamese, and more | Available as `IndicConformer-600M-multilingual`; RNN-T variant generally outperforms CTC in this benchmark. |
| **Vaani FastConformer** | FastConformer (NeMo) encoder, streaming-capable |  Vaani project | Indic languages incl. low-resource/NE India varieties | NeMo-based, optimized for lower latency/streaming vs. standard Conformer. |

### Commercial APIs
 
| Model | Type | Provider | 
|---|---|---|
| **Sarvam Saaras v3** | Commercial multilingual Indic ASR API | Sarvam AI | 
| **Gnani Prisma v2.5** | Commercial multilingual Indic ASR API | Gnani.ai | 


---
## Test Datasets Used
 
| Dataset | Type / Source | Languages | Test Dataset Link | Notes | 
|---|---|---|---|---|
| **Kathbath** | Read-speech corpus, crowd-sourced via Karya platform | 12 Indian languages |[link](https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath.zip) |Released by AI4Bharat; ~1,700 hrs total across languages, recorded by native speakers reading local news/Wikipedia-style text. Has a **Noisy 8kHz** variant (telephone-quality) used by some commercial APIs (e.g. Gnani), which is *not* directly comparable to clean 16kHz evaluation — a key caveat in this benchmark. |
| **IndicVoices** | Natural, spontaneous conversational speech corpus |22+ Indian languages, incl. low-resource NE India varieties|[link](https://huggingface.co/datasets/ai4bharat/IndicVoices)  | Released by AI4Bharat; captures diverse real-world speaking styles, dialects, and code-mixing, harder and more realistic than read-speech corpora like Kathbath, useful for stress-testing generalization beyond scripted audio. |


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
    └── Gnani_Multilingual/
        └── evaluate_gnani.py
        └── gnani_results
    └── Sarvam_Multilingual/
        └── evaluate.py
        └── sarvam_results
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
| `Gnani_Multilingual` | Dataset loading, model evaluation, WER/CER for All 5 lang(Hindi,Tamil,Telugu,Marathi,Bengali) |
| `Sarvam_Multilingual` | Dataset loading, model evaluation, WER/CER for All 5 lang(Hindi,Tamil,Telugu,Marathi,Bengali) |




> ⚠️ **Start with `speech_signal_processing.ipynb`** before running any language notebook.

Datasets are loaded directly from [HuggingFace Datasets](https://huggingface.co/datasets) — no manual download required.

---


## Getting API Keys / Tokens
 
| Source | Where to Get It | Notes |
|--------|------------------|-------|
| Gnani Prisma v2.5 | [https://gnani.ai/prisma/](https://www.gnani.ai/speech-to-text-api) (request access / contact sales) | Commercial — you'll get an API key + secret pair |
| Sarvam Saaras v3 | [https://dashboard.sarvam.ai/](https://dashboard.sarvam.ai/) → Sign up → API Keys | Free tier available |
| Hugging Face | [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token (read access is enough for downloading models) | Needed to pull IndicConformer, IndicWhisper, IndicWav2Vec, Vaani-FastConformer checkpoints |

Once you have the keys, set them as environment variables (same pattern as **Setup** above):
 
```bash
# macOS/Linux
export GNANI_API_KEY=your_key_here
export GNANI_API_SECRET=your_secret_here
export SARVAM_API_KEY=your_key_here
export HF_TOKEN=your_huggingface_token_here
 
# Windows CMD
set GNANI_API_KEY=your_key_here
set GNANI_API_SECRET=your_secret_here
set SARVAM_API_KEY=your_key_here
set HF_TOKEN=your_huggingface_token_here
```


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

| S. No. | Model | Source Type| Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|------|---------|--------------|---------|---------|
| 1 | Vaani_FastConformer(Hindi)|Open| IndicVoices-Hindi | 5530 | 15.11 | 7.09 |
| 2 | IndicConformer(indic-conformer-600m-multilingual)|Open | IndicVoices-Hindi | 5530 | 16.6(CTC) | 7.3(CTC) |
|   |         |                                          |                   |      | 15.3(RNNT | 7.2(RNNT)|       
| 3 | IndicWav2Vec-Hindi | Open|IndicVoices-Hindi | 5530 | 38.6 | 22.5 |
| 4 | IndicWhisper-Hindi |Open|Kathbath-Hindi | 1929 | 10.23 | 3.65 |
| 5 | Sarvam Saaras v3 | Closed|Kathbath-Hindi | 1929 | 14.78 | 4.59 |
| 6 | Gnani Prisma v2.5| Closed|Kathbath-Hindi | 1929 | 8.78 | 3.26 |
| 7 | Vaani_FastConformer(Hindi)|Open| Kathbath-Hindi | 1929 | 12.49 | 4.3 |
| 8 | IndicConformer(indic-conformer-600m-multilingual)|Open | Kathbath-Hindi | 1929 | 9.74(CTC) | 3.17(CTC) |
|   |         |                                          |                   |      | 9.44(RNNT | 3.07(RNNT)|   
| 9 | IndicWav2Vec-Hindi | Open|Kathbath-Hindi | 1929 | 12.49| 4.3 |
| 10 | ARTPARK-IISc/SraVaani-1.0 | Open | Kathbath-Hindi | 1929 | 8.96 | 3.19 |

### Bengali

| S. No. | Model | Source Type|Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual)|Opem | IndicVoices-Bengali | 3906 | 14.23(CTC) | 5.41(CTC) |
|   |                  |                                 |                     |      | 13.49(RNNT)| 5.36(RNNT)|
| 2 | IndicWav2Vec(Bengali)|Open | IndicVoices-Bengali | 3906 | 46.7 | 21.2 |
| 3 | IndicWhisper(Bengali)|Open | Kathbath-Bengali| 1783 | 19.22 | 5.51 |
| 4 | Sarvam Saaras v3|Closed | Kathbath-Bengali| 1783 | 21.21 | 5.07 |
| 5 | Gnani Prisma v2.5|Closed | Kathbath-Bengali | 1783 | 13.94 | 4.79 |
| 6 | IndicWav2Vec(Bengali)|Open | Kathbath-Bengali | 1783 | 22.69 | 4.68 |
| 7 | IndicConformer(indic-conformer-600m-multilingual)|Open | Kathbath-Bengali | 1783 | 13.58(CTC) | 3.94(CTC) |
|   |                  |                                 |                     |      | 13.2(RNNT)| 3.82(RNNT)|
| 8 | ARTPARK-IISc/SraVaani-1.0 | Open | Kathbath-Bengali | 1783 | 12.27 | 4.09 |


### Marathi

| S. No. | Model|Source Type | Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual)|Open | IndicVoices-Marathi | 3552 | 16(CTC) | 5.69(CTC) |
|  |  | |  |  | 14.9(RNNT) | 5.61(RNNT) |
| 2 | IndicWhisper(Marathi)|Open | Kathbath-Marathi | 1631 | 20.25 | 6.52 |
| 3 | Sarvam Saaras v3|Closed|Kathbath-Marathi | 1631 | 20.46 | 5.96 |
| 4 | Gnani Prisma v2.5|Closed | Kathbath-Marathi | 1631 | 16.89 | 5.72 |
| 5 | IndicConformer(indic-conformer-600m-multilingual)|Open | Kathbath-Marathi | 1631 | 17.61(CTC) | 5.62(CTC) |
|  |  | |  |  | 17.14(RNNT) | 5.5(RNNT) |
| 6 | ARTPARK-IISc/SraVaani-1.0 | Open | Kathbath-Marathi | 1631 | 17.02 | 6.19 |

### Telugu

| S. No. | Model | Source Type|Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual)|Open | IndicVoices-Telugu | 3295 | 27.86(CTC), | 8.53(CTC) |
|   |  | | |  | 26.28(RNNT) | 8.42(RNNT) |
| 2 | ARTPARK-IISc/Vaani-FastConformer-Telugu|Open | IndicVoices-Telugu | 3295 | 27.31 | 8.77 |
| 3 | ARTPARK-IISc/whisper-medium-vaani-telugu| Open|IndicVoices-Telugu | 3295 | 61.69 | 25.69 |
| 4 | Sarvam Saaras v3| Closed|Kathbath-Telugu | 1492 | 24.8 | 3.7 |
| 5 | IndicWhisper-Telugu| Open|Kathbath-Telugu | 1492 | 36.9 | 12.9 |
| 6 | Gnani Prisma v2.5| Closed |Kathbath-Telugu | 1492 | 25.53 | 7.55 |
| 7 | IndicConformer(indic-conformer-600m-multilingual)|Open | Kathbath-Telugu | 1492 | 21.92(CTC), | 3.63(CTC) |
|   |  | | |  | 21.37(RNNT) | 3.55(RNNT) |
| 8 | ARTPARK-IISc/Vaani-FastConformer-Telugu|Open | Kathbath-Telugu | 1492 | 23.29 | 4.06 |
| 9 | ARTPARK-IISc/SraVaani-1.0 | Open | Kathbath-Telugu | 1492 | 21.00 | 3.16 |


### Tamil

| S. No. | Model | Source Type|Dataset | Test Samples | WER (%) | CER (%) |
|------|-------|------|---------|--------------|---------|---------|
| 1 | IndicConformer(indic-conformer-600m-multilingual)|Open | IndicVoices-Tamil | 5276 | 34.7(CTC) | 10.1(CTC) |
|  |  | | |  | 30.7(RNNT) | 9.1(RNNT) |
| 2 | IndicWhisper(Tamil)|Open | Kathbath-Tamil | 1642 | 24.9 | 4.47 |
| 3 | Sarvam Saaras v3|Closed | Kathbath-Tamil | 1642 | 28.65 | 4.74 |
| 4 | Gnani Prisma v2.5|Closed| Kathbath-Tamil | 1642 | 24.96 | 5.51 |
| 5 | IndicConformer(indic-conformer-600m-multilingual)|Open | Kathbath-Tamil | 1642 | 24.59(CTC) | 4.29(CTC) |
|  |  | | |  | 23.37(RNNT) | 3.95(RNNT) |
| 8 | ARTPARK-IISc/SraVaani-1.0 | Open | Kathbath-Tamil | 1642 | 22.63 | 3.69 |
---

## Conclusion

This project establishes a reproducible benchmark for ASR across India's five most spoken languages, evaluated on both natural/spontaneous speech (IndicVoices) and clean read-speech (Kathbath). Key observations:

- **IndicConformer-600M-Multilingual (RNNT) is the strongest and most consistent open-source model.** RNNT decoding outperforms CTC in every single language/dataset pairing measured, with the largest gaps on Tamil (30.7% vs 34.7% WER, IndicVoices) and Telugu (26.28% vs 27.86% WER, IndicVoices).

- **On Kathbath, IndicConformer RNNT beats *both* commercial APIs in 3 of 5 languages** — Bengali (13.2% WER / 3.82% CER vs. Gnani's 13.94%/4.79%), Telugu (21.37%/3.55% vs. Sarvam's 24.8%/3.7% and Gnani's 25.53%/7.55%), and Tamil (23.37%/3.95%, best of all five models tested, ahead of IndicWhisper, Sarvam, and Gnani). This is a meaningfully different picture than "commercial beats open-source" — it mostly doesn't, once RNNT decoding is used.

- **Hindi and Marathi are the two languages where a commercial API (Gnani) edges out IndicConformer on WER** — Hindi (8.78% vs. 9.44%) and Marathi (16.89% vs. 17.14%) — though IndicConformer still wins or ties on CER in both cases (3.07% vs. 3.26% Hindi; 5.5% vs. 5.72% Marathi). Gnani's Hindi result (8.78% WER) is the single best number in the whole benchmark.


- **On IndicVoices (harder, spontaneous speech), Vaani-FastConformer-Hindi is the only model that edges out IndicConformer**, on Hindi specifically (15.11% vs. 15.3% WER, RNNT) — making it a reasonable pick for Hindi-only, low-latency deployments. On Telugu, Vaani-FastConformer-Telugu (27.31%/8.77%) trails IndicConformer RNNT (26.28%/8.42%) by less than a point, making both viable depending on latency/streaming requirements.

- **IndicWav2Vec is not production-viable.** It trails IndicConformer badly on every language tested — 38.6% vs. 15.3% WER on Hindi (IndicVoices) and 46.7% vs. 13.49% on Bengali (IndicVoices) — and remains behind even on the easier Kathbath set.

- **Tamil is the hardest language in the benchmark** (best result: 23.37% WER on Kathbath, 30.7% on IndicVoices), roughly double Bengali's error rate (13.2%/13.49%), pointing to lower training-data coverage and/or higher acoustic/script complexity. **Bengali is the easiest**, with the strongest results across nearly every model tested.

**Overall recommendation:** IndicConformer-600M-Multilingual (RNNT) is the recommended default across all five languages — it is either the best or a very close second in every language on both datasets, and beats the commercial APIs outright on Bengali, Telugu, and Tamil. Gnani Prisma v2.5 is worth considering specifically for Hindi and Marathi. Vaani-FastConformer remains the pick where streaming/low-latency inference matters more than the last fraction of a WER point.

---

## References

- Bhogale, K. S., Sundaresan, S., Raman, A., Javed, T., Khapra, M. M., & Kumar, P. (2023). Vistaar: Diverse Benchmarks and Training Sets for Indian Language ASR. arXiv:2305.15386. https://arxiv.org/abs/2305.15386
- Javed, T., Nawale, J. A., George, E. I., Joshi, S., Bhogale, K. S., Mehendale, D., Sethi, I. V., Ananthanarayanan, A., Faquih, H., Palit, P., Ravishankar, S., Sukumaran, S., Panchagnula, T., Murali, S., Gandhi, K. S., R, A., M, M. K., Vaijayanthi, C. V., Karunganni, K. S. R., Kumar, P., & Khapra, M. M. (2024). IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian Languages. arXiv:2403.01926. https://arxiv.org/abs/2403.01926
- Javed, T., Bhogale, K. S., Raman, A., Kunchukuttan, A., Kumar, P., & Khapra, M. M. (2022). IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian languages. arXiv:2208.11761. https://arxiv.org/abs/2208.11761
- Sarvam AI. Evaluating Indian Language ASR. Sarvam AI Blog. https://www.sarvam.ai/blogs/evaluating-indian-language-asr
- Gnani.ai. Gnani.ai — Conversational AI Platform. https://www.gnani.ai/




---

<p align="center">
  Made with ❤️ for Indian language NLP research
</p>
