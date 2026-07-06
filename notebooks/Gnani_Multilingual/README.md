# Gnani ASR Evaluation Pipeline

Evaluates **Gnani Prisma v2.5** (Gnani's state-of-the-art ASR) on the **Kathbath** dataset.

## Expected Dataset Layout

```
kathbath/
└── hindi/
    ├── manifest.json       ← one JSON object per line: audio_filepath, duration, text
    └── wav/
        ├── file1.wav
        └── file2.wav
```

Each line in `manifest.json` looks like:
```json
{"audio_filepath": "wav/file1.wav", "duration": 4.2, "text": "यह एक परीक्षण वाक्य है"}
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Sarvam API key
#    Get one free at: https://dashboard.sarvam.ai/
set SARVAM_API_KEY=your_key_here      # Windows CMD
# export SARVAM_API_KEY=your_key_here # macOS/Linux
```

## Run

### Quick test (first 20 samples)
```bash
python evaluate.py \
  --dataset_root "C:\path\to\kathbath" \
  --languages hindi \
  --max_samples 20
```

### Full evaluation
```bash
python evaluate.py \
  --dataset_root "C:\path\to\kathbath" \
  --languages hindi
```

### Multiple languages
```bash
python evaluate.py \
  --dataset_root "C:\path\to\kathbath" \
  --languages hindi tamil marathi
```

## Outputs  (saved to `./sarvam_results/`)

| File | Description |
|------|-------------|
| `hindi_predictions.csv` | Per-sample: reference, hypothesis, WER, CER |
| `hindi_summary.json` | Aggregate metrics + metadata |
| `hindi_checkpoint.json` | Auto-resume state (safe to delete after run) |
| `evaluation_report.txt` | Human-readable summary table |

## Resume After Interruption

The pipeline checkpoints every 10 samples. Just re-run the same command —
already-processed samples are skipped automatically.

## Supported Languages

| Language | Code |
|----------|------|
| hindi    | hi-IN |
| tamil    | ta-IN |
| telugu   | te-IN |
| bengali  | bn-IN |
| marathi  | mr-IN |

## Notes

- Gnani REST API accepts audio **≤ 30 seconds** per file.
  Files longer than 30s will return an error (status="error" in CSV).
- Default delay between requests is **1.1 sec** (safe for free tier).
  Increase throughput by lowering `REQUEST_DELAY_SEC` in `evaluate.py` if you have a paid key.
- WER/CER are computed with `editdistance` — no external Java/Perl tools required.
