# Gnani ASR Evaluation Pipeline
 
Evaluates **Prisma v2.5** (Gnani.ai's ASR model) on the **Kathbath** dataset.
 
> **Note:** Gnani Prisma v2.5 is benchmarked against the **Noisy 8kHz** variant of Kathbath
> (not the clean 16kHz variant used for the open-source models / Sarvam Saaras v3).
> Keep this in mind when comparing WER/CER results across models — it is not a like-for-like
> comparison unless you resample and re-run consistently.
 
## Expected Dataset Layout
 
```
kathbath_noisy_8khz/
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
 
# 3. Set your Gnani API credentials
#    Request access at: https://gnani.ai/prisma/
set GNANI_API_KEY=your_key_here          # Windows CMD
set GNANI_API_SECRET=your_secret_here    # Windows CMD
# export GNANI_API_KEY=your_key_here        # macOS/Linux
# export GNANI_API_SECRET=your_secret_here  # macOS/Linux
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
  --dataset_root "C:\path\to\kathbath_noisy_8khz" \
  --languages hindi
```
 
### Multiple languages
 
```bash
python evaluate.py \
  --dataset_root "C:\path\to\kathbath_noisy_8khz" \
  --languages hindi tamil marathi
```
 
## Outputs  (saved to `./gnani_results/`)
 
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
 
| Language | Code  |
|----------|-------|
| hindi    | hi-IN |
| tamil    | ta-IN |
| telugu   | te-IN |
| bengali  | bn-IN |
| marathi  | mr-IN |
 
## Notes
 
- The Gnani Prisma v2.5 API expects **8kHz** audio input; files at other sample rates
  should be resampled before upload or they may be silently downsampled by the server,
  which can distort results.
- Audio is scored against the **Noisy** Kathbath split — this variant includes background
  noise augmentation, so raw WER/CER numbers will generally run higher than on clean-audio
  benchmarks and should not be directly compared to models evaluated on clean Kathbath.
- Default delay between requests is **1.1 sec** (safe for free/trial tier).
  Increase throughput by lowering `REQUEST_DELAY_SEC` in `evaluate.py` if you have a paid key.
- WER/CER are computed with `editdistance` — no external Java/Perl tools required.
