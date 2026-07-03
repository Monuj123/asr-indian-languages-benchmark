"""
Sarvam ASR Evaluation Pipeline
Evaluates Saaras v3 on Kathbath dataset (Hindi).
Outputs: per-sample CSV, summary JSON, and a text report.
"""

import os
import json
import csv
import time
import argparse
import requests
import editdistance
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── CONFIG ──────────────────────────────────────────────────────────────────

API_URL   = "https://api.sarvam.ai/speech-to-text"
MODEL     = "saaras:v3"
MODE      = "transcribe"          # standard transcription with normalization

LANG_CODE = {
    "hindi":   "hi-IN",
    "tamil":   "ta-IN",
    "telugu":  "te-IN",
    "bengali": "bn-IN",
    "marathi": "mr-IN",
}

# Checkpoint: save progress every N samples so you can resume on failure
CHECKPOINT_EVERY = 10

# Sarvam free-tier: ~1 req/sec is safe; increase if you have a paid key
REQUEST_DELAY_SEC = 1.1

# ─── METRICS ─────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + collapse whitespace. Unicode-safe (works for Devanagari)."""
    return " ".join(text.strip().lower().split())

def compute_wer(ref: str, hyp: str) -> float:
    ref_words = normalize(ref).split()
    hyp_words = normalize(hyp).split()
    if not ref_words:
        return 0.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)

def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(normalize(ref).replace(" ", ""))
    hyp_chars = list(normalize(hyp).replace(" ", ""))
    if not ref_chars:
        return 0.0
    return editdistance.eval(ref_chars, hyp_chars) / len(ref_chars)

# ─── SARVAM API ───────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, language_code: str, api_key: str) -> Optional[str]:
    """
    Send one audio file to Sarvam STT REST API.
    Returns the transcript string, or None on failure.
    """
    headers = {"api-subscription-key": api_key}
    data    = {
        "model":         MODEL,
        "language_code": language_code,
        "mode":          MODE,
    }
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        try:
            resp = requests.post(API_URL, headers=headers, data=data, files=files, timeout=60)
            resp.raise_for_status()
            return resp.json().get("transcript", "")
        except requests.exceptions.HTTPError as e:
            print(f"  [HTTP ERROR] {e} | body: {resp.text[:200]}")
            return None
        except Exception as e:
            print(f"  [ERROR] {e}")
            return None

# ─── CHECKPOINT ───────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_path: Path, completed: dict):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False, indent=2)

# ─── EVALUATION LOOP ─────────────────────────────────────────────────────────

def evaluate_language(
    manifest_path: Path,
    wav_dir: Path,
    language: str,
    api_key: str,
    output_dir: Path,
    dataset_root: Path,
    max_samples: Optional[int] = None,
):
    lang_code = LANG_CODE[language.lower()]
    print(f"\n{'='*60}")
    print(f"  Language : {language}  ({lang_code})")
    print(f"  Model    : {MODEL} / mode={MODE}")
    print(f"  Manifest : {manifest_path}")
    print(f"{'='*60}")

    # Load manifest
    with open(manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if max_samples:
        entries = entries[:max_samples]
        print(f"  Evaluating first {max_samples} samples (--max_samples flag)")

    print(f"  Total samples: {len(entries)}")

    # Paths
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path        = output_dir / f"{language}_predictions.csv"
    checkpoint_path = output_dir / f"{language}_checkpoint.json"
    completed       = load_checkpoint(checkpoint_path)

    skipped = len(completed)
    if skipped:
        print(f"  Resuming — {skipped} samples already done")

    # Open CSV (append mode so resume works)
    csv_exists = csv_path.exists()
    csv_file   = open(csv_path, "a", newline="", encoding="utf-8")
    writer     = csv.DictWriter(csv_file, fieldnames=[
        "sample_id", "audio_filepath", "duration",
        "reference", "hypothesis", "wer", "cer", "status"
    ])
    if not csv_exists:
        writer.writeheader()

    wer_scores, cer_scores = [], []
    error_count = 0

    for i, entry in enumerate(entries):
        sample_id = entry.get("audio_filepath", str(i))

        if sample_id in completed:
            # Already done — collect metrics for summary
            wer_scores.append(completed[sample_id]["wer"])
            cer_scores.append(completed[sample_id]["cer"])
            continue

        # Skip entries with empty reference text
        reference = entry["text"].strip()
        if not reference:
            print(f"  [{i+1:>4}/{len(entries)}] SKIPPED (empty reference)")
            continue

        # Resolve audio path:
        # manifest stores paths like "kathbath/hindi/wavs/file.wav"
        # which is relative to the dataset_root parent
        raw = Path(entry["audio_filepath"])
        candidates = [
            raw,                                      # absolute path
            dataset_root.parent / raw,                # dataset_root/../kathbath/hindi/wavs/file.wav
            dataset_root / raw,                       # dataset_root/kathbath/hindi/wavs/file.wav
            wav_dir / raw.name,                       # wav_dir/file.wav
            manifest_path.parent / raw.name,          # language_dir/file.wav
        ]
        audio_filepath = next((p for p in candidates if p.exists()), None)
        if audio_filepath is None:
            print(f"  [{i+1:>4}/{len(entries)}] FILE NOT FOUND: {raw.name}")
            error_count += 1
            writer.writerow({
                "sample_id": sample_id, "audio_filepath": str(raw),
                "duration": entry.get("duration", ""), "reference": reference,
                "hypothesis": "", "wer": "", "cer": "", "status": "file_not_found",
            })
            csv_file.flush()
            continue

        print(f"  [{i+1:>4}/{len(entries)}] {audio_filepath.name} ... ", end="", flush=True)

        t0         = time.time()
        hypothesis = transcribe(audio_filepath, lang_code, api_key)
        latency    = time.time() - t0

        if hypothesis is None:
            print("FAILED")
            error_count += 1
            status = "error"
            wer, cer = None, None
            row = {
                "sample_id":     sample_id,
                "audio_filepath": str(audio_filepath),
                "duration":      entry.get("duration", ""),
                "reference":     reference,
                "hypothesis":    "",
                "wer":           "",
                "cer":           "",
                "status":        status,
            }
        else:
            wer    = compute_wer(reference, hypothesis)
            cer    = compute_cer(reference, hypothesis)
            status = "ok"
            wer_scores.append(wer)
            cer_scores.append(cer)
            print(f"WER={wer:.3f}  CER={cer:.3f}  ({latency:.1f}s)")
            row = {
                "sample_id":     sample_id,
                "audio_filepath": str(audio_filepath),
                "duration":      entry.get("duration", ""),
                "reference":     reference,
                "hypothesis":    hypothesis,
                "wer":           round(wer, 4),
                "cer":           round(cer, 4),
                "status":        status,
            }

        writer.writerow(row)
        csv_file.flush()

        if wer is not None:
            completed[sample_id] = {"wer": wer, "cer": cer}

        # Checkpoint every N samples
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint_path, completed)

        time.sleep(REQUEST_DELAY_SEC)

    csv_file.close()
    save_checkpoint(checkpoint_path, completed)

    # ── Summary ──────────────────────────────────────────────────────────────
    total    = len(entries)
    success  = len(wer_scores)
    avg_wer  = sum(wer_scores) / success if success else None
    avg_cer  = sum(cer_scores) / success if success else None

    summary = {
        "language":      language,
        "language_code": lang_code,
        "model":         MODEL,
        "mode":          MODE,
        "total_samples": total,
        "success":       success,
        "errors":        error_count,
        "avg_wer":       round(avg_wer, 4) if avg_wer is not None else None,
        "avg_cer":       round(avg_cer, 4) if avg_cer is not None else None,
        "timestamp":     datetime.now().isoformat(),
    }

    summary_path = output_dir / f"{language}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Done  |  Avg WER: {avg_wer:.4f}  |  Avg CER: {avg_cer:.4f}")
    print(f"  Saved → {csv_path.name}  |  {summary_path.name}")
    return summary

# ─── REPORT ──────────────────────────────────────────────────────────────────

def write_report(summaries: list[dict], output_dir: Path):
    report_path = output_dir / "evaluation_report.txt"

    header = [
        "=" * 60,
        "  Sarvam ASR Evaluation Report",
        f"  Model     : {MODEL}  (mode={MODE})",
        "=" * 60,
        "",
        f"  {'Language':<12} {'Samples':>8} {'Success':>8} {'Errors':>7} {'WER':>8} {'CER':>8}",
        "  " + "-" * 55,
    ]

    # Load existing rows (if report already exists)
    existing_rows = {}
    if report_path.exists():
        with open(report_path) as f:
            for line in f:
                # Data rows start with spaces and a language name (not dashes or equals)
                stripped = line.strip()
                if stripped and not stripped.startswith("=") and not stripped.startswith("-") \
                        and not stripped.startswith("Sarvam") and not stripped.startswith("Model") \
                        and not stripped.startswith("Generated") and not stripped.startswith("Language"):
                    lang_name = stripped.split()[0].lower()
                    existing_rows[lang_name] = line.rstrip()

    # Overwrite with new summaries (new run takes priority)
    for s in summaries:
        existing_rows[s["language"].lower()] = (
            f"  {s['language']:<12} {s['total_samples']:>8} {s['success']:>8} "
            f"{s['errors']:>7} {s['avg_wer']:>8.4f} {s['avg_cer']:>8.4f}"
        )

    lines = header + list(existing_rows.values()) + ["", "=" * 60]
    report = "\n".join(lines)
    print("\n" + report)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved → {report_path}")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sarvam ASR Evaluation Pipeline")
    parser.add_argument(
        "--dataset_root", required=True,
        help="Root dir of Kathbath dataset.  Expects <root>/<language>/manifest.json and <root>/<language>/wav/"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["hindi"],
        choices=list(LANG_CODE.keys()),
        help="Languages to evaluate (default: hindi)"
    )
    parser.add_argument(
        "--api_key", default=os.environ.get("SARVAM_API_KEY"),
        help="Sarvam API key (or set SARVAM_API_KEY env var)"
    )
    parser.add_argument(
        "--output_dir", default="./sarvam_results",
        help="Directory for results (default: ./sarvam_results)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit samples per language for a quick test run"
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "No API key found. Pass --api_key or set SARVAM_API_KEY environment variable."
        )

    dataset_root = Path(args.dataset_root)
    output_dir   = Path(args.output_dir)
    summaries    = []

    for lang in args.languages:
        lang_dir      = dataset_root / lang
        manifest_path = lang_dir / "manifest.json"
        wav_dir       = lang_dir / "wav"

        if not manifest_path.exists():
            print(f"[SKIP] manifest not found: {manifest_path}")
            continue

        summary = evaluate_language(
            manifest_path = manifest_path,
            wav_dir       = wav_dir,
            language      = lang,
            api_key       = args.api_key,
            output_dir    = output_dir,
            dataset_root  = dataset_root,
            max_samples   = args.max_samples,
        )
        summaries.append(summary)

    if summaries:
        write_report(summaries, output_dir)

if __name__ == "__main__":
    main()