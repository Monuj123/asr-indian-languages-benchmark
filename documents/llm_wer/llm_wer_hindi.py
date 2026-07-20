"""
LLM-WER / LLM-CER for Hindi (based on sarvamai/llm_wer methodology)
=====================================================================

What this does (same pipeline as https://github.com/sarvamai/llm_wer):

1. Normalize reference & predicted text (Indic normalization + punctuation strip).
2. Compute the ORIGINAL WER/CER (plain jiwer-based, exactly like the repo's
   `wer()` / `cer()` helpers).
3. Diff reference vs. prediction word-by-word (difflib.SequenceMatcher) to
   find the exact mismatched segments (same as the repo's `get_segments`).
4. Send every *unique* mismatched (reference, prediction) segment pair to an
   LLM once, asking it to judge whether they are semantically + phonetically
   equivalent (same rules as the repo's prompt_template.txt: punctuation,
   numerals, transliteration, spoken-form, minor spelling variants, etc.).
5. Any segment the LLM marks "equivalent" is treated as correct -> rebuild a
   "corrected" transcript and recompute WER/CER on that ("llm_wer", "llm_cer").

Differences from the original repo (kept for portability):
  - Uses the free-tier Google Gemini API (via the `google-genai` SDK) as the
    judge model instead of Vertex AI + a paid service account. Gemini's
    Developer API free tier needs only an API key from
    https://aistudio.google.com/apikey (no credit card required).
  - Drops the Google-Sheets push and the Whisper-tokenizer normalizer path
    (not needed for Hindi-only use); Hindi normalization uses indic-nlp-library.
  - Everything else (get_segments, wer/cer formulas, prompt rules, caching,
    reconstruction logic) mirrors the original as closely as possible.

FIX (this version): pure insertions/deletions (a mismatched segment with
NOTHING on one side, e.g. an inserted/duplicated word) used to be skipped
entirely and hard-coded as an error, because judging a bare word in
isolation ("" vs "नहीं") is meaningless without seeing the sentence it sits
in. This version now sends those segments to the LLM too, alongside the
full reference/prediction sentence as context, so it can tell a stray
repeated helper word (not a real error) apart from an insertion that
actually reverses meaning (a real error) -- see PROMPT_TEMPLATE section 5.

Install:
    pip install pandas jiwer indic-nlp-library google-genai

Get a free API key (no credit card needed):
    https://aistudio.google.com/apikey
    export GEMINI_API_KEY=your_key_here

Run:
    python llm_wer_hindi.py \
        --input my_data.csv \
        --reference-col reference \
        --predicted-col prediction \
        --output-dir outputs

CSV requirements:
    - Must have a reference column and a predicted column (Hindi text).
    - No language column needed - language is fixed to Hindi via --lang.

Note on the free tier: Gemini's Flash free tier allows only a handful of
requests per minute and roughly 1,500 requests/day (exact numbers change -
check https://ai.google.dev/gemini-api/docs/rate-limits). This script batches
multiple segment pairs into a single call and caches results to disk, so
re-runs and large files stay well within those limits.

Note on model naming: Google frequently retires specific model-version IDs
(e.g. "gemini-2.5-flash" has been removed for new API keys/projects). This
script defaults to the alias "gemini-flash-latest", which Google keeps
pointed at whatever their current free-tier Flash model is, so you shouldn't
need to update it manually. If you ever get a 404 "model no longer
available" error, list what's actually enabled for your key with:
    from google import genai
    for m in genai.Client().models.list(): print(m.name)
and pass one of those names via --model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import jiwer

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
except ImportError as e:
    raise ImportError(
        "indic-nlp-library is required. Install with: pip install indic-nlp-library"
    ) from e

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:
    raise ImportError(
        "google-genai is required. Install with: pip install google-genai"
    ) from e


# ---------------------------------------------------------------------------
# 1. WER / CER  (identical formulas to sarvamai/llm_wer utilities.py)
# ---------------------------------------------------------------------------

def wer(ref: str, hyp: str, clamp: bool = True,
        insertion_weight: float = 1, deletion_weight: float = 1,
        substitution_weight: float = 1) -> float:
    ref, hyp = str(ref).strip(), str(hyp).strip()
    N, M = len(ref.split()), len(hyp.split())
    if N == 0 and M == 0:
        return 0.0
    if N == 0 and M > 0:
        return insertion_weight
    if N > 0 and M == 0:
        return deletion_weight
    out = jiwer.process_words(ref, hyp)
    S, D, I = out.substitutions, out.deletions, out.insertions
    denom = max(M, N) if clamp else N
    return (S * substitution_weight + D * deletion_weight + I * insertion_weight) / denom


def cer(ref: str, hyp: str, clamp: bool = True,
        insertion_weight: float = 1, deletion_weight: float = 1.0,
        substitution_weight: float = 1.0) -> float:
    ref, hyp = str(ref).strip(), str(hyp).strip()
    N, M = len(ref), len(hyp)
    if N == 0 and M == 0:
        return 0.0
    if N == 0 and M > 0:
        return insertion_weight
    if N > 0 and M == 0:
        return deletion_weight
    out = jiwer.process_characters(ref, hyp)
    S, D, I = out.substitutions, out.deletions, out.insertions
    denom = max(M, N) if clamp else N
    return (S * substitution_weight + D * deletion_weight + I * insertion_weight) / denom


# ---------------------------------------------------------------------------
# 2. Hindi normalization (indic-nlp-library, mirrors IndicNormalizer path)
# ---------------------------------------------------------------------------

INDIC_PUNCTUATION = "।॥॰''\"‛‟′″´˝^°¤।॥॰¯'—–‑°¬´\u200b\u200c\u200d\u200e\u200f"
_hi_normalizer = IndicNormalizerFactory().get_normalizer("hi")


def normalize_hindi(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "" if not isinstance(text, str) else text
    text = re.sub(r'([,\-\.\(\)\[\]\{\}/\\])\B', r' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation + INDIC_PUNCTUATION))
    text = text.lower()
    text = _hi_normalizer.normalize(text)
    text = re.sub(' +', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. Segment diffing (identical to get_segments in main.py)
# ---------------------------------------------------------------------------

def get_segments(reference_string: str, predicted_string: str, key: Any) -> List[Dict[str, Any]]:
    ref_words = reference_string.strip().split()
    pred_words = predicted_string.strip().split()
    if not ref_words and not pred_words:
        return []
    matcher = SequenceMatcher(None, ref_words, pred_words)
    return [
        {
            "reference": " ".join(ref_words[i1:i2]),
            "prediction": " ".join(pred_words[j1:j2]),
            "tag": tag,
            "key": key,
            "segment_idx": seg_idx,
        }
        for seg_idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes())
    ]


# ---------------------------------------------------------------------------
# 4. LLM equivalence prompt
#    This is the sarvamai/llm_wer prompt_template.txt, plus one added rule
#    (section 5) covering empty-side segments, and a batching addendum so
#    the model returns a JSON array covering a whole batch of pairs in one
#    call instead of one object per call - needed to stay within Gemini's
#    free-tier requests-per-minute limit when scoring many segments.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """# Persona
You are an expert linguistic analyst specializing in Indian languages.

# Primary Goal
Your primary goal is to precisely compare two transcripts and determine if they are essentially equivalent based on a set of equivalence rules. Adapt the phonetic and structural principles for other Indic languages and their respective scripts as appropriate.

# Equivalence Rules
To determine equivalence, you MUST adhere strictly to the following rules.

## 1. Formatting and Symbol Equivalence
- **Ignore Punctuation:** Disregard all punctuation marks (e.g., ।, ?, ,, ॥, -, .).
- **Hyphenation:** Treat hyphenated words as identical to their multi-word or single-word counterparts.
  - Example (Hindi): "धीरे-धीरे" (dheere-dheere) = "धीरे धीरे" = "धीरेधीरे"
  - Example (Tamil): "அங்கு-இங்கு" (angu-ingu) = "அங்கு இங்கு" = "அங்குஇங்கு"
  - Example (Bengali): "পাশে-পাশে" (pashe-pashe) = "পাশে পাশে" = "পাশপাশে"
  - Example (Gujarati): "સાથે-સાથે" (sathe-sathe) = "સાથે સાથે" = "સાથેસાથે"
- **Numbers:** Convert all numbers, whether in digit or word form, to a standard numeric value for comparison. Combine consecutive number words.
  - Example (Hindi): "उन्नीस सौ नब्बे" (unnees sau nabbe) = "1990"
  - Example (Tamil): "இரண்டு ஆயிரம் இருபத்தி மூன்று" (irandu aayiram irupathi moondru) = "2023"
  - Example (Telugu): "పంతొమ్మిది వందల తొంభై" (panthommidi vandala thombai) = "1990"
  - Example (Punjabi): "ਉੱਨੀ ਸੌ ਨੱਬੇ" (unni sau nabbe) = "1990"
- **Symbols:** Words representing symbols are equivalent to the symbols themselves.
  - Example (Hindi): "रुपये" (rupaye) = '₹'
  - Example (Tamil): "சதவீதம்" (sadhaveedham) = '%'
  - Example (Bengali): "টাকা" (taka) = '₹'
  - Example (Gujarati): "ટકા" (taka) = '%'
- **Numbers and Currencies:** Numbers with symbols representing the same value are equivalent.
  - Example (Hindi): "सौ रुपये" (sau rupaye) = "₹100"
  - Example (Hindi): "एक सौ चार रुपये पचास पैसे" (ek sau chaar rupaye pachaas paise) = "₹104.50"
  - Example (Malayalam): "നൂറു രൂപ" (nooru roopa) = "₹100"

## 2. Spoken vs. Written Form Equivalence
Account for common differences between spoken and written forms.
- **Acronyms and Initialisms:** Spoken-out letters of an acronym are equivalent to the consolidated written form.
  - Example (Hindi): "पी एन बी" = "पीएनबी"
  - Example (Tamil): "டி சி எஸ்" = "டிசிஎஸ்"
- **Phonetic Spelling of Brands/Names:** Phonetic spellings of proper nouns or brands are equivalent to their standard written form.
  - Example (Hindi): "रेडियो मिर्ची" = "Radio Mirchi"
  - Example (Tamil): "கோக கோலா" = "Coca-Cola"
  - Example (Bengali): "পেটিএম" (Paytm) = "Paytm"

## 3. Language and Script Equivalence
- **Cross-Script Equivalence:** Words that sound the same but are written in different scripts (e.g., Roman and a native Indian script) are equivalent.
  - Example (Hindi): "Amazon" = "अमेज़न"
  - Example (Tamil): "WhatsApp" = "வாட்ஸ்அப்"
  - Example (Telugu): "Facebook" = "ఫేస్బుక్"
  - Example (Punjabi): "Youtube" = "ਯੂਟਿਊਬ"
- **Common Spelling Variations:** Minor, common spelling variations that do not significantly alter pronunciation are equivalent. This includes variations in spacing for the same word.
  - Examples: "दोबारा" = "दुबारा", "கட்டிடம்" = "கட்டிடம", "वहाँ" = "वहां", "मज़ा" = "मजा", "केला" = "केलं", "दिलजीत" = "दिलजित" = "दिलचीत"
  - Example (Bengali): "জন্য" (jonnyo) = "জন্যে" (jonne)
  - Example (Telugu): "వెళ్తున్నాను" (velthunnanu) = "వెళ్తున్నా" (velthunna)

## 4. Phonetic Contractions or Reductions
- **Phonetic Contractions/Reductions:** Words that are phonetic reductions or contractions of another word are equivalent if their pronunciation is somewhat similar.
  - Example (Hindi): 'पर' (par) = 'पे' (pe)
  - Example (Hindi): 'ये' (ye) = 'यह' (yah)
  - Example (Bengali): 'তাহার' (tahar) = 'তার' (tar)

## 5. Empty-Side (Insertion/Deletion) Segments
Some pairs you receive will have an EMPTY reference or an EMPTY prediction --
that means the aligner found a word (or short phrase) inserted into, or
deleted from, the predicted transcript relative to the reference, with
nothing on the other side to compare it to directly. For these you will
also be given `full_reference_sentence` and `full_prediction_sentence` --
use them to judge the inserted/deleted span in context, not in isolation:
- **Stray repeated helper word:** if the inserted/deleted span is a short
  helper word (negation particles, fillers, discourse markers) that is
  simply duplicated adjacent to an identical word already in the sentence
  (e.g. reference "यह सही नहीं है", prediction "यह सही नहीं नहीं है" --
  segment is an inserted "नहीं"), this is a disfluency/stutter artifact,
  NOT a meaning change. Mark it **equivalent**.
- **Meaning-changing insertion/deletion:** if the inserted/deleted span
  changes what the sentence asserts -- most importantly inserting or
  removing a negation that is NOT simply duplicating an adjacent word
  (e.g. reference "मुझे यह पसंद है" / prediction "मुझे यह नहीं पसंद है" --
  "I like this" vs "I do NOT like this"), or inserts/removes any other
  content word that alters the sentence's meaning, this is a genuine
  error. Mark it **not equivalent**, regardless of how short the span is.
The test is always: does the full predicted sentence assert the same
thing as the full reference sentence? A duplicated word does not change
what is asserted; a negation (or other content change) does.

# Input Format:
You will be given a list of JSON objects, of the following format:

```json
{
  "index": int,
  "reference": str,
  "prediction": str,
  "full_reference_sentence": str,   // OPTIONAL, only present for empty-side segments
  "full_prediction_sentence": str   // OPTIONAL, only present for empty-side segments
}
```

# Output Format
Your final output must be a single JSON object with the keys: `index`, `equivalence` and `reasoning`.

- **`index`**: An integer value. Ensure that the `index` is the same as the input index for the corresponding reference and prediction pair.
- **`equivalence`**: A boolean value. Use `True` for an exact match and `False` for a mismatch after carefully considering all the rules and analyzing the transcripts.
- **`reasoning`**: A string. Provide a brief, clear explanation for the equivalence value, highlighting the specific words or sequences along with it's translation as well as transliteration to English.

```json
{
    "index": int,
    "equivalence": bool,
    "reasoning": str
}
```

# Batch Output Note (addendum)
You will be given MULTIPLE input objects at once (a JSON array). For each one, produce an
object following the Output Format above. Return ONLY a single JSON array containing one
such object per input object, in any order, with no markdown fences and no other text."""


def build_batch_prompt(pairs: List[Dict[str, str]]) -> str:
    payload = []
    for i, p in enumerate(pairs):
        item: Dict[str, Any] = {
            "index": i,
            "reference": p["reference"],
            "prediction": p["prediction"],
        }
        # Only attach sentence-level context when we actually have it (empty-side
        # segments) -- keeps the prompt small for the common replace/replace case.
        if p.get("context_reference") is not None:
            item["full_reference_sentence"] = p["context_reference"]
            item["full_prediction_sentence"] = p["context_prediction"]
        payload.append(item)
    return "**INPUT:**\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def query_llm_batch(client: "genai.Client", model: str,
                     pairs: List[Dict[str, str]]) -> Dict[int, Dict[str, Any]]:
    """Send one batch of unique segment pairs to Gemini, return index -> verdict."""
    if not pairs:
        return {}
    prompt = build_batch_prompt(pairs)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=PROMPT_TEMPLATE,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    raw = (resp.text or "").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to locate the JSON array within the text
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else []
    return {item["index"]: item for item in parsed if "index" in item}


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    reference_col: str,
    predicted_col: str,
    output_dir: str,
    model: str = "gemini-flash-latest",
    batch_size: int = 20,
    api_key: str | None = None,
):
    path_obj = Path(input_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{input_path} does not exist")
    df = pd.read_csv(path_obj)
    for col in (reference_col, predicted_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Columns present: {list(df.columns)}")

    print(f"Loaded {len(df)} rows from {input_path}")

    # --- normalize + original WER/CER ---
    df["norm_reference"] = df[reference_col].astype(str).map(normalize_hindi)
    df["norm_prediction"] = df[predicted_col].astype(str).map(normalize_hindi)
    df["original_wer"] = [wer(r, p) for r, p in zip(df["norm_reference"], df["norm_prediction"])]
    df["original_cer"] = [cer(r, p) for r, p in zip(df["norm_reference"], df["norm_prediction"])]

    # --- diff every row into segments ---
    # Dedup key: (reference_segment, prediction_segment, context_reference,
    # context_prediction). For a normal replace/replace segment (both sides
    # non-empty) context is left as (None, None) so identical mismatches
    # still collapse into one LLM query regardless of which sentence they
    # came from, exactly as before.
    #
    # For an EMPTY-SIDE segment (pure insertion/deletion), the bare text is
    # not enough to disambiguate: the same inserted word ("" -> "नहीं") can
    # be a harmless repeated helper word in one sentence and a genuine
    # meaning-reversing negation in another. Deduping those two cases down
    # to one shared key -- as an earlier version of this fix did -- makes
    # every occurrence of that bare pair get the SAME verdict regardless of
    # which sentence it actually appeared in. So empty-side segments include
    # the full sentence pair in the key, forcing a separate LLM judgment
    # per distinct context.
    row_segment_map: Dict[int, List[Dict]] = {}
    PairKey = Tuple[str, str, Optional[str], Optional[str]]
    unique_segments: Dict[PairKey, List[Dict]] = {}

    def make_key(seg_ref: str, seg_pred: str, row_ref: str, row_pred: str) -> PairKey:
        is_empty_side = not seg_ref.strip() or not seg_pred.strip()
        if is_empty_side:
            return (seg_ref, seg_pred, row_ref, row_pred)
        return (seg_ref, seg_pred, None, None)

    for idx, row in df.iterrows():
        segments = get_segments(row["norm_reference"], row["norm_prediction"], key=idx)
        row_segment_map[idx] = segments
        for seg in segments:
            if seg["tag"] == "equal":
                continue
            # FIX 1: previously required both seg["reference"] and
            # seg["prediction"] to be non-empty, which silently skipped every
            # pure insertion/deletion (one side empty) and defaulted them to
            # "error" with no LLM judgment at all. Now every non-equal
            # segment is queried.
            key = make_key(seg["reference"], seg["prediction"], row["norm_reference"], row["norm_prediction"])
            unique_segments.setdefault(key, []).append(
                {"row_idx": idx, "segment_idx": seg["segment_idx"]}
            )

    print(f"Found {len(unique_segments)} unique mismatched segment pairs to check with the LLM.")

    # --- cache setup ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"{path_obj.stem}_llm_cache.jsonl"
    cache: Dict[PairKey, Dict[str, Any]] = {}
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache_key = (
                        item["reference"],
                        item["prediction"],
                        item.get("context_reference"),
                        item.get("context_prediction"),
                    )
                    cache[cache_key] = item
                except (json.JSONDecodeError, KeyError):
                    continue

    to_query = [pair for pair in unique_segments if pair not in cache]
    print(f"{len(unique_segments) - len(to_query)} pairs found in cache. Querying LLM for {len(to_query)} new pairs.")

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        raise SystemExit(
            "No Gemini API key found. Either pass --api-key YOUR_KEY, or set it in this same "
            "shell/session with:\n"
            "  export GEMINI_API_KEY=your_key_here   (macOS/Linux)\n"
            "  $env:GEMINI_API_KEY='your_key_here'    (Windows PowerShell)\n"
            "Get a free key at https://aistudio.google.com/apikey. Note: 'export' only applies to "
            "the terminal session you ran it in - a new terminal, notebook kernel, or IDE run "
            "button won't see it unless you set it there too."
        )
    client = genai.Client(api_key=resolved_key)
    log_records = []

    with cache_file.open("a", encoding="utf-8") as cache_f:
        for i in range(0, len(to_query), batch_size):
            batch_pairs = to_query[i:i + batch_size]
            batch_dicts = [
                {
                    "reference": ref,
                    "prediction": pred,
                    **(
                        {"context_reference": ctx_ref, "context_prediction": ctx_pred}
                        if ctx_ref is not None
                        else {}
                    ),
                }
                for (ref, pred, ctx_ref, ctx_pred) in batch_pairs
            ]
            print(f"  Querying LLM for segments {i}-{i + len(batch_pairs)} / {len(to_query)}")
            verdicts = query_llm_batch(client, model, batch_dicts)
            for j, (ref, pred, ctx_ref, ctx_pred) in enumerate(batch_pairs):
                verdict = verdicts.get(j, {"equivalence": False, "reasoning": "LLM did not return a verdict"})
                record = {
                    "reference": ref,
                    "prediction": pred,
                    "context_reference": ctx_ref,
                    "context_prediction": ctx_pred,
                    "equivalent": bool(verdict.get("equivalence", verdict.get("equivalent", False))),
                    "reasoning": verdict.get("reasoning", ""),
                }
                cache[(ref, pred, ctx_ref, ctx_pred)] = record
                cache_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- build final verdict map (row_idx, segment_idx) -> bool, using cache for everything ---
    equivalent_flags: Dict[Tuple[int, int], bool] = {}
    for key, occurrences in unique_segments.items():
        item = cache.get(key)
        is_equiv = bool(item and item.get("equivalent", False))
        if item:
            log_records.append(item)
        if is_equiv:
            for occ in occurrences:
                equivalent_flags[(occ["row_idx"], occ["segment_idx"])] = True

    # --- reconstruct corrected transcripts & rescore ---
    corrected_predictions, corrected_references = [], []
    for idx, row in df.iterrows():
        segments = row_segment_map.get(idx, [])
        pred_parts, ref_parts = [], []
        for seg in segments:
            is_equiv = equivalent_flags.get((idx, seg["segment_idx"]), False)
            if seg["tag"] == "equal" or is_equiv:
                pred_parts.append(seg["reference"])
            else:
                pred_parts.append(seg["prediction"])
            ref_parts.append(seg["reference"])
        corrected_predictions.append(" ".join(pred_parts))
        corrected_references.append(" ".join(ref_parts))

    df["corrected_prediction"] = corrected_predictions
    df["corrected_reference"] = corrected_references
    df["llm_wer"] = [wer(r, p) for r, p in zip(df["corrected_reference"], df["corrected_prediction"])]
    df["llm_cer"] = [cer(r, p) for r, p in zip(df["corrected_reference"], df["corrected_prediction"])]

    print("\n===== Results =====")
    print(f"Original WER (mean): {df['original_wer'].mean():.4f}")
    print(f"Original CER (mean): {df['original_cer'].mean():.4f}")
    print(f"LLM WER      (mean): {df['llm_wer'].mean():.4f}")
    print(f"LLM CER      (mean): {df['llm_cer'].mean():.4f}")

    out_csv = out_dir / f"{path_obj.stem}_llm_wer.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull results saved to: {out_csv}")

    if log_records:
        logs_csv = out_dir / f"{path_obj.stem}_llm_logs.csv"
        pd.DataFrame(log_records).drop_duplicates(subset=["reference", "prediction"]).to_csv(logs_csv, index=False)
        print(f"LLM reasoning logs saved to: {logs_csv}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute LLM-WER / LLM-CER for Hindi ASR output (sarvamai/llm_wer methodology).")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--reference-col", default="reference", help="Column name with ground-truth text.")
    parser.add_argument("--predicted-col", default="prediction", help="Column name with ASR/predicted text.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write results/cache to.")
    parser.add_argument("--model", default="gemini-flash-latest",
                         help="Gemini model name to use as the judge (free tier). "
                              "'gemini-flash-latest' auto-tracks Google's current free Flash model. "
                              "If that 404s for your account, try 'gemini-2.5-flash-lite' or run "
                              "`from google import genai; [m.name for m in genai.Client().models.list()]` "
                              "to see exactly what's enabled for your key.")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of segment pairs per LLM call (keep small to respect free-tier RPM limits).")
    parser.add_argument("--api-key", default=None,
                         help="Gemini API key. If omitted, reads GEMINI_API_KEY or GOOGLE_API_KEY "
                              "from the environment. Get a free key at https://aistudio.google.com/apikey")
    args = parser.parse_args()

    run(
        input_path=args.input,
        reference_col=args.reference_col,
        predicted_col=args.predicted_col,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        api_key=args.api_key,
    )