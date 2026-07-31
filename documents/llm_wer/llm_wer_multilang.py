# """
# LLM-WER / LLM-CER for Hindi, Bengali, Tamil, Marathi, Telugu
# =====================================================================
# (multilingual extension of sarvamai/llm_wer methodology)

# What this does:

# 1. Normalize reference & hypothesis text (Indic normalization + punctuation
#    strip), per-language.
# 2. Compute the ORIGINAL WER/CER (plain jiwer-based).
# 3. Diff reference vs. hypothesis word-by-word (difflib.SequenceMatcher) to
#    find the exact mismatched segments.
# 4. Send every *unique* mismatched (reference, hypothesis) segment pair to an
#    LLM judge, asking whether they are semantically + phonetically
#    equivalent (punctuation, numerals, transliteration, spoken-form, minor
#    spelling variants, cross-script, etc.) -- rules generalized to cover all
#    5 languages.
# 5. Any segment the LLM marks "equivalent" is treated as correct -> rebuild a
#    "corrected" transcript and recompute WER/CER on that ("llm_wer", "llm_cer").

# LLM judge: multi-provider fallback chain
# -----------------------------------------
#   1. Gemini (primary)            - google-genai SDK, free tier
#   2. OpenRouter free models       - open-weight models (Llama, Qwen, Gemma),
#                                      OpenAI-compatible REST API, free tier
#   3. Groq free tier               - open-weight Llama models, very fast,
#                                      OpenAI-compatible REST API, free tier
# Each provider also tries a short list of model names, so if one specific
# model 404s / is retired / hits its own quota, the script moves to the next
# model, then the next provider, before giving up on a batch. Every provider
# in this chain is only used when its API key env var is set, so you can run
# with just a Gemini key and no fallbacks configured -- it just won't have
# anywhere to fall back to.

# Install:
#     pip install pandas jiwer indic-nlp-library google-genai requests

# API keys (set whichever you have -- more keys = more fallback resilience):
#     export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey
#     export OPENROUTER_API_KEY=...    # https://openrouter.ai/keys (free tier)
#     export GROQ_API_KEY=...          # https://console.groq.com/keys (free tier)

# CSV schema (matches your files):
#     sample_id,audio_filepath,duration,reference,hypothesis,wer,cer,status
# The script reads `reference` and `hypothesis` columns by default (override
# with --reference-col / --hypothesis-col). Any existing `wer`/`cer` columns
# in your CSV are left untouched and simply carried through to the output as
# `reported_wer` / `reported_cer` for comparison against the freshly computed
# `original_wer` / `original_cer`.

# Run - single language:
#     python llm_wer_multilingual.py \
#         --input hindi_results.csv --lang hi --output-dir outputs

# Run - all 5 languages in one go (batch mode):
#     python llm_wer_multilingual.py \
#         --batch-dir ./csvs --output-dir outputs
#     # expects filenames that contain a language hint, e.g.:
#     #   hindi_results.csv / results_hi.csv / hi_test.csv       -> hi
#     #   bengali_results.csv / results_bn.csv / bangla_test.csv -> bn
#     #   tamil_results.csv / results_ta.csv                     -> ta
#     #   marathi_results.csv / results_mr.csv                   -> mr
#     #   telugu_results.csv / results_te.csv                    -> te

# Persistent cross-language summary:
#     Every run (single-file or batch) appends a block to
#     `<output-dir>/wer_summary_all_languages.txt` -- original WER/CER and
#     LLM WER/CER for that run, with a timestamp and input filename. The file
#     is opened in APPEND mode only, so running Hindi today and Telugu next
#     week just adds a new block; nothing already in the file is ever
#     overwritten or deleted, regardless of which language or file is run.
# """

# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import string
# import time
# from datetime import datetime
# from difflib import SequenceMatcher
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import pandas as pd
# import jiwer
# import requests

# try:
#     from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
# except ImportError as e:
#     raise ImportError(
#         "indic-nlp-library is required. Install with: pip install indic-nlp-library"
#     ) from e

# try:
#     from google import genai
#     from google.genai import types as genai_types
# except ImportError as e:
#     raise ImportError(
#         "google-genai is required. Install with: pip install google-genai"
#     ) from e


# # ---------------------------------------------------------------------------
# # 0. Language configuration
# # ---------------------------------------------------------------------------

# # code -> (display name, indic-nlp-library normalizer code)
# LANGUAGES: Dict[str, Dict[str, str]] = {
#     "hi": {"name": "Hindi", "indic_code": "hi"},
#     "bn": {"name": "Bengali", "indic_code": "bn"},
#     "ta": {"name": "Tamil", "indic_code": "ta"},
#     "mr": {"name": "Marathi", "indic_code": "mr"},
#     "te": {"name": "Telugu", "indic_code": "te"},
# }

# # filename hints used by --batch-dir to auto-detect language from a file name
# LANG_FILENAME_HINTS: Dict[str, List[str]] = {
#     "hi": ["hindi", "_hi", "-hi", "hi_", "hi-"],
#     "bn": ["bengali", "bangla", "_bn", "-bn", "bn_", "bn-"],
#     "ta": ["tamil", "_ta", "-ta", "ta_", "ta-"],
#     "mr": ["marathi", "_mr", "-mr", "mr_", "mr-"],
#     "te": ["telugu", "_te", "-te", "te_", "te-"],
# }


# def detect_language_from_filename(path: Path) -> Optional[str]:
#     stem = path.stem.lower()
#     for lang_code, hints in LANG_FILENAME_HINTS.items():
#         for hint in hints:
#             if hint in stem:
#                 return lang_code
#     return None


# # ---------------------------------------------------------------------------
# # 1. WER / CER (identical formulas to sarvamai/llm_wer utilities.py)
# # ---------------------------------------------------------------------------

# def wer(ref: str, hyp: str, clamp: bool = True,
#         insertion_weight: float = 1, deletion_weight: float = 1,
#         substitution_weight: float = 1) -> float:
#     ref, hyp = str(ref).strip(), str(hyp).strip()
#     N, M = len(ref.split()), len(hyp.split())
#     if N == 0 and M == 0:
#         return 0.0
#     if N == 0 and M > 0:
#         return insertion_weight
#     if N > 0 and M == 0:
#         return deletion_weight
#     out = jiwer.process_words(ref, hyp)
#     S, D, I = out.substitutions, out.deletions, out.insertions
#     denom = max(M, N) if clamp else N
#     return (S * substitution_weight + D * deletion_weight + I * insertion_weight) / denom


# def cer(ref: str, hyp: str, clamp: bool = True,
#         insertion_weight: float = 1, deletion_weight: float = 1.0,
#         substitution_weight: float = 1.0) -> float:
#     ref, hyp = str(ref).strip(), str(hyp).strip()
#     N, M = len(ref), len(hyp)
#     if N == 0 and M == 0:
#         return 0.0
#     if N == 0 and M > 0:
#         return insertion_weight
#     if N > 0 and M == 0:
#         return deletion_weight
#     out = jiwer.process_characters(ref, hyp)
#     S, D, I = out.substitutions, out.deletions, out.insertions
#     denom = max(M, N) if clamp else N
#     return (S * substitution_weight + D * deletion_weight + I * insertion_weight) / denom


# # ---------------------------------------------------------------------------
# # 2. Indic normalization (generalized across the 5 languages)
# # ---------------------------------------------------------------------------

# INDIC_PUNCTUATION = "।॥॰''\"‛‟′″´˝^°¤।॥॰¯'—–‑°¬´\u200b\u200c\u200d\u200e\u200f"

# _NORMALIZER_CACHE: Dict[str, Any] = {}


# def _get_normalizer(indic_code: str):
#     if indic_code not in _NORMALIZER_CACHE:
#         _NORMALIZER_CACHE[indic_code] = IndicNormalizerFactory().get_normalizer(indic_code)
#     return _NORMALIZER_CACHE[indic_code]


# def normalize_text(text: str, lang: str) -> str:
#     if not isinstance(text, str) or not text:
#         return "" if not isinstance(text, str) else text
#     indic_code = LANGUAGES[lang]["indic_code"]
#     normalizer = _get_normalizer(indic_code)
#     text = re.sub(r'([,\-\.\(\)\[\]\{\}/\\])\B', r' ', text)
#     text = text.translate(str.maketrans('', '', string.punctuation + INDIC_PUNCTUATION))
#     text = text.lower()
#     text = normalizer.normalize(text)
#     text = re.sub(' +', ' ', text).strip()
#     return text


# # ---------------------------------------------------------------------------
# # 3. Segment diffing (identical to get_segments in main.py)
# # ---------------------------------------------------------------------------

# def get_segments(reference_string: str, predicted_string: str, key: Any) -> List[Dict[str, Any]]:
#     ref_words = reference_string.strip().split()
#     pred_words = predicted_string.strip().split()
#     if not ref_words and not pred_words:
#         return []
#     matcher = SequenceMatcher(None, ref_words, pred_words)
#     return [
#         {
#             "reference": " ".join(ref_words[i1:i2]),
#             "prediction": " ".join(pred_words[j1:j2]),
#             "tag": tag,
#             "key": key,
#             "segment_idx": seg_idx,
#         }
#         for seg_idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes())
#     ]


# # ---------------------------------------------------------------------------
# # 4. LLM equivalence prompt (generalized for Hindi / Bengali / Tamil /
# #    Marathi / Telugu)
# # ---------------------------------------------------------------------------

# PROMPT_TEMPLATE = """# Persona
# You are an expert linguistic analyst specializing in Indian languages, fluent in
# Hindi, Bengali, Tamil, Marathi and Telugu.

# # Primary Goal
# Your primary goal is to precisely compare two transcripts (in one of the above
# five languages, specified per item as `language`) and determine if they are
# essentially equivalent based on a set of equivalence rules. Adapt the phonetic
# and structural principles to whichever language and script the item is in.

# # Equivalence Rules
# To determine equivalence, you MUST adhere strictly to the following rules.

# ## 1. Formatting and Symbol Equivalence
# - **Ignore Punctuation:** Disregard all punctuation marks (e.g., ।, ?, ,, ॥, -, .).
# - **Hyphenation:** Treat hyphenated words as identical to their multi-word or single-word counterparts.
#   - Example (Hindi): "धीरे-धीरे" (dheere-dheere) = "धीरे धीरे" = "धीरेधीरे"
#   - Example (Tamil): "அங்கு-இங்கு" (angu-ingu) = "அங்கு இங்கு" = "அங்குஇங்கு"
#   - Example (Bengali): "পাশে-পাশে" (pashe-pashe) = "পাশে পাশে" = "পাশপাশে"
#   - Example (Marathi): "हळू-हळू" (halu-halu) = "हळू हळू" = "हळूहळू"
#   - Example (Telugu): "నెమ్మది-నెమ్మది" (nemmadi-nemmadi) = "నెమ్మది నెమ్మది"
# - **Numbers:** Convert all numbers, whether in digit or word form, to a standard numeric value for comparison. Combine consecutive number words.
#   - Example (Hindi): "उन्नीस सौ नब्बे" (unnees sau nabbe) = "1990"
#   - Example (Tamil): "இரண்டு ஆயிரம் இருபத்தி மூன்று" (irandu aayiram irupathi moondru) = "2023"
#   - Example (Telugu): "పంతొమ్మిది వందల తొంభై" (panthommidi vandala thombai) = "1990"
#   - Example (Marathi): "एकोणीसशे नव्वद" (ekonisshe navvad) = "1990"
# - **Symbols:** Words representing symbols are equivalent to the symbols themselves.
#   - Example (Hindi): "रुपये" (rupaye) = '₹'
#   - Example (Tamil): "சதவீதம்" (sadhaveedham) = '%'
#   - Example (Bengali): "টাকা" (taka) = '₹'
#   - Example (Marathi): "टक्के" (takke) = '%'
#   - Example (Telugu): "శాతం" (shatam) = '%'
# - **Numbers and Currencies:** Numbers with symbols representing the same value are equivalent.
#   - Example (Hindi): "सौ रुपये" (sau rupaye) = "₹100"
#   - Example (Hindi): "एक सौ चार रुपये पचास पैसे" (ek sau chaar rupaye pachaas paise) = "₹104.50"

# ## 2. Spoken vs. Written Form Equivalence
# Account for common differences between spoken and written forms.
# - **Acronyms and Initialisms:** Spoken-out letters of an acronym are equivalent to the consolidated written form.
#   - Example (Hindi): "पी एन बी" = "पीएनबी"
#   - Example (Tamil): "டி சி எஸ்" = "டிசிஎஸ்"
# - **Phonetic Spelling of Brands/Names:** Phonetic spellings of proper nouns or brands are equivalent to their standard written form.
#   - Example (Hindi): "रेडियो मिर्ची" = "Radio Mirchi"
#   - Example (Tamil): "கோக கோலா" = "Coca-Cola"
#   - Example (Bengali): "পেটিএম" (Paytm) = "Paytm"
#   - Example (Marathi): "फेसबुक" (Facebook) = "Facebook"

# ## 3. Language and Script Equivalence
# - **Cross-Script Equivalence:** Words that sound the same but are written in different scripts (e.g., Roman and a native Indian script) are equivalent.
#   - Example (Hindi): "Amazon" = "अमेज़न"
#   - Example (Tamil): "WhatsApp" = "வாட்ஸ்அப்"
#   - Example (Telugu): "Facebook" = "ఫేస్బుక్"
#   - Example (Marathi): "Youtube" = "यूट्यूब"
# - **Common Spelling Variations:** Minor, common spelling variations that do not significantly alter pronunciation are equivalent. This includes variations in spacing for the same word.
#   - Examples: "दोबारा" = "दुबारा", "கட்டிடம்" = "கட்டிடம", "वहाँ" = "वहां", "मज़ा" = "मजा"
#   - Example (Bengali): "জন্য" (jonnyo) = "জন্যে" (jonne)
#   - Example (Telugu): "వెళ్తున్నాను" (velthunnanu) = "వెళ్తున్నా" (velthunna)
#   - Example (Marathi): "करतोय" (kartoy) = "करतो आहे" (karto aahe)

# ## 4. Phonetic Contractions or Reductions
# - **Phonetic Contractions/Reductions:** Words that are phonetic reductions or contractions of another word are equivalent if their pronunciation is somewhat similar.
#   - Example (Hindi): 'पर' (par) = 'पे' (pe)
#   - Example (Hindi): 'ये' (ye) = 'यह' (yah)
#   - Example (Bengali): 'তাহার' (tahar) = 'তার' (tar)
#   - Example (Marathi): 'त्याला' (tyala) = 'त्याले' (tyale) (colloquial)

# ## 5. Empty-Side (Insertion/Deletion) Segments
# Some pairs you receive will have an EMPTY reference or an EMPTY hypothesis --
# that means the aligner found a word (or short phrase) inserted into, or
# deleted from, the hypothesis transcript relative to the reference, with
# nothing on the other side to compare it to directly. For these you will
# also be given `full_reference_sentence` and `full_hypothesis_sentence` --
# use them to judge the inserted/deleted span in context, not in isolation:
# - **Stray repeated helper word:** if the inserted/deleted span is a short
#   helper word (negation particles, fillers, discourse markers) that is
#   simply duplicated adjacent to an identical word already in the sentence
#   (e.g. reference "यह सही नहीं है", hypothesis "यह सही नहीं नहीं है" --
#   segment is an inserted "नहीं"), this is a disfluency/stutter artifact,
#   NOT a meaning change. Mark it **equivalent**.
# - **Meaning-changing insertion/deletion:** if the inserted/deleted span
#   changes what the sentence asserts -- most importantly inserting or
#   removing a negation that is NOT simply duplicating an adjacent word
#   (e.g. reference "मुझे यह पसंद है" / hypothesis "मुझे यह नहीं पसंद है" --
#   "I like this" vs "I do NOT like this"), or inserts/removes any other
#   content word that alters the sentence's meaning, this is a genuine
#   error. Mark it **not equivalent**, regardless of how short the span is.
# The test is always: does the full hypothesis sentence assert the same
# thing as the full reference sentence? A duplicated word does not change
# what is asserted; a negation (or other content change) does.

# # Input Format:
# You will be given a list of JSON objects, of the following format:

# ```json
# {
#   "index": int,
#   "language": str,                       // one of: Hindi, Bengali, Tamil, Marathi, Telugu
#   "reference": str,
#   "hypothesis": str,
#   "full_reference_sentence": str,         // OPTIONAL, only present for empty-side segments
#   "full_hypothesis_sentence": str         // OPTIONAL, only present for empty-side segments
# }
# ```

# # Output Format
# Your final output must be a single JSON object with the keys: `index`, `equivalence` and `reasoning`.

# - **`index`**: An integer value. Ensure that the `index` is the same as the input index for the corresponding reference and hypothesis pair.
# - **`equivalence`**: A boolean value. Use `true` for an exact match and `false` for a mismatch after carefully considering all the rules and analyzing the transcripts.
# - **`reasoning`**: A string. Provide a brief, clear explanation for the equivalence value, highlighting the specific words or sequences along with its translation as well as transliteration to English.

# ```json
# {
#     "index": int,
#     "equivalence": bool,
#     "reasoning": str
# }
# ```

# # Batch Output Note (addendum)
# You will be given MULTIPLE input objects at once (a JSON array). For each one, produce an
# object following the Output Format above. Return ONLY a single JSON array containing one
# such object per input object, in any order, with no markdown fences and no other text."""


# def build_batch_prompt(pairs: List[Dict[str, str]], lang_name: str) -> str:
#     payload = []
#     for i, p in enumerate(pairs):
#         item: Dict[str, Any] = {
#             "index": i,
#             "language": lang_name,
#             "reference": p["reference"],
#             "hypothesis": p["prediction"],
#         }
#         if p.get("context_reference") is not None:
#             item["full_reference_sentence"] = p["context_reference"]
#             item["full_hypothesis_sentence"] = p["context_prediction"]
#         payload.append(item)
#     return "**INPUT:**\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
#     raw = (raw or "").strip()
#     raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
#     try:
#         return json.loads(raw)
#     except json.JSONDecodeError:
#         match = re.search(r"\[.*\]", raw, re.DOTALL)
#         if match:
#             return json.loads(match.group(0))
#         raise


# # ---------------------------------------------------------------------------
# # 4b. LLM providers -- Gemini primary, open-source free-tier fallbacks
# # ---------------------------------------------------------------------------

# class ProviderError(Exception):
#     pass


# class GeminiProvider:
#     name = "gemini"

#     def __init__(self, api_key: str, models: List[str]):
#         self.client = genai.Client(api_key=api_key)
#         self.models = models

#     def call(self, model: str, system_prompt: str, user_prompt: str) -> str:
#         resp = self.client.models.generate_content(
#             model=model,
#             contents=user_prompt,
#             config=genai_types.GenerateContentConfig(
#                 system_instruction=system_prompt,
#                 response_mime_type="application/json",
#                 temperature=0.0,
#             ),
#         )
#         text = resp.text
#         if not text:
#             raise ProviderError("Gemini returned empty response")
#         return text


# class OpenAICompatibleProvider:
#     """Works for any OpenAI-chat-completions-compatible free-tier endpoint
#     (OpenRouter, Groq, etc.) serving open-weight models."""

#     def __init__(self, name: str, api_key: str, base_url: str, models: List[str],
#                  extra_headers: Optional[Dict[str, str]] = None):
#         self.name = name
#         self.api_key = api_key
#         self.base_url = base_url
#         self.models = models
#         self.extra_headers = extra_headers or {}

#     def call(self, model: str, system_prompt: str, user_prompt: str) -> str:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#         headers.update(self.extra_headers)
#         body = {
#             "model": model,
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             "temperature": 0.0,
#             "response_format": {"type": "json_object"} if False else None,
#         }
#         # Not every free-tier open model reliably honors response_format /
#         # json_object mode, so we don't force it here -- instead we ask for
#         # a JSON array explicitly in the prompt and parse leniently below.
#         body.pop("response_format", None)
#         resp = requests.post(f"{self.base_url}/chat/completions", headers=headers,
#                               data=json.dumps(body), timeout=90)
#         if resp.status_code == 429:
#             raise ProviderError(f"{self.name}/{model} rate-limited (429)")
#         if resp.status_code >= 400:
#             raise ProviderError(f"{self.name}/{model} HTTP {resp.status_code}: {resp.text[:300]}")
#         data = resp.json()
#         try:
#             text = data["choices"][0]["message"]["content"]
#         except (KeyError, IndexError):
#             raise ProviderError(f"{self.name}/{model} unexpected response shape: {data}")
#         if not text:
#             raise ProviderError(f"{self.name}/{model} returned empty content")
#         return text


# def build_provider_chain(requested: List[str]) -> List[Any]:
#     """Build the ordered fallback chain from whichever API keys are set."""
#     chain: List[Any] = []

#     gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
#     if "gemini" in requested and gemini_key:
#         chain.append(GeminiProvider(
#             api_key=gemini_key,
#             models=["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"],
#         ))

#     openrouter_key = os.environ.get("OPENROUTER_API_KEY")
#     if "openrouter" in requested and openrouter_key:
#         chain.append(OpenAICompatibleProvider(
#             name="openrouter",
#             api_key=openrouter_key,
#             base_url="https://openrouter.ai/api/v1",
#             models=[
#                 "meta-llama/llama-3.3-70b-instruct:free",
#                 "qwen/qwen-2.5-72b-instruct:free",
#                 "google/gemma-2-9b-it:free",
#             ],
#             extra_headers={"HTTP-Referer": "https://local-script", "X-Title": "llm-wer-multilingual"},
#         ))

#     groq_key = os.environ.get("GROQ_API_KEY")
#     if "groq" in requested and groq_key:
#         chain.append(OpenAICompatibleProvider(
#             name="groq",
#             api_key=groq_key,
#             base_url="https://api.groq.com/openai/v1",
#             models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
#         ))

#     if not chain:
#         raise SystemExit(
#             "No LLM provider API keys found. Set at least one of:\n"
#             "  export GEMINI_API_KEY=...      (https://aistudio.google.com/apikey)\n"
#             "  export OPENROUTER_API_KEY=...  (https://openrouter.ai/keys, free tier)\n"
#             "  export GROQ_API_KEY=...        (https://console.groq.com/keys, free tier)\n"
#             "Gemini is recommended as the primary judge; the others are open-source "
#             "free-tier fallbacks used only if Gemini's quota is exhausted or it errors."
#         )
#     return chain


# def query_llm_batch_with_fallback(
#     providers: List[Any],
#     pairs: List[Dict[str, str]],
#     lang_name: str,
# ) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
#     """Try each provider (and each model within it) in order. Returns
#     (index -> verdict dict, provider/model string used) or ({}, None) if
#     every provider/model failed."""
#     if not pairs:
#         return {}, None

#     user_prompt = build_batch_prompt(pairs, lang_name)

#     for provider in providers:
#         for model in provider.models:
#             for attempt in range(2):  # one retry per model on transient errors
#                 try:
#                     raw = provider.call(model, PROMPT_TEMPLATE, user_prompt)
#                     parsed = _extract_json_array(raw)
#                     verdicts = {item["index"]: item for item in parsed if "index" in item}
#                     if verdicts:
#                         return verdicts, f"{provider.name}/{model}"
#                     raise ProviderError("parsed empty/invalid verdict list")
#                 except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall back
#                     wait = 3 if attempt == 0 else 0
#                     print(f"    [warn] {provider.name}/{model} failed ({exc}); "
#                           f"{'retrying' if attempt == 0 else 'moving to next model/provider'}")
#                     if wait:
#                         time.sleep(wait)
#     return {}, None


# SUMMARY_TXT_NAME = "wer_summary_all_languages.txt"


# def append_summary_txt(output_dir: str, lang_name: str, lang_code: str,
#                         input_file: str, df: pd.DataFrame) -> Path:
#     """Append this run's results as a new block in a shared, cross-language
#     summary .txt file. Always opened in append mode -- existing blocks from
#     earlier runs (this language or any other) are never touched or removed,
#     so results accumulate across every language you run over time."""
#     out_dir = Path(output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     summary_path = out_dir / SUMMARY_TXT_NAME
#     is_new = not summary_path.exists()

#     n = len(df)
#     orig_wer = df["original_wer"].mean()
#     orig_cer = df["original_cer"].mean()
#     llm_wer_v = df["llm_wer"].mean()
#     llm_cer_v = df["llm_cer"].mean()
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     with summary_path.open("a", encoding="utf-8") as f:
#         if is_new:
#             f.write("LLM-WER / LLM-CER SUMMARY -- all languages, all runs\n")
#             f.write("Each run appends a new block below; nothing is ever overwritten or deleted.\n")
#             f.write("=" * 78 + "\n\n")
#         f.write(f"[{timestamp}]  Language: {lang_name} ({lang_code})\n")
#         f.write(f"  Input file     : {input_file}\n")
#         f.write(f"  Rows evaluated : {n}\n")
#         f.write(f"  Original WER   : {orig_wer:.4f}\n")
#         f.write(f"  Original CER   : {orig_cer:.4f}\n")
#         f.write(f"  LLM WER        : {llm_wer_v:.4f}\n")
#         f.write(f"  LLM CER        : {llm_cer_v:.4f}\n")
#         f.write("-" * 78 + "\n\n")

#     print(f"[{lang_name}] Appended summary to: {summary_path}")
#     return summary_path


# # ---------------------------------------------------------------------------
# # 5. Full pipeline (single language)
# # ---------------------------------------------------------------------------

# def run(
#     input_path: str,
#     lang: str,
#     reference_col: str = "reference",
#     hypothesis_col: str = "hypothesis",
#     output_dir: str = "outputs",
#     batch_size: int = 10,
#     provider_order: Optional[List[str]] = None,
# ):
#     if lang not in LANGUAGES:
#         raise ValueError(f"Unsupported --lang '{lang}'. Choose from: {list(LANGUAGES)}")
#     lang_name = LANGUAGES[lang]["name"]

#     path_obj = Path(input_path)
#     if not path_obj.exists():
#         raise FileNotFoundError(f"{input_path} does not exist")
#     df = pd.read_csv(path_obj)
#     for col in (reference_col, hypothesis_col):
#         if col not in df.columns:
#             raise ValueError(f"Column '{col}' not found in CSV. Columns present: {list(df.columns)}")

#     print(f"[{lang_name}] Loaded {len(df)} rows from {input_path}")

#     # keep any pre-existing wer/cer/status columns as "reported_*" for comparison
#     for existing_col, new_col in [("wer", "reported_wer"), ("cer", "reported_cer")]:
#         if existing_col in df.columns:
#             df[new_col] = df[existing_col]

#     # --- normalize + original WER/CER ---
#     df["norm_reference"] = df[reference_col].astype(str).map(lambda t: normalize_text(t, lang))
#     df["norm_hypothesis"] = df[hypothesis_col].astype(str).map(lambda t: normalize_text(t, lang))
#     df["original_wer"] = [wer(r, p) for r, p in zip(df["norm_reference"], df["norm_hypothesis"])]
#     df["original_cer"] = [cer(r, p) for r, p in zip(df["norm_reference"], df["norm_hypothesis"])]

#     # --- diff every row into segments ---
#     row_segment_map: Dict[int, List[Dict]] = {}
#     PairKey = Tuple[str, str, Optional[str], Optional[str]]
#     unique_segments: Dict[PairKey, List[Dict]] = {}

#     def make_key(seg_ref: str, seg_pred: str, row_ref: str, row_pred: str) -> PairKey:
#         is_empty_side = not seg_ref.strip() or not seg_pred.strip()
#         if is_empty_side:
#             return (seg_ref, seg_pred, row_ref, row_pred)
#         return (seg_ref, seg_pred, None, None)

#     for idx, row in df.iterrows():
#         segments = get_segments(row["norm_reference"], row["norm_hypothesis"], key=idx)
#         row_segment_map[idx] = segments
#         for seg in segments:
#             if seg["tag"] == "equal":
#                 continue
#             key = make_key(seg["reference"], seg["prediction"], row["norm_reference"], row["norm_hypothesis"])
#             unique_segments.setdefault(key, []).append(
#                 {"row_idx": idx, "segment_idx": seg["segment_idx"]}
#             )

#     print(f"[{lang_name}] Found {len(unique_segments)} unique mismatched segment pairs to check with the LLM.")

#     # --- cache setup (per-language, per-input-file) ---
#     out_dir = Path(output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     cache_file = out_dir / f"{path_obj.stem}_{lang}_llm_cache.jsonl"
#     cache: Dict[PairKey, Dict[str, Any]] = {}
#     if cache_file.exists():
#         with cache_file.open("r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     item = json.loads(line)
#                     cache_key = (
#                         item["reference"],
#                         item["prediction"],
#                         item.get("context_reference"),
#                         item.get("context_prediction"),
#                     )
#                     cache[cache_key] = item
#                 except (json.JSONDecodeError, KeyError):
#                     continue

#     to_query = [pair for pair in unique_segments if pair not in cache]
#     print(f"[{lang_name}] {len(unique_segments) - len(to_query)} pairs found in cache. "
#           f"Querying LLM for {len(to_query)} new pairs.")

#     providers = build_provider_chain(provider_order or ["gemini", "openrouter", "groq"])
#     print(f"[{lang_name}] Provider fallback chain: "
#           f"{' -> '.join(p.name + ':' + '/'.join(p.models) for p in providers)}")

#     log_records = []

#     with cache_file.open("a", encoding="utf-8") as cache_f:
#         for i in range(0, len(to_query), batch_size):
#             batch_pairs = to_query[i:i + batch_size]
#             batch_dicts = [
#                 {
#                     "reference": ref,
#                     "prediction": pred,
#                     **(
#                         {"context_reference": ctx_ref, "context_prediction": ctx_pred}
#                         if ctx_ref is not None
#                         else {}
#                     ),
#                 }
#                 for (ref, pred, ctx_ref, ctx_pred) in batch_pairs
#             ]
#             print(f"[{lang_name}]   Querying LLM for segments {i}-{i + len(batch_pairs)} / {len(to_query)}")
#             verdicts, used = query_llm_batch_with_fallback(providers, batch_dicts, lang_name)
#             if used:
#                 print(f"[{lang_name}]   -> answered by {used}")
#             else:
#                 print(f"[{lang_name}]   -> ALL providers failed for this batch; "
#                       f"defaulting these {len(batch_pairs)} pairs to 'not equivalent'")
#             for j, (ref, pred, ctx_ref, ctx_pred) in enumerate(batch_pairs):
#                 verdict = verdicts.get(j, {"equivalence": False, "reasoning": "All LLM providers failed"})
#                 record = {
#                     "language": lang_name,
#                     "reference": ref,
#                     "prediction": pred,
#                     "context_reference": ctx_ref,
#                     "context_prediction": ctx_pred,
#                     "equivalent": bool(verdict.get("equivalence", verdict.get("equivalent", False))),
#                     "reasoning": verdict.get("reasoning", ""),
#                     "llm_provider": used or "none",
#                 }
#                 cache[(ref, pred, ctx_ref, ctx_pred)] = record
#                 cache_f.write(json.dumps(record, ensure_ascii=False) + "\n")

#     # --- build final verdict map (row_idx, segment_idx) -> bool ---
#     equivalent_flags: Dict[Tuple[int, int], bool] = {}
#     for key, occurrences in unique_segments.items():
#         item = cache.get(key)
#         is_equiv = bool(item and item.get("equivalent", False))
#         if item:
#             log_records.append(item)
#         if is_equiv:
#             for occ in occurrences:
#                 equivalent_flags[(occ["row_idx"], occ["segment_idx"])] = True

#     # --- reconstruct corrected transcripts & rescore ---
#     corrected_hypotheses, corrected_references = [], []
#     for idx, row in df.iterrows():
#         segments = row_segment_map.get(idx, [])
#         pred_parts, ref_parts = [], []
#         for seg in segments:
#             is_equiv = equivalent_flags.get((idx, seg["segment_idx"]), False)
#             if seg["tag"] == "equal" or is_equiv:
#                 pred_parts.append(seg["reference"])
#             else:
#                 pred_parts.append(seg["prediction"])
#             ref_parts.append(seg["reference"])
#         corrected_hypotheses.append(" ".join(pred_parts))
#         corrected_references.append(" ".join(ref_parts))

#     df["corrected_hypothesis"] = corrected_hypotheses
#     df["corrected_reference"] = corrected_references
#     df["llm_wer"] = [wer(r, p) for r, p in zip(df["corrected_reference"], df["corrected_hypothesis"])]
#     df["llm_cer"] = [cer(r, p) for r, p in zip(df["corrected_reference"], df["corrected_hypothesis"])]
#     df["language"] = lang_name

#     print(f"\n===== Results: {lang_name} =====")
#     print(f"Original WER (mean): {df['original_wer'].mean():.4f}")
#     print(f"Original CER (mean): {df['original_cer'].mean():.4f}")
#     print(f"LLM WER      (mean): {df['llm_wer'].mean():.4f}")
#     print(f"LLM CER      (mean): {df['llm_cer'].mean():.4f}")

#     out_csv = out_dir / f"{path_obj.stem}_{lang}_llm_wer.csv"
#     df.to_csv(out_csv, index=False)
#     print(f"Full results saved to: {out_csv}")

#     if log_records:
#         logs_csv = out_dir / f"{path_obj.stem}_{lang}_llm_logs.csv"
#         pd.DataFrame(log_records).drop_duplicates(subset=["reference", "prediction"]).to_csv(logs_csv, index=False)
#         print(f"LLM reasoning logs saved to: {logs_csv}")

#     append_summary_txt(output_dir, lang_name, lang, str(path_obj), df)

#     return df


# # ---------------------------------------------------------------------------
# # 6. Batch driver across all 5 languages
# # ---------------------------------------------------------------------------

# def run_batch(
#     batch_dir: str,
#     output_dir: str = "outputs",
#     reference_col: str = "reference",
#     hypothesis_col: str = "hypothesis",
#     batch_size: int = 10,
#     provider_order: Optional[List[str]] = None,
# ):
#     directory = Path(batch_dir)
#     if not directory.is_dir():
#         raise FileNotFoundError(f"{batch_dir} is not a directory")

#     csv_files = sorted(directory.glob("*.csv"))
#     if not csv_files:
#         raise FileNotFoundError(f"No CSV files found in {batch_dir}")

#     all_results = []
#     skipped = []
#     for csv_path in csv_files:
#         lang = detect_language_from_filename(csv_path)
#         if lang is None:
#             skipped.append(csv_path.name)
#             continue
#         print(f"\n########## {csv_path.name} -> detected language: {LANGUAGES[lang]['name']} ({lang}) ##########")
#         df = run(
#             input_path=str(csv_path),
#             lang=lang,
#             reference_col=reference_col,
#             hypothesis_col=hypothesis_col,
#             output_dir=output_dir,
#             batch_size=batch_size,
#             provider_order=provider_order,
#         )
#         all_results.append(df)

#     if skipped:
#         print(f"\n[warn] Could not detect language for these files, skipped: {skipped}. "
#               f"Rename them to include a language hint (hindi/hi, bengali/bn, tamil/ta, "
#               f"marathi/mr, telugu/te) or run them individually with --input/--lang.")

#     if all_results:
#         combined = pd.concat(all_results, ignore_index=True)
#         out_dir = Path(output_dir)
#         combined_csv = out_dir / "combined_llm_wer_all_languages.csv"
#         combined.to_csv(combined_csv, index=False)
#         print(f"\nCombined results for all languages saved to: {combined_csv}")

#         print("\n===== Summary across all languages =====")
#         summary = combined.groupby("language")[["original_wer", "original_cer", "llm_wer", "llm_cer"]].mean()
#         print(summary.round(4).to_string())

#     return all_results


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Compute LLM-WER / LLM-CER for Hindi/Bengali/Tamil/Marathi/Telugu ASR output."
#     )
#     parser.add_argument("--input", help="Path to a single input CSV file (use with --lang).")
#     parser.add_argument("--lang", choices=list(LANGUAGES), help="Language code for --input: hi, bn, ta, mr, te.")
#     parser.add_argument("--batch-dir", help="Directory containing multiple per-language CSV files; "
#                                              "language is auto-detected from each filename.")
#     parser.add_argument("--reference-col", default="reference", help="Column name with ground-truth text.")
#     parser.add_argument("--hypothesis-col", default="hypothesis", help="Column name with ASR hypothesis text.")
#     parser.add_argument("--output-dir", default="outputs", help="Directory to write results/cache to.")
#     parser.add_argument("--batch-size", type=int, default=10,
#                          help="Number of segment pairs per LLM call (keep small to respect free-tier RPM limits).")
#     parser.add_argument("--providers", default="gemini,openrouter,groq",
#                          help="Comma-separated fallback order. Providers without an API key set are "
#                               "skipped automatically. Default: gemini,openrouter,groq")
#     args = parser.parse_args()

#     provider_order = [p.strip() for p in args.providers.split(",") if p.strip()]

#     if args.batch_dir:
#         run_batch(
#             batch_dir=args.batch_dir,
#             output_dir=args.output_dir,
#             reference_col=args.reference_col,
#             hypothesis_col=args.hypothesis_col,
#             batch_size=args.batch_size,
#             provider_order=provider_order,
#         )
#     elif args.input and args.lang:
#         run(
#             input_path=args.input,
#             lang=args.lang,
#             reference_col=args.reference_col,
#             hypothesis_col=args.hypothesis_col,
#             output_dir=args.output_dir,
#             batch_size=args.batch_size,
#             provider_order=provider_order,
#         )
#     else:
#         parser.error("Either provide --batch-dir (process a folder of per-language CSVs), "
#                       "or both --input and --lang (process a single CSV).")

"""
LLM-WER / LLM-CER for Hindi, Bengali, Tamil, Marathi, Telugu
=====================================================================
(multilingual extension of sarvamai/llm_wer methodology)

What this does:

1. Normalize reference & hypothesis text (Indic normalization + punctuation
   strip), per-language.
2. Compute the ORIGINAL WER/CER (plain jiwer-based).
3. Diff reference vs. hypothesis word-by-word (difflib.SequenceMatcher) to
   find the exact mismatched segments.
4. Send every *unique* mismatched (reference, hypothesis) segment pair to an
   LLM judge, asking whether they are semantically + phonetically
   equivalent (punctuation, numerals, transliteration, spoken-form, minor
   spelling variants, cross-script, etc.) -- rules generalized to cover all
   5 languages.
5. Any segment the LLM marks "equivalent" is treated as correct -> rebuild a
   "corrected" transcript and recompute WER/CER on that ("llm_wer", "llm_cer").

LLM judge: multi-provider fallback chain
-----------------------------------------
  1. Gemini (primary)            - google-genai SDK, free tier
  2. OpenRouter free models       - open-weight models (Llama, Qwen, Gemma),
                                     OpenAI-compatible REST API, free tier
  3. Groq free tier               - open-weight Llama models, very fast,
                                     OpenAI-compatible REST API, free tier
Each provider also tries a short list of model names, so if one specific
model 404s / is retired / hits its own quota, the script moves to the next
model, then the next provider, before giving up on a batch. Every provider
in this chain is only used when its API key env var is set, so you can run
with just a Gemini key and no fallbacks configured -- it just won't have
anywhere to fall back to.

Install:
    pip install pandas jiwer indic-nlp-library google-genai requests

API keys (set whichever you have -- more keys = more fallback resilience):
    export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey
    export OPENROUTER_API_KEY=...    # https://openrouter.ai/keys (free tier)
    export GROQ_API_KEY=...          # https://console.groq.com/keys (free tier)

CSV schema (matches your files):
    sample_id,audio_filepath,duration,reference,hypothesis,wer,cer,status
The script reads `reference` and `hypothesis` columns by default (override
with --reference-col / --hypothesis-col). Any existing `wer`/`cer` columns
in your CSV are left untouched and simply carried through to the output as
`reported_wer` / `reported_cer` for comparison against the freshly computed
`original_wer` / `original_cer`.

Run - single language:
    python llm_wer_multilingual.py \
        --input hindi_results.csv --lang hi --output-dir outputs

Run - all 5 languages in one go (batch mode):
    python llm_wer_multilingual.py \
        --batch-dir ./csvs --output-dir outputs
    # expects filenames that contain a language hint, e.g.:
    #   hindi_results.csv / results_hi.csv / hi_test.csv       -> hi
    #   bengali_results.csv / results_bn.csv / bangla_test.csv -> bn
    #   tamil_results.csv / results_ta.csv                     -> ta
    #   marathi_results.csv / results_mr.csv                   -> mr
    #   telugu_results.csv / results_te.csv                    -> te

Persistent cross-language summary:
    Every run (single-file or batch) appends a block to
    `<output-dir>/wer_summary_all_languages.txt` -- original WER/CER and
    LLM WER/CER for that run, with a timestamp and input filename. The file
    is opened in APPEND mode only, so running Hindi today and Telugu next
    week just adds a new block; nothing already in the file is ever
    overwritten or deleted, regardless of which language or file is run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import jiwer
import requests

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
# 0. Language configuration
# ---------------------------------------------------------------------------

# code -> (display name, indic-nlp-library normalizer code)
LANGUAGES: Dict[str, Dict[str, str]] = {
    "hi": {"name": "Hindi", "indic_code": "hi"},
    "bn": {"name": "Bengali", "indic_code": "bn"},
    "ta": {"name": "Tamil", "indic_code": "ta"},
    "mr": {"name": "Marathi", "indic_code": "mr"},
    "te": {"name": "Telugu", "indic_code": "te"},
}

# filename hints used by --batch-dir to auto-detect language from a file name
LANG_FILENAME_HINTS: Dict[str, List[str]] = {
    "hi": ["hindi", "_hi", "-hi", "hi_", "hi-"],
    "bn": ["bengali", "bangla", "_bn", "-bn", "bn_", "bn-"],
    "ta": ["tamil", "_ta", "-ta", "ta_", "ta-"],
    "mr": ["marathi", "_mr", "-mr", "mr_", "mr-"],
    "te": ["telugu", "_te", "-te", "te_", "te-"],
}


def detect_language_from_filename(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for lang_code, hints in LANG_FILENAME_HINTS.items():
        for hint in hints:
            if hint in stem:
                return lang_code
    return None


# ---------------------------------------------------------------------------
# 1. WER / CER (identical formulas to sarvamai/llm_wer utilities.py)
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
# 2. Indic normalization (generalized across the 5 languages)
# ---------------------------------------------------------------------------

INDIC_PUNCTUATION = "।॥॰''\"‛‟′″´˝^°¤।॥॰¯'—–‑°¬´\u200b\u200c\u200d\u200e\u200f"

_NORMALIZER_CACHE: Dict[str, Any] = {}


def _get_normalizer(indic_code: str):
    if indic_code not in _NORMALIZER_CACHE:
        _NORMALIZER_CACHE[indic_code] = IndicNormalizerFactory().get_normalizer(indic_code)
    return _NORMALIZER_CACHE[indic_code]


def normalize_text(text: str, lang: str) -> str:
    if not isinstance(text, str) or not text:
        return "" if not isinstance(text, str) else text
    indic_code = LANGUAGES[lang]["indic_code"]
    normalizer = _get_normalizer(indic_code)
    text = re.sub(r'([,\-\.\(\)\[\]\{\}/\\])\B', r' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation + INDIC_PUNCTUATION))
    text = text.lower()
    text = normalizer.normalize(text)
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
# 4. LLM equivalence prompt (generalized for Hindi / Bengali / Tamil /
#    Marathi / Telugu)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """# Persona
You are an expert linguistic analyst specializing in Indian languages, fluent in
Hindi, Bengali, Tamil, Marathi and Telugu.

# Primary Goal
Your primary goal is to precisely compare two transcripts (in one of the above
five languages, specified per item as `language`) and determine if they are
essentially equivalent based on a set of equivalence rules. Adapt the phonetic
and structural principles to whichever language and script the item is in.

# Equivalence Rules
To determine equivalence, you MUST adhere strictly to the following rules.

## 1. Formatting and Symbol Equivalence
- **Ignore Punctuation:** Disregard all punctuation marks (e.g., ।, ?, ,, ॥, -, .).
- **Hyphenation:** Treat hyphenated words as identical to their multi-word or single-word counterparts.
  - Example (Hindi): "धीरे-धीरे" (dheere-dheere) = "धीरे धीरे" = "धीरेधीरे"
  - Example (Tamil): "அங்கு-இங்கு" (angu-ingu) = "அங்கு இங்கு" = "அங்குஇங்கு"
  - Example (Bengali): "পাশে-পাশে" (pashe-pashe) = "পাশে পাশে" = "পাশপাশে"
  - Example (Marathi): "हळू-हळू" (halu-halu) = "हळू हळू" = "हळूहळू"
  - Example (Telugu): "నెమ్మది-నెమ్మది" (nemmadi-nemmadi) = "నెమ్మది నెమ్మది"
- **Numbers:** Convert all numbers, whether in digit or word form, to a standard numeric value for comparison. Combine consecutive number words.
  - Example (Hindi): "उन्नीस सौ नब्बे" (unnees sau nabbe) = "1990"
  - Example (Tamil): "இரண்டு ஆயிரம் இருபத்தி மூன்று" (irandu aayiram irupathi moondru) = "2023"
  - Example (Telugu): "పంతొమ్మిది వందల తొంభై" (panthommidi vandala thombai) = "1990"
  - Example (Marathi): "एकोणीसशे नव्वद" (ekonisshe navvad) = "1990"
- **Symbols:** Words representing symbols are equivalent to the symbols themselves.
  - Example (Hindi): "रुपये" (rupaye) = '₹'
  - Example (Tamil): "சதவீதம்" (sadhaveedham) = '%'
  - Example (Bengali): "টাকা" (taka) = '₹'
  - Example (Marathi): "टक्के" (takke) = '%'
  - Example (Telugu): "శాతం" (shatam) = '%'
- **Numbers and Currencies:** Numbers with symbols representing the same value are equivalent.
  - Example (Hindi): "सौ रुपये" (sau rupaye) = "₹100"
  - Example (Hindi): "एक सौ चार रुपये पचास पैसे" (ek sau chaar rupaye pachaas paise) = "₹104.50"

## 2. Spoken vs. Written Form Equivalence
Account for common differences between spoken and written forms.
- **Acronyms and Initialisms:** Spoken-out letters of an acronym are equivalent to the consolidated written form.
  - Example (Hindi): "पी एन बी" = "पीएनबी"
  - Example (Tamil): "டி சி எஸ்" = "டிசிஎஸ்"
- **Phonetic Spelling of Brands/Names:** Phonetic spellings of proper nouns or brands are equivalent to their standard written form.
  - Example (Hindi): "रेडियो मिर्ची" = "Radio Mirchi"
  - Example (Tamil): "கோக கோலா" = "Coca-Cola"
  - Example (Bengali): "পেটিএম" (Paytm) = "Paytm"
  - Example (Marathi): "फेसबुक" (Facebook) = "Facebook"

## 3. Language and Script Equivalence
- **Cross-Script Equivalence:** Words that sound the same but are written in different scripts (e.g., Roman and a native Indian script) are equivalent.
  - Example (Hindi): "Amazon" = "अमेज़न"
  - Example (Tamil): "WhatsApp" = "வாட்ஸ்அப்"
  - Example (Telugu): "Facebook" = "ఫేస్బుక్"
  - Example (Marathi): "Youtube" = "यूट्यूब"
- **Common Spelling Variations:** Minor, common spelling variations that do not significantly alter pronunciation are equivalent. This includes variations in spacing for the same word.
  - Examples: "दोबारा" = "दुबारा", "கட்டிடம்" = "கட்டிடம", "वहाँ" = "वहां", "मज़ा" = "मजा"
  - Example (Bengali): "জন্য" (jonnyo) = "জন্যে" (jonne)
  - Example (Telugu): "వెళ్తున్నాను" (velthunnanu) = "వెళ్తున్నా" (velthunna)
  - Example (Marathi): "करतोय" (kartoy) = "करतो आहे" (karto aahe)

## 4. Phonetic Contractions or Reductions
- **Phonetic Contractions/Reductions:** Words that are phonetic reductions or contractions of another word are equivalent if their pronunciation is somewhat similar.
  - Example (Hindi): 'पर' (par) = 'पे' (pe)
  - Example (Hindi): 'ये' (ye) = 'यह' (yah)
  - Example (Bengali): 'তাহার' (tahar) = 'তার' (tar)
  - Example (Marathi): 'त्याला' (tyala) = 'त्याले' (tyale) (colloquial)

## 5. Empty-Side (Insertion/Deletion) Segments
Some pairs you receive will have an EMPTY reference or an EMPTY hypothesis --
that means the aligner found a word (or short phrase) inserted into, or
deleted from, the hypothesis transcript relative to the reference, with
nothing on the other side to compare it to directly. For these you will
also be given `full_reference_sentence` and `full_hypothesis_sentence` --
use them to judge the inserted/deleted span in context, not in isolation:
- **Stray repeated helper word:** if the inserted/deleted span is a short
  helper word (negation particles, fillers, discourse markers) that is
  simply duplicated adjacent to an identical word already in the sentence
  (e.g. reference "यह सही नहीं है", hypothesis "यह सही नहीं नहीं है" --
  segment is an inserted "नहीं"), this is a disfluency/stutter artifact,
  NOT a meaning change. Mark it **equivalent**.
- **Meaning-changing insertion/deletion:** if the inserted/deleted span
  changes what the sentence asserts -- most importantly inserting or
  removing a negation that is NOT simply duplicating an adjacent word
  (e.g. reference "मुझे यह पसंद है" / hypothesis "मुझे यह नहीं पसंद है" --
  "I like this" vs "I do NOT like this"), or inserts/removes any other
  content word that alters the sentence's meaning, this is a genuine
  error. Mark it **not equivalent**, regardless of how short the span is.
The test is always: does the full hypothesis sentence assert the same
thing as the full reference sentence? A duplicated word does not change
what is asserted; a negation (or other content change) does.

# Input Format:
You will be given a list of JSON objects, of the following format:

```json
{
  "index": int,
  "language": str,                       // one of: Hindi, Bengali, Tamil, Marathi, Telugu
  "reference": str,
  "hypothesis": str,
  "full_reference_sentence": str,         // OPTIONAL, only present for empty-side segments
  "full_hypothesis_sentence": str         // OPTIONAL, only present for empty-side segments
}
```

# Output Format
Your final output must be a single JSON object with the keys: `index`, `equivalence` and `reasoning`.

- **`index`**: An integer value. Ensure that the `index` is the same as the input index for the corresponding reference and hypothesis pair.
- **`equivalence`**: A boolean value. Use `true` for an exact match and `false` for a mismatch after carefully considering all the rules and analyzing the transcripts.
- **`reasoning`**: A string. Provide a brief, clear explanation for the equivalence value, highlighting the specific words or sequences along with its translation as well as transliteration to English.

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


def build_batch_prompt(pairs: List[Dict[str, str]], lang_name: str) -> str:
    payload = []
    for i, p in enumerate(pairs):
        item: Dict[str, Any] = {
            "index": i,
            "language": lang_name,
            "reference": p["reference"],
            "hypothesis": p["prediction"],
        }
        if p.get("context_reference") is not None:
            item["full_reference_sentence"] = p["context_reference"]
            item["full_hypothesis_sentence"] = p["context_prediction"]
        payload.append(item)
    return "**INPUT:**\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ---------------------------------------------------------------------------
# 4b. LLM providers -- Gemini primary, open-source free-tier fallbacks
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    pass


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, models: List[str]):
        self.client = genai.Client(api_key=api_key)
        self.models = models

    def call(self, model: str, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        text = resp.text
        if not text:
            raise ProviderError("Gemini returned empty response")
        return text


class OpenAICompatibleProvider:
    """Works for any OpenAI-chat-completions-compatible free-tier endpoint
    (OpenRouter, Groq, etc.) serving open-weight models."""

    def __init__(self, name: str, api_key: str, base_url: str, models: List[str],
                 extra_headers: Optional[Dict[str, str]] = None):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.models = models
        self.extra_headers = extra_headers or {}

    def call(self, model: str, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"} if False else None,
        }
        # Not every free-tier open model reliably honors response_format /
        # json_object mode, so we don't force it here -- instead we ask for
        # a JSON array explicitly in the prompt and parse leniently below.
        body.pop("response_format", None)
        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers,
                              data=json.dumps(body), timeout=90)
        if resp.status_code == 429:
            raise ProviderError(f"{self.name}/{model} rate-limited (429)")
        if resp.status_code >= 400:
            raise ProviderError(f"{self.name}/{model} HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ProviderError(f"{self.name}/{model} unexpected response shape: {data}")
        if not text:
            raise ProviderError(f"{self.name}/{model} returned empty content")
        return text


def build_provider_chain(requested: List[str]) -> List[Any]:
    """Build the ordered fallback chain from whichever API keys are set."""
    chain: List[Any] = []

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if "gemini" in requested and gemini_key:
        chain.append(GeminiProvider(
            api_key=gemini_key,
            models=["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"],
        ))

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if "openrouter" in requested and openrouter_key:
        chain.append(OpenAICompatibleProvider(
            name="openrouter",
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            models=[
                "meta-llama/llama-3.3-70b-instruct:free",
                "qwen/qwen-2.5-72b-instruct:free",
                "google/gemma-2-9b-it:free",
            ],
            extra_headers={"HTTP-Referer": "https://local-script", "X-Title": "llm-wer-multilingual"},
        ))

    groq_key = os.environ.get("GROQ_API_KEY")
    if "groq" in requested and groq_key:
        chain.append(OpenAICompatibleProvider(
            name="groq",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        ))

    if not chain:
        raise SystemExit(
            "No LLM provider API keys found. Set at least one of:\n"
            "  export GEMINI_API_KEY=...      (https://aistudio.google.com/apikey)\n"
            "  export OPENROUTER_API_KEY=...  (https://openrouter.ai/keys, free tier)\n"
            "  export GROQ_API_KEY=...        (https://console.groq.com/keys, free tier)\n"
            "Gemini is recommended as the primary judge; the others are open-source "
            "free-tier fallbacks used only if Gemini's quota is exhausted or it errors."
        )
    return chain


def query_llm_batch_with_fallback(
    providers: List[Any],
    pairs: List[Dict[str, str]],
    lang_name: str,
    max_retries_per_model: int = 3,
) -> Tuple[Dict[int, Dict[str, Any]], Optional[str]]:
    """Try each provider (and each model within it) in order. Returns
    (index -> verdict dict, provider/model string used) or ({}, None) if
    every provider/model failed. Rate-limit (429) errors get a longer,
    increasing backoff before retrying the same model, since those are
    transient and usually resolve within seconds -- burning through
    fallback models for a 429 wastes a model that would've worked fine."""
    if not pairs:
        return {}, None

    user_prompt = build_batch_prompt(pairs, lang_name)

    for provider in providers:
        for model in provider.models:
            for attempt in range(max_retries_per_model):
                try:
                    raw = provider.call(model, PROMPT_TEMPLATE, user_prompt)
                    parsed = _extract_json_array(raw)
                    verdicts = {item["index"]: item for item in parsed if "index" in item}
                    if verdicts:
                        return verdicts, f"{provider.name}/{model}"
                    raise ProviderError("parsed empty/invalid verdict list")
                except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall back
                    is_last_attempt = attempt == max_retries_per_model - 1
                    is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower() or "quota" in str(exc).lower()
                    if is_last_attempt:
                        wait = 0
                    elif is_rate_limit:
                        wait = 15 * (attempt + 1)  # 15s, 30s, 45s... give the quota window time to reset
                    else:
                        wait = 3
                    print(f"    [warn] {provider.name}/{model} failed ({exc}); "
                          f"{f'retrying in {wait}s' if wait else 'moving to next model/provider'}")
                    if wait:
                        time.sleep(wait)
    return {}, None


SUMMARY_TXT_NAME = "wer_summary_all_languages.txt"


def append_summary_txt(output_dir: str, lang_name: str, lang_code: str,
                        input_file: str, df: pd.DataFrame, unresolved_count: int = 0) -> Path:
    """Append this run's results as a new block in a shared, cross-language
    summary .txt file. Always opened in append mode -- existing blocks from
    earlier runs (this language or any other) are never touched or removed,
    so results accumulate across every language you run over time."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_TXT_NAME
    is_new = not summary_path.exists()

    n = len(df)
    orig_wer = df["original_wer"].mean()
    orig_cer = df["original_cer"].mean()
    llm_wer_v = df["llm_wer"].mean()
    llm_cer_v = df["llm_cer"].mean()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with summary_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("LLM-WER / LLM-CER SUMMARY -- all languages, all runs\n")
            f.write("Each run appends a new block below; nothing is ever overwritten or deleted.\n")
            f.write("=" * 78 + "\n\n")
        f.write(f"[{timestamp}]  Language: {lang_name} ({lang_code})\n")
        f.write(f"  Input file     : {input_file}\n")
        f.write(f"  Rows evaluated : {n}\n")
        f.write(f"  Original WER   : {orig_wer:.4f}\n")
        f.write(f"  Original CER   : {orig_cer:.4f}\n")
        f.write(f"  LLM WER        : {llm_wer_v:.4f}\n")
        f.write(f"  LLM CER        : {llm_cer_v:.4f}\n")
        if unresolved_count:
            f.write(f"  NOTE           : {unresolved_count} segment(s) unresolved (all LLM "
                     f"providers failed) -- LLM WER/CER above are conservative for this run; "
                     f"re-run the same command to retry just those segments.\n")
        f.write("-" * 78 + "\n\n")

    print(f"[{lang_name}] Appended summary to: {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# 5. Full pipeline (single language)
# ---------------------------------------------------------------------------

def run(
    input_path: str,
    lang: str,
    reference_col: str = "reference",
    hypothesis_col: str = "hypothesis",
    output_dir: str = "outputs",
    batch_size: int = 10,
    provider_order: Optional[List[str]] = None,
    request_delay: float = 2.0,
):
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported --lang '{lang}'. Choose from: {list(LANGUAGES)}")
    lang_name = LANGUAGES[lang]["name"]

    path_obj = Path(input_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{input_path} does not exist")
    df = pd.read_csv(path_obj)
    for col in (reference_col, hypothesis_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Columns present: {list(df.columns)}")

    print(f"[{lang_name}] Loaded {len(df)} rows from {input_path}")

    # keep any pre-existing wer/cer/status columns as "reported_*" for comparison
    for existing_col, new_col in [("wer", "reported_wer"), ("cer", "reported_cer")]:
        if existing_col in df.columns:
            df[new_col] = df[existing_col]

    # --- normalize + original WER/CER ---
    df["norm_reference"] = df[reference_col].astype(str).map(lambda t: normalize_text(t, lang))
    df["norm_hypothesis"] = df[hypothesis_col].astype(str).map(lambda t: normalize_text(t, lang))
    df["original_wer"] = [wer(r, p) for r, p in zip(df["norm_reference"], df["norm_hypothesis"])]
    df["original_cer"] = [cer(r, p) for r, p in zip(df["norm_reference"], df["norm_hypothesis"])]

    # --- diff every row into segments ---
    row_segment_map: Dict[int, List[Dict]] = {}
    PairKey = Tuple[str, str, Optional[str], Optional[str]]
    unique_segments: Dict[PairKey, List[Dict]] = {}

    def make_key(seg_ref: str, seg_pred: str, row_ref: str, row_pred: str) -> PairKey:
        is_empty_side = not seg_ref.strip() or not seg_pred.strip()
        if is_empty_side:
            return (seg_ref, seg_pred, row_ref, row_pred)
        return (seg_ref, seg_pred, None, None)

    for idx, row in df.iterrows():
        segments = get_segments(row["norm_reference"], row["norm_hypothesis"], key=idx)
        row_segment_map[idx] = segments
        for seg in segments:
            if seg["tag"] == "equal":
                continue
            key = make_key(seg["reference"], seg["prediction"], row["norm_reference"], row["norm_hypothesis"])
            unique_segments.setdefault(key, []).append(
                {"row_idx": idx, "segment_idx": seg["segment_idx"]}
            )

    print(f"[{lang_name}] Found {len(unique_segments)} unique mismatched segment pairs to check with the LLM.")

    # --- cache setup (per-language, per-input-file) ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"{path_obj.stem}_{lang}_llm_cache.jsonl"
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
    print(f"[{lang_name}] {len(unique_segments) - len(to_query)} pairs found in cache. "
          f"Querying LLM for {len(to_query)} new pairs.")

    providers = build_provider_chain(provider_order or ["gemini", "openrouter", "groq"])
    print(f"[{lang_name}] Provider fallback chain: "
          f"{' -> '.join(p.name + ':' + '/'.join(p.models) for p in providers)}")

    log_records = []
    unresolved_count = 0

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
            print(f"[{lang_name}]   Querying LLM for segments {i}-{i + len(batch_pairs)} / {len(to_query)}")
            verdicts, used = query_llm_batch_with_fallback(providers, batch_dicts, lang_name)
            if used:
                print(f"[{lang_name}]   -> answered by {used}")
            else:
                print(f"[{lang_name}]   -> ALL providers failed for this batch; "
                      f"scoring these {len(batch_pairs)} pairs as 'not equivalent' for THIS run only "
                      f"(not cached -- they'll be retried automatically next run)")
            for j, (ref, pred, ctx_ref, ctx_pred) in enumerate(batch_pairs):
                key = (ref, pred, ctx_ref, ctx_pred)
                if j in verdicts:
                    verdict = verdicts[j]
                    record = {
                        "language": lang_name,
                        "reference": ref,
                        "prediction": pred,
                        "context_reference": ctx_ref,
                        "context_prediction": ctx_pred,
                        "equivalent": bool(verdict.get("equivalence", verdict.get("equivalent", False))),
                        "reasoning": verdict.get("reasoning", ""),
                        "llm_provider": used,
                    }
                    # Real verdict: persist to disk cache so we never re-pay for it.
                    cache[key] = record
                    cache_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    # Every provider failed on this pair. Do NOT write to the
                    # cache file -- keep it session-only so it's picked up as
                    # "not yet cached" and retried on the next run, instead of
                    # being permanently frozen in as a false "not equivalent".
                    unresolved_count += 1
                    cache[key] = {
                        "language": lang_name,
                        "reference": ref,
                        "prediction": pred,
                        "context_reference": ctx_ref,
                        "context_prediction": ctx_pred,
                        "equivalent": False,
                        "reasoning": "UNRESOLVED: all LLM providers failed for this pair "
                                      "(rate limit / outage) -- scored as not-equivalent for "
                                      "this run only, will retry automatically next run.",
                        "llm_provider": "none",
                        "_unresolved": True,
                    }

            if request_delay and i + batch_size < len(to_query):
                time.sleep(request_delay)

    if unresolved_count:
        print(f"[{lang_name}] {unresolved_count} segment pair(s) could not be judged by any "
              f"provider this run. They were scored conservatively (not-equivalent) for this "
              f"run's numbers but were NOT cached, so simply re-running the same command later "
              f"will retry just those pairs.")

    # --- build final verdict map (row_idx, segment_idx) -> bool ---
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
    corrected_hypotheses, corrected_references = [], []
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
        corrected_hypotheses.append(" ".join(pred_parts))
        corrected_references.append(" ".join(ref_parts))

    df["corrected_hypothesis"] = corrected_hypotheses
    df["corrected_reference"] = corrected_references
    df["llm_wer"] = [wer(r, p) for r, p in zip(df["corrected_reference"], df["corrected_hypothesis"])]
    df["llm_cer"] = [cer(r, p) for r, p in zip(df["corrected_reference"], df["corrected_hypothesis"])]
    df["language"] = lang_name

    print(f"\n===== Results: {lang_name} =====")
    print(f"Original WER (mean): {df['original_wer'].mean():.4f}")
    print(f"Original CER (mean): {df['original_cer'].mean():.4f}")
    print(f"LLM WER      (mean): {df['llm_wer'].mean():.4f}")
    print(f"LLM CER      (mean): {df['llm_cer'].mean():.4f}")

    out_csv = out_dir / f"{path_obj.stem}_{lang}_llm_wer.csv"
    df.to_csv(out_csv, index=False)
    print(f"Full results saved to: {out_csv}")

    if log_records:
        logs_csv = out_dir / f"{path_obj.stem}_{lang}_llm_logs.csv"
        pd.DataFrame(log_records).drop_duplicates(subset=["reference", "prediction"]).to_csv(logs_csv, index=False)
        print(f"LLM reasoning logs saved to: {logs_csv}")

    append_summary_txt(output_dir, lang_name, lang, str(path_obj), df, unresolved_count)

    return df


# ---------------------------------------------------------------------------
# 6. Batch driver across all 5 languages
# ---------------------------------------------------------------------------

def run_batch(
    batch_dir: str,
    output_dir: str = "outputs",
    reference_col: str = "reference",
    hypothesis_col: str = "hypothesis",
    batch_size: int = 10,
    provider_order: Optional[List[str]] = None,
    request_delay: float = 2.0,
):
    directory = Path(batch_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"{batch_dir} is not a directory")

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {batch_dir}")

    all_results = []
    skipped = []
    for csv_path in csv_files:
        lang = detect_language_from_filename(csv_path)
        if lang is None:
            skipped.append(csv_path.name)
            continue
        print(f"\n########## {csv_path.name} -> detected language: {LANGUAGES[lang]['name']} ({lang}) ##########")
        df = run(
            input_path=str(csv_path),
            lang=lang,
            reference_col=reference_col,
            hypothesis_col=hypothesis_col,
            output_dir=output_dir,
            batch_size=batch_size,
            provider_order=provider_order,
            request_delay=request_delay,
        )
        all_results.append(df)

    if skipped:
        print(f"\n[warn] Could not detect language for these files, skipped: {skipped}. "
              f"Rename them to include a language hint (hindi/hi, bengali/bn, tamil/ta, "
              f"marathi/mr, telugu/te) or run them individually with --input/--lang.")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_dir = Path(output_dir)
        combined_csv = out_dir / "combined_llm_wer_all_languages.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nCombined results for all languages saved to: {combined_csv}")

        print("\n===== Summary across all languages =====")
        summary = combined.groupby("language")[["original_wer", "original_cer", "llm_wer", "llm_cer"]].mean()
        print(summary.round(4).to_string())

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute LLM-WER / LLM-CER for Hindi/Bengali/Tamil/Marathi/Telugu ASR output."
    )
    parser.add_argument("--input", help="Path to a single input CSV file (use with --lang).")
    parser.add_argument("--lang", choices=list(LANGUAGES), help="Language code for --input: hi, bn, ta, mr, te.")
    parser.add_argument("--batch-dir", help="Directory containing multiple per-language CSV files; "
                                             "language is auto-detected from each filename.")
    parser.add_argument("--reference-col", default="reference", help="Column name with ground-truth text.")
    parser.add_argument("--hypothesis-col", default="hypothesis", help="Column name with ASR hypothesis text.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write results/cache to.")
    parser.add_argument("--batch-size", type=int, default=10,
                         help="Number of segment pairs per LLM call (keep small to respect free-tier RPM limits).")
    parser.add_argument("--request-delay", type=float, default=2.0,
                         help="Seconds to sleep between LLM batch calls, to stay under free-tier "
                              "RPM limits. Increase this (e.g. 5-10) if you keep seeing 429s. Default: 2.0")
    parser.add_argument("--providers", default="gemini,openrouter,groq",
                         help="Comma-separated fallback order. Providers without an API key set are "
                              "skipped automatically. Default: gemini,openrouter,groq")
    args = parser.parse_args()

    provider_order = [p.strip() for p in args.providers.split(",") if p.strip()]

    if args.batch_dir:
        run_batch(
            batch_dir=args.batch_dir,
            output_dir=args.output_dir,
            reference_col=args.reference_col,
            hypothesis_col=args.hypothesis_col,
            batch_size=args.batch_size,
            provider_order=provider_order,
            request_delay=args.request_delay,
        )
    elif args.input and args.lang:
        run(
            input_path=args.input,
            lang=args.lang,
            reference_col=args.reference_col,
            hypothesis_col=args.hypothesis_col,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            provider_order=provider_order,
            request_delay=args.request_delay,
        )
    else:
        parser.error("Either provide --batch-dir (process a folder of per-language CSVs), "
                      "or both --input and --lang (process a single CSV).")