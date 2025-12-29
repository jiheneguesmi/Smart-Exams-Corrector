# -*- coding: utf-8 -*-
"""Post-traitement du texte OCR pour examens

Provides safe normalization, lightweight spell correction and an
optional LLM-backed line-by-line cleaner. Designed to be imported
by the main pipeline.
"""
from typing import Optional
import re
import os
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None

try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker(language="fr")
except Exception:
    _SPELL = None

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


def normalize_characters(text: str) -> str:
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "—": "-",
        "–": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "  ": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def clean_structure_safe(text: str) -> str:
    # Keep paragraph/line breaks but normalize repeated spaces
    text = re.sub(r"[^\S\n]{2,}", " ", text)   # multiple spaces except newlines
    text = re.sub(r"\n{4,}", "\n\n\n", text)   # cap excessive newlines
    return text.strip()


def is_suspicious(word: str) -> bool:
    if len(word) <= 2:
        return False
    if word.isupper():
        return False
    if zipf_frequency is None:
        # conservative fallback: treat short uncommon-like tokens as safe
        return False
    try:
        return zipf_frequency(word.lower(), "fr") < 1.5
    except Exception:
        return False


def correct_word(word: str) -> str:
    if _SPELL is None:
        return word
    try:
        corr = _SPELL.correction(word)
        return corr if corr else word
    except Exception:
        return word


def correct_line_locally(line: str) -> str:
    corrected_words = []
    for word in line.split():
        if is_suspicious(word):
            corrected_words.append(correct_word(word))
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


def llm_secure_ocr_correction(line: str, client: InferenceClient, model: str) -> str:
    prompt = f"""You are correcting OCR errors in a French exam copy.

TASK:
- Correct spelling and OCR mistakes only.

STRICT RULES:
- Do NOT rephrase.
- Do NOT add content.
- Do NOT remove content.
- Do NOT merge or split lines.
- Preserve punctuation and numbering.
- Preserve the line exactly as a single line.

TEXT:
{line}

Return ONLY the corrected text.
"""
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM correction failed: {e}")
        return line


def llm_safe_line_correction(text: str, hf_api_token: Optional[str] = None, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> str:
    if not hf_api_token or InferenceClient is None:
        return text
    try:
        client = InferenceClient(token=hf_api_token)
    except Exception as e:
        logger.warning(f"Could not initialize HF InferenceClient: {e}")
        return text

    lines = text.split("\n")
    corrected_lines = []
    for line in lines:
        if not line.strip() or len(line.strip()) < 4:
            corrected_lines.append(line)
            continue
        corrected = llm_secure_ocr_correction(line, client=client, model=model)
        corrected_lines.append(corrected)
    return "\n".join(corrected_lines)


def full_ocr_postprocess_SAFE(ocr_text: str, use_llm: bool = False, hf_api_token: Optional[str] = None, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> str:
    step1 = normalize_characters(ocr_text)
    step2 = clean_structure_safe(step1)

    locally_corrected = [correct_line_locally(line) for line in step2.split("\n")]
    step3 = "\n".join(locally_corrected)

    if use_llm and hf_api_token:
        final_text = llm_safe_line_correction(step3, hf_api_token=hf_api_token, model=model)
    else:
        final_text = step3

    return final_text


if __name__ == "__main__":
    # tiny local demo
    sample = "Ceci est un exämple de téxt avec des erreurs OCR fiﬂ — et des  spaces."
    print(full_ocr_postprocess_SAFE(sample))
# -*- coding: utf-8 -*-
