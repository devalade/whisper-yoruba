# Yorùbá WER evaluation — normalization protocol

**Problem**: raw WER on Yorùbá Whisper output is dominated by punctuation, diacritic conventions, and word-boundary differences that have nothing to do with model quality. The first fine-tuned checkpoint produced a raw 89% WER on FLEURS yo_ng N=50, which would normally suggest the fine-tune destroyed the model. Inspection of REF/HYP samples showed the model was actually transcribing correctly — the metric was lying.

This doc fixes the metric.

## What inflates raw Yorùbá WER

Looking at one real REF/HYP pair from the test notebook:

```
REF: fotosintesisi ni ọ̀nà tí àwọn ewé ma n gbà gi ṣe ounje wọn láti orun
HYP: Fọtosintesis ni ọna ti awọn ewe ma n gba giin ṣe ounjẹ wọn lati ọrun
```

Every word is semantically correct. The "errors" are:

1. **Capitalization** (`Fọtosintesis`) — Whisper restores casing; FLEURS refs are lowercase.
2. **Punctuation** — Whisper inserts commas and periods; refs have none.
3. **Diacritics** — refs use full tone marks (`ọ̀nà`); hyps often drop them (`ọna`). And vice versa: some refs *also* drop them while hyps add them. Even *within* FLEURS yo_ng, ref diacritization is inconsistent across samples.
4. **Word boundaries** — `torípé` ↔ `to rii pe` is the same content but counts as three substitutions plus an insertion.
5. **Code-switching** — Whisper sometimes falls back to English (`kọ́pa` ↔ `copper`). Semantically right, lexically a miss.

A naive `jiwer.wer(ref, hyp)` counts every one of these as a word error.

## The Whisper-paper convention

The original Whisper paper applies `whisper.normalizers.BasicTextNormalizer` to every non-English language before computing WER. It lowercases, strips punctuation, normalises Unicode, and collapses whitespace. Without this step, no Yorùbá ASR number on the internet is comparable to any other Yorùbá ASR number.

We do the same, and additionally report a "permissive" WER that strips diacritics.

## Three numbers to report

For each evaluation, compute and report all three:

| Name | What it strips | What it tells you |
|---|---|---|
| **Raw WER** | nothing | Surface mismatch including all formatting noise. Don't compare it to anything published. Useful only to demonstrate the normalization gap. |
| **Normalized WER** | case, punctuation, extra whitespace; preserves diacritics | **The number to compare against published Yorùbá ASR results.** Matches Whisper-paper convention. |
| **Permissive WER** | normalized + all Unicode combining marks (`ọ→o`, `ẹ→e`, `ṣ→s`, tone marks) | Upper bound on content correctness. Tells you whether the model got the *consonants and vowels* right, independent of how it diacritized. |

The gap between normalized and permissive WER **is the diacritic error rate** — a useful number on its own for the thesis, because Yorùbá tone marks carry lexical meaning.

## Implementation

Both transforms are in `Whisper_test.ipynb` and short enough to copy elsewhere:

```python
import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")

def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Keeps diacritics."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def strip_diacritics(s: str) -> str:
    """NFD-decompose and drop combining marks. ọ→o, ẹ→e, ṣ→s, tone marks gone."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)
```

Apply `normalize` to both refs and hyps for normalized WER. Apply `normalize` then `strip_diacritics` for permissive WER.

## What "good" looks like

Rough targets for whisper-large-v3-class models on FLEURS yo_ng:

- Normalized WER ≤ 50% — competitive with published Yorùbá ASR.
- Normalized WER ≤ 35% — state-of-the-art territory.
- Permissive WER ≤ 25% — the model knows the words; the gap to normalized WER is your diacritic budget.

If raw WER is 80%+ but normalized WER is 50%, the model is fine — you need to either ship the normalizer with the artifact or adjust the training data's diacritization conventions. If both raw and permissive WER are above 70%, the fine-tune has genuinely regressed.

## Reporting in the thesis / experiments table

Always log all three numbers plus `N`. Format suggestion for `docs/experiments/*.md`:

```
FLEURS yo_ng — N=200
  raw WER         = 87.4%
  normalized WER  = 48.6%   ← headline number
  permissive WER  = 31.2%
  diacritic gap   = 17.4 pp
```

Don't quote raw WER alone. It's actively misleading.

## Open question: which normalizer is "fair"?

Other published Yorùbá ASR papers don't always disclose their normalizer. Whisper-paper-style basic normalization is the most defensible baseline. If a paper reports a number with an unstated normalizer, treat it as comparable-with-caveats and note it in the table.
