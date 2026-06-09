# Experiment: small-run training loop + conversational eval set

**Date:** 2026-06-06
**Goal:** Build a fast iteration loop (small-run training with raw audio holdout + a conversational eval set) so future fine-tune trials can be judged on the failure mode that actually matters for deployment, not just FLEURS WER.
**Outcome:** ⚠️ Tooling shipped, not a model run. No new checkpoint produced; this commit unlocks shorter cycles for the next training trial.

## What was added (commit `d59b5ae`)

- `scripts/finetune_whisper.py` gained `--total-samples`, `--holdout-n`, `--seed` — runs on ~800 samples in 15–25 min and pops `N` raw WAVs (with manifest) before feature extraction so they're excluded from train and eval.
- `scripts/test_holdout.py` — runs M1 against the held-out clips locally for per-clip and aggregate WER.
- `data/eval/convo_phrases.jsonl` — 25 short conversational phrases across greetings, identity, question words, numbers, requests.
- `scripts/record_convo.py` + `scripts/eval_convo.py` — record via mic helper, score with the same diacritic-stripped normalization as FLEURS so numbers are directly comparable.
- `docs/conversational-eval.md`, `docs/small-run-with-holdout.md` — full recipes.
- `docs/methodology.md` — section pointing at the new eval.

## Why

The 2026-06-05 fine-tune mis-segmented `Kíni orúkọ rẹ?` as `Kini Oru gọ ọrẹ?` — a failure FLEURS yo_ng (studio-read news prose) does not surface. Aggregate WER on FLEURS isn't a useful signal for "did this trial fix the conversational failures?" — we need:

1. A held-out **audio** eval (not just text WER) so qualitative judgement is in the loop.
2. A conversational eval set scored with the same normalization as FLEURS so numbers are comparable.
3. A small-run mode so a hypothesis costs 15–25 min and ~$0.30, not 2.5 h and ~$2.

## First baseline numbers (to fill once recorded)

After recording `data/audio/eval_convo/*.wav`:

| Model | Convo overall WER | Convo identity WER | Convo greeting WER |
|---|---|---|---|
| `openai/whisper-large-v3` (zero-shot baseline) | _TBD — `make wer-convo-hf` with base model_ | _TBD_ | _TBD_ |
| `devalade/whisper-large-v3-yoruba` (current ft) | _TBD — `make wer-convo-hf`_ | _TBD_ | _TBD_ |
| `mlx-whisper-large-v3` (mlx baseline) | _TBD — `make wer-convo-mlx`_ | _TBD_ | _TBD_ |

This gives the next training trial a concrete target to beat on the identity bucket without regressing the others.

## Decision

⚠️ No model shipped — this is infrastructure. Next training trial uses `make finetune-small` against this eval to test hypotheses before the next full run.

Open follow-ups:
- Native-speaker review of `data/eval/convo_phrases.jsonl` (diacritics + word choice) before treating the numbers as authoritative.
- Record the 25 phrases and fill the baseline table above.
- First hypothesis to test with the small-run loop: _TBD — decide next._
