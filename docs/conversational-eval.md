# Conversational eval set for M1

## Why this exists

The M1 fine-tune was trained on `Hidi-agili/yoruba_tts_dataset`: 9,499 clips, **single speaker**, TTS-style read sentences, durations 0.3–23.3 s with most clips in the multi-second range. The headline FLEURS yo_ng eval is also studio-read news prose.

Neither distribution looks anything like the actual deployment input — short conversational microphone queries spoken by the end user. The first end-to-end test of the fine-tuned model exposed exactly this gap: `Kíni orúkọ rẹ?` ("What is your name?") was transcribed as `Kini Oru gọ ọrẹ?`, breaking the entire downstream chain even though every other module behaved correctly.

This eval set measures the conversational slice directly, so future fine-tune iterations can be judged on whether they fix this class of failure rather than just moving FLEURS WER.

## What's in it

25 phrases across five categories — greetings, identity, question words, numbers, requests — chosen to mirror the kinds of utterances a real user of the voice pipeline would produce:

- Source of truth: `data/eval/convo_phrases.jsonl` (one phrase per line)
- Audio: `data/audio/eval_convo/<id>.wav` (user-recorded via the project mic helper)
- References are **diacritized**, matching the training-target convention

The phrase list was authored by Claude as a starting point — diacritics and word choice should be reviewed by a native speaker before recording. The file is intentionally small and easy to extend.

## How to run

```bash
# 1. Record. Push-to-talk loop, skips phrases already on disk.
make record-convo

# Re-record one:
python -m scripts.record_convo --rerecord ident_01

# 2. Evaluate. Same diacritic-stripped normalization as scripts/eval_wer.py,
#    so numbers are comparable to FLEURS WER.
make wer-convo-hf      # the fine-tuned HF model
make wer-convo-mlx     # the mlx baseline
make wer-convo         # both, sequentially
```

Output: per-phrase line (✓/blank, WER, ref vs hyp), per-category aggregate, overall aggregate. Full results written to `logs/wer_convo_<backend>.jsonl`.

## How to interpret

- **Overall WER** — headline number for the conversational slice. Compare against the FLEURS WER to see how much harder this distribution is.
- **Per-category WER** — a regression in `identity` (the bucket containing the original failure) without movement in `greeting` means the fix didn't generalize.
- A category WER of 100% on a 5-phrase bucket is not statistically meaningful — it's a smoke signal pointing at where to grow the training set.

## When to extend the set

- Add a phrase whenever a fine-tune surfaces a new failure mode that isn't covered.
- Keep categories balanced (~5 phrases each) so per-category aggregates stay comparable.
- Re-record everything if you switch microphones — acoustic environment is part of the eval.

## Limitations

- Single speaker (the thesis author) — same acoustic-diversity limitation the training corpus has. The eval measures *deployment-condition* WER for this specific user, not generalization.
- 25 phrases is too small for tight CIs. Treat per-category numbers as directional, not statistically significant.
- Diacritic-stripped metric only — diacritic-restoration quality is M2's job and tracked separately.
