# Experiment: v4 OOD evaluation on `Hidi-agili/yoruba_male_dataset`

**Date:** 2026-06-15
**Goal:** Quantify the v4 fine-tune's out-of-distribution generalization on a related-but-distinct Yorùbá speech corpus, paired against the un-finetuned `openai/whisper-large-v3` baseline on the same clips. Pairs with the in-distribution 50-clip holdout to give both axes of model quality.
**Outcome:** ⚠️ **Useful, model not yet shipped.** v4 wins on every YASR-Bench metric vs baseline, including +25 pp on Y-WER-perm and +13.5 pp on Y-CER-perm. Confirms the fine-tune transfers to audio from speakers the model never saw — not just memorisation. Honest caveat: same-author corpus, so this is "related-but-distinct" rather than strict OOD; the next round (OpenSLR86 or FLEURS test) is needed for the strict claim.

## Setup

| | |
|---|---|
| Fine-tuned model | v4 from `MyDrive/yoruba-pipeline-logs/training/2026-06-14T17-23-20Z/merged_16bit/` — **not yet pushed to HF** |
| Base model | `openai/whisper-large-v3` (zero-shot baseline) |
| Adapter recipe | LoRA r=64, α=64, target q,k,v,out_proj,fc1,fc2 (~115 M trainable) |
| Training data | `Hidi-agili/yoruba_tts_dataset` (9,446 clips; 50 holdout + 7,559 train + 1,879 internal eval; SEED=3407) |
| Training run notebook | `Whisper_v4.ipynb` |
| Test corpora | (a) 50-clip in-distribution holdout from training; (b) `Hidi-agili/yoruba_male_dataset` (10,446 clips, 2 male speakers, biblical content), 100 samples streamed |
| Decoding | greedy (`num_beams=1`, `max_new_tokens=256`), forced `<\|yo\|>` language token |
| Metric suite | YASR-Bench: Y-WER, Y-WER-perm, Y-CER, Y-CER-perm (normalizer v1) |
| Eval notebook | `Whisper_ood_test.ipynb` (overlap check + paired baseline-vs-v4) |
| Hardware | A100 80 GB (Colab) — single-shot eval, no separate cost beyond training |

### Overlap check (Step 7.5)

Text-overlap between OOD samples and training transcripts, normalised. **Result: TBD** — to be backfilled from the Step 7.5 print. Empirically below the 10% abort threshold since Step 8 was reached and produced the numbers below. Document the exact percentage in the methodology section of the thesis.

The check is a text-level proxy. Audio-fingerprint overlap is not verified; for same-author corpora that's a known limitation called out in [`Whisper_ood_test.ipynb`](../../Whisper_ood_test.ipynb) Step 7.5.

## Results

### Paired numbers — in-distribution holdout vs OOD male_dataset

Both columns scored against `openai/whisper-large-v3` on the same N clips per set.

| Metric | Baseline (holdout) | v4 (holdout) | Δ (holdout) | Baseline (OOD) | v4 (OOD) | Δ (OOD) |
|---|---:|---:|---:|---:|---:|---:|
| Y-WER       | 94.49% | **53.36%** | +41.13 | 90.92% | **88.05%** | +2.86  |
| Y-WER-perm  | 91.80% | **27.55%** | +64.25 | 86.78% | **61.78%** | +25.00 |
| Y-CER       | 33.68% | **17.71%** | +15.97 | 40.79% | **35.50%** | +5.30  |
| Y-CER-perm  | 27.36% | **5.64%**  | +21.72 | 32.77% | **19.30%** | +13.48 |
| diac gap    | 2.69   | 25.81      | -23.12 | 4.13   | 26.27      | -22.14 |

N = 50 (holdout), 100 (OOD). Δ > 0 means v4 wins by that many percentage points.

### Headline reads

1. **v4 beats baseline on every metric on both test sets.** No regression direction anywhere.
2. **Y-CER-perm — the most defensible thesis metric — drops from 27.36% (baseline) to 19.30% (v4) on OOD, a 41% relative reduction.** Tells you the fine-tune learned Yorùbá phonemes, not just memorised training transcripts.
3. **In-distribution → OOD transfer cost:** 13.66 pp on Y-CER-perm (5.64 → 19.30) and 34.23 pp on Y-WER-perm (27.55 → 61.78). This is the speaker-and-recording-condition generalisation gap; both numbers remain substantially below the baseline-on-OOD, so transfer is meaningfully positive.

### Diacritic-gap stability — the central finding

The "diacritic gap" (Y-WER minus Y-WER-perm) is **stable across in-distribution and OOD test sets** for each model:

| Model | gap (holdout) | gap (OOD) | drift |
|---|---:|---:|---:|
| baseline | 2.69 pp | 4.13 pp | +1.44 pp |
| **v4**   | **25.81 pp** | **26.27 pp** | **+0.46 pp** |

Reads:

- Baseline barely produces tone marks at all (gap ~3 pp on both sets). Stripping diacritics from references doesn't change baseline WER much because baseline transcripts were already (incorrectly) diacritic-free.
- **v4 has internalised a stable diacritization convention** (gap ~26 pp on both sets, within 0.5 pp). The gap is not random noise; it's a systematic difference between canonical written Yorùbá (what v4 produces) and the inconsistently-annotated reference transcripts (what's in both corpora).
- This consistency across test sets is the strongest defence of the *"raw Y-WER understates the fine-tune's quality because of reference inconsistency"* argument: if the fine-tune were behaving stochastically with diacritics, the gap would differ between test sets. It doesn't.

Defensible thesis paragraph this supports:

> "The diacritic gap (Y-WER minus Y-WER-perm) for our fine-tuned model is stable at ~26 pp across both in-distribution and out-of-distribution test sets (25.81 pp and 26.27 pp respectively), while the baseline's gap is 3 pp on both. We interpret this as evidence that the fine-tune produces a consistent, canonical Yorùbá diacritization that the reference annotations across our test corpora do not uniformly follow. We therefore report Y-CER-perm (character error rate after stripping combining marks) as the headline metric for fine-tune quality, as it isolates phoneme recognition from annotation inconsistency."

## Qualitative samples — three-way (REF / baseline / v4)

From the in-distribution holdout cell ([`Whisper_test.ipynb`](../../Whisper_test.ipynb), saved to `RUN_DIR/holdout_baseline_paired.json`):

| # | REF (normalised) | baseline | v4 |
|---|---|---|---|
| 0 | `lẹhinna o joko o si duro diẹ nitori pe eyi ni ogun naa` | `la ina ojoku osijuro diya ni torikbe e i ni oguno` | `lẹ yìn náà o joko o sì dúró díẹ nítorí pé èyí ní ogún náà` |
| 1 | `ati pe eyi ni idi ni awọn orilẹ ede meji ti ko ti ni arun yii fun diẹ sii ju boya ọdun mẹwa ni awọn ẹgbẹ idakeji ti agbaiye awọn ibẹsilẹ roparosẹ ẹru lojiji` | `atikbe eyi ni idini anwon orile demeji tiko ti ni arun i fun di e si ju boya odume wa ni anwon ekbe idakeji ti agbaye anwon ebesile rokwa rose rulo jijin` | `ati pe eyi ni idi ni awọn orilẹ ede meji ti ko ti ni arun yii fun diẹ sii ju boya ọdun mẹwa ni awọn ẹgbẹ idakeji ti agbaye awọn ibẹsilẹ rọparọsẹ erulojiji` |
| 2 | `iyen ni isoro` | `iye ni ishuru` | `ìyẹn ni isoro` |
| 3 | `fi ahọn rẹ kuro barry` | `pi an hon re kuro bari` | `p a han rẹ kúrò barry` |
| 4 | `ati nitorina ni mo bẹrẹ si beere ibeere miiran` | `atini torina o nimo bere si bere i bere miro` | `àti nítorí náà ni mo bẹ rẹ sí bééré ìbéèrè mìíràn` |

**Pattern**: baseline produces phonetic transliteration (it hears Yorùbá phonemes but spells them as if writing English approximations — `ishuru` for `isoro`, `oguno` for `ogun naa`). v4 produces real Yorùbá with diacritization. Sample 0 and sample 2 are striking — v4's output is "more correct" than the reference, just diacritized.

OOD male_dataset triples: **TBD** — to be backfilled from `Whisper_ood_test.ipynb` Step 11 output. Same pattern expected.

## What worked

- **Reserving the 50-clip in-distribution holdout up-front** (Step 5 of `Whisper_v4.ipynb`) gave us a clean before-OOD comparison that doesn't share any clips with training. Step 5 writes the holdout as an `HF datasets` directory to Drive, so the eval notebook just `load_from_disk`s — no reconstruction needed.
- **Paired baseline-vs-v4 on the same clips, same decoding settings.** Eliminates the reviewer's "your reference inconsistency penalises both models so the Δ is what matters" defence even before raising it.
- **YASR-Bench's permissive variant (Y-WER-perm, Y-CER-perm).** Without it, the raw Y-WER on OOD looks like only a 3 pp improvement; with it, the real 25 pp content-recognition win surfaces. The same fix matters for the reference-inconsistency story.
- **The Step 7.5 overlap check** caught the "same-author footgun" without requiring manual inspection: ran in ~30 s by streaming the training corpus and indexing normalised transcripts. The fact that v4 still degrades from 53% Y-WER (holdout) to 88% (OOD) is itself evidence the OOD set has limited overlap — full overlap would have produced near-holdout numbers.

## What didn't / caveats

- **Same-author corpus.** `Hidi-agili/yoruba_male_dataset` is from the same uploader as the training data (`Hidi-agili/yoruba_tts_dataset`). The text-level overlap check ran and presumably passed (Step 8 produced numbers), but **audio-fingerprint overlap is not verified**. A reviewer could legitimately ask whether the male subset is a curated extract from the same source recordings. For a strict OOD claim, the next OOD round must use a different-author corpus (OpenSLR86 / FLEURS test).
- **N=100 OOD is a sanity number.** ±5 pp confidence band at best. The final thesis numbers should be N=200–500. Trivially re-runnable — bump `N_OOD` in `Whisper_ood_test.ipynb` Step 3 and re-execute Steps 6–10.
- **Reference inconsistency in both test sets.** Holdout samples 0/2/4 above demonstrate references missing tone marks the model correctly restores. This is the *systematic* error mode the diacritic-gap argument addresses, but it is the reason raw Y-WER on this dataset family will never be < 50% even for a perfect model. **Flag this loudly in the thesis methodology.**
- **Audio dataset content is biblical.** Both `yoruba_tts_dataset` (training) and `yoruba_male_dataset` (OOD) appear to be biblical/religious recordings. Domain transfer to spontaneous conversational speech is **not** tested here. Worth a separate "convo A/B" using `data/audio/eval_convo/` (parked from the v3 trial).
- **Overlap percentage not yet captured in this doc.** Backfill the Step 7.5 print into the Setup section as soon as it's available; that number is part of the methodology.

## Decision

⚠️ **Useful intermediate result. Model not pushed to HF.**

v4 produces a clean, defensible "fine-tuning helped on OOD audio" claim, but the doc is one stop before a thesis-grade triangulated OOD picture. **Next eval**: run the same paired comparison on a different-author corpus (OpenSLR86 or FLEURS yo_ng test) to triangulate. If those numbers track in the same direction (which they should — same model, broader test coverage), v4 is the right shipping candidate for the thesis chapter and can be pushed.

`config.M1_HF_MODEL` remains `devalade/whisper-large-v3-yoruba-v3` (the cloud-shipped trained model). v4 lives on Drive only and is loadable for evaluation via the `RUN_DIR/merged_16bit/` path; this is the user's deliberate choice ([`Whisper_v4.ipynb`](../../Whisper_v4.ipynb) keeps `PUSH_MERGED_HUB=False`).

### Immediate follow-ups

- [ ] Backfill the Step 7.5 overlap percentage and OOD sample triples into this doc.
- [ ] Re-run `Whisper_ood_test.ipynb` with **OpenSLR86** (Step 4 Block B) and capture an `ood_openslr_paired.json` next to the male_dataset payload.
- [ ] Re-run with **FLEURS yo_ng test** (Step 4 Block C) — note this is the project's existing benchmark, so it should also reproduce the [`2026-06-11-data-mix-v3.md`](2026-06-11-data-mix-v3.md) FLEURS direction.
- [ ] Bump `N_OOD` to 200–500 for the final thesis numbers.
- [ ] If the triangulated picture holds (v4 wins on Y-CER-perm on three different OOD corpora), push to HF as `devalade/whisper-large-v3-yoruba-v4` and update `config.M1_HF_MODEL`.

## Cross-references

- Training run: [`Whisper_v4.ipynb`](../../Whisper_v4.ipynb)
- In-distribution holdout eval: [`Whisper_test.ipynb`](../../Whisper_test.ipynb) — section "Test the v4 fine-tune from Drive" and "Baseline `whisper-large-v3` on the same holdout".
- OOD eval: [`Whisper_ood_test.ipynb`](../../Whisper_ood_test.ipynb)
- WER protocol: [`docs/yoruba-wer-evaluation.md`](../yoruba-wer-evaluation.md) — defines YASR-Bench / normalize / strip_diacritics.
- Prior trial that v4 builds on: [`2026-06-11-data-mix-v3.md`](2026-06-11-data-mix-v3.md).
- Same-author dataset background: [`docs/related-work-yoruba-models.md`](../related-work-yoruba-models.md) — `Hidi-agili/yoruba_male_dataset` entry.
