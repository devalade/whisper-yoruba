# Omnilingual ASR — evaluation for the M1 foundation model

**Drafted:** 2026-06-10
**Status:** evaluation proposal — no experiments run yet
**Purpose:** capture the argument for adding Meta's Omnilingual ASR as a second foundation-model arm in the thesis, alongside the existing Whisper-large-v3 + LoRA arm. This doc is the source the thesis chapter pulls from.

For the why-this-decomposition argument, see [`methodology.md`](methodology.md). For the runtime pipeline this would plug into, see [`architecture.md`](architecture.md). For the v3 Whisper fine-tune in flight, see [`experiments/_active.md`](experiments/_active.md).

---

## 1. Motivation

The thesis is committed to building a Yoruba voice-query pipeline that runs on Apple Silicon. M1 (ASR) is the bottleneck stage: a 65.6% WER on FLEURS (v2 fine-tune of Whisper-large-v3) is far from production-usable and is the single biggest contributor to end-to-end error.

The v2 → v3 trial isolates the **data axis** (single-speaker corpus → demographically balanced mix) on a fixed foundation model. But there is a second axis the thesis has not yet explored: the **choice of foundation model itself**. Meta's release of Omnilingual ASR on 2025-11-10, with explicit investment in under-resourced languages, makes this axis worth a controlled arm.

This document scopes that arm.

## 2. What Omnilingual ASR is

Open-source ASR system released by Meta FAIR in November 2025, updated with v2 checkpoints in December 2025. Distributed under Apache 2.0.

- **Coverage:** 1,600+ languages (vs Whisper-large-v3's ~100).
- **Architecture families:** three, layered on a shared wav2vec 2.0 SSL encoder:
  - **W2V** — self-supervised pretraining checkpoints (no decoder).
  - **CTC** — wav2vec 2.0 encoder + CTC head. Fast, single-pass.
  - **LLM** — wav2vec 2.0 encoder + LLaMA decoder. Slower, language-conditionable, supports decoding with text context.
- **Sizes:** 300M / 1B / 3B / 7B parameters for each family.
- **Reported headline:** the 7B-LLM model reaches CER < 10 for 78% of supported languages.
- **Tooling:** built on `fairseq2` (Meta's research toolkit, not HuggingFace). Inference SDK is a clean `ASRInferencePipeline` API.
- **Notable limit:** standard CTC/LLM variants only accept audio ≤ 40 seconds. The `LLM_Unlimited` variants remove this cap but currently do not support fine-tuning.

### Released checkpoints relevant to this thesis

| Model | Family | Params | VRAM (infer) | Real-time factor | Fine-tunable |
|---|---|---|---|---|---|
| `omniASR_CTC_300M_v2` | CTC | 325M | ~2 GiB | 96× faster than LLM-7B | yes |
| `omniASR_CTC_1B_v2` | CTC | 975M | ~3 GiB | 48× | yes |
| `omniASR_LLM_3B_v2` | LLM | 4.4B | ~10 GiB | ~1× | yes |
| `omniASR_LLM_7B_v2` | LLM | 7.8B | ~17 GiB | ~1× | yes |
| `omniASR_LLM_Unlimited_7B_v2` | LLM | 7.8B | ~17 GiB | ~1× | **no** |

(Real-time factors per batch=1, audio=30s, BF16, A100, taken from the upstream README.)

## 3. Why this is the right comparison for Yoruba

Whisper-large-v3 was trained on ~680,000 hours of audio. Yoruba's share of that was reported by OpenAI as ~1.7 hours — among the lowest-resource languages in the model. The model's strong stock WER (88.6% on FLEURS yo_ng) reflects that.

Omnilingual ASR's training data was scaled and *balanced* specifically to cover under-resourced languages. The Meta paper documents on-the-ground corpus creation in Pakistan and Liberia (Yoruba is widely spoken in Nigeria, Benin, and parts of Togo, but the data-collection methodology applies). Concrete consequences for Yoruba:

1. **Larger pretraining footprint.** Likely 10–100× more Yoruba audio at pretraining than Whisper, though Meta has not published a per-language hours table yet.
2. **Language conditioning is first-class.** Whisper has `language="yo"` as a decoder hint that often drifts; Omnilingual's LLM variants accept a `lang="yor_Latn"` argument explicitly.
3. **Released zero-shot variant** (`omniASR_LLM_7B_ZS`) — if Yoruba weren't supported, you could add it from a handful of paired examples.
4. **Tokenizer covers Yoruba diacritics.** `omniASR_tokenizer_written_v2` should emit clean diacritized text, which could reduce or eliminate the need for the M2 (diacritic restoration) stage downstream.

The first cheap experiment to run is **zero-shot CER/WER on FLEURS yo_ng** with `omniASR_LLM_7B_v2` and `omniASR_CTC_1B_v2`. The per-language results table shipped with the release (`per_language_results_table_7B_llm_asr.csv`) reports their own evaluation; cross-checking against our N=200 / beams=5 protocol grounds the number against our other baselines.

## 4. Comparison: Whisper-large-v3 vs Omnilingual

| Axis | Whisper-large-v3 + LoRA (Arm A, v3) | Omnilingual ASR + fine-tune (Arm B, proposed) |
|---|---|---|
| Yoruba pretraining hours | ~1.7 (per OpenAI) | undisclosed but materially higher |
| Architecture | Encoder-decoder, autoregressive | wav2vec 2.0 encoder + CTC head (CTC) or LLM decoder (LLM) |
| Stock FLEURS yo_ng WER | 88.6% (measured) | TBD — first experiment to run |
| Released under | MIT | Apache 2.0 |
| Framework | HuggingFace Transformers | fairseq2 |
| Best size for Apple Silicon target | large-v3, ~6 GB fp16 | CTC 1B (~3 GB) or LLM 3B (~10 GB) |
| Inference speed on target | ~1× (autoregressive) | CTC: ~48× faster; LLM: ~1× |
| Long audio | Sliding-window chunking, robust | Hard 40s cap (Unlimited variant exists but cannot be fine-tuned) |
| Language conditioning | `language="yo"` decoder hint | `lang="yor_Latn"` first-class |
| Diacritic output | Trained model dependent — current Yoruba ft emits diacritics | Tokenizer designed for diacritized output |
| Fine-tuning method | LoRA (community) | Recipes published in `workflows/recipes/wav2vec2/asr/` |
| Adapter ecosystem | Mature (PEFT, bitsandbytes) | Younger; fairseq2 native |
| Stops M2 from being needed? | No, M2 still required for now | Plausibly yes — measurable experiment |

## 5. Proposed methodology — two-arm comparison

The cleanest thesis structure is to **vary only the foundation model** while holding everything else constant.

### Controlled axes (same across both arms)

- Training mix: `config.FT_DATA_MIX` (Hidi-agili + chukypedro demographic shards + FLEURS yo_ng train)
- Demographic balance: 1k–5k per source, all kid shards retained
- Eval protocol: FLEURS yo_ng test, N=200, beams=5 (where applicable)
- Held-out evaluations: Hidi-agili held-out split, convo eval set
- WER normalization: `scripts/eval_wer.normalize` (strips diacritics for cross-arm fairness; separately track diacritic accuracy)

### Free axes (one per arm)

- **Arm A — Whisper:** LoRA r=32 q+v, 3 epochs, 1e-5 LR with warmup. Already wired (`scripts/finetune_whisper.py`, v3 trial in flight).
- **Arm B — Omnilingual:** fairseq2 fine-tuning recipe per the upstream `workflows/recipes/wav2vec2/asr/README.md`. Likely full fine-tune of CTC 1B and LLM 3B as the two interesting Omnilingual sub-arms.

### Sequence of trials

| Order | Trial | Output | Cost |
|---|---|---|---|
| T1 (done) | Whisper-large-v3 v1 baseline | 66.1% FLEURS WER | shipped |
| T2 (done) | Whisper-large-v3 v2 (more epochs) | 65.6% FLEURS WER, single-speaker overfit confirmed | shipped |
| T3 (in flight, 2026-06-10) | Whisper v3 — balanced data mix | TBD | ~$10 / ~6h |
| T4 (proposed) | Omnilingual zero-shot baseline (CTC 1B + LLM 3B + LLM 7B) | sets the foundation-only ceiling | ~$5 / ~2h, GPU |
| T5 (proposed) | Omnilingual CTC 1B fine-tune on the same data mix | direct Arm A vs Arm B comparison | ~$8 / ~5h |
| T6 (optional) | Omnilingual LLM 3B fine-tune on the same data mix | quality-vs-speed trade study | ~$15 / ~8h |

### Decision rule for the thesis

Three outcomes, each defensible in the writeup:

- ✅ **Arm B wins on FLEURS and convo eval.** Conclusion: foundation-model choice matters more than fine-tuning strategy for low-resource languages. Ship Omnilingual as M1, write the foundation-comparison chapter as the headline result.
- ⚠️ **Arm A wins after fine-tuning.** Conclusion: aggressive LoRA + balanced data on a familiar foundation beats a stronger but less-tuned alternative. Ship Whisper v3 as M1; the comparison chapter still has value as a methodology contribution.
- ❌ **Both arms plateau within noise.** Conclusion: data and training recipe are the limiting factor, not foundation. Pivot the next trial toward data scaling (more Yoruba audio collection) rather than model swaps. Equally publishable as a negative result.

## 6. Integration into the M1 module

The pipeline contract (`PipelineModule` in `modules/base.py`) is already foundation-agnostic. Adding Omnilingual is a new sibling to `M1ASR` (mlx) and `M1ASRHF` (HuggingFace Whisper).

### Module skeleton (`modules/m1_asr_omni.py`)

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from modules.base import PipelineModule
import config


class M1ASROmni(PipelineModule):
    def __init__(self, model_card: str = "omniASR_CTC_1B_v2", lang: str = "yor_Latn"):
        super().__init__()
        self.model_card = model_card
        self.lang = lang
        self._pipeline = None

    def initialize(self) -> None:
        self._pipeline = ASRInferencePipeline(model_card=self.model_card)
        self._ready = True

    def process(self, audio_path: str) -> dict:
        self._require_ready()
        # Read 16 kHz mono; chunk to ≤40s if not using Unlimited variant.
        transcripts = self._pipeline.transcribe(
            [str(audio_path)], lang=[self.lang], batch_size=1
        )
        return {"text": transcripts[0], "model": self.model_card, "language": self.lang}
```

Wiring it into `pipeline.py`:

```python
class YorubaPipeline:
    def __init__(self, asr_backend: str = "mlx", ...):
        if asr_backend == "mlx":
            self.m1 = M1ASR()
        elif asr_backend == "hf":
            self.m1 = M1ASRHF()
        elif asr_backend == "omni":
            self.m1 = M1ASROmni()
```

CLI: `pipeline.py --asr omni` joins the existing `--asr mlx|hf` choices. The variant matrix table in `architecture.md` grows one row.

### Configuration

Add to `config.py`:

```python
M1_OMNI_MODEL = "omniASR_CTC_1B_v2"   # CTC sub-arm by default; switch to LLM for the slower/stronger sub-arm
M1_OMNI_LANG = "yor_Latn"
```

### Downstream effects

- **M2 (diacritic restoration) may become redundant.** Run a side-experiment: pass raw Omnilingual output (no M2) through M3 and compare end-to-end answer accuracy vs Omnilingual + M2 + M3. If no degradation, the pipeline simplifies by one stage and that simplification is itself a thesis contribution.
- **M5 (TTS) is unaffected** — it's downstream of M3 (English), so the M1 swap doesn't reach it.

## 7. Apple Silicon deployment — open questions

The thesis target is M1/M2/M3 Macs via the existing `M1ASR` (mlx-whisper) path. Omnilingual must run on Apple Silicon to be a real M1 candidate, not just a research curiosity.

1. **fairseq2 MPS support.** Needs verification. The library targets CUDA primarily; CPU fallback exists but may be too slow for interactive use.
2. **ONNX export path.** If fairseq2 supports it, exporting the CTC 1B encoder to ONNX and running it via Core ML or ONNX Runtime could close the gap. This is itself a contribution worth a thesis subsection.
3. **MLX port.** mlx-whisper exists because the Apple MLX team ported Whisper. No equivalent for Omnilingual yet — a research-grade port would be ambitious but feasible for a strong thesis chapter.

The pragmatic minimum: get Omnilingual running on the Vast box for training and evaluation, and report the Apple-Silicon deployment story honestly (CPU-only fallback or future-work flag).

## 8. Risks and limits

| Risk | Likelihood | Mitigation |
|---|---|---|
| Omnilingual zero-shot is already so good that fine-tuning helps marginally | Moderate, if Yoruba is in the 78% with CER < 10 | Frame Arm B as "demonstrating the new SOTA on Yoruba." Marginal fine-tuning gain is itself a clean result. |
| Omnilingual zero-shot underperforms Whisper-v3 on Yoruba | Possible — Yoruba dialect / Nigerian-English code-switching may not be well represented | Run the cheap zero-shot test first (T4) before committing to T5. |
| fairseq2 / Apple Silicon mismatch | Likely (CUDA-first library) | Decouple training (cloud GPU) from deployment (Mac). Report the deployment status as a separate experiment. |
| 40s audio cap blocks long-form use | Real for some Yoruba conversational test clips | Voice-query pipeline natural turns are ≤10s; not a blocker for M1's use case. Flag explicitly. |
| `LLM_Unlimited` cannot be fine-tuned (as of Dec 2025) | Confirmed in upstream README | If we need long-audio fine-tune, stick with chunked CTC/LLM standard variants. |
| Reproducibility — fairseq2 is research-grade, less stable than HF Transformers | Moderate | Pin exact `omnilingual-asr` package version in `requirements-omni.txt`. Snapshot model cards used. |

## 9. Concrete next steps

1. Wait for Whisper v3 trial (T3) to finish. Record the numbers.
2. Spin up a fresh Vast box (or reuse the existing one). Install `omnilingual-asr` and `omnilingual-asr[data]`.
3. **T4:** Run zero-shot inference of `omniASR_CTC_1B_v2`, `omniASR_LLM_3B_v2`, and `omniASR_LLM_7B_v2` on the FLEURS yo_ng test split at N=200. Compute WER and CER with our normalizer.
4. Compare T4 numbers against Whisper v1/v2/v3 baselines. If T4 already beats v3, the foundation question is answered without T5.
5. **T5 (if warranted):** Fine-tune `omniASR_CTC_1B_v2` on `config.FT_DATA_MIX` using the upstream fairseq2 recipe. Push to HF under `devalade/omnilingual-yoruba-v1`.
6. Re-evaluate on FLEURS / Hidi-agili held-out / convo eval. Write the comparison chapter.

## 10. Citation

```bibtex
@misc{omnilingualasrteam2025omnilingualasropensourcemultilingual,
      title={Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages},
      author={Omnilingual ASR team and Gil Keren and Artyom Kozhevnikov and Yen Meng and Christophe Ropers and Matthew Setzler and Skyler Wang and Ife Adebara and Michael Auli and Can Balioglu and Kevin Chan and Chierh Cheng and Joe Chuang and Caley Droof and Mark Duppenthaler and Paul-Ambroise Duquenne and Alexander Erben and Cynthia Gao and Gabriel Mejia Gonzalez and Kehan Lyu and Sagar Miglani and Vineel Pratap and Kaushik Ram Sadagopan and Safiyyah Saleem and Arina Turkatenko and Albert Ventayol-Boada and Zheng-Xin Yong and Yu-An Chung and Jean Maillard and Rashel Moritz and Alexandre Mourachko and Mary Williamson and Shireen Yates},
      year={2025},
      eprint={2511.09690},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.09690},
}
```

Related background to cite alongside this in the thesis:

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper paper).
- Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (MMS paper, Omnilingual's direct predecessor).
- Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* (the SSL foundation under both MMS and Omnilingual).
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (the fine-tuning method used in Arm A).

## 11. Related project docs

- [`methodology.md`](methodology.md) — five-stage decomposition rationale.
- [`architecture.md`](architecture.md) — module contracts; this proposal grows the M1 row.
- [`literature-review.md`](literature-review.md) — existing literature framing.
- [`fine-tuning-runbook.md`](fine-tuning-runbook.md) — Whisper fine-tuning procedure (Arm A).
- [`experiments/_active.md`](experiments/_active.md) — v3 trial currently in flight.
- [`thesis-plan.md`](thesis-plan.md) — chapter outline that needs updating if Arm B lands.
