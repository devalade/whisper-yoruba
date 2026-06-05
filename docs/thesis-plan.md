# Thesis plan — 1 week (2026-06-05 → 2026-06-12)

## Contribution statement

A voice-driven Yoruba search system that lets non-English-literate Yoruba speakers query an English-language knowledge base (Wikipedia) and receive spoken Yoruba answers. The system bridges a low-resource speech language to a high-resource text knowledge base, with a fine-tuned Whisper model for Yoruba ASR as the core empirical contribution.

## Scope cuts (be honest)

Given one week, the following are explicitly **out of scope** — note them in the "Future work" section of the thesis:

- Multiple dataset mixes (only `Hidi-agili/yoruba_tts_dataset`)
- Hyperparameter sweeps (single LoRA r, lr, epochs combination)
- User studies (no human evaluation of output Yoruba audio quality)
- Latency optimization beyond what mlx-whisper already provides
- Direct comparison against commercial systems (Google STT, Azure, OpenAI Whisper API)

## Day-by-day

### Day 1 (today, Thu 2026-06-05) — system complete
- [x] Fine-tuning pipeline working end-to-end (smoke test passed)
- [ ] Finish disk-space cleanup, merge LoRA, push to HF Hub
- [ ] Backfill docs: methodology, lit review skeleton, setup, runbook, troubleshooting
- [ ] Kick off real 3-epoch fine-tune run on Vast (~2.5h, can run overnight)

### Day 2 (Fri 2026-06-06) — evidence
- [ ] Verify fine-tuned model in pipeline locally on Mac (`make wer-hf N=200`)
- [ ] Capture results: baseline (RafatK) vs fine-tune WER, train/eval loss curves from TensorBoard
- [ ] Write `docs/experiments/2026-06-06-large-v3-yoruba-ft.md` with all numbers
- [ ] Generate 10 end-to-end qualitative examples (audio → transcript → translation → answer → spoken answer)

### Day 3 (Sat 2026-06-07) — methodology + system chapters
- [ ] Draft methodology chapter (use `docs/methodology.md` as base)
- [ ] Draft system architecture chapter — include the M1–M5 diagram
- [ ] Add architecture diagram (Mermaid or hand-drawn) to README + thesis

### Day 4 (Sun 2026-06-08) — literature review + introduction
- [ ] Fill citation stubs in `docs/literature-review.md`
- [ ] Write introduction: problem framing (Yoruba speaker accessing English knowledge), motivation, contribution
- [ ] Write related work section using the lit review

### Day 5 (Mon 2026-06-09) — results + discussion
- [ ] Write results section: ASR WER table, qualitative examples, error analysis
- [ ] Write discussion: what worked, what surprised you, limitations
- [ ] Write conclusion + future work

### Day 6 (Tue 2026-06-10) — polish + demo
- [ ] Record a 2-minute video demo of the system
- [ ] Take screenshots of the pipeline running
- [ ] Format thesis to ESGIS requirements (template, citations style, page count)
- [ ] First full read-through, fix typos and prose

### Day 7 (Wed 2026-06-11) — buffer
- [ ] Reserved for unexpected problems
- [ ] Final review with supervisor if possible
- [ ] Submission Thu 2026-06-12 morning

## What to commit to git daily

Each evening, commit:
- That day's docs additions/changes
- Any code or config changes
- The training-run experiment log if applicable

This creates a git history that itself documents the iteration — useful artifact for the defense.

## Defense prep (parallel)

While writing, keep notes for likely questions:

- **Why LoRA over full fine-tuning?** Compute/cost (32 GB consumer GPU vs 80 GB A100), overfitting risk on a 9.5k-sample corpus, easier to share/version adapters.
- **Why whisper-large-v3 over small/medium?** Best zero-shot baseline on Yoruba per FLEURS; LoRA makes it tractable. (Cite original FLEURS evaluation.)
- **Why this dataset?** Open, diacritized targets, single speaker so cleaner than crowd-sourced. Acknowledge: single-speaker = limited acoustic diversity.
- **Why RAG with English Wikipedia?** Yoruba Wikipedia has ~30k articles vs English ~6.7M. Cross-lingual retrieval expands coverage by 200×.
- **Why translation step instead of end-to-end Yoruba LLM?** No production-quality Yoruba LLM available; NLLB+English LLM is more reliable than current Yoruba-only models.
- **Limitations:** single-speaker training data, no real-world noise robustness eval, no Yoruba dialect coverage beyond what's in the dataset, latency not measured systematically.
