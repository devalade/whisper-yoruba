# Related work — Yorùbá speech models

A running registry of public Yorùbá speech models the thesis touches. One row per model. Add a section when one becomes relevant to a pipeline module (M1 ASR, M5 TTS, etc.) or to the related-work writeup.

For Whisper-large-v3 fine-tunes specifically, see `docs/experiments/` for our own ablations.

---

## TTS

### `FloatinggOnion/yoruba-cfm-dit`

- **URL**: https://huggingface.co/FloatinggOnion/yoruba-cfm-dit
- **Task**: Text-to-Speech. **Not** an ASR model — opposite direction of our M1 work. Relevant only as a candidate for **M5 (TTS)** or as a related-work citation.
- **Architecture**: Conditional Flow Matching with a Diffusion Transformer backbone. Predicts the velocity field `v = x₁ − x₀` that transports Gaussian noise to a real EnCodec latent.
  - Text encoder: 4-layer Transformer.
  - DiT: 10 blocks, model dim 512.
  - Decoder: pre-trained EnCodec 24 kHz (not trained, used as-is).
  - ~57 M parameters total.
- **Lineage**: same family as Matcha-TTS, F5-TTS, Voicebox (all flow-matching TTS). Author advertises this as an original CFM+DiT, not a fork.
- **Training data**:
  - `PlotweaverAI/yoruba-tts-selected-speakers`
  - `Hidi-agili/yoruba_male_dataset` — 10,446 male-speaker samples. **Note**: this is *not* the same shard as `Hidi-agili/yoruba_tts_dataset` we fine-tune Whisper on. Same author/collection, different curation. Card says "single speaker dataset; voice diversity is limited."
- **Training setup**: 120,000 steps, batch 8, AdamW (lr 2e-4, β=(0.9, 0.95), wd 1e-2), fp16, grad-clip 1.0, EMA 0.999. From scratch (no foundation-model adaptation).
- **Inference**: 24 ODE steps by default.
- **Audio config**: 24 kHz, EnCodec latents (128-dim, max 2048 frames).
- **Reported metrics**: **none**. README is purely qualitative — no MOS, no UTMOS, no WER-on-synthesised-speech. Don't cite quality numbers from the card.

**How it could be used in our work**

1. **M5 TTS candidate**: drop-in for whatever Yorùbá TTS we currently use. Worth A/B-testing.
2. **Numerical comparison via M1**: synthesize a fixed set of Yorùbá sentences with this model, transcribe with our fine-tuned Whisper, compute WER. That gives a defensible "TTS quality" proxy without needing human raters. Both this model and our M1 must agree on text — careful with diacritic normalization (`docs/yoruba-wer-evaluation.md`).
3. **Related-work citation**: methodologically interesting contrast. Theirs: small (57 M), from scratch, flow-matching in EnCodec latents. Ours (M1): large (1.5 B), adapter on a pre-trained foundation model, discriminative. Worth a paragraph in the thesis on the "small specialised model vs. large adapted model" trade-off for low-resource languages.

**What we don't know**
- Real speaker count beyond "single speaker" caveat.
- Whether the EnCodec decoder introduces audible Yorùbá-specific artefacts (tone marks, nasalisation).
- How it handles long utterances — card warns of "silence or truncation."

---

## ASR

*(none registered yet beyond our own fine-tunes — see `docs/experiments/` for ours)*

---

## Adding a new entry

Same structure as above:
- URL, task, architecture, lineage.
- Training data (with exact dataset IDs and sizes if known).
- Training setup (steps, batch, optimizer).
- Reported metrics + caveats. Be explicit if metrics are missing.
- How it relates to our pipeline modules.
- Open questions / unknowns.

Keep the registry honest about what's *not* reported — citing unverified numbers from a model card is a footgun.
