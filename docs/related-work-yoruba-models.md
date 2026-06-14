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

## Cascaded pipeline systems

### Oyesanmi & Olukanmi (2026) — *Towards Yorùbà-Speaking Google Maps Navigation*

- **Citation**: Oyesanmi, F. & Olukanmi, P. (2026). Towards Yorùbà-Speaking Google Maps Navigation. *SAIEE Africa Research Journal*, **117**(2). https://www.scielo.org.za/scielo.php?pid=S1991-16962026000200003&script=sci_arttext&tlng=en
- **Task**: English → Yorùbá speech-to-speech translation for navigation narration. **Cascaded ASR → MT → TTS architecture** — same shape as ours but inverted direction and no retrieval stage.
- **Contribution**: not a new model — a survey-style evaluation of 5 ASR + 4 MT + 5 TTS systems to assess feasibility of Yorùbá navigation assistance.
- **Models evaluated**:
  - **ASR**: WhisperAI, Facebook FAIRSEQ S2T, Microsoft UniSpeech-SAT, Nvidia Conformer-Transducer, Google Speech Library.
  - **MT**: DeepL, LibreTranslate, NLLB (Meta), Google Translate.
  - **TTS**: Seamless TTS, MMS TTS, SpeechT5, Coqui TTS, YourTTS.
- **Evaluation**: Word Error Rate via raw Levenshtein distance. Tested in serene + noisy conditions. Focus on numerals and location names.
- **Reported numbers**:
  - **ASR**: WhisperAI best at 33% WER in noise; meaning preserved despite errors. 4 of 5 ASR systems usable.
  - **MT**: 41–48% WER on direction narration. Severe failures on Yorùbá numerals and proper nouns.
  - **TTS**: only MMS-TTS-yor produces usable Yorùbá; pronunciation "suboptimal", flagged as needing fine-tuning.
- **Datasets they cite**: OpenSLR86 (~4 h), Lagos-NWU (~2 h 45 min), Bibeli Mimo/NIV (~93 h, religious-only). ~7 h combined excluding religious text.

**How it relates to our pipeline**

| Module | Our choice | Theirs (best of their cohort) | Notes |
|---|---|---|---|
| M1 ASR | `whisper-large-v3` fine-tuned on Hidi-agili + chukypedro + FLEURS | WhisperAI (base) | Confirms Whisper as the right family. Their 33% WER (raw Levenshtein) is not directly comparable to our YASR-Bench numbers (`docs/yoruba-wer-evaluation.md`) but it's a useful order-of-magnitude reference. |
| M2 Diacritic | `Davlan/mT5_base_yoruba_adr` | *not addressed* | They acknowledge tone marks matter but don't address restoration. **This is something we have that they don't** — frame in the thesis as a contribution. |
| M3 / M5 MT | `facebook/nllb-200-distilled-600M` (both directions) | Google Translate (their claim) | They state "only Google Translate supports English-to-Yorùbà translation." This is incorrect at publication date — NLLB-200 supports yor_Latn in both directions. Worth flagging as a small correction. |
| M5 TTS | `facebook/mms-tts-yor` | MMS TTS | Independent validation of the choice. Their "needs fine-tuning" caveat is a citable hook for future work in our thesis. |
| M4 RAG | Mistral-7B + FAISS over Yorùbá Wikipedia | *no analogue* | Their task is narration, not Q&A — they don't need retrieval. No comparison point. |

**Methodological points worth borrowing**

1. **Noisy-condition evaluation**: their best ASR number is in noise (33%), not clean speech. We only evaluate on FLEURS (clean read). Adding a torchaudio-noise-augmented variant of YASR-Bench would strengthen the thesis and is cheap to add.
2. **WER ≠ semantic preservation**: the paper explicitly argues raw WER is insufficient for assessing meaning. This is the same argument we make for `Y-WER-perm` / `Y-CER` (`docs/yoruba-wer-evaluation.md`) — citing this paper in support is honest framing.
3. **Numeral handling**: they flag Yorùbá numerals as a systematic failure mode for MT. We haven't probed this in M3/M5. Worth a small slice in evaluation: "WER on the numerals subset of FLEURS".

**Limitations they acknowledge** (citable as motivation for our design choices):

- Cascaded latency from multiple sequential models.
- Insufficient training data for low-resource Yorùbá.
- Complex Yorùbá morphology and numeral systems poorly handled by current MT.
- WER metric insufficient for assessing semantic preservation.
- ~7 hours of usable public Yorùbá audio (excluding religious text).

**Differences worth being clear about in the thesis**

- Their data regime (~7 h) is much smaller than ours (Hidi-agili `yoruba_tts_dataset` alone is ~9 k clips ≈ 15+ h; the full v3 mix is multiples of that). We are operating well past their data ceiling.
- They evaluate off-the-shelf models. We *fine-tune*. Different question, but their numbers give the "starting point" against which our deltas should be reported.

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
