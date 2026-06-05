# Literature review

> Skeleton — fill in BibTeX keys and add commentary as you read each reference.
> Aim for ~2–3 paragraphs of prose per section, not a list. The defense panel will ask which paper most influenced each design choice.

## 1. Multilingual and low-resource ASR

### Whisper

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* OpenAI technical report.
  - The base architecture for M1. Trained on 680k hours of multilingual web-scraped speech. Documents Yoruba as one of 99 supported languages but with low pretraining hours (note the exact number from Table 1 for the thesis).
  - Relevance: justifies why fine-tuning is necessary — base Whisper has seen relatively little Yoruba.

### Common Voice for Yoruba

- Ardila, R., et al. (2020). *Common Voice: A Massively-Multilingual Speech Corpus.* LREC.
- Any subsequent paper analysing Common Voice Yoruba performance — add as you find them.

### MMS

- Pratap, V., et al. (2023). *Scaling Speech Technology to 1000+ Languages.* Meta AI.
  - Source of `mms-tts-yor` (M5). Also has ASR variants; mention why we chose Whisper over MMS-ASR: Whisper has stronger long-form decoding and better integration with HF tooling.

### Wav2Vec2 for African languages

- Conneau, A., et al. (2021). *Unsupervised Cross-Lingual Representation Learning for Speech Recognition.* Interspeech.
  - The wav2vec2-XLS-R baseline we considered for M1 but did not pick.

## 2. Parameter-efficient fine-tuning

### LoRA

- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
  - The technique used to fine-tune Whisper. Justify r=32 by citing the paper's recommendations and ablations.

### PEFT library

- Mangrulkar, S., et al. (2022). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.* HuggingFace.
  - Cite for reproducibility — explain which PEFT version was used.

### Whisper PEFT recipes

- HuggingFace blog posts on Whisper LoRA fine-tuning — cite the canonical one. They establish the recipe we follow (target `q_proj` + `v_proj`, gradient checkpointing, label collator pattern).

## 3. Diacritic restoration for Yoruba

### Davlan mT5 ADR

- Adelani, D. I., et al. (2021). *The Effect of Domain and Diacritics in Yorùbá–English Neural Machine Translation.* arXiv:2103.08647.
  - Source of `Davlan/mT5_base_yoruba_adr` (M2). Reports the 64.63 / 70.27 BLEU figures cited in the methodology.

### Earlier work

- Orife, I. (2018). *Attentive Sequence-to-Sequence Learning for Diacritic Restoration of Yorùbá Language Text.*
  - Earlier baseline for Yoruba ADR. Cite as the prior state-of-the-art before mT5-based approaches.

## 4. Multilingual translation

### NLLB

- NLLB Team, Costa-jussà, M. R., et al. (2022). *No Language Left Behind: Scaling Human-Centred Machine Translation.* Meta AI.
  - Source of M3 and M5's translation component. Reports BLEU scores for `yor_Latn ↔ eng_Latn`. Cite the specific table for the thesis.

## 5. Cross-lingual retrieval and RAG

### Cross-lingual IR

- Pick one foundational paper. Suggestion: Litschko, R., et al. (2018). *Unsupervised Cross-Lingual Information Retrieval Using Monolingual Data Only.* SIGIR.
- And one recent: a 2023–2024 paper on retrieval over a high-resource KB for low-resource queries.

### Retrieval-augmented generation

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
  - Foundational RAG paper, motivates the M4 architecture.

### Sentence Transformers

- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
  - Justifies the use of MiniLM-L6 in M4.

## 6. Yoruba TTS

### MMS-TTS

- Pratap, V., et al. (2023). — (same as MMS reference above; the TTS subsystem is documented in the same paper).

### Earlier Yoruba TTS

- Find one paper on rule-based or HMM-based Yoruba TTS to contextualise why MMS is a step change.

## 7. End-to-end speech-to-speech systems

- Pick one reference for full audio-in / audio-out pipelines. The system architecture here is closer to traditional ASR → MT → TTS chaining than to direct speech-to-speech models like Translatotron, but cite Translatotron (Jia et al., 2019) for completeness and to justify the modular choice.

## Notes for the defense

When the panel asks "which paper most influenced your design," you should be able to name one for each component:
- M1 → Radford 2022 (Whisper) + Hu 2021 (LoRA)
- M2 → Adelani 2021 (Davlan mT5 ADR)
- M3 → Costa-jussà 2022 (NLLB)
- M4 → Lewis 2020 (RAG) + Reimers 2019 (Sentence-BERT)
- M5 → Pratap 2023 (MMS)
