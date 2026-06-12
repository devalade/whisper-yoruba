"""Central configuration for the Yoruba voice query pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
WIKI_DIR = DATA_DIR / "wikipedia"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = ROOT / "logs"

# M1 - ASR (mlx-whisper, default fast path on Apple Silicon)
M1_MODEL = "mlx-community/whisper-large-v3-mlx"
M1_LANGUAGE = "yo"  # Yoruba
M1_SAMPLE_RATE = 16000

# M1 - alternate HF Whisper backend (Yoruba-fine-tuned). Selected with --asr hf.
# Switched 2026-06-12 to v3 (multi-source data-mix). See
# docs/experiments/2026-06-11-data-mix-v3.md. Prior versions kept as comments
# for quick A/B / rollback:
#   M1_HF_MODEL = "RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT"  # external baseline
#   M1_HF_PROCESSOR = "openai/whisper-large-v2"
#   M1_HF_MODEL = "devalade/whisper-large-v3-yoruba"     # v1 (Hidi-agili only, 2026-06-05)
#   M1_HF_MODEL = "devalade/whisper-large-v3-yoruba-v2"  # v2 (3 epochs on v1 data, 2026-06-09)
M1_HF_MODEL = "devalade/whisper-large-v3-yoruba-v3"
M1_HF_PROCESSOR = "openai/whisper-large-v3"
M1_HF_CHUNK_S = 15
M1_HF_NUM_BEAMS = 5
M1_HF_MAX_NEW_TOKENS = 440

# M2 - Diacritic restoration (Yoruba ADR). Davlan's mT5-base finetune,
# reported 64.63 BLEU on Global Voices, 70.27 on Menyo-20k.
M2_MODEL_ORIFE = "Davlan/mT5_base_yoruba_adr"

# M3 - Translation YO -> EN
M3_MODEL = "facebook/nllb-200-distilled-600M"
M3_SRC_LANG = "yor_Latn"
M3_TGT_LANG = "eng_Latn"

# M4 - RAG
M4_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
M4_LLM_PATH = ROOT / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
M4_CHUNK_TOKENS = 200
M4_CHUNK_OVERLAP = 50
M4_SIM_THRESHOLD = 0.5
M4_TOP_K = 4

# M5 - TTS
M5_TTS_MODEL = "facebook/mms-tts-yor"
M5_EN_YO_MODEL = "facebook/nllb-200-distilled-600M"  # EN -> YO leg
M5_SRC_LANG = "eng_Latn"
M5_TGT_LANG = "yor_Latn"

# M1 fine-tuning (LoRA on whisper-large-v3; run on CUDA — see scripts/finetune_whisper.py).
# Training targets are diacritized Yoruba text, so the resulting M1 outputs diacritics
# directly; M2 stays in the chain as a safety net.
FT_BASE_MODEL = "openai/whisper-large-v3"
FT_OUTPUT_DIR = ROOT / "models" / "whisper-yo-lora"
FT_MERGED_DIR = ROOT / "models" / "whisper-yo-merged"
# Legacy simple list — still consumed by --datasets on the CLI for backward compat.
# v1/v2 fine-tunes used this single-source path.
FT_DATASETS = ["Hidi-agili/yoruba_tts_dataset"]
FT_MAX_AUDIO_SECONDS = 30
FT_LORA_R = 32
FT_LORA_ALPHA = 64

# v3 trial — balanced multi-source mix. Each entry resolves to one HF dataset
# load: `repo` is required, `config`/`data_files`/`split` follow HF semantics,
# `text_column` overrides auto-detection (FLEURS has both `transcription` and
# `raw_transcription`; we want the diacritized one), `max_rows` deterministically
# subselects after a shuffle with `--seed`. Demographic split for chukypedro
# matches the parquet sharding we discovered in `data/inspect/`.
FT_DATA_MIX: list[dict] = [
    # Anchor — prevents catastrophic forgetting of the v2 training distribution.
    {"repo": "Hidi-agili/yoruba_tts_dataset", "split": "train",
     "max_rows": 5000, "tag": "hidi-agili"},

    # The bet — multi-demographic Yoruba. Speaker_id is unreliable, so we treat
    # each demographic shard as a separable voice profile and balance by row count.
    {"repo": "chukypedro/clean_yoruba_dataset",
     "data_files": "*/yoruba_male_18-29.parquet",
     "max_rows": 2000, "tag": "chuky-male-18-29"},
    {"repo": "chukypedro/clean_yoruba_dataset",
     "data_files": "*/yoruba_male_30-over.parquet",
     "max_rows": 2000, "tag": "chuky-male-30-over"},
    {"repo": "chukypedro/clean_yoruba_dataset",
     "data_files": "*/yoruba_female_18-29.parquet",
     "max_rows": 2000, "tag": "chuky-female-18-29"},
    {"repo": "chukypedro/clean_yoruba_dataset",
     "data_files": "*/yoruba_female_30-over.parquet",
     "max_rows": 2000, "tag": "chuky-female-30-over"},
    # Kid shards are tiny (~600 rows each) — take them all to preserve the rare
    # demographic in the mix even though they won't move overall metrics.
    {"repo": "chukypedro/clean_yoruba_dataset",
     "data_files": "*/yoruba_male_6-17.parquet",
     "max_rows": None, "tag": "chuky-male-6-17"},
    {"repo": "chukypedro/clean_yoruba_dataset",
     "data_files": "*/yoruba_female_6-17.parquet",
     "max_rows": None, "tag": "chuky-female-6-17"},

    # Multi-speaker read speech — same domain as our FLEURS eval (test split),
    # so this *does* train somewhat toward the eval. That's a known trade-off
    # logged in the trial notes; the eval split is held out.
    {"repo": "google/fleurs", "config": "yo_ng", "split": "train",
     "text_column": "raw_transcription", "max_rows": 2500, "tag": "fleurs-train"},
]
