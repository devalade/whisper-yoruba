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
# Switched 2026-06-05 to our own fine-tune of whisper-large-v3 on Hidi-agili/yoruba_tts_dataset.
# Prior baseline kept as a comment for reference / quick A/B:
#   M1_HF_MODEL = "RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT"
#   M1_HF_PROCESSOR = "openai/whisper-large-v2"
M1_HF_MODEL = "devalade/whisper-large-v3-yoruba"
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
FT_DATASETS = ["Hidi-agili/yoruba_tts_dataset"]
FT_MAX_AUDIO_SECONDS = 30
FT_LORA_R = 32
FT_LORA_ALPHA = 64
