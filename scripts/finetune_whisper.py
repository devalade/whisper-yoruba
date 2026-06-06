"""Fine-tune Whisper-large-v3 on Yoruba data with PEFT LoRA.

Designed for a single CUDA box (Vast.ai, RunPod, Colab A100). Trains on one or
more HuggingFace datasets in (audio, text) shape — defaults to
`Hidi-agili/yoruba_tts_dataset`, which carries diacritized Yoruba text. The
resulting LoRA adapter is saved to `config.FT_OUTPUT_DIR`; merge it into a
plain Whisper checkpoint with `scripts/merge_lora.py` before swapping
`config.M1_HF_MODEL`.

Usage:
    # Smoke test (a few minutes on any GPU):
    python -m scripts.finetune_whisper --epochs 1 \\
        --max-train-samples 50 --max-eval-samples 10

    # Real run:
    python -m scripts.finetune_whisper --epochs 3 --batch-size 8 \\
        --push-to-hub --hub-model-id <user>/whisper-large-v3-yoruba-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import config
from scripts.eval_wer import normalize
from utils.logging import get_logger

log = get_logger("finetune")


# ---- data ---------------------------------------------------------------

def _audio_column_name(ds) -> str:
    # Datasets author chose `audio` for Hidi-agili; tolerate `path` too.
    for cand in ("audio", "path"):
        if cand in ds.column_names:
            return cand
    raise ValueError(f"no audio column in {ds.column_names}")


def _text_column_name(ds) -> str:
    for cand in ("text", "transcription", "sentence"):
        if cand in ds.column_names:
            return cand
    raise ValueError(f"no text column in {ds.column_names}")


def load_corpus(dataset_ids: list[str]) -> "datasets.Dataset":
    """Load + normalize + concatenate every dataset into a single (audio, text) set."""
    parts = []
    for ds_id in dataset_ids:
        log.info("loading %s", ds_id)
        ds = load_dataset(ds_id, split="train")
        audio_col = _audio_column_name(ds)
        text_col = _text_column_name(ds)
        keep = [audio_col, text_col]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        if audio_col != "audio":
            ds = ds.rename_column(audio_col, "audio")
        if text_col != "text":
            ds = ds.rename_column(text_col, "text")
        ds = ds.cast_column("audio", Audio(sampling_rate=config.M1_SAMPLE_RATE))
        parts.append(ds)
        log.info("  %s: %d rows", ds_id, len(ds))
    combined = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    log.info("combined corpus: %d rows", len(combined))
    return combined


def extract_holdout(corpus, n: int, out_dir: Path, seed: int) -> "datasets.Dataset":
    """Pop `n` raw samples from `corpus`, save them as WAV + manifest, return the rest.

    Runs before feature prep so we have the original audio array (not mel
    features) on disk. The same seed produces the same holdout — switching
    seeds across runs is fine, but reusing a seed means reusing samples.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shuffled = corpus.shuffle(seed=seed)
    holdout = shuffled.select(range(n))
    rest = shuffled.select(range(n, len(shuffled)))

    sr = config.M1_SAMPLE_RATE
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, row in enumerate(holdout):
            wav_path = out_dir / f"holdout_{i:02d}.wav"
            arr = np.asarray(row["audio"]["array"], dtype=np.float32)
            sf.write(wav_path, arr, sr, subtype="PCM_16")
            mf.write(json.dumps({
                "id": f"holdout_{i:02d}",
                "wav": str(wav_path.relative_to(config.ROOT)),
                "yo": row["text"],
                "duration_s": round(len(arr) / sr, 2),
            }, ensure_ascii=False) + "\n")
            log.info("holdout %d: %s (%.2fs)  ref=%r",
                     i, wav_path.name, len(arr) / sr, row["text"])
    log.info("wrote %d holdout samples + manifest to %s", n, out_dir)
    return rest


def build_prepare_fn(processor: WhisperProcessor):
    sr = config.M1_SAMPLE_RATE
    max_samples = int(config.FT_MAX_AUDIO_SECONDS * sr)

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio"]
        # Whisper cannot handle >30s clips — these are dropped after the map.
        if len(audio["array"]) > max_samples:
            return {"input_features": None, "labels": None}
        text = (batch["text"] or "").strip()
        if not text:
            return {"input_features": None, "labels": None}
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=sr
        ).input_features[0]
        labels = processor.tokenizer(text).input_ids
        return {"input_features": feats, "labels": labels}

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Standard HF Whisper training collator — pads audio + labels independently."""

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper prepends the decoder_start_token; strip it from labels so
        # cross-entropy doesn't learn to re-emit it.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---- metric -------------------------------------------------------------

def build_compute_metrics(processor: WhisperProcessor):
    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = pad_id
        hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        # Same normalization the WER eval script uses — strips diacritics so
        # the metric is fair against the existing baselines.
        hyps_n = [normalize(h) for h in hyps]
        refs_n = [normalize(r) for r in refs]
        wer = jiwer.wer(refs_n, hyps_n)
        return {"wer": wer}

    return compute_metrics


# ---- main ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=config.FT_BASE_MODEL)
    p.add_argument("--output-dir", default=str(config.FT_OUTPUT_DIR))
    p.add_argument("--datasets", nargs="+", default=config.FT_DATASETS,
                   help="HF dataset IDs with (audio, text) columns")
    p.add_argument("--language", default=config.M1_LANGUAGE)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=config.FT_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=config.FT_LORA_ALPHA)
    p.add_argument("--eval-frac", type=float, default=0.05)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training rows — use for fast smoke tests")
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--total-samples", type=int, default=None,
                   help="Cap the raw corpus to N rows BEFORE the train/eval split "
                        "(deterministic shuffle by --seed). Use for a small targeted run.")
    p.add_argument("--holdout-n", type=int, default=0,
                   help="Pop N raw audio samples before training, save as WAV + "
                        "manifest under --holdout-dir. These are excluded from train+eval.")
    p.add_argument("--holdout-dir", default=str(config.AUDIO_DIR / "holdout"),
                   help="Where to write the held-out WAVs + manifest.jsonl")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for holdout extraction + train/test split")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        log.warning(
            "CUDA not available — large-v3 LoRA training will be unusably slow. "
            "Continuing anyway because you asked, but you almost certainly want "
            "to run this on a GPU box."
        )

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task="transcribe"
    )

    corpus = load_corpus(args.datasets)

    # 1) Hold out raw audio FIRST — we want the original arrays on disk,
    #    not the mel features. The same seed is reused for the train/test
    #    split below so holdout, train, eval are all deterministic.
    if args.holdout_n > 0:
        corpus = extract_holdout(
            corpus, args.holdout_n, Path(args.holdout_dir), seed=args.seed,
        )

    # 2) Cap the corpus before feature prep — saves wall-time on small runs.
    if args.total_samples:
        capped = min(args.total_samples, len(corpus))
        corpus = corpus.shuffle(seed=args.seed).select(range(capped))
        log.info("capped corpus to %d rows (--total-samples)", capped)

    prepare = build_prepare_fn(processor)
    log.info("preparing features (this materializes mel spectrograms)…")
    corpus = corpus.map(
        prepare,
        remove_columns=corpus.column_names,
        num_proc=args.num_workers,
        desc="prepare",
    )
    before = len(corpus)
    corpus = corpus.filter(lambda r: r["input_features"] is not None)
    log.info("kept %d/%d rows after length/empty-text filter", len(corpus), before)

    split = corpus.train_test_split(test_size=args.eval_frac, seed=args.seed)
    if args.max_train_samples:
        split["train"] = split["train"].select(range(min(args.max_train_samples, len(split["train"]))))
    if args.max_eval_samples:
        split["test"] = split["test"].select(range(min(args.max_eval_samples, len(split["test"]))))
    log.info("train=%d eval=%d", len(split["train"]), len(split["test"]))

    log.info("loading base model %s", args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    # Force Yoruba+transcribe so generate() doesn't drift across languages.
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # No task_type — PEFT's SEQ_2_SEQ_LM template injects `input_ids`, which
    # Whisper rejects (it takes `input_features`). Letting task_type default
    # passes kwargs through unchanged.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        # use_reentrant=False is the modern checkpointing impl — required with
        # PEFT because the legacy reentrant version errors out when the base
        # model is frozen ("element 0 of tensors does not require grad").
        gradient_checkpointing_kwargs={"use_reentrant": False},
        predict_with_generate=True,
        generation_max_length=225,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # keep only best checkpoint — saves disk on small boxes
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    log.info("starting training")
    trainer.train()

    log.info("saving adapter + processor to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
