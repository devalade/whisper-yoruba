.PHONY: help install index sample run talk chat talk-hf chat-hf wer wer-mlx wer-hf wer-ft finetune finetune-smoke merge-lora gpu-ssh test test-m1 test-m2 test-m3 test-m4 test-m5 test-chain clean-logs clean-outputs

PYTHON ?= $(HOME)/miniforge3/envs/yoruba/bin/python
SAMPLE_WAV := data/audio/fleurs_yo_sample.wav
OUTPUT_WAV := data/outputs/response.wav
INPUT ?= $(SAMPLE_WAV)
OUTPUT ?= $(OUTPUT_WAV)

# --- Remote GPU (Vast.ai) ---
# Override per-instance: make gpu-ssh GPU_HOST=ssh9.vast.ai GPU_PORT=12345
GPU_HOST ?= ssh7.vast.ai
GPU_PORT ?= 22141
GPU_USER ?= root

help:
	@echo "Yoruba voice query pipeline — make targets"
	@echo ""
	@echo "  make install        pip install -r requirements.txt"
	@echo "  make index          build FAISS index from Wikipedia (M4)"
	@echo "  make sample         download a Yoruba FLEURS sample WAV"
	@echo "  make run            run the pipeline on the FLEURS sample"
	@echo "  make run INPUT=path/to.wav OUTPUT=out.wav"
	@echo "  make talk           push-to-talk conversation loop (M4 = RAG)"
	@echo "  make chat           push-to-talk conversation loop (M4 = free-form Mistral, no retrieval)"
	@echo "  make talk-hf        same as talk, but M1 = HF Whisper-Yoruba fine-tune"
	@echo "  make chat-hf        same as chat, but M1 = HF Whisper-Yoruba fine-tune"
	@echo ""
	@echo "  make wer-mlx [N=50]  WER eval of M1 mlx backend on FLEURS yo_ng"
	@echo "  make wer-hf  [N=50]  WER eval of M1 HF backend on FLEURS yo_ng"
	@echo "  make wer-ft  [N=200] WER eval after fine-tune (requires merged model + config.M1_HF_MODEL updated)"
	@echo ""
	@echo "  make finetune-smoke  Quick LoRA training smoke test (~10 min on any GPU)"
	@echo "  make finetune        Full LoRA fine-tune of whisper-large-v3 on Yoruba — CUDA only"
	@echo "  make merge-lora      Merge LoRA adapter into a plain Whisper checkpoint for inference"
	@echo "  make gpu-ssh         SSH into the rented Vast.ai GPU (override GPU_HOST/GPU_PORT per-instance)"
	@echo ""
	@echo "  make test           run all per-module tests"
	@echo "  make test-m1..m5    run a single module test"
	@echo "  make test-chain     run M1→M3 and M1→M4 chained tests"
	@echo ""
	@echo "  make clean-logs     remove logs/*.jsonl"
	@echo "  make clean-outputs  remove generated WAVs in data/outputs"

install:
	$(PYTHON) -m pip install -r requirements.txt

index:
	$(PYTHON) -m scripts.build_index

sample:
	$(PYTHON) tests/fetch_yoruba_sample.py

run:
	$(PYTHON) pipeline.py $(INPUT) $(OUTPUT)

talk:
	$(PYTHON) pipeline.py --mic

chat:
	$(PYTHON) pipeline.py --mic --chat

talk-hf:
	$(PYTHON) pipeline.py --mic --asr hf

chat-hf:
	$(PYTHON) pipeline.py --mic --chat --asr hf

N ?= 50
SPLIT ?= validation

wer-mlx:
	$(PYTHON) -m scripts.eval_wer --asr mlx --n $(N) --split $(SPLIT) --out logs/wer_mlx.jsonl

wer-hf:
	$(PYTHON) -m scripts.eval_wer --asr hf  --n $(N) --split $(SPLIT) --out logs/wer_hf.jsonl

wer-ft:
	$(PYTHON) -m scripts.eval_wer --asr hf  --n $(N) --split $(SPLIT) --out logs/wer_ft.jsonl

wer: wer-mlx wer-hf

# --- fine-tuning (run on a CUDA box) ---
EPOCHS ?= 3
BATCH ?= 8

finetune-smoke:
	$(PYTHON) -m scripts.finetune_whisper --epochs 1 --max-train-samples 50 --max-eval-samples 10

finetune:
	$(PYTHON) -m scripts.finetune_whisper --epochs $(EPOCHS) --batch-size $(BATCH)

merge-lora:
	$(PYTHON) -m scripts.merge_lora

gpu-ssh:
	ssh -p $(GPU_PORT) $(GPU_USER)@$(GPU_HOST) -L 8080:localhost:8080

test: test-m1 test-m2 test-m3 test-m4 test-m5

test-m1:
	$(PYTHON) -m tests.test_m1

test-m2:
	$(PYTHON) -m tests.test_m2

test-m3:
	$(PYTHON) -m tests.test_m3

test-m4:
	$(PYTHON) -m tests.test_m4

test-m5:
	$(PYTHON) -m tests.test_m5

test-chain:
	$(PYTHON) -m tests.test_chain_m1_m2_m3
	$(PYTHON) -m tests.test_chain_m1_to_m4

clean-logs:
	rm -f logs/*.jsonl

clean-outputs:
	rm -f data/outputs/*.wav
