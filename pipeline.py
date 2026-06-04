"""End-to-end Yoruba voice query pipeline (M1 -> M5).

Usage:
    python pipeline.py <input.wav> [output.wav]
    python pipeline.py --mic                 # conversation loop (Ctrl+C to quit)
    python pipeline.py --mic --chat          # free-form chat instead of RAG
    python pipeline.py <input.wav> --chat
"""
import signal
import sys
from pathlib import Path

import config
from modules.m1_asr import M1ASR
from modules.m1_asr_hf import M1ASRHF
from modules.m2_diacritic import M2Diacritic
from modules.m3_translate import M3Translate
from modules.m4_chat import M4Chat
from modules.m4_rag import M4RAG
from modules.m5_tts import M5TTS
from utils.logging import get_logger, log_stage

log = get_logger("pipeline")


class YorubaPipeline:
    """Owns all five modules. initialize() loads everything once; run() executes
    one audio-in / audio-out turn and writes per-stage JSONL logs."""

    def __init__(self, chat_mode: bool = False, asr: str = "mlx") -> None:
        self.m1 = M1ASRHF() if asr == "hf" else M1ASR()
        self.m2 = M2Diacritic()
        self.m3 = M3Translate()
        self.m4 = M4Chat() if chat_mode else M4RAG()
        self.m5 = M5TTS()

    def initialize(self) -> None:
        log.info("warming up pipeline (loads M1..M5 weights)…")
        self.m1.initialize()
        self.m2.initialize()
        self.m3.initialize()
        self.m4.initialize()
        self.m5.initialize()
        log.info("pipeline ready")

    def run(self, audio_in: str | Path, audio_out: str | Path) -> dict:
        run_id = f"run-{Path(audio_in).stem}"
        r1 = self.m1.process(audio_in);                 log_stage("M1", run_id, r1)
        r2 = self.m2.process(r1["text"]);               log_stage("M2", run_id, r2)
        r3 = self.m3.process(r2["text"]);               log_stage("M3", run_id, r3)
        r4 = self.m4.process(r3["text"]);               log_stage("M4", run_id, r4)
        r5 = self.m5.process(r4["answer"], output_path=audio_out)
        log_stage("M5", run_id, r5)
        return {"m1": r1, "m2": r2, "m3": r3, "m4": r4, "m5": r5, "run_id": run_id}


def _print_summary(res: dict) -> None:
    print("\n=== M1 raw YO    ===\n", res["m1"]["text"])
    print("\n=== M2 diacrit.  ===\n", res["m2"]["text"])
    print("\n=== M3 EN query  ===\n", res["m3"]["text"])
    print(f"\n=== M4 EN answer (max_sim={res['m4']['max_sim']:.3f}) ===\n", res["m4"]["answer"])
    print("\n=== M5 YO answer ===\n", res["m5"]["yo_diacritized"])
    print(f"\nWAV: {res['m5']['audio_path']}  ({res['m5']['duration_s']:.2f}s)")
    print(f"log: logs/{res['run_id']}.jsonl")


def _run_mic_loop(pipe: "YorubaPipeline") -> None:
    """Conversation loop: record → run → play → wait for next turn."""
    from utils.mic import play_wav, record_push_to_talk

    mic_wav = config.AUDIO_DIR / "mic_capture.wav"
    turn = 0
    print("\nConversation mode. Ctrl+C to quit.\n")
    while True:
        turn += 1
        print(f"--- turn {turn} ---")
        try:
            record_push_to_talk(mic_wav)
        except KeyboardInterrupt:
            print("\nbye.")
            return

        audio_out = config.OUTPUT_DIR / f"response_{turn:03d}.wav"
        try:
            res = pipe.run(mic_wav, audio_out)
        except Exception as e:
            log.error("pipeline failed on turn %d: %s", turn, e)
            continue

        _print_summary(res)
        play_wav(res["m5"]["audio_path"])
        print()


def _parse_asr_flag(args: list[str]) -> tuple[str, list[str]]:
    """Pull `--asr <mlx|hf>` (or `--asr=hf`) out of argv; default 'mlx'."""
    asr = "mlx"
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--asr" and i + 1 < len(args):
            asr = args[i + 1]; i += 2; continue
        if a.startswith("--asr="):
            asr = a.split("=", 1)[1]; i += 1; continue
        out.append(a); i += 1
    if asr not in {"mlx", "hf"}:
        print(f"unknown --asr value: {asr!r} (expected 'mlx' or 'hf')")
        sys.exit(2)
    return asr, out


def main() -> None:
    args = sys.argv[1:]
    chat_mode = "--chat" in args
    args = [a for a in args if a != "--chat"]
    asr, args = _parse_asr_flag(args)

    if not args:
        print("usage: python pipeline.py <input.wav> [output.wav] [--chat] [--asr mlx|hf]")
        print("       python pipeline.py --mic [--chat] [--asr mlx|hf]")
        sys.exit(1)

    mic_mode = args[0] == "--mic"

    pipe = YorubaPipeline(chat_mode=chat_mode, asr=asr)
    pipe.initialize()
    log.info("M1 backend: %s | M4 mode: %s",
             asr, "chat (no retrieval)" if chat_mode else "RAG")

    if mic_mode:
        # Re-assert the default SIGINT handler — sounddevice/PortAudio installs
        # its own on macOS that can swallow the first Ctrl+C.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            _run_mic_loop(pipe)
        except KeyboardInterrupt:
            print("\nbye.")
        return

    audio_in = args[0]
    audio_out = args[1] if len(args) >= 2 else str(config.OUTPUT_DIR / "response.wav")
    res = pipe.run(audio_in, audio_out)
    _print_summary(res)


if __name__ == "__main__":
    main()
