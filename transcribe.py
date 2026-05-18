#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from core import (
    AUDIO_EXTS,
    format_timestamp,
    load_diarization_pipeline,
    load_whisper,
    transcribe_stream,
    write_outputs,
)

load_dotenv()


def transcribe_file(audio: Path, output_root: Path, whisper_model, language, vad: bool, diar_pipeline) -> None:
    out_dir = output_root / audio.stem
    txt_path = out_dir / f"{audio.stem}.txt"
    srt_path = out_dir / f"{audio.stem}.srt"

    if txt_path.exists() and srt_path.exists():
        print(f"  -> déjà transcrit, on saute ({txt_path})")
        return

    print(f"\n=== {audio.name} ===")
    start = time.time()
    segments = []
    info = None
    for event in transcribe_stream(audio, whisper_model, diar_pipeline=diar_pipeline, language=language, vad=vad):
        t = event["type"]
        if t == "status":
            print(event["message"])
        elif t == "info":
            print(f"Langue: {event['language']} ({event['language_probability']:.2f}) | Durée: {event['duration']:.1f}s")
            info = event
        elif t == "segment":
            speaker = event.get("speaker")
            prefix = f"{speaker} | " if speaker else ""
            print(f"[{event['progress']:5.1f}%] {format_timestamp(event['start'])} {prefix}{event['text']}")
        elif t == "done":
            segments = event["segments"]

    write_outputs(out_dir, audio.stem, segments)
    elapsed = time.time() - start
    speed = (info["duration"] / elapsed) if info and elapsed > 0 else 0
    print(f"Terminé en {elapsed:.1f}s ({speed:.1f}x temps réel) -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcription par lot (input/ -> output/<nom>/).")
    parser.add_argument("-i", "--input-dir", type=Path, default=Path("input"))
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    parser.add_argument("-m", "--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("-l", "--language", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Erreur: dossier d'entrée introuvable: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"Aucun fichier audio trouvé dans {args.input_dir}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diar_pipeline = None
    if args.diarize:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("Erreur: --diarize nécessite un token HuggingFace (voir README).", file=sys.stderr)
            return 1
        print("Chargement du pipeline de diarisation pyannote…")
        diar_pipeline = load_diarization_pipeline(hf_token)

    print(f"{len(files)} fichier(s) à transcrire.")
    print(f"Chargement du modèle Whisper '{args.model}' (device={args.device}, compute={args.compute_type})…")
    whisper_model = load_whisper(args.model, device=args.device, compute_type=args.compute_type)

    for f in files:
        try:
            transcribe_file(f, args.output_dir, whisper_model, args.language, args.vad, diar_pipeline)
        except Exception as e:
            print(f"  !! Erreur sur {f.name}: {e}", file=sys.stderr)

    print("\nTout est fini.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
