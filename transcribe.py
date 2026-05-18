#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

load_dotenv()

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".aac"}


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_diarization_pipeline(hf_token: str):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    try:
        import torch
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
        elif torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass
    return pipeline


def load_audio_for_pyannote(audio_path: Path, sample_rate: int = 16000):
    import torch
    audio = decode_audio(str(audio_path), sampling_rate=sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(audio: Path, pipeline):
    audio_input = load_audio_for_pyannote(audio)
    diarization = pipeline(audio_input)
    turns = []
    raw_labels = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
        if speaker not in raw_labels:
            raw_labels.append(speaker)
    label_map = {raw: f"Speaker {i + 1}" for i, raw in enumerate(raw_labels)}
    return [(s, e, label_map[spk]) for s, e, spk in turns], len(raw_labels)


def assign_speaker(seg_start: float, seg_end: float, turns) -> str:
    best_speaker = "Unknown"
    best_overlap = 0.0
    for t_start, t_end, speaker in turns:
        overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def transcribe_file(audio: Path, output_root: Path, model: WhisperModel, language, vad: bool, diar_pipeline):
    out_dir = output_root / audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{audio.stem}.txt"
    srt_path = out_dir / f"{audio.stem}.srt"

    if txt_path.exists() and srt_path.exists():
        print(f"  -> déjà transcrit, on saute ({txt_path})")
        return

    print(f"\n=== {audio.name} ===")
    start = time.time()

    turns = []
    if diar_pipeline is not None:
        print("Diarisation en cours...")
        t0 = time.time()
        turns, n_speakers = diarize(audio, diar_pipeline)
        print(f"Diarisation terminée en {time.time() - t0:.1f}s — {n_speakers} locuteur(s) détecté(s)")

    print("Transcription...")
    segments, info = model.transcribe(
        str(audio),
        language=language,
        vad_filter=vad,
        beam_size=5,
    )
    print(f"Langue: {info.language} ({info.language_probability:.2f}) | Durée: {info.duration:.1f}s")

    current_speaker = None
    with txt_path.open("w", encoding="utf-8") as txt_f, srt_path.open("w", encoding="utf-8") as srt_f:
        for i, seg in enumerate(segments, start=1):
            text = seg.text.strip()
            speaker = assign_speaker(seg.start, seg.end, turns) if turns else None

            if speaker is not None:
                if speaker != current_speaker:
                    if current_speaker is not None:
                        txt_f.write("\n")
                    txt_f.write(f"{speaker}:\n")
                    current_speaker = speaker
                txt_f.write(f"  {text}\n")
                srt_text = f"[{speaker}] {text}"
            else:
                txt_f.write(text + "\n")
                srt_text = text

            srt_f.write(f"{i}\n")
            srt_f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            srt_f.write(f"{srt_text}\n\n")

            pct = min(100.0, seg.end / info.duration * 100) if info.duration else 0
            prefix = f"{speaker} | " if speaker else ""
            print(f"[{pct:5.1f}%] {format_timestamp(seg.start)} {prefix}{text}")

    elapsed = time.time() - start
    speed = info.duration / elapsed if elapsed > 0 else 0
    print(f"Terminé en {elapsed:.1f}s ({speed:.1f}x temps réel) -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcription par lot avec faster-whisper (input/ -> output/<nom>/)."
    )
    parser.add_argument("-i", "--input-dir", type=Path, default=Path("input"), help="Dossier d'entrée (défaut: input/)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"), help="Dossier de sortie (défaut: output/)")
    parser.add_argument("-m", "--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Modèle Whisper (défaut: small)")
    parser.add_argument("-l", "--language", default=None, help="Code langue (ex: fr, en). Auto-détection si omis.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Périphérique d'inférence Whisper (défaut: auto)")
    parser.add_argument("--compute-type", default="int8", help="Type de calcul Whisper (défaut: int8)")
    parser.add_argument("--vad", action="store_true", help="Active le filtre VAD (saute les silences)")
    parser.add_argument("--diarize", action="store_true", help="Active la diarisation (Speaker 1, Speaker 2, ...)")
    parser.add_argument("--hf-token", default=None, help="Token HuggingFace pour pyannote (sinon variable d'env HF_TOKEN)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Erreur: dossier d'entrée introuvable: {args.input_dir}", file=sys.stderr)
        return 1

    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        print(f"Aucun fichier audio trouvé dans {args.input_dir} (extensions: {', '.join(sorted(AUDIO_EXTS))})")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    diar_pipeline = None
    if args.diarize:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            print(
                "Erreur: --diarize nécessite un token HuggingFace.\n"
                "  1. Crée un compte sur https://huggingface.co\n"
                "  2. Accepte les conditions sur les trois modèles :\n"
                "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "     - https://huggingface.co/pyannote/segmentation-3.0\n"
                "     - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "  3. Génère un token (Read) sur https://huggingface.co/settings/tokens\n"
                "  4. Passe-le via --hf-token TOKEN, .env (HF_TOKEN=...), ou export HF_TOKEN=TOKEN",
                file=sys.stderr,
            )
            return 1
        print("Chargement du pipeline de diarisation pyannote...")
        diar_pipeline = load_diarization_pipeline(hf_token)

    print(f"{len(files)} fichier(s) à transcrire.")
    print(f"Chargement du modèle Whisper '{args.model}' (device={args.device}, compute={args.compute_type})...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    for f in files:
        try:
            transcribe_file(f, args.output_dir, model, args.language, args.vad, diar_pipeline)
        except Exception as e:
            print(f"  !! Erreur sur {f.name}: {e}", file=sys.stderr)

    print("\nTout est fini.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
