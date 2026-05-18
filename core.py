"""Logique partagée entre la CLI (transcribe.py) et le serveur web (server.py)."""
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".webm", ".aac"}


def format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_whisper(model: str, device: str = "auto", compute_type: str = "int8") -> WhisperModel:
    return WhisperModel(model, device=device, compute_type=compute_type)


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


def _load_audio_for_pyannote(audio_path: Path, sample_rate: int = 16000):
    import torch
    audio = decode_audio(str(audio_path), sampling_rate=sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(audio: Path, pipeline):
    output = pipeline(_load_audio_for_pyannote(audio))
    annotation = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", output)
    )
    turns = []
    raw_labels = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
        if speaker not in raw_labels:
            raw_labels.append(speaker)
    label_map = {raw: f"Speaker {i + 1}" for i, raw in enumerate(raw_labels)}
    return [(s, e, label_map[spk]) for s, e, spk in turns]


def assign_speaker(seg_start: float, seg_end: float, turns) -> Optional[str]:
    if not turns:
        return None
    best_speaker = "Unknown"
    best_overlap = 0.0
    for t_start, t_end, speaker in turns:
        overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def pick_speaker_samples(turns, min_duration: float = 2.0, max_duration: float = 5.0):
    """Pour chaque speaker, retourne (start, end) d'un turn représentatif."""
    samples = {}
    for t_start, t_end, speaker in turns:
        if speaker in samples:
            continue
        if (t_end - t_start) >= min_duration:
            samples[speaker] = (t_start, min(t_end, t_start + max_duration))
    for t_start, t_end, speaker in turns:
        if speaker not in samples:
            samples[speaker] = (t_start, min(t_end, t_start + max_duration))
    return samples


def extract_audio_segment(audio_path: Path, start: float, end: float, sample_rate: int = 16000) -> np.ndarray:
    audio = decode_audio(str(audio_path), sampling_rate=sample_rate)
    return audio[int(start * sample_rate): int(end * sample_rate)]


def write_outputs(out_dir: Path, stem: str, segments, name_map: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{stem}.txt"
    srt_path = out_dir / f"{stem}.srt"
    name_map = name_map or {}

    current_speaker = None
    with txt_path.open("w", encoding="utf-8") as txt_f, srt_path.open("w", encoding="utf-8") as srt_f:
        for i, seg in enumerate(segments, start=1):
            text = seg["text"]
            speaker = seg.get("speaker")
            display = name_map.get(speaker, speaker) if speaker else None

            if display:
                if display != current_speaker:
                    if current_speaker is not None:
                        txt_f.write("\n")
                    txt_f.write(f"{display}:\n")
                    current_speaker = display
                txt_f.write(f"  {text}\n")
                srt_text = f"[{display}] {text}"
            else:
                txt_f.write(text + "\n")
                srt_text = text

            srt_f.write(f"{i}\n")
            srt_f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            srt_f.write(f"{srt_text}\n\n")
    return txt_path, srt_path


def transcribe_stream(
    audio_path: Path,
    whisper_model: WhisperModel,
    diar_pipeline=None,
    language: Optional[str] = None,
    vad: bool = False,
) -> Iterator[dict]:
    """Génère des events :
        {type: 'status', message}
        {type: 'info', language, language_probability, duration, n_speakers}
        {type: 'segment', index, start, end, text, speaker, progress}
        {type: 'done', segments, speakers: [{label, sample_start, sample_end}]}
    """
    turns = []
    n_speakers = 0
    if diar_pipeline is not None:
        yield {"type": "status", "message": "Diarisation en cours…"}
        turns = diarize(audio_path, diar_pipeline)
        n_speakers = len({spk for _, _, spk in turns})
        yield {"type": "status", "message": f"Diarisation terminée — {n_speakers} locuteur(s) détecté(s)"}

    yield {"type": "status", "message": "Transcription en cours…"}
    segments_iter, info = whisper_model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=vad,
        beam_size=5,
    )
    yield {
        "type": "info",
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "n_speakers": n_speakers,
    }

    all_segments = []
    for i, seg in enumerate(segments_iter, start=1):
        text = seg.text.strip()
        speaker = assign_speaker(seg.start, seg.end, turns) if turns else None
        entry = {
            "index": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
            "speaker": speaker,
        }
        all_segments.append(entry)
        progress = min(100.0, seg.end / info.duration * 100) if info.duration else 0
        yield {"type": "segment", **entry, "progress": progress}

    speakers_list = []
    if turns:
        sample_ranges = pick_speaker_samples(turns)
        seen = []
        for _, _, spk in turns:
            if spk not in seen:
                seen.append(spk)
        speakers_list = [
            {"label": spk, "sample_start": sample_ranges[spk][0], "sample_end": sample_ranges[spk][1]}
            for spk in seen
        ]

    yield {"type": "done", "segments": all_segments, "speakers": speakers_list}
