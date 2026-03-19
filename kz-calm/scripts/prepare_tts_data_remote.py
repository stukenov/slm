#!/usr/bin/env python3
"""Download KazakhTTS + KazEmoTTS, process, QC, and push to HuggingFace.

Run on a vast.ai instance:
    python prepare_tts_data_remote.py --hf-repo stukenov/kzcalm-tts-kk-v1

Writes processed wav files to disk to avoid OOM, then builds HF Dataset
from file paths using Audio feature.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 24_000
MIN_DURATION = 1.0
MAX_DURATION = 30.0
MIN_SNR_DB = 15.0
CLIP_THRESHOLD = 0.99

WORK_DIR = Path("/root/tts_data_prep")
OUTPUT_AUDIO_DIR = WORK_DIR / "processed_audio"
MANIFEST_PATH = WORK_DIR / "manifest.jsonl"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_audio(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0)


def estimate_snr(audio: np.ndarray) -> float:
    eps = 1e-10
    frame_len = 1024
    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return 0.0
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    powers = np.mean(frames ** 2, axis=1)
    sorted_powers = np.sort(powers)
    noise_floor = np.mean(sorted_powers[: max(1, n_frames // 10)]) + eps
    signal_power = np.mean(powers) + eps
    return 10 * np.log10(signal_power / noise_floor)


def qc_check(audio: np.ndarray, sr: int) -> str | None:
    duration = len(audio) / sr
    if duration < MIN_DURATION:
        return "too_short"
    if duration > MAX_DURATION:
        return "too_long"
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return "silent"
    if peak >= CLIP_THRESHOLD:
        return "clipped"
    snr = estimate_snr(audio)
    if snr < MIN_SNR_DB:
        return "low_snr"
    return None


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text


# ---------------------------------------------------------------------------
# Process wav+txt pair → save to disk
# ---------------------------------------------------------------------------

_counter = 0


def process_and_save(wav_path: Path, text: str, speaker: str, source: str,
                     emotion: str = "neutral", manifest_f=None) -> bool:
    """Read wav, resample, QC, save processed wav to disk, write manifest line."""
    global _counter
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:
        return False

    audio = to_mono(audio)
    audio = resample_audio(audio, sr)

    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95

    reason = qc_check(audio, TARGET_SR)
    if reason:
        return False

    text = normalize_text(text)
    if len(text) < 2:
        return False

    duration = len(audio) / TARGET_SR

    # Save processed audio
    out_name = f"{_counter:08d}.wav"
    out_path = OUTPUT_AUDIO_DIR / out_name
    sf.write(str(out_path), audio, TARGET_SR)
    _counter += 1

    record = {
        "audio": out_name,
        "text": text,
        "speaker_id": speaker,
        "source": source,
        "emotion": emotion,
        "duration": round(duration, 2),
    }
    manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


# ---------------------------------------------------------------------------
# KazakhTTS
# ---------------------------------------------------------------------------

def download_kazakh_tts() -> Path:
    from huggingface_hub import hf_hub_download
    import tarfile

    extract_dir = WORK_DIR / "kazakh_tts"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ISSAI_KazakhTTS.tar.gz (11GB)...")
    tar_path = hf_hub_download("issai/KazakhTTS", "ISSAI_KazakhTTS.tar.gz", repo_type="dataset")
    logger.info("Extracting KazakhTTS...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)
    logger.info("KazakhTTS extracted.")

    parts = [f"ISSAI_KazakhTTS2_by_parts/ISSAI_KazakhTTS2.tar.gz.part{s}"
             for s in ("aa", "ab", "ac", "ad", "ae", "af")]
    part_paths = []
    for part in parts:
        logger.info(f"Downloading {part}...")
        p = hf_hub_download("issai/KazakhTTS", part, repo_type="dataset")
        part_paths.append(p)

    combined = extract_dir / "ISSAI_KazakhTTS2.tar.gz"
    logger.info("Concatenating KazakhTTS2 parts...")
    with open(combined, "wb") as out:
        for pp in part_paths:
            with open(pp, "rb") as inp:
                while True:
                    chunk = inp.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

    logger.info("Extracting KazakhTTS2...")
    with tarfile.open(str(combined), "r:gz") as tf:
        tf.extractall(extract_dir)
    combined.unlink()
    logger.info("KazakhTTS2 extracted.")
    return extract_dir


def _find_audio_transcript_pairs(base_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for d in sorted(base_dir.rglob("Transcripts")):
        if not d.is_dir():
            continue
        parent = d.parent
        audio_dir = parent / "Audios"
        if not audio_dir.exists():
            audio_dir = parent / "Audio"
        if not audio_dir.exists():
            continue
        pairs.append((audio_dir, d))
    return pairs


def process_kazakh_tts(extract_dir: Path, manifest_f):
    accepted = 0
    rejected = 0

    pairs_dirs = _find_audio_transcript_pairs(extract_dir)
    logger.info(f"Found {len(pairs_dirs)} audio+transcript directory pairs")

    for audio_dir, txt_dir in pairs_dirs:
        rel = audio_dir.relative_to(extract_dir)
        parts = rel.parts
        speaker = "unknown"
        for p in parts:
            if re.match(r"^[MF]\d", p):
                speaker = p.split("_")[0]
                break

        wav_files = {f.stem: f for f in audio_dir.glob("*.wav")}
        txt_files = {f.stem: f for f in txt_dir.glob("*.txt")}
        common = sorted(set(wav_files.keys()) & set(txt_files.keys()))
        logger.info(f"[KazakhTTS/{'/'.join(parts)}] {len(common)} pairs")

        for i, stem in enumerate(common):
            if i % 5000 == 0 and i > 0:
                logger.info(f"  [{speaker}] {i}/{len(common)} (total accepted={accepted})")

            text = txt_files[stem].read_text(encoding="utf-8").strip()
            ok = process_and_save(wav_files[stem], text, speaker=speaker,
                                  source="KazakhTTS", manifest_f=manifest_f)
            if ok:
                accepted += 1
            else:
                rejected += 1

    logger.info(f"KazakhTTS: {accepted} accepted, {rejected} rejected")
    return accepted


# ---------------------------------------------------------------------------
# KazEmoTTS
# ---------------------------------------------------------------------------

def download_kazemotts() -> Path:
    from huggingface_hub import hf_hub_download

    extract_dir = WORK_DIR / "kazemotts"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading EmoKaz.zip (9.7GB)...")
    zip_path = hf_hub_download("issai/KazEmoTTS", "EmoKaz.zip", repo_type="dataset")
    logger.info("Extracting KazEmoTTS...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    logger.info("KazEmoTTS extracted.")
    return extract_dir


def process_kazemotts(extract_dir: Path, manifest_f):
    accepted = 0
    rejected = 0

    emokaz_dir = extract_dir / "EmoKaz"
    if not emokaz_dir.exists():
        emokaz_dir = extract_dir

    for speaker_dir in sorted(emokaz_dir.iterdir()):
        if not speaker_dir.is_dir() or speaker_dir.name.startswith("."):
            continue
        if not speaker_dir.name[0].isdigit():
            continue

        speaker_id = speaker_dir.name

        for split_dir in sorted(speaker_dir.iterdir()):
            if not split_dir.is_dir():
                continue

            wav_files = {f.stem: f for f in split_dir.glob("*.wav")}
            txt_files = {f.stem: f for f in split_dir.glob("*.txt")}
            common = sorted(set(wav_files.keys()) & set(txt_files.keys()))
            logger.info(f"[KazEmoTTS/{speaker_id}/{split_dir.name}] {len(common)} pairs")

            for i, stem in enumerate(common):
                if i % 5000 == 0 and i > 0:
                    logger.info(f"  [{speaker_id}/{split_dir.name}] {i}/{len(common)}")

                parts = stem.split("_")
                emotion = parts[1] if len(parts) >= 3 else "unknown"
                if emotion == "fear":
                    emotion = "scared"

                text = txt_files[stem].read_text(encoding="utf-8").strip()
                ok = process_and_save(wav_files[stem], text, speaker=speaker_id,
                                      source="KazEmoTTS", emotion=emotion,
                                      manifest_f=manifest_f)
                if ok:
                    accepted += 1
                else:
                    rejected += 1

    logger.info(f"KazEmoTTS: {accepted} accepted, {rejected} rejected")
    return accepted


# ---------------------------------------------------------------------------
# Build HF Dataset from manifest + audio files
# ---------------------------------------------------------------------------

def build_and_push(hf_repo: str):
    from datasets import Dataset, Audio

    logger.info("Reading manifest...")
    records = []
    with open(MANIFEST_PATH) as f:
        for line in f:
            r = json.loads(line)
            # Convert audio filename to full path
            r["audio"] = str(OUTPUT_AUDIO_DIR / r["audio"])
            records.append(r)

    logger.info(f"Building HF Dataset from {len(records)} records...")
    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    # Stats
    total_hours = sum(r["duration"] for r in records) / 3600
    speakers = set(r["speaker_id"] for r in records)
    emotions = set(r["emotion"] for r in records)
    sources = {}
    for r in records:
        sources[r["source"]] = sources.get(r["source"], 0) + 1

    logger.info("=== Dataset Stats ===")
    logger.info(f"Samples:  {len(records)}")
    logger.info(f"Hours:    {total_hours:.1f}h")
    logger.info(f"Speakers: {speakers}")
    logger.info(f"Emotions: {emotions}")
    logger.info(f"Sources:  {sources}")

    logger.info(f"Pushing to {hf_repo}...")
    ds.push_to_hub(
        hf_repo,
        private=False,
        commit_message=f"KazakhTTS + KazEmoTTS unified ({len(records)} samples, {total_hours:.1f}h)",
    )
    logger.info(f"Pushed to {hf_repo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="Ignored (cloud launcher compat)")
    parser.add_argument("--hf-repo", default="stukenov/kzcalm-tts-kk-v1")
    parser.add_argument("--skip-kazakh-tts", action="store_true")
    parser.add_argument("--skip-kazemotts", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-processing", action="store_true",
                        help="Skip processing, just push existing manifest+audio")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_processing:
        total = 0
        with open(MANIFEST_PATH, "w") as manifest_f:
            if not args.skip_kazakh_tts:
                try:
                    ktt_dir = WORK_DIR / "kazakh_tts"
                    if not args.skip_download or not ktt_dir.exists():
                        ktt_dir = download_kazakh_tts()
                    else:
                        logger.info("Using existing KazakhTTS extraction")
                    n = process_kazakh_tts(ktt_dir, manifest_f)
                    total += n
                except Exception:
                    logger.exception("KazakhTTS failed")

            if not args.skip_kazemotts:
                try:
                    emo_dir = WORK_DIR / "kazemotts"
                    if not args.skip_download or not emo_dir.exists():
                        emo_dir = download_kazemotts()
                    else:
                        logger.info("Using existing KazEmoTTS extraction")
                    n = process_kazemotts(emo_dir, manifest_f)
                    total += n
                except Exception:
                    logger.exception("KazEmoTTS failed")

        if total == 0:
            logger.error("No records! Exiting.")
            sys.exit(1)

        logger.info(f"Total processed: {total}")

    # Push to HF
    build_and_push(args.hf_repo)
    logger.info("Done!")


if __name__ == "__main__":
    main()
