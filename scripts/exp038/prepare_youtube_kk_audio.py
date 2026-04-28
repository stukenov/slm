#!/usr/bin/env python3
"""exp038: bootstrap recent Kazakh speech chunks from YouTube news channels.

Pipeline:
  1. Inventory recent videos with yt-dlp in flat-playlist mode.
  2. Filter/rank by title, recency, duration, and per-channel quotas.
  3. Download only selected audio.
  4. Optionally isolate vocals with Demucs.
  5. Gate language with faster-whisper on a short preview.
  6. Split speech with Silero VAD into <=30s chunks.
  7. Upload chunk shards to a Hugging Face dataset repo.

This script is intentionally recent-first and quota-based. It should never
be used to crawl whole channels end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, create_repo, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from slm.utils import load_config

logger = logging.getLogger("exp038")

KAZAKH_CHARS = set("әғқңөұүһі")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recent-first YouTube Kazakh speech bootstrap")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument(
        "--step",
        choices=["inventory", "process", "all"],
        default="inventory",
        help="Which stage to run",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional selected manifest path override",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan work without downloading or uploading audio",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Process only N videos and save intermediates to data/exp038_review/ for manual inspection",
    )
    parser.add_argument(
        "--refresh-inventory",
        action="store_true",
        help="Force re-crawl channel metadata even if HF cache is fresh",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker index for parallel runs — isolates HF paths (shards/wN, state/wN)",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def resolve_yt_dlp_command() -> list[str]:
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp:
        return [yt_dlp]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "yt-dlp", "yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


def _hf_retry(fn, *, max_retries: int = 5):
    """Call fn() with exponential backoff on HF 429 rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = 60 * (2 ** attempt)  # 60, 120, 240, 480s
                logger.warning("HF rate limit (attempt %d/%d), retrying in %ds: %s", attempt + 1, max_retries, wait, exc)
                time.sleep(wait)
            else:
                raise


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=capture,
    )


def load_audio_config(config_path: str) -> tuple[dict, dict]:
    config = load_config(config_path)
    audio_cfg = config.get("audio_collection")
    if not audio_cfg:
        raise ValueError("Config must define top-level 'audio_collection'")
    return config, audio_cfg


def get_hf_token(required: bool = False) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
    if required and not token:
        raise RuntimeError("HF_TOKEN is required")
    return token or ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    normalized = normalize_text(text)
    hits: list[str] = []
    for kw in keywords:
        if not kw:
            continue
        pattern = re.compile(rf"(?<![0-9a-zа-яёәғқңөұүһі]){re.escape(kw.lower())}(?![0-9a-zа-яёәғқңөұүһі])")
        if pattern.search(normalized):
            hits.append(kw)
    return hits


def parse_upload_date(value: str | None) -> datetime | None:
    if not value or len(value) != 8:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def days_old(upload_date: str | None) -> int | None:
    parsed = parse_upload_date(upload_date)
    if parsed is None:
        return None
    now = datetime.now(timezone.utc)
    return (now.date() - parsed.date()).days


def inventory_channel(channel_cfg: dict, output_dir: Path, *, cache_days: int = 0, force_refresh: bool = False) -> list[dict]:
    out_path = output_dir / f"{channel_cfg['name']}_inventory.jsonl"

    if not force_refresh and cache_days > 0 and out_path.exists():
        age_s = (datetime.now(timezone.utc).timestamp() - out_path.stat().st_mtime)
        age_d = age_s / 86400
        if age_d < cache_days:
            logger.info(
                "Using cached inventory for %s (%.1f days old, limit %d days)",
                channel_cfg["name"], age_d, cache_days,
            )
            return [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    yt_dlp_cmd = resolve_yt_dlp_command()
    raw_path = output_dir / f"{channel_cfg['name']}_inventory_raw.json"
    cmd = yt_dlp_cmd + ["--flat-playlist"]
    if channel_cfg.get("playlist_end"):
        cmd += ["--playlist-end", str(channel_cfg["playlist_end"])]
    cmd += ["--dump-single-json", channel_cfg["url"]]
    result = run(cmd, capture=True)
    raw_path.write_text(result.stdout, encoding="utf-8")

    payload = json.loads(result.stdout)
    entries = payload.get("entries") or []
    inventory: list[dict] = []
    for order_index, entry in enumerate(entries, start=1):
        title = (entry.get("title") or "").strip()
        duration = entry.get("duration")
        live_status = entry.get("live_status")
        uploaded = entry.get("upload_date")
        url = entry.get("url")
        if url and not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={entry['id']}"

        allow_hits = keyword_hits(title, channel_cfg.get("require_any_title_keywords", []))
        deny_hits = keyword_hits(title, channel_cfg.get("deny_title_keywords", []))
        age_days = days_old(uploaded)

        reasons: list[str] = []
        if live_status and live_status != "not_live":
            reasons.append(f"live_status:{live_status}")
        if duration is None:
            reasons.append("missing_duration")
        else:
            min_dur = channel_cfg.get("min_duration_seconds")
            max_dur = channel_cfg.get("max_duration_seconds")
            if min_dur is not None and duration < min_dur:
                reasons.append("too_short")
            elif max_dur is not None and duration > max_dur:
                reasons.append("too_long")
        recency_days = channel_cfg.get("recency_days")
        if recency_days is not None and age_days is not None and age_days > recency_days:
            reasons.append("too_old")
        if deny_hits:
            reasons.append("deny_keyword")
        if channel_cfg.get("require_any_title_keywords") and not allow_hits:
            reasons.append("missing_allow_keyword")

        score = 0.0
        if age_days is not None:
            score += max(0.0, 120.0 - age_days)
        else:
            playlist_end = channel_cfg.get("playlist_end", 10000)
            score += max(0.0, playlist_end - order_index) / 10.0
        if duration:
            score += min(float(duration), 1800.0) / 60.0
        score += len(allow_hits) * 20.0
        score -= len(deny_hits) * 50.0
        if live_status == "not_live":
            score += 5.0

        inventory.append(
            {
                "channel": channel_cfg["name"],
                "channel_url": channel_cfg["url"],
                "video_id": entry.get("id"),
                "title": title,
                "playlist_order": order_index,
                "duration_seconds": duration,
                "upload_date": uploaded,
                "age_days": age_days,
                "live_status": live_status,
                "webpage_url": url or f"https://www.youtube.com/watch?v={entry.get('id')}",
                "allow_hits": allow_hits,
                "deny_hits": deny_hits,
                "score": round(score, 3),
                "status": "candidate" if not reasons else "rejected",
                "reasons": reasons,
            }
        )

    out_path = output_dir / f"{channel_cfg['name']}_inventory.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in inventory:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return inventory


def select_videos(audio_cfg: dict, inventory_rows: list[dict], output_dir: Path) -> list[dict]:
    total_candidates = sum(1 for r in inventory_rows if r["status"] == "candidate")
    target_video_count = audio_cfg.get("target_video_count", total_candidates)
    by_channel: dict[str, list[dict]] = {}
    channel_cfg_map = {channel["name"]: channel for channel in audio_cfg["channels"]}
    ordered_channels = sorted(audio_cfg["channels"], key=lambda item: item["priority"])
    primary_channel_name = ordered_channels[0]["name"]
    for row in inventory_rows:
        if row["status"] != "candidate":
            continue
        by_channel.setdefault(row["channel"], []).append(row)

    for rows in by_channel.values():
        rows.sort(key=lambda item: (item["score"], item.get("upload_date") or ""), reverse=True)

    selected: list[dict] = []
    counts: Counter[str] = Counter()

    for channel_name, channel_cfg in sorted(channel_cfg_map.items(), key=lambda item: item[1]["priority"]):
        if len(selected) >= target_video_count:
            break
        max_selected = min(channel_cfg.get("max_selected", len(by_channel.get(channel_name, []))), len(by_channel.get(channel_name, [])))
        channel_cap = min(max_selected, math.floor(target_video_count * channel_cfg["max_share"]))
        if channel_cap <= 0 and max_selected > 0:
            channel_cap = 1
        if channel_name != primary_channel_name:
            primary_count = counts[primary_channel_name]
            if primary_count <= 0:
                channel_cap = 0
            else:
                ratio_cap = math.floor(primary_count * channel_cfg["max_share"] / (1.0 - channel_cfg["max_share"]))
                channel_cap = min(channel_cap, ratio_cap)
        remaining_slots = target_video_count - len(selected)
        for row in by_channel.get(channel_name, [])[:min(channel_cap, remaining_slots)]:
            row = dict(row)
            row["selection_reason"] = "per_channel_cap"
            selected.append(row)
            counts[channel_name] += 1

    if len(selected) < target_video_count:
        used_ids = {row["video_id"] for row in selected}
        leftovers = sorted(
            (
                row
                for row in inventory_rows
                if row["status"] == "candidate" and row["video_id"] not in used_ids
            ),
            key=lambda item: (item["score"], item.get("upload_date") or ""),
            reverse=True,
        )
        for row in leftovers:
            if len(selected) >= target_video_count:
                break
            channel_cfg = channel_cfg_map[row["channel"]]
            max_share_count = max(1, math.floor(target_video_count * channel_cfg["max_share"]))
            if counts[row["channel"]] >= max_share_count:
                continue
            if row["channel"] != primary_channel_name:
                primary_count = counts[primary_channel_name]
                ratio_cap = math.floor(primary_count * channel_cfg["max_share"] / (1.0 - channel_cfg["max_share"]))
                if counts[row["channel"]] >= ratio_cap:
                    continue
            row = dict(row)
            row["selection_reason"] = "best_remaining"
            selected.append(row)
            counts[row["channel"]] += 1

    selected = selected[:target_video_count]
    selected.sort(key=lambda item: (item["channel"], item["score"]), reverse=True)
    manifest_path = output_dir / audio_cfg["selected_manifest"]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "target_video_count": target_video_count,
        "selected_video_count": len(selected),
        "selected_by_channel": dict(counts),
        "rejected_by_reason": dict(
            Counter(
                reason
                for row in inventory_rows
                for reason in row.get("reasons", [])
            )
        ),
    }
    (output_dir / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Selected %d videos: %s", len(selected), dict(counts))
    return selected


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")


def _cookies_opt() -> list[str]:
    """Return yt-dlp cookies flag if a cookies file is present.

    yt-dlp rewrites the cookies file on each run, stripping essential cookies
    (SID, LOGIN_INFO, etc.).  We keep a read-only backup and restore it before
    every download so the full cookie set is always available.
    """
    backup = Path("/workspace/cookies_original.txt")
    target = Path("/workspace/cookies.txt")
    if backup.exists():
        shutil.copy2(backup, target)
        return ["--cookies", str(target)]
    for candidate in [target, Path.home() / "cookies.txt", Path("cookies.txt")]:
        if candidate.exists():
            return ["--cookies", str(candidate)]
    return []


def download_audio(video: dict, raw_dir: Path, *, retries: int = 3) -> Path:
    ensure_ffmpeg()
    yt_dlp_cmd = resolve_yt_dlp_command()
    channel_dir = raw_dir / video["channel"]
    channel_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{video.get('upload_date') or 'unknown'}_{video['video_id']}"
    wav_path = channel_dir / f"{stem}.wav"
    if wav_path.exists():
        return wav_path

    cookies_opt = _cookies_opt()
    with TemporaryDirectory(prefix="exp038_dl_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        prefix = tmpdir / stem
        download_cmd = yt_dlp_cmd + cookies_opt + [
            "-f",
            "bestaudio/best",
            "--sleep-interval", "1",
            "--max-sleep-interval", "3",
            "--concurrent-fragments", "10",
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "-o",
            str(prefix) + ".%(ext)s",
            video["webpage_url"],
        ]
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                run(download_cmd)
                last_exc = None
                break
            except subprocess.CalledProcessError as exc:
                last_exc = exc
                wait = 30 * (2 ** attempt)
                logger.warning("Download failed (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, retries, wait, video["video_id"])
                import time; time.sleep(wait)
        if last_exc is not None:
            raise last_exc
        downloaded = sorted(tmpdir.glob(f"{stem}.*"))
        if not downloaded:
            raise RuntimeError(f"yt-dlp produced no files for {video['webpage_url']}")
        source_audio = downloaded[0]
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_audio),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
        run(ffmpeg_cmd)
    return wav_path


def maybe_extract_voice(audio_cfg: dict, input_path: Path, work_dir: Path) -> Path:
    backend = audio_cfg["processing"]["separator_backend"]
    if backend == "none":
        return input_path

    if backend != "demucs":
        raise ValueError(f"Unsupported separator backend: {backend}")

    output_root = work_dir / "demucs"
    model_name = audio_cfg["processing"]["separator_model"]
    expected = output_root / model_name / input_path.stem / "vocals.wav"
    normalized = work_dir / "speech_only" / f"{input_path.stem}_vocals.wav"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    if normalized.exists():
        return normalized

    demucs_cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems=vocals",
        "-n",
        model_name,
        "--device",
        "cuda",
        "--out",
        str(output_root),
        str(input_path),
    ]
    try:
        run(demucs_cmd)
    except Exception:
        if audio_cfg["processing"].get("separator_required", False):
            raise
        logger.warning("Demucs failed for %s, falling back to raw audio", input_path.name)
        return input_path

    if not expected.exists():
        if audio_cfg["processing"].get("separator_required", False):
            raise RuntimeError(f"Demucs output missing: {expected}")
        logger.warning("Demucs output missing for %s, falling back to raw audio", input_path.name)
        return input_path

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(expected),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(normalized),
    ]
    run(ffmpeg_cmd)
    return normalized


def maybe_enhance_voice(audio_cfg: dict, input_path: Path, work_dir: Path) -> Path:
    backend = audio_cfg["processing"].get("enhancer_backend", "none")
    if backend == "none" or not backend:
        return input_path

    if backend != "noisereduce":
        logger.warning("Unknown enhancer_backend %r, skipping", backend)
        return input_path

    out_dir = work_dir / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_enhanced.wav"
    if out_path.exists():
        return out_path

    try:
        import noisereduce as nr
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(input_path))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        reduced = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.85)
        sf.write(str(out_path), reduced.astype(np.float32), sr)
        return out_path
    except Exception as exc:
        logger.warning("noisereduce failed for %s: %s — using raw voice audio", input_path.name, exc)
        return input_path


class WhisperLanguageGate:
    def __init__(self, processing_cfg: dict) -> None:
        self.processing_cfg = processing_cfg
        self.model = None

    def _load(self):
        if self.model is not None:
            return self.model
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.processing_cfg["language_model"],
            device=self.processing_cfg["language_device"],
            compute_type=self.processing_cfg["language_compute_type"],
        )
        return self.model

    def check(self, audio_path: Path) -> dict:
        preview_path = audio_path.with_suffix(".preview.wav")
        skip_s = self.processing_cfg.get("language_preview_skip_seconds", 0)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(skip_s),
            "-t", str(self.processing_cfg["language_preview_seconds"]),
            "-ac", "1",
            "-ar", "16000",
            str(preview_path),
        ]
        run(ffmpeg_cmd)
        model = self._load()
        segments, info = model.transcribe(
            str(preview_path),
            beam_size=1,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        transcript_lower = transcript.lower()
        kk_chars = sum(1 for char in transcript_lower if char in KAZAKH_CHARS)
        alpha_chars = sum(1 for char in transcript_lower if char.isalpha())
        kk_ratio = kk_chars / alpha_chars if alpha_chars else 0.0
        accepted = (
            info.language in set(self.processing_cfg["accepted_languages"])
            and info.language_probability >= self.processing_cfg["min_language_probability"]
            and len(transcript) >= self.processing_cfg["min_transcript_chars"]
            and kk_ratio >= self.processing_cfg["min_kazakh_chars_ratio"]
        )
        return {
            "accepted": accepted,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "transcript_preview": transcript[:500],
            "kazakh_char_ratio": round(kk_ratio, 4),
        }


class SileroChunker:
    def __init__(self, processing_cfg: dict) -> None:
        self.processing_cfg = processing_cfg
        self.model = None

    def _load(self):
        if self.model is not None:
            return self.model
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        return self.model

    def chunk(self, audio_path: Path) -> list[tuple[float, float]]:
        import numpy as np
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps

        model = self._load()
        audio, sr = sf.read(str(audio_path))
        if sr != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sr}")
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        wav = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=self.processing_cfg["threshold"],
            min_silence_duration_ms=self.processing_cfg["min_silence_duration_ms"],
            speech_pad_ms=self.processing_cfg["speech_pad_ms"],
        )
        segments = [(ts["start"] / 16000.0, ts["end"] / 16000.0) for ts in timestamps]
        return merge_segments(
            segments,
            max_gap=self.processing_cfg["max_merge_gap_seconds"],
            min_len=self.processing_cfg["min_chunk_seconds"],
            max_len=self.processing_cfg["max_chunk_seconds"],
        )


def merge_segments(
    segments: list[tuple[float, float]],
    *,
    max_gap: float,
    min_len: float,
    max_len: float,
) -> list[tuple[float, float]]:
    if not segments:
        return []

    merged: list[tuple[float, float]] = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start - cur_end <= max_gap and end - cur_start <= max_len:
            cur_end = end
            continue
        merged.extend(split_segment(cur_start, cur_end, min_len=min_len, max_len=max_len))
        cur_start, cur_end = start, end
    merged.extend(split_segment(cur_start, cur_end, min_len=min_len, max_len=max_len))
    return merged


def split_segment(start: float, end: float, *, min_len: float, max_len: float) -> list[tuple[float, float]]:
    duration = end - start
    if duration < min_len:
        return []
    if duration <= max_len:
        return [(start, end)]
    chunks: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + max_len, end)
        if chunk_end - cursor >= min_len:
            chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


class ChunkUploader:
    def __init__(
        self,
        upload_cfg: dict,
        base_dir: Path,
        *,
        api: HfApi,
        token: str,
        start_shard_index: int = 0,
        start_chunk_index: int = 0,
    ) -> None:
        self.upload_cfg = upload_cfg
        self.base_dir = base_dir
        self.queue_dir = base_dir / "hf_queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.api = api
        self.shard_index = start_shard_index
        self.chunk_index = start_chunk_index
        self.chunk_count = 0
        self.current_shard_dir = self._new_shard_dir()
        self.metadata_handle = (self.current_shard_dir / "metadata.jsonl").open("a", encoding="utf-8")

        self.token = token

        create_repo(
            self.upload_cfg["repo_id"],
            repo_type=self.upload_cfg["repo_type"],
            private=self.upload_cfg["private"],
            exist_ok=True,
            token=self.token,
        )

    def _new_shard_dir(self) -> Path:
        shard_dir = self.queue_dir / f"shard-{self.shard_index:05d}"
        (shard_dir / "audio").mkdir(parents=True, exist_ok=True)
        return shard_dir

    def next_chunk_path(self, video: dict) -> tuple[Path, str]:
        filename = f"{video['channel']}_{video['video_id']}_{self.chunk_index:06d}.wav"
        self.chunk_index += 1
        return self.current_shard_dir / "audio" / filename, filename

    def add_metadata(self, record: dict) -> None:
        self.metadata_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.metadata_handle.flush()
        self.chunk_count += 1
        if self.chunk_count >= self.upload_cfg["shard_size_chunks"]:
            self.flush()

    def flush(self) -> bool:
        if self.chunk_count == 0:
            return False
        self.metadata_handle.close()
        shard_name = self.current_shard_dir.name
        logger.info("Uploading shard %s (%d chunks)", shard_name, self.chunk_count)
        _hf_retry(lambda: self.api.upload_folder(
            folder_path=str(self.current_shard_dir),
            repo_id=self.upload_cfg["repo_id"],
            repo_type=self.upload_cfg["repo_type"],
            path_in_repo=f"{self.upload_cfg['path_in_repo']}/{shard_name}",
            token=self.token,
            commit_message=f"exp038: add {shard_name}",
        ))
        self.shard_index += 1
        self.chunk_count = 0
        self.current_shard_dir = self._new_shard_dir()
        self.metadata_handle = (self.current_shard_dir / "metadata.jsonl").open("a", encoding="utf-8")
        return True

    def close(self) -> None:
        self.flush()
        self.metadata_handle.close()


class HubState:
    def __init__(self, upload_cfg: dict, base_dir: Path, *, target_duration_hours: float) -> None:
        self.upload_cfg = upload_cfg
        self.base_dir = base_dir
        self.target_duration_hours = target_duration_hours
        self.state_dir = base_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.token = get_hf_token(required=False)
        self.api = HfApi(token=self.token) if self.token else None
        self.remote_enabled = bool(upload_cfg["enabled"] and self.token and self.api)
        self.seen_video_ids: set[str] = set()
        self.processed_video_ids: set[str] = set()
        self.rejected_video_ids: set[str] = set()
        self.selected_video_ids: set[str] = set()
        self.status_path = self.state_dir / "status.json"
        self.processed_path = self.state_dir / "processed_videos.jsonl"
        self.rejected_path = self.state_dir / "rejected_videos.jsonl"
        self.selected_path = self.state_dir / "selected_videos.jsonl"
        self.stats = {
            "selected_videos": 0,
            "processed_videos": 0,
            "rejected_videos": 0,
            "uploaded_chunks": 0,
            "uploaded_shards": 0,
            "total_duration_s": 0.0,
        }
        if self.remote_enabled:
            try:
                create_repo(
                    upload_cfg["repo_id"],
                    repo_type=upload_cfg["repo_type"],
                    private=upload_cfg["private"],
                    exist_ok=True,
                    token=self.token,
                )
                if upload_cfg.get("sync_state_from_hub"):
                    self._sync_from_hub()
            except Exception as exc:
                logger.warning("HF remote state disabled: %s", exc)
                self.remote_enabled = False
        self._load_local_state()
        self.write_status(reason="state_initialized")

    def _sync_from_hub(self) -> None:
        assert self.token
        try:
            remote_dir = Path(
                snapshot_download(
                    repo_id=self.upload_cfg["repo_id"],
                    repo_type=self.upload_cfg["repo_type"],
                    token=self.token,
                    allow_patterns=[
                        f"{self.upload_cfg['status_path']}/*",
                        "inventory/*.jsonl",
                    ],
                )
            )
        except Exception as exc:
            logger.info("No remote state to sync yet: %s", exc)
            return
        # sync state/
        src_dir = remote_dir / self.upload_cfg["status_path"]
        if src_dir.exists():
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.state_dir / item.name)
        # sync inventory/ (cached channel metadata)
        inv_src = remote_dir / "inventory"
        if inv_src.exists():
            inv_dst = self.base_dir / "inventory"
            inv_dst.mkdir(parents=True, exist_ok=True)
            for item in inv_src.iterdir():
                if item.is_file() and item.suffix == ".jsonl":
                    dst = inv_dst / item.name
                    if not dst.exists():
                        shutil.copy2(item, dst)
                        logger.info("Restored cached inventory: %s", item.name)

    def push_inventory(self, inventory_dir: Path) -> None:
        if not self.remote_enabled or not self.token or not self.api:
            return
        for jsonl_path in inventory_dir.glob("*_inventory.jsonl"):
            try:
                _hf_retry(lambda p=jsonl_path: self.api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"inventory/{p.name}",
                    repo_id=self.upload_cfg["repo_id"],
                    repo_type=self.upload_cfg["repo_type"],
                    token=self.token,
                    commit_message=f"exp038: cache inventory {p.name}",
                ))
                logger.info("Pushed inventory cache: %s", jsonl_path.name)
            except Exception as exc:
                logger.warning("Failed to push inventory %s: %s", jsonl_path.name, exc)

    def _load_jsonl_ids(self, path: Path) -> set[str]:
        ids: set[str] = set()
        if not path.exists():
            return ids
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                video_id = payload.get("video_id")
                if video_id:
                    ids.add(video_id)
        return ids

    def _load_local_state(self) -> None:
        self.processed_video_ids = self._load_jsonl_ids(self.processed_path)
        self.rejected_video_ids = self._load_jsonl_ids(self.rejected_path)
        self.selected_video_ids = self._load_jsonl_ids(self.selected_path)
        self.seen_video_ids = set().union(self.processed_video_ids, self.rejected_video_ids)
        if self.status_path.exists():
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            self.stats.update(status.get("stats", {}))
        else:
            self.stats.update(
                {
                    "selected_videos": len(self.selected_video_ids),
                    "processed_videos": len(self.processed_video_ids),
                    "rejected_videos": len(self.rejected_video_ids),
                    "uploaded_chunks": 0,
                    "uploaded_shards": 0,
                }
            )

    def write_status(self, *, reason: str, current_chunk_count: int | None = None, total_duration_s: float | None = None) -> None:
        if current_chunk_count is not None:
            self.stats["uploaded_chunks"] = current_chunk_count
        if total_duration_s is not None:
            self.stats["total_duration_s"] = total_duration_s
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "target_duration_hours": self.target_duration_hours,
            "progress_hours": round(float(self.stats.get("total_duration_s", 0)) / 3600, 2),
            "stats": {
                **self.stats,
                "selected_videos": len(self.selected_video_ids),
                "processed_videos": len(self.processed_video_ids),
                "rejected_videos": len(self.rejected_video_ids),
                "seen_videos": len(self.seen_video_ids),
            },
        }
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_selected(self, rows: list[dict]) -> None:
        existing = self.selected_video_ids.copy()
        with self.selected_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                video_id = row.get("video_id")
                if not video_id or video_id in existing:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.add(video_id)
        self.selected_video_ids = existing
        self.stats["selected_videos"] = len(self.selected_video_ids)
        self.write_status(reason="inventory_complete")

    def mark_processed(self, row: dict, *, chunks_produced: int, total_chunks: int, total_duration_s: float) -> None:
        with self.processed_path.open("a", encoding="utf-8") as handle:
            payload = dict(row)
            payload["chunks_produced"] = chunks_produced
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        video_id = row.get("video_id")
        if video_id:
            self.processed_video_ids.add(video_id)
            self.seen_video_ids.add(video_id)
        self.stats["processed_videos"] = len(self.processed_video_ids)
        self.write_status(reason="video_processed", current_chunk_count=total_chunks, total_duration_s=total_duration_s)

    def mark_rejected(self, row: dict, *, total_chunks: int, total_duration_s: float) -> None:
        with self.rejected_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        video_id = row.get("video_id")
        if video_id:
            self.rejected_video_ids.add(video_id)
            self.seen_video_ids.add(video_id)
        self.stats["rejected_videos"] = len(self.rejected_video_ids)
        self.write_status(reason=row.get("status", "video_rejected"), current_chunk_count=total_chunks, total_duration_s=total_duration_s)

    def note_uploaded_shard(self, *, total_chunks: int) -> None:
        self.stats["uploaded_shards"] = int(self.stats.get("uploaded_shards", 0)) + 1
        self.write_status(reason="shard_uploaded", current_chunk_count=total_chunks)

    def flush_remote(self, *, reason: str) -> None:
        if not self.remote_enabled or not self.token or not self.api:
            return
        self.write_status(reason=reason)
        for local_path in [self.status_path, self.selected_path, self.processed_path, self.rejected_path]:
            if not local_path.exists():
                continue
            _hf_retry(lambda lp=local_path: self.api.upload_file(
                path_or_fileobj=str(lp),
                path_in_repo=f"{self.upload_cfg['status_path']}/{lp.name}",
                repo_id=self.upload_cfg["repo_id"],
                repo_type=self.upload_cfg["repo_type"],
                token=self.token,
            ))


def cleanup_video_files(output_root: Path, video: dict) -> None:
    """Delete raw, demucs, and enhanced files for a video after processing."""
    video_id = video.get("video_id", "")
    upload_date = video.get("upload_date") or "unknown"
    stem = f"{upload_date}_{video_id}"
    channel = video.get("channel", "")

    for path in [
        output_root / "raw" / channel / f"{stem}.wav",
        output_root / "speech_only" / f"{stem}_vocals.wav",
        output_root / "enhanced" / f"{stem}_vocals_enhanced.wav",
        output_root / "enhanced" / f"{stem}_vocals_enhanced.preview.wav",
    ]:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    # Also clean demucs output directory for this video
    demucs_dir = output_root / "demucs" / "htdemucs" / f"{stem}"
    if demucs_dir.exists():
        import shutil as _shutil
        try:
            _shutil.rmtree(demucs_dir)
        except Exception:
            pass


def extract_chunk(source_audio: Path, target_audio: Path, start_s: float, end_s: float) -> None:
    duration = max(0.0, end_s - start_s)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_audio),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_audio),
    ]
    run(ffmpeg_cmd)


def process_videos(audio_cfg: dict, manifest_path: Path, dry_run: bool, sample_n: int = 0) -> None:
    rows = load_manifest(manifest_path)
    if sample_n > 0:
        rows = rows[:sample_n]
        logger.info("Sample mode: processing %d videos only", sample_n)
    output_root = Path(audio_cfg["output_root"])
    raw_dir = output_root / "raw"
    speech_dir = output_root / "speech_only"
    chunk_dir = output_root / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    processed_manifest = output_root / audio_cfg["processed_manifest"]
    rejected_manifest = output_root / audio_cfg["rejected_manifest"]

    target_duration_hours = audio_cfg.get("target_duration_hours", 5000)
    review_dir = output_root / "review"
    review_entries: list[dict] = []
    if sample_n > 0:
        review_dir.mkdir(parents=True, exist_ok=True)

    hub_state = HubState(audio_cfg["upload"], output_root, target_duration_hours=target_duration_hours)
    language_gate = WhisperLanguageGate(audio_cfg["processing"])
    chunker = SileroChunker(audio_cfg["processing"])
    uploader = None
    if audio_cfg["upload"]["enabled"] and not dry_run and hub_state.remote_enabled and hub_state.token and hub_state.api:
        uploader = ChunkUploader(
            audio_cfg["upload"],
            output_root,
            api=hub_state.api,
            token=hub_state.token,
            start_shard_index=int(hub_state.stats.get("uploaded_shards", 0)),
            start_chunk_index=int(hub_state.stats.get("uploaded_chunks", 0)),
        )

    total_chunks = int(hub_state.stats.get("uploaded_chunks", 0))
    total_duration_s = float(hub_state.stats.get("total_duration_s", 0.0))
    target_duration_s = target_duration_hours * 3600.0
    status_updates = 0
    processed_manifest.parent.mkdir(parents=True, exist_ok=True)

    with processed_manifest.open("a", encoding="utf-8") as processed_handle, \
            rejected_manifest.open("a", encoding="utf-8") as rejected_handle:
        for row in rows:
            if row.get("video_id") in hub_state.seen_video_ids:
                logger.info("Skipping already-seen video %s", row["video_id"])
                continue
            if total_duration_s >= target_duration_s:
                logger.info("Reached duration target: %.1f h", total_duration_s / 3600)
                break

            logger.info("Processing %s | %s", row["channel"], row["title"])
            if dry_run:
                processed_handle.write(json.dumps({"video_id": row["video_id"], "status": "planned"}, ensure_ascii=False) + "\n")
                continue

            try:
                raw_audio = download_audio(row, raw_dir)
                demucs_audio = maybe_extract_voice(audio_cfg, raw_audio, output_root)
                voice_audio = maybe_enhance_voice(audio_cfg, demucs_audio, output_root)
                if audio_cfg["processing"].get("skip_language_gate", False):
                    lang_result = {"accepted": True, "language": "kk", "language_probability": 1.0,
                                   "transcript_preview": "", "kazakh_char_ratio": 1.0}
                else:
                    lang_result = language_gate.check(voice_audio)
                    if not lang_result["accepted"]:
                        rejected = dict(row)
                        rejected["status"] = "rejected_language_gate"
                        rejected["language_gate"] = lang_result
                        rejected_handle.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                        rejected_handle.flush()
                        hub_state.mark_rejected(rejected, total_chunks=total_chunks, total_duration_s=total_duration_s)
                        status_updates += 1
                        if status_updates % audio_cfg["upload"]["upload_state_every_videos"] == 0:
                            hub_state.flush_remote(reason="periodic_state_sync")
                        continue

                segments = chunker.chunk(voice_audio)
                if not segments:
                    rejected = dict(row)
                    rejected["status"] = "rejected_vad_empty"
                    rejected["language_gate"] = lang_result
                    rejected_handle.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                    rejected_handle.flush()
                    hub_state.mark_rejected(rejected, total_chunks=total_chunks, total_duration_s=total_duration_s)
                    status_updates += 1
                    if status_updates % audio_cfg["upload"]["upload_state_every_videos"] == 0:
                        hub_state.flush_remote(reason="periodic_state_sync")
                    continue

                produced = 0
                for segment_index, (start_s, end_s) in enumerate(segments):
                    if total_duration_s >= target_duration_s:
                        break

                    duration_s = round(end_s - start_s, 3)
                    if uploader is not None:
                        target_audio, rel_audio = uploader.next_chunk_path(row)
                    else:
                        rel_audio = f"{row['channel']}_{row['video_id']}_{segment_index:04d}.wav"
                        target_audio = chunk_dir / rel_audio

                    extract_chunk(voice_audio, target_audio, start_s, end_s)
                    record = {
                        "audio": f"audio/{rel_audio}" if uploader is not None else rel_audio,
                        "channel": row["channel"],
                        "video_id": row["video_id"],
                        "source_url": row["webpage_url"],
                        "title": row["title"],
                        "upload_date": row.get("upload_date"),
                        "chunk_index": segment_index,
                        "start_s": round(start_s, 3),
                        "end_s": round(end_s, 3),
                        "duration_s": duration_s,
                        "language": lang_result["language"],
                        "language_probability": lang_result["language_probability"],
                        "transcript_preview": lang_result["transcript_preview"],
                    }
                    if uploader is not None:
                        uploader.add_metadata(record)

                    processed = dict(row)
                    processed["status"] = "processed"
                    processed["chunk"] = record
                    processed_handle.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    processed_handle.flush()

                    produced += 1
                    total_chunks += 1
                    total_duration_s += duration_s

                if uploader is not None and produced > 0:
                    if uploader.flush():
                        hub_state.note_uploaded_shard(total_chunks=total_chunks)
                cleanup_video_files(output_root, row)
                hub_state.mark_processed(row, chunks_produced=produced, total_chunks=total_chunks, total_duration_s=total_duration_s)
                if sample_n > 0:
                    chunk_paths = sorted((chunk_dir / f).name for f in chunk_dir.iterdir() if row["video_id"] in f.name) if chunk_dir.exists() else []
                    review_entries.append({
                        "video_id": row["video_id"],
                        "title": row["title"],
                        "channel": row["channel"],
                        "raw_audio": str(raw_audio),
                        "demucs_audio": str(demucs_audio),
                        "enhanced_audio": str(voice_audio),
                        "lang_result": lang_result,
                        "chunks_produced": produced,
                        "chunk_dir": str(chunk_dir),
                    })
                    (review_dir / "review_manifest.json").write_text(
                        json.dumps(review_entries, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                status_updates += 1
                if status_updates % audio_cfg["upload"]["upload_state_every_videos"] == 0:
                    hub_state.flush_remote(reason="periodic_state_sync")

            except Exception as exc:
                rejected = dict(row)
                rejected["status"] = "failed"
                rejected["error"] = str(exc)
                rejected_handle.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                rejected_handle.flush()
                cleanup_video_files(output_root, row)
                hub_state.mark_rejected(rejected, total_chunks=total_chunks, total_duration_s=total_duration_s)
                status_updates += 1
                if status_updates % audio_cfg["upload"]["upload_state_every_videos"] == 0:
                    hub_state.flush_remote(reason="periodic_state_sync")
                logger.exception("Failed processing %s", row["video_id"])

    if uploader is not None:
        uploader.close()
    hub_state.flush_remote(reason="process_finished")

    logger.info("Finished with %d chunks", total_chunks)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _, audio_cfg = load_audio_config(args.config)
    if args.worker_id > 0:
        w = f"w{args.worker_id}"
        audio_cfg["upload"]["path_in_repo"] = f"{audio_cfg['upload']['path_in_repo']}/{w}"
        audio_cfg["upload"]["status_path"] = f"{audio_cfg['upload']['status_path']}/{w}"
        audio_cfg["output_root"] = str(Path(audio_cfg["output_root"]).parent / f"{Path(audio_cfg['output_root']).name}_{w}")
    output_root = Path(audio_cfg["output_root"])
    inventory_dir = output_root / audio_cfg["inventory_subdir"]
    inventory_dir.mkdir(parents=True, exist_ok=True)

    if args.step in {"inventory", "all"}:
        hub_state = HubState(audio_cfg["upload"], output_root, target_duration_hours=audio_cfg.get("target_duration_hours", 5000))
        cache_days = int(audio_cfg.get("inventory_cache_days", 0))
        force_refresh = args.refresh_inventory
        all_rows: list[dict] = []
        for channel_cfg in audio_cfg["channels"]:
            rows = inventory_channel(channel_cfg, inventory_dir, cache_days=cache_days, force_refresh=force_refresh)
            rows = [row for row in rows if row.get("video_id") not in hub_state.seen_video_ids]
            logger.info(
                "Channel %s: %d/%d candidates",
                channel_cfg["name"],
                sum(1 for row in rows if row["status"] == "candidate"),
                len(rows),
            )
            all_rows.extend(rows)
        selected = select_videos(audio_cfg, all_rows, inventory_dir)
        hub_state.mark_selected(selected)
        hub_state.push_inventory(inventory_dir)
        hub_state.flush_remote(reason="inventory_complete")
        logger.info("Inventory complete. Selected manifest: %s", inventory_dir / audio_cfg["selected_manifest"])
        if args.step == "inventory":
            logger.info("Selected %d videos. Stop here and inspect the manifest before full processing.", len(selected))
            return

    manifest_path = Path(args.manifest) if args.manifest else inventory_dir / audio_cfg["selected_manifest"]
    if args.step in {"process", "all"}:
        process_videos(audio_cfg, manifest_path, args.dry_run, sample_n=args.sample)


if __name__ == "__main__":
    main()
