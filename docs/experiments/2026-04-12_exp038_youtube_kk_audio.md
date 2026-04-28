# EXP-038: Recent-only Kazakh Speech Bootstrap from YouTube

**Date**: 2026-04-12  
**Status**: Designed / ready for inventory  
**Goal**: Build `5K+` unlabeled Kazakh speech chunks for audio-encoder pretraining from recent YouTube news/talk videos, without crawling full channels.

## Why This Experiment Exists

The target corpus is not “all YouTube from two channels”.  
That would be the wrong objective because:

- both channels are effectively unbounded streams
- `QazaqstanTV` mixes useful talk/news with serials, entertainment, promos, and non-speech-heavy content
- `aqparatkz` is useful as a freshness source, but should remain a small fraction of the final corpus
- downloading everything first is the slowest and most expensive possible order of operations

The correct order is:

1. metadata inventory only
2. recent-window candidate selection
3. audio download only for selected videos
4. speech-only isolation
5. Kazakh-language gate
6. VAD chunking
7. rolling upload to Hugging Face

## Scope Constraints

- `QazaqstanTV` is the main source.
- `aqparatkz` must remain at **5-10%** of the final corpus.
- Full historical crawl is forbidden by design.
- The pipeline operates on **recent windows only** and stops after the chunk target is reached.

## Current Channel Observations

Local metadata probe on 2026-04-12 using `yt-dlp --flat-playlist` shows:

- `QazaqstanTV` recent uploads include a mix of useful talk/news (`Әлем және біз`, `Ашық алаң`, `Парасат майданы`) and clearly irrelevant serial / entertainment items.
- `aqparatkz` recent uploads are mostly short news items and full bulletin editions, which is useful for freshness but too repetitive for a dominant share.

That confirms the collection strategy should be:

- title/duration filtering before download
- hard per-channel quotas
- recent-first ordering

## Inventory Snapshot

Inventory was run locally on 2026-04-12 with the current `exp038` config.

Results:

- candidate pool after metadata triage:
  - `qazaqstan_tv`: `102 / 800`
  - `aqparatkz`: `40 / 450`
- selected manifest:
  - total selected videos: `113`
  - `qazaqstan_tv`: `102`
  - `aqparatkz`: `11`
- primary reject reasons:
  - `missing_allow_keyword`: `1107`
  - `deny_keyword`: `216`
  - `too_short`: `15`
  - `too_long`: `4`

Top selected examples were exactly the intended format mix:

- `Парасат майданы. Генуэз конференциясы`
- `Ашық алаң. Ұлттық ғылымның әлеуеті`
- `Let's talk` talk-show episodes

That is a good sign: the selector is currently biased toward discussion/news programs instead of serials and entertainment uploads.

## Recommended Stack

### 1. Inventory / triage

- Tool: `yt-dlp`
- Mode: `--flat-playlist`
- Purpose: collect IDs, titles, durations, upload dates, and URLs without media download

### 2. Speech-only extraction

- First choice: `Demucs` vocal stem extraction
- Fallback: raw audio when Demucs fails and the clip is still speech-dominant

Why: broadcast/news content often contains intro music, beds, jingles, and archival inserts. Removing non-vocal stems before VAD improves chunk purity.

### 3. Kazakh language gate

- Tool: `faster-whisper`
- Strategy: transcribe only the first `~90s`
- Keep only clips where:
  - language is `kk`
  - probability is high enough
  - transcript has enough text
  - Kazakh-specific character ratio is high enough

This is much cheaper than full-video ASR and gives a strong early rejection signal.

### 4. Chunking

- Tool: `silero-vad`
- Output: speech chunks `<= 30s`
- Merge small speech islands with short gaps
- Reject too-short fragments

### 5. Upload pattern

- Upload in rolling shards to a Hugging Face **dataset** repo
- Do not commit one file per chunk
- Use shard folders with `metadata.jsonl` + audio files

This keeps progress durable during long runs on RunPod without waiting for the entire crawl to finish.

## Why RunPod

This workload is mostly:

- network-bound during metadata fetch and download
- disk-bound during audio staging
- moderately GPU-accelerated during Demucs / Whisper

That makes a single 4090 / A6000 / L40S pod a good first target.  
The experiment should launch on GPU types with mature CUDA support only.

## Files Added

- [configs/experiments/exp038_youtube_recent_kk_audio.yaml](/Users/sakentukenov/slm/configs/experiments/exp038_youtube_recent_kk_audio.yaml:1)
- [scripts/exp038/prepare_youtube_kk_audio.py](/Users/sakentukenov/slm/scripts/exp038/prepare_youtube_kk_audio.py:1)
- [scripts/exp038/launch_runpod.py](/Users/sakentukenov/slm/scripts/exp038/launch_runpod.py:1)
- [scripts/exp038/deploy_to_pod.sh](/Users/sakentukenov/slm/scripts/exp038/deploy_to_pod.sh:1)
- [scripts/exp038/run_on_pod.sh](/Users/sakentukenov/slm/scripts/exp038/run_on_pod.sh:1)

## Default Behavior

The pod bootstrap script intentionally starts with:

```bash
python3 scripts/exp038/prepare_youtube_kk_audio.py \
  --config configs/experiments/exp038_youtube_recent_kk_audio.yaml \
  --step inventory
```

That forces a metadata review before full processing.

## Resume / Status Model

The experiment is now explicitly resumable.

State is stored both locally and on Hugging Face dataset repo
`stukenov/sozkz-corpus-raw-kk-youtube-speech-v1` under `state/`:

- `state/status.json`
- `state/selected_videos.jsonl`
- `state/processed_videos.jsonl`
- `state/rejected_videos.jsonl`

Resume behavior:

- inventory syncs prior `state/` from HF first
- already seen `video_id` values are excluded from new selection
- processing skips any video already present in processed/rejected state
- progress is pushed back to HF periodically
- chunk upload happens before processed status is finalized, so resumed runs continue from unique content instead of re-counting unfinished progress

This means the pod can be stopped and resumed later while continuing from unseen videos/chunks.

Current synced state after inventory:

- `selected_videos = 113`
- `processed_videos = 0`
- `rejected_videos = 0`
- `uploaded_chunks = 0`
- `uploaded_shards = 0`

## Research Notes

This design follows the practical pattern used in large-scale speech mining:

- metadata-first triage to avoid waste
- short-preview language gating before full processing
- VAD after cleanup, not before raw download triage
- incremental durable upload instead of “download everything then upload once”

## External References

- `yt-dlp` flat-playlist / metadata-first extraction: https://github.com/yt-dlp/yt-dlp
- `pyannote.audio` overview of speech activity detection / diarization components: https://github.com/pyannote/pyannote-audio
- WhisperX notes on VAD + alignment + diarization for long-form speech workflows: https://github.com/m-bain/whisperX
- Hugging Face upload guidance, including scheduled/chunked uploads: https://huggingface.co/docs/huggingface_hub/main/guides/upload
