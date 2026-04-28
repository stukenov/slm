"""
Rebuild rtrk/kazakh-traditional-audio as parquet with embedded audio bytes.
Uses soundfile to read WAV, manually embeds bytes into Audio feature.
"""
import os
import json
import io
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

REPO_ID = "rtrk/kazakh-traditional-audio"
HF_TOKEN = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
DATA_DIR = Path("/tmp/kazakh-traditional-audio/data")
OUTPUT_DIR = Path("/tmp/audio_parquet_output")
OUTPUT_DIR.mkdir(exist_ok=True)

SHARD_SIZE_BYTES = 200 * 1024 * 1024  # 200MB per shard

print("=== Step 1: Load metadata ===")
metadata_path = DATA_DIR / "metadata.jsonl"
records = []
with open(metadata_path) as f:
    for line in f:
        obj = json.loads(line)
        audio_path = DATA_DIR / obj["file_name"]
        if not audio_path.exists():
            print(f"  SKIP (missing): {obj['file_name']}")
            continue
        records.append({
            "file_name": obj["file_name"],
            "audio_path": str(audio_path),
            "title": obj["title"],
            "type": obj["type"],
            "language": obj["language"],
            "duration": obj["duration"],
        })
print(f"Loaded {len(records)} records")

print("=== Step 2: Build parquet shards with embedded audio ===")
shard_idx = 0
current_shard_bytes = 0
current_rows = {
    "audio": [],  # list of {"bytes": bytes, "path": str}
    "title": [],
    "type": [],
    "language": [],
    "duration": [],
}

def write_shard():
    global shard_idx, current_shard_bytes, current_rows
    if not current_rows["title"]:
        return

    # Build audio struct array manually
    audio_bytes_arr = pa.array([a["bytes"] for a in current_rows["audio"]], type=pa.binary())
    audio_path_arr = pa.array([a["path"] for a in current_rows["audio"]], type=pa.string())
    audio_struct = pa.StructArray.from_arrays(
        [audio_bytes_arr, audio_path_arr],
        names=["bytes", "path"],
    )

    table = pa.table({
        "audio": audio_struct,
        "title": pa.array(current_rows["title"]),
        "type": pa.array(current_rows["type"]),
        "language": pa.array(current_rows["language"]),
        "duration": pa.array(current_rows["duration"], type=pa.float64()),
    })

    out_path = OUTPUT_DIR / f"train-{shard_idx:05d}-of-XXXXX.parquet"
    pq.write_table(table, out_path, row_group_size=50)
    n_rows = len(current_rows["title"])
    print(f"  Shard {shard_idx}: {n_rows} rows, {current_shard_bytes / 1024 / 1024:.1f}MB -> {out_path.name}")

    shard_idx += 1
    current_shard_bytes = 0
    current_rows = {"audio": [], "title": [], "type": [], "language": [], "duration": []}

for i, rec in enumerate(records):
    # Read WAV file as raw bytes (keep original format)
    with open(rec["audio_path"], "rb") as f:
        audio_bytes = f.read()

    current_rows["audio"].append({"bytes": audio_bytes, "path": rec["file_name"]})
    current_rows["title"].append(rec["title"])
    current_rows["type"].append(rec["type"])
    current_rows["language"].append(rec["language"])
    current_rows["duration"].append(rec["duration"])
    current_shard_bytes += len(audio_bytes)

    if current_shard_bytes >= SHARD_SIZE_BYTES:
        write_shard()

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(records)} files...")

# Write remaining
write_shard()

# Fix shard filenames with actual total
total_shards = shard_idx
for i in range(total_shards):
    old_name = OUTPUT_DIR / f"train-{i:05d}-of-XXXXX.parquet"
    new_name = OUTPUT_DIR / f"train-{i:05d}-of-{total_shards:05d}.parquet"
    old_name.rename(new_name)

print(f"Total shards: {total_shards}")

print("=== Step 3: Upload parquet shards to Hub ===")
api = HfApi()
api.upload_folder(
    folder_path=str(OUTPUT_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo="data",
    token=HF_TOKEN,
    commit_message="rebuild: parquet with embedded audio, 200MB shards, row_group_size=50",
)

print("=== Done! ===")
