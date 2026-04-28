# Dataset: kz-transformers/multidomain-kazakh-dataset (MDBKD)

Multi-Domain Bilingual Kazakh Dataset.

## Source

- HuggingFace: `kz-transformers/multidomain-kazakh-dataset`
- License: Apache 2.0
- Languages: kk (Kazakh), ru (Russian)

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Text content (one language per row) |
| `predicted_language` | string | Language label: `kaz` or `rus` |
| `contains_kaz_symbols` | int64 | 1 if text contains Kazakh-specific characters, 0 otherwise |
| `id` | string | UUID |

## Statistics

- Total rows: ~80M (79,954,702 estimated)
- Unique texts: ~24.9M (per dataset card)
- Language distribution: overwhelmingly Kazakh (`kaz`), small fraction Russian (`rus`)
- Splits: `train` only (no validation or test split)
- Parquet size: ~2.5 GB on disk, ~5 GB in memory

## Usage Notes

- Filter by `predicted_language == "kaz"` for Kazakh-only data
- Filter by `predicted_language == "rus"` for Russian data
- `contains_kaz_symbols == 1` is a secondary indicator (catches text with Kazakh-specific Cyrillic: i, u, etc.)
- No built-in val split — must create manually (e.g., 95/5 train/val)
- Dataset is large — consider streaming or subset for experiments (`split="train[:1%]"`)
