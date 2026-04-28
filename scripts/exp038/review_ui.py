#!/usr/bin/env python3
"""Streamlit UI for reviewing exp038 audio pipeline stages.

Usage:
    streamlit run scripts/exp038/review_ui.py
    streamlit run scripts/exp038/review_ui.py -- --manifest data/exp038_youtube_recent_kk_audio/review/review_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data/exp038_youtube_recent_kk_audio/review/review_manifest.json"
)
FEEDBACK_PATH = DEFAULT_MANIFEST.parent / "feedback.json"

STAGE_LABELS = {
    "raw_audio": "1. Raw (прямо с YouTube)",
    "demucs_audio": "2. После Demucs (только голос)",
    "enhanced_audio": "3. После noisereduce (очищенный голос)",
}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_feedback(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_feedback(path: Path, feedback: dict) -> None:
    path.write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8")


def audio_player(label: str, file_path: str) -> None:
    p = Path(file_path)
    if not p.exists():
        st.caption(f"_{label}: файл не найден ({p.name})_")
        return
    st.caption(label)
    st.audio(str(p))


def chunk_players(chunk_dir: str, video_id: str, max_chunks: int = 5) -> None:
    d = Path(chunk_dir)
    if not d.exists():
        st.caption("_чанки не найдены_")
        return
    chunks = sorted(f for f in d.iterdir() if video_id in f.name and f.suffix == ".wav")
    if not chunks:
        st.caption("_чанки не найдены_")
        return
    st.caption(f"Чанки ({len(chunks)} всего, показываю первые {min(len(chunks), max_chunks)}):")
    for chunk in chunks[:max_chunks]:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.caption(chunk.name)
        with col2:
            st.audio(str(chunk))


def main() -> None:
    st.set_page_config(page_title="exp038 Quality Review", layout="wide")
    st.title("exp038 — Проверка качества аудио")
    st.caption("Слушай каждый этап и оставляй инсайты. Данные сохраняются автоматически.")

    # Manifest path override
    manifest_path = DEFAULT_MANIFEST
    if len(sys.argv) > 1 and sys.argv[-1].endswith(".json"):
        manifest_path = Path(sys.argv[-1])

    with st.sidebar:
        st.header("Настройки")
        manifest_input = st.text_input("Путь к manifest", value=str(manifest_path))
        manifest_path = Path(manifest_input)
        st.divider()
        st.markdown("**Как пользоваться:**")
        st.markdown("""
1. Запусти dry run:
```bash
python scripts/exp038/prepare_youtube_kk_audio.py \\
  --config configs/experiments/exp038_youtube_recent_kk_audio.yaml \\
  --step all --sample 5
```
2. Открой этот UI и слушай
3. Оставляй комментарии по каждому видео
4. Отправь инсайты разработчику
        """)

    entries = load_manifest(manifest_path)
    if not entries:
        st.warning(f"Manifest не найден: `{manifest_path}`")
        st.info("Сначала запусти пайплайн с флагом `--sample 5`")
        return

    feedback = load_feedback(FEEDBACK_PATH)
    feedback_changed = False

    st.success(f"Загружено {len(entries)} видео из `{manifest_path.name}`")

    for idx, entry in enumerate(entries):
        video_id = entry.get("video_id", f"video_{idx}")
        title = entry.get("title", video_id)
        channel = entry.get("channel", "")
        lang_result = entry.get("lang_result", {})

        with st.expander(f"**{idx + 1}. {title}** — {channel}", expanded=(idx == 0)):
            col_info, col_lang = st.columns([2, 1])
            with col_info:
                st.markdown(f"**video_id**: `{video_id}`")
                st.markdown(f"**чанков создано**: {entry.get('chunks_produced', '?')}")
            with col_lang:
                st.markdown(f"**язык**: `{lang_result.get('language', '?')}`")
                st.markdown(f"**вероятность**: `{lang_result.get('language_probability', '?')}`")
                st.markdown(f"**казахских символов**: `{lang_result.get('kazakh_char_ratio', '?')}`")
                if lang_result.get("transcript_preview"):
                    st.text_area("Превью транскрипта", lang_result["transcript_preview"][:300], height=80, key=f"tr_{video_id}")

            st.divider()
            st.subheader("Этапы обработки")

            cols = st.columns(3)
            for col, (stage_key, stage_label) in zip(cols, STAGE_LABELS.items()):
                with col:
                    audio_player(stage_label, entry.get(stage_key, ""))

            st.divider()
            st.subheader("Финальные чанки (VAD)")
            chunk_players(entry.get("chunk_dir", ""), video_id)

            st.divider()
            prev_fb = feedback.get(video_id, {})

            st.markdown("**Твои инсайты по этому видео:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                raw_ok = st.selectbox(
                    "Raw аудио",
                    ["не проверено", "хорошее", "шум/музыка", "не казахский"],
                    index=["не проверено", "хорошее", "шум/музыка", "не казахский"].index(
                        prev_fb.get("raw_ok", "не проверено")
                    ),
                    key=f"raw_{video_id}",
                )
            with c2:
                demucs_ok = st.selectbox(
                    "После Demucs",
                    ["не проверено", "чисто", "остался шум", "голос обрезан"],
                    index=["не проверено", "чисто", "остался шум", "голос обрезан"].index(
                        prev_fb.get("demucs_ok", "не проверено")
                    ),
                    key=f"demucs_{video_id}",
                )
            with c3:
                chunks_ok = st.selectbox(
                    "Чанки",
                    ["не проверено", "хорошие", "много тишины", "слишком короткие", "артефакты"],
                    index=["не проверено", "хорошие", "много тишины", "слишком короткие", "артефакты"].index(
                        prev_fb.get("chunks_ok", "не проверено")
                    ),
                    key=f"chunks_{video_id}",
                )

            notes = st.text_area(
                "Заметки (любые наблюдения)",
                value=prev_fb.get("notes", ""),
                key=f"notes_{video_id}",
                placeholder="например: слышна фоновая музыка, голос металлический после demucs, чанки обрезаются на полуслове...",
            )

            new_fb = {"raw_ok": raw_ok, "demucs_ok": demucs_ok, "chunks_ok": chunks_ok, "notes": notes}
            if new_fb != prev_fb:
                feedback[video_id] = new_fb
                feedback_changed = True

    if feedback_changed:
        save_feedback(FEEDBACK_PATH, feedback)
        st.toast("Фидбек сохранён", icon="✅")

    st.divider()
    st.subheader("Сводка фидбека")
    if feedback:
        rows = []
        for vid_id, fb in feedback.items():
            entry_map = {e["video_id"]: e for e in entries}
            title = entry_map.get(vid_id, {}).get("title", vid_id)[:50]
            rows.append({
                "Видео": title,
                "Raw": fb.get("raw_ok", "—"),
                "Demucs": fb.get("demucs_ok", "—"),
                "Чанки": fb.get("chunks_ok", "—"),
                "Заметки": fb.get("notes", "")[:80],
            })
        st.dataframe(rows, use_container_width=True)

        st.download_button(
            "Скачать фидбек (JSON)",
            data=json.dumps(feedback, ensure_ascii=False, indent=2),
            file_name="exp038_feedback.json",
            mime="application/json",
        )
    else:
        st.caption("Фидбек пока не оставлен")


if __name__ == "__main__":
    main()
