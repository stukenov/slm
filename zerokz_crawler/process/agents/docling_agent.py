"""Convert HTML to structured Docling format (JSON + markdown with structure).

For HTML input docling uses a pure structural parser (no OCR, no vision models).
Cyrillic (Russian/Kazakh) passes through fine — language doesn't matter here.
GPU is not used for HTML processing.
"""
from .base import BaseAgent


class DoclingAgent(BaseAgent):
    columns = ["docling_md", "docling_json"]

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter, InputFormat, FormatOption
            from docling.pipeline.simple_pipeline import SimplePipeline
            from docling.backend.html_backend import HTMLDocumentBackend

            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.HTML],
                format_options={
                    InputFormat.HTML: FormatOption(
                        pipeline_cls=SimplePipeline,
                        backend=HTMLDocumentBackend,
                    )
                },
            )
        return self._converter

    def process(self, row: dict) -> dict:
        html = row.get("html", "")
        if not html or len(html) < 200:
            return {"docling_md": None, "docling_json": None}

        import tempfile, os, json
        converter = self._get_converter()

        with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                         encoding="utf-8", delete=False) as f:
            f.write(html)
            tmp_path = f.name

        try:
            result = converter.convert(tmp_path)
            doc = result.document
            docling_md = doc.export_to_markdown()
            docling_json = json.dumps(doc.export_to_dict(), ensure_ascii=False)
        finally:
            os.unlink(tmp_path)

        return {
            "docling_md": docling_md if docling_md else None,
            "docling_json": docling_json if docling_json != "{}" else None,
        }
