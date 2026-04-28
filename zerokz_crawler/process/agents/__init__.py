from .title import TitleAgent
from .text import TextAgent
from .markdown import MarkdownAgent
from .language import LanguageAgent
from .docling_agent import DoclingAgent

# Default pipeline — runs on every HTML file during pipeline.py
# DoclingAgent is included — produces docling_md and docling_json columns
DEFAULT_AGENTS = [
    TitleAgent(),
    TextAgent(),
    MarkdownAgent(),
    DoclingAgent(),
    LanguageAgent(),  # last: uses text from TextAgent
]
