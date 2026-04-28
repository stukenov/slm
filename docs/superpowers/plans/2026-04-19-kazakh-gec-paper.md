# Kazakh GEC Paper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a morphology-aware minimal-edit GEC pipeline for Kazakh with a 3-level error taxonomy, dual-model architecture (NLLB seq2seq + edit tagger), multi-reference evaluation, and all artifacts published — targeting an arXiv preprint.

**Architecture:** Three iterative rounds. Round 1 builds the foundation (taxonomy, synthetic data, NLLB baseline, metrics). Round 2 scales data (organic sources), adds the edit tagger, and builds the multi-reference benchmark. Round 3 adds the Qwen-distilled morpheme segmenter, runs ablations, and produces the paper.

**Tech Stack:** Python 3.10+, PyTorch, transformers, datasets, openai (GPT-4o API), apertium-kaz, XLM-RoBERTa, NLLB-200-distilled-600M, HuggingFace Hub.

**Spec:** `docs/superpowers/specs/2026-04-19-kazakh-gec-paper-design.md`

---

## File Map

```
gec-paper/
├── pyproject.toml                      # Project deps + entry points
├── configs/
│   ├── round1_nllb_baseline.yaml
│   ├── round2_nllb_organic.yaml
│   ├── round2_tagger.yaml
│   └── round3_ablation_morph.yaml
├── src/gecpaper/
│   ├── __init__.py
│   ├── taxonomy/
│   │   ├── __init__.py
│   │   ├── schema.py                   # 3-level enum taxonomy + ErrorAnnotation dataclass
│   │   └── classifier.py              # Rule-based L1/L2 + heuristic L3 classifier
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py               # GPT-4o taxonomy-aware corruption generator
│   │   ├── organic_wiki.py            # Wikipedia kk edit history extractor
│   │   ├── organic_social.py          # Social media → GPT-4o correction
│   │   ├── mixer.py                   # Merge sources, dedupe, split, add identity
│   │   └── multi_ref.py              # LLM multi-reference generator
│   ├── morph/
│   │   ├── __init__.py
│   │   ├── apertium.py               # apertium-kaz wrapper: text → segmented text
│   │   └── segmenter.py              # Qwen-distilled char-level segmenter (Round 3)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── nllb_gec.py               # NLLB-200 fine-tuning: data collator + training loop
│   │   ├── edit_tagger.py            # XLM-R token classifier + edit tag vocab builder
│   │   └── reranker.py               # QE reranker (Round 3, optional)
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                # word_f05, gleu, cer, multi-ref wrappers
│   │   ├── benchmark.py              # Full benchmark runner: load model, run scoring, save JSON
│   │   └── analysis.py              # Per-L1/L2 breakdown, bootstrap significance
│   └── pipeline.py                   # Dual-model inference: morph → tagger → NLLB → rerank
├── scripts/
│   ├── generate_synthetic.py          # CLI: generate taxonomy-tagged synthetic pairs
│   ├── collect_wiki_edits.py          # CLI: download + extract Wikipedia kk edits
│   ├── collect_social_data.py         # CLI: scrape + GPT-4o correct social media texts
│   ├── mix_dataset.py                 # CLI: merge all sources → final dataset
│   ├── generate_multi_ref.py          # CLI: generate multi-ref benchmark
│   ├── train_nllb.py                  # CLI: fine-tune NLLB-600M
│   ├── train_tagger.py               # CLI: train XLM-R edit tagger
│   ├── train_morph_segmenter.py       # CLI: train Qwen-distilled segmenter
│   ├── run_scoring.py                 # CLI: run full model assessment
│   └── run_pipeline.py               # CLI: interactive dual-model inference
├── tests/
│   ├── test_taxonomy.py
│   ├── test_metrics.py
│   ├── test_edit_tags.py
│   ├── test_morph.py
│   └── test_pipeline.py
├── data/                              # Local cache (gitignored)
├── paper/
│   ├── main.tex
│   ├── figures/
│   └── tables/
└── README.md
```

---

# ROUND 1: BASELINE

---

### Task 1: Project Scaffold

**Files:**
- Create: `gec-paper/pyproject.toml`
- Create: `gec-paper/src/gecpaper/__init__.py`
- Create: `gec-paper/.gitignore`
- Create: `gec-paper/README.md`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p gec-paper/{configs,src/gecpaper/{taxonomy,data,morph,models,scoring},scripts,tests,data,paper/{figures,tables}}
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "gecpaper"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "datasets>=2.18",
    "openai>=1.0",
    "tqdm",
    "huggingface-hub>=0.20",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
morph = ["apertium-streamparser"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create __init__.py files**

Create empty `__init__.py` in each package directory:
- `gec-paper/src/gecpaper/__init__.py`
- `gec-paper/src/gecpaper/taxonomy/__init__.py`
- `gec-paper/src/gecpaper/data/__init__.py`
- `gec-paper/src/gecpaper/morph/__init__.py`
- `gec-paper/src/gecpaper/models/__init__.py`
- `gec-paper/src/gecpaper/scoring/__init__.py`

- [ ] **Step 4: Create .gitignore**

```
data/
*.pyc
__pycache__/
*.egg-info/
dist/
.venv/
outputs/
logs/
wandb/
```

- [ ] **Step 5: Create README.md**

```markdown
# Morphology-Aware Minimal-Edit GEC for Kazakh

Research subproject: dual-model pipeline (NLLB seq2seq + edit tagger) with 3-level error taxonomy and multi-reference assessment.

## Setup

cd gec-paper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

## Structure

- `src/gecpaper/` — library code
- `scripts/` — CLI entry points
- `configs/` — experiment YAML configs
- `tests/` — unit tests
- `paper/` — LaTeX source
```

- [ ] **Step 6: Install in editable mode and verify**

```bash
cd gec-paper && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import gecpaper; print('OK')"
```

- [ ] **Step 7: Commit**

```bash
git add gec-paper/
git commit -m "feat(gec-paper): scaffold research subproject for Kazakh GEC paper"
```

---

### Task 2: Error Taxonomy Schema

**Files:**
- Create: `gec-paper/src/gecpaper/taxonomy/schema.py`
- Create: `gec-paper/tests/test_taxonomy.py`

- [ ] **Step 1: Write tests for taxonomy schema**

```python
# gec-paper/tests/test_taxonomy.py
from gecpaper.taxonomy.schema import (
    Level1, Level2, Level3,
    ErrorAnnotation,
    ALL_L2_FOR_L1,
    ALL_L3_FOR_L2,
    parse_annotation,
)


def test_level1_has_three_categories():
    assert len(Level1) == 3
    assert Level1.ORTHOGRAPHY.value == "orthography"
    assert Level1.MORPHOSYNTAX.value == "morphosyntax"
    assert Level1.SYNTAX_DISCOURSE.value == "syntax_discourse"


def test_level2_parent_mapping():
    l2_ortho = ALL_L2_FOR_L1[Level1.ORTHOGRAPHY]
    assert Level2.SPELLING in l2_ortho
    assert Level2.VOWEL_HARMONY in l2_ortho
    assert Level2.SPACING in l2_ortho
    assert Level2.PUNCTUATION in l2_ortho
    assert Level2.CASE not in l2_ortho


def test_level3_only_for_morphosyntax():
    for l2 in ALL_L2_FOR_L1[Level1.ORTHOGRAPHY]:
        assert l2 not in ALL_L3_FOR_L2 or len(ALL_L3_FOR_L2[l2]) == 0
    assert len(ALL_L3_FOR_L2[Level2.CASE]) == 7


def test_error_annotation_creation():
    ann = ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.CASE, l3=Level3.DATIVE)
    assert ann.tag == "morphosyntax/case/dative"


def test_error_annotation_no_l3():
    ann = ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)
    assert ann.tag == "orthography/spelling"


def test_parse_annotation_roundtrip():
    ann = ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.PLURAL, l3=Level3.ALLOMORPH)
    parsed = parse_annotation(ann.tag)
    assert parsed == ann


def test_all_l2_categories_count():
    total = sum(len(v) for v in ALL_L2_FOR_L1.values())
    assert total >= 15
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd gec-paper && python -m pytest tests/test_taxonomy.py -v
```
Expected: ImportError — module not yet implemented.

- [ ] **Step 3: Implement taxonomy schema**

```python
# gec-paper/src/gecpaper/taxonomy/schema.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Level1(Enum):
    ORTHOGRAPHY = "orthography"
    MORPHOSYNTAX = "morphosyntax"
    SYNTAX_DISCOURSE = "syntax_discourse"


class Level2(Enum):
    SPELLING = "spelling"
    VOWEL_HARMONY = "vowel_harmony"
    SPACING = "spacing"
    PUNCTUATION = "punctuation"
    CASE = "case"
    POSSESSIVE = "possessive"
    PERSONAL_ENDING = "personal_ending"
    PLURAL = "plural"
    NEGATION = "negation"
    TENSE = "tense"
    POSTPOSITION = "postposition"
    AGREEMENT = "agreement"
    DERIVATION = "derivation"
    WORD_ORDER = "word_order"
    CLAUSE_STRUCTURE = "clause_structure"
    MISSING_ELEMENT = "missing_element"
    REDUNDANT_ELEMENT = "redundant_element"
    DISCOURSE = "discourse"


class Level3(Enum):
    FRONT_BACK_MISMATCH = "front_back_mismatch"
    ROUNDING_HARMONY = "rounding_harmony"
    BOUNDARY_HARMONY = "boundary_harmony"
    MISSING_SPACE = "missing_space"
    EXTRA_SPACE = "extra_space"
    NOMINATIVE = "nominative"
    GENITIVE = "genitive"
    DATIVE = "dative"
    ACCUSATIVE = "accusative"
    LOCATIVE = "locative"
    ABLATIVE = "ablative"
    INSTRUMENTAL = "instrumental"
    PERSON_MISMATCH = "person_mismatch"
    NUMBER_MISMATCH = "number_mismatch"
    PERSON = "person"
    NUMBER = "number"
    TENSE_AGREEMENT = "tense_agreement"
    EXTRA_PLURAL = "extra_plural"
    MISSING_PLURAL = "missing_plural"
    ALLOMORPH = "allomorph"
    DOUBLE_NEGATION = "double_negation"
    WRONG_FORM = "wrong_form"
    PAST_PRESENT = "past_present"
    PRESENT_FUTURE = "present_future"
    PAST_FUTURE = "past_future"
    WRONG_POSTPOSITION = "wrong_postposition"
    CASE_GOVERNMENT = "case_government"
    SUBJECT_VERB = "subject_verb"
    MODIFIER_HEAD = "modifier_head"
    WRONG_DERIVATIONAL_SUFFIX = "wrong_derivational_suffix"
    VERB_POSITION = "verb_position"
    MODIFIER_POSITION = "modifier_position"
    FRAGMENTED = "fragmented"
    RUN_ON = "run_on"
    DROPPED_ARGUMENT = "dropped_argument"
    MISSING_COPULA = "missing_copula"
    REPEATED_WORD = "repeated_word"
    PLEONASM = "pleonasm"
    CONNECTOR_MISUSE = "connector_misuse"


ALL_L2_FOR_L1: dict[Level1, list[Level2]] = {
    Level1.ORTHOGRAPHY: [
        Level2.SPELLING, Level2.VOWEL_HARMONY, Level2.SPACING, Level2.PUNCTUATION,
    ],
    Level1.MORPHOSYNTAX: [
        Level2.CASE, Level2.POSSESSIVE, Level2.PERSONAL_ENDING, Level2.PLURAL,
        Level2.NEGATION, Level2.TENSE, Level2.POSTPOSITION, Level2.AGREEMENT,
        Level2.DERIVATION,
    ],
    Level1.SYNTAX_DISCOURSE: [
        Level2.WORD_ORDER, Level2.CLAUSE_STRUCTURE, Level2.MISSING_ELEMENT,
        Level2.REDUNDANT_ELEMENT, Level2.DISCOURSE,
    ],
}

ALL_L3_FOR_L2: dict[Level2, list[Level3]] = {
    Level2.VOWEL_HARMONY: [Level3.FRONT_BACK_MISMATCH, Level3.ROUNDING_HARMONY, Level3.BOUNDARY_HARMONY],
    Level2.SPACING: [Level3.MISSING_SPACE, Level3.EXTRA_SPACE],
    Level2.CASE: [Level3.NOMINATIVE, Level3.GENITIVE, Level3.DATIVE, Level3.ACCUSATIVE,
                  Level3.LOCATIVE, Level3.ABLATIVE, Level3.INSTRUMENTAL],
    Level2.POSSESSIVE: [Level3.PERSON_MISMATCH, Level3.NUMBER_MISMATCH],
    Level2.PERSONAL_ENDING: [Level3.PERSON, Level3.NUMBER, Level3.TENSE_AGREEMENT],
    Level2.PLURAL: [Level3.EXTRA_PLURAL, Level3.MISSING_PLURAL, Level3.ALLOMORPH],
    Level2.NEGATION: [Level3.DOUBLE_NEGATION, Level3.WRONG_FORM],
    Level2.TENSE: [Level3.PAST_PRESENT, Level3.PRESENT_FUTURE, Level3.PAST_FUTURE],
    Level2.POSTPOSITION: [Level3.WRONG_POSTPOSITION, Level3.CASE_GOVERNMENT],
    Level2.AGREEMENT: [Level3.SUBJECT_VERB, Level3.MODIFIER_HEAD],
    Level2.DERIVATION: [Level3.WRONG_DERIVATIONAL_SUFFIX],
    Level2.WORD_ORDER: [Level3.VERB_POSITION, Level3.MODIFIER_POSITION],
    Level2.CLAUSE_STRUCTURE: [Level3.FRAGMENTED, Level3.RUN_ON],
    Level2.MISSING_ELEMENT: [Level3.DROPPED_ARGUMENT, Level3.MISSING_COPULA],
    Level2.REDUNDANT_ELEMENT: [Level3.REPEATED_WORD, Level3.PLEONASM],
    Level2.DISCOURSE: [Level3.CONNECTOR_MISUSE],
}

_L2_TO_L1: dict[Level2, Level1] = {}
for _l1, _l2s in ALL_L2_FOR_L1.items():
    for _l2 in _l2s:
        _L2_TO_L1[_l2] = _l1

_L3_TO_L2: dict[Level3, Level2] = {}
for _l2, _l3s in ALL_L3_FOR_L2.items():
    for _l3 in _l3s:
        _L3_TO_L2[_l3] = _l2


@dataclass(frozen=True)
class ErrorAnnotation:
    l1: Level1
    l2: Level2
    l3: Level3 | None = None

    @property
    def tag(self) -> str:
        parts = [self.l1.value, self.l2.value]
        if self.l3 is not None:
            parts.append(self.l3.value)
        return "/".join(parts)

    def to_dict(self) -> dict:
        d = {"l1": self.l1.value, "l2": self.l2.value}
        if self.l3 is not None:
            d["l3"] = self.l3.value
        return d


def parse_annotation(tag: str) -> ErrorAnnotation:
    parts = tag.split("/")
    l1 = Level1(parts[0])
    l2 = Level2(parts[1])
    l3 = Level3(parts[2]) if len(parts) > 2 else None
    return ErrorAnnotation(l1=l1, l2=l2, l3=l3)


def get_l1_for_l2(l2: Level2) -> Level1:
    return _L2_TO_L1[l2]


def get_l2_for_l3(l3: Level3) -> Level2:
    return _L3_TO_L2[l3]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd gec-paper && python -m pytest tests/test_taxonomy.py -v
```
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add gec-paper/src/gecpaper/taxonomy/ gec-paper/tests/test_taxonomy.py
git commit -m "feat(gec-paper): implement 3-level error taxonomy schema with 3/18/~40 categories"
```

---

### Task 3: Taxonomy Classifier

**Files:**
- Create: `gec-paper/src/gecpaper/taxonomy/classifier.py`

- [ ] **Step 1: Implement rule-based L1/L2 classifier**

```python
# gec-paper/src/gecpaper/taxonomy/classifier.py
from __future__ import annotations

import re
from difflib import SequenceMatcher

from gecpaper.taxonomy.schema import (
    ErrorAnnotation, Level1, Level2, Level3,
)

BACK_VOWELS = set("аоұы")
FRONT_VOWELS = set("әөүі")
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS

POSTPOSITIONS = {
    "туралы", "үшін", "арқылы", "бойынша", "сияқты", "тәрізді",
    "кейін", "бұрын", "дейін", "бері", "сайын", "жайлы", "басқа",
}

PLURAL_SUFFIXES = {"лар", "лер", "дар", "дер", "тар", "тер"}


def classify(source: str, target: str) -> ErrorAnnotation:
    src = source.strip()
    tgt = target.strip()

    if src == tgt:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)

    src_words = src.split()
    tgt_words = tgt.split()

    src_nospace = src.replace(" ", "")
    tgt_nospace = tgt.replace(" ", "")
    if src_nospace == tgt_nospace:
        l3 = Level3.EXTRA_SPACE if len(src_words) > len(tgt_words) else Level3.MISSING_SPACE
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPACING, l3=l3)

    if len(src_words) == len(tgt_words) and sorted(src_words) == sorted(tgt_words):
        return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.WORD_ORDER)

    src_letters = re.sub(r"[^\w\s]", "", src)
    tgt_letters = re.sub(r"[^\w\s]", "", tgt)
    if src_letters == tgt_letters:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.PUNCTUATION)

    if len(src_words) != len(tgt_words):
        diff = abs(len(src_words) - len(tgt_words))
        if diff >= 2:
            if len(src_words) > len(tgt_words):
                return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.REDUNDANT_ELEMENT)
            return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.MISSING_ELEMENT)

    changed_pairs = []
    if len(src_words) == len(tgt_words):
        for sw, tw in zip(src_words, tgt_words):
            if sw != tw:
                changed_pairs.append((sw, tw))

    if not changed_pairs and len(src_words) != len(tgt_words):
        return ErrorAnnotation(l1=Level1.SYNTAX_DISCOURSE, l2=Level2.CLAUSE_STRUCTURE)

    if len(changed_pairs) == 1:
        sw, tw = changed_pairs[0]
        ann = _classify_word_pair(sw, tw)
        if ann is not None:
            return ann

    if changed_pairs:
        all_suffix = True
        for sw, tw in changed_pairs:
            ratio = SequenceMatcher(None, sw, tw).ratio()
            if ratio < 0.5:
                all_suffix = False
                break
        if all_suffix:
            return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.AGREEMENT)

    overall = SequenceMatcher(None, src, tgt).ratio()
    if overall > 0.85:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)

    return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.CASE)


def _classify_word_pair(src_word: str, tgt_word: str) -> ErrorAnnotation | None:
    sw_lower = src_word.lower()
    tw_lower = tgt_word.lower()

    if sw_lower in POSTPOSITIONS or tw_lower in POSTPOSITIONS:
        return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.POSTPOSITION)

    prefix_len = 0
    for a, b in zip(sw_lower, tw_lower):
        if a == b:
            prefix_len += 1
        else:
            break

    src_suffix = sw_lower[prefix_len:]
    tgt_suffix = tw_lower[prefix_len:]

    src_vowels = [c for c in src_suffix if c in ALL_VOWELS]
    tgt_vowels = [c for c in tgt_suffix if c in ALL_VOWELS]
    if src_vowels and tgt_vowels:
        src_back = any(v in BACK_VOWELS for v in src_vowels)
        tgt_back = any(v in BACK_VOWELS for v in tgt_vowels)
        if src_back != tgt_back:
            return ErrorAnnotation(
                l1=Level1.ORTHOGRAPHY, l2=Level2.VOWEL_HARMONY,
                l3=Level3.FRONT_BACK_MISMATCH,
            )

    if src_suffix in PLURAL_SUFFIXES or tgt_suffix in PLURAL_SUFFIXES:
        return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.PLURAL, l3=Level3.ALLOMORPH)

    if prefix_len >= 2 and len(src_suffix) <= 4 and len(tgt_suffix) <= 4:
        return ErrorAnnotation(l1=Level1.MORPHOSYNTAX, l2=Level2.CASE)

    if SequenceMatcher(None, sw_lower, tw_lower).ratio() > 0.8:
        return ErrorAnnotation(l1=Level1.ORTHOGRAPHY, l2=Level2.SPELLING)

    return None
```

- [ ] **Step 2: Verify on known examples**

```bash
cd gec-paper && python -c "
from gecpaper.taxonomy.classifier import classify
pairs = [
    ('Қабырғаларде өрнектер', 'Қабырғаларда өрнектер'),
    ('Студенттар оқиды', 'Студенттер оқиды'),
    ('Жақсы оқиды ол', 'Ол жақсы оқиды'),
]
for s, t in pairs:
    print(f'{classify(s, t).tag:40s} {s} -> {t}')
"
```

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/taxonomy/classifier.py
git commit -m "feat(gec-paper): add rule-based error classifier for L1/L2/L3 taxonomy"
```

---

### Task 4: Scoring Metrics

**Files:**
- Create: `gec-paper/src/gecpaper/scoring/metrics.py`
- Create: `gec-paper/tests/test_metrics.py`

- [ ] **Step 1: Write tests for metrics**

```python
# gec-paper/tests/test_metrics.py
from gecpaper.scoring.metrics import (
    char_error_rate,
    word_level_edits,
    compute_word_f05,
    compute_gleu,
    multi_ref_word_f05,
    multi_ref_cer,
)


def test_cer_identical():
    assert char_error_rate("hello", "hello") == 0.0


def test_cer_completely_different():
    assert char_error_rate("abc", "xyz") == 1.0


def test_cer_one_edit():
    assert abs(char_error_rate("helo", "hello") - 0.2) < 0.01


def test_word_edits_no_change():
    assert word_level_edits("a b c", "a b c") == set()


def test_word_edits_substitution():
    edits = word_level_edits("a b c", "a x c")
    assert len(edits) > 0


def test_f05_perfect():
    source = "He go to school"
    target = "He goes to school"
    prediction = "He goes to school"
    result = compute_word_f05(source, prediction, target)
    assert result["f05"] == 1.0


def test_f05_no_correction():
    source = "He go to school"
    target = "He goes to school"
    prediction = "He go to school"
    result = compute_word_f05(source, prediction, target)
    assert result["f05"] == 0.0


def test_multi_ref_f05_takes_max():
    source = "He go to school"
    refs = ["He goes to school", "He went to school"]
    prediction = "He went to school"
    score = multi_ref_word_f05(source, prediction, refs)
    assert score == 1.0


def test_multi_ref_cer_takes_min():
    refs = ["hello world", "hi world"]
    assert multi_ref_cer("hello world", refs) == 0.0


def test_gleu_perfect():
    score = compute_gleu("a b c", "a x c", "a x c")
    assert score > 0.9
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd gec-paper && python -m pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement metrics**

```python
# gec-paper/src/gecpaper/scoring/metrics.py
from __future__ import annotations

from collections import Counter


def char_error_rate(prediction: str, reference: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    m, n = len(prediction), len(reference)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if prediction[i - 1] == reference[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / n


def word_level_edits(source: str, text: str) -> set[tuple]:
    src_words = source.split()
    txt_words = text.split()
    edits = set()
    m, n = len(src_words), len(txt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_words[i - 1] == txt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i, j = m, n
    while i > 0 and j > 0:
        if src_words[i - 1] == txt_words[j - 1]:
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            edits.add(("del", i - 1, src_words[i - 1]))
            i -= 1
        else:
            edits.add(("ins", j - 1, txt_words[j - 1]))
            j -= 1
    while i > 0:
        edits.add(("del", i - 1, src_words[i - 1]))
        i -= 1
    while j > 0:
        edits.add(("ins", j - 1, txt_words[j - 1]))
        j -= 1
    return edits


def compute_word_f05(
    source: str, prediction: str, reference: str, beta: float = 0.5,
) -> dict:
    gold_edits = word_level_edits(source, reference)
    pred_edits = word_level_edits(source, prediction)
    tp = len(gold_edits & pred_edits)
    fp = len(pred_edits - gold_edits)
    fn = len(gold_edits - pred_edits)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = beta**2 * precision + recall
    f05 = (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "f05": f05, "tp": tp, "fp": fp, "fn": fn}


def _ngrams(words: list[str], n: int) -> Counter:
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def compute_gleu(source: str, prediction: str, reference: str, max_n: int = 4) -> float:
    src_words = source.split()
    pred_words = prediction.split()
    ref_words = reference.split()
    all_src_ngrams = Counter()
    all_ref_ngrams = Counter()
    all_pred_ngrams = Counter()
    for n in range(1, max_n + 1):
        all_src_ngrams += _ngrams(src_words, n)
        all_ref_ngrams += _ngrams(ref_words, n)
        all_pred_ngrams += _ngrams(pred_words, n)
    ref_diff = all_ref_ngrams - all_src_ngrams
    key_ngrams = ref_diff + all_ref_ngrams
    num = sum((all_pred_ngrams & key_ngrams).values())
    denom = max(sum(all_pred_ngrams.values()), 1)
    return num / denom


def multi_ref_word_f05(source: str, prediction: str, references: list[str]) -> float:
    return max(compute_word_f05(source, prediction, ref)["f05"] for ref in references)


def multi_ref_cer(prediction: str, references: list[str]) -> float:
    return min(char_error_rate(prediction, ref) for ref in references)


def multi_ref_exact_match(prediction: str, references: list[str]) -> bool:
    pred = prediction.strip()
    return any(pred == ref.strip() for ref in references)


def multi_ref_gleu(source: str, prediction: str, references: list[str]) -> float:
    return sum(compute_gleu(source, prediction, ref) for ref in references) / len(references)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd gec-paper && python -m pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add gec-paper/src/gecpaper/scoring/metrics.py gec-paper/tests/test_metrics.py
git commit -m "feat(gec-paper): implement GEC scoring metrics with multi-reference support"
```

---

### Task 5: Synthetic Data Generator (Round 1 — 10K)

**Files:**
- Create: `gec-paper/src/gecpaper/data/synthetic.py`
- Create: `gec-paper/scripts/generate_synthetic.py`

- [ ] **Step 1: Implement synthetic generator with taxonomy prompts**

Create `gec-paper/src/gecpaper/data/synthetic.py` — GPT-4o-based taxonomy-aware corruption engine. Per L2 category, uses a specific Kazakh-language prompt describing the error type. Generates balanced ~N per L2 category. Outputs JSONL with fields: `input`, `target`, `error_tag`, `error_description`, `source`.

Key function: `generate_balanced_dataset(client, seed_texts, target_per_l2=600, model="gpt-4o", output_path=None)` — iterates over all L2 categories, generates batches, deduplicates via MD5 hash, appends to output JSONL.

- [ ] **Step 2: Create CLI script `gec-paper/scripts/generate_synthetic.py`**

Loads seeds from MDBKD (streaming, filter 20-300 chars), calls `generate_balanced_dataset`, reports hit rate.

- [ ] **Step 3: Dry run with 2 per category**

```bash
cd gec-paper && python scripts/generate_synthetic.py --target_per_l2 2 --max_seeds 100 --output data/synthetic_test.jsonl
head -3 data/synthetic_test.jsonl
```

- [ ] **Step 4: Commit**

```bash
git add gec-paper/src/gecpaper/data/synthetic.py gec-paper/scripts/generate_synthetic.py
git commit -m "feat(gec-paper): add taxonomy-aware synthetic GEC data generator via GPT-4o"
```

---

### Task 6: NLLB-600M Fine-Tuning Script

**Files:**
- Create: `gec-paper/src/gecpaper/models/nllb_gec.py`
- Create: `gec-paper/scripts/train_nllb.py`
- Create: `gec-paper/configs/round1_nllb_baseline.yaml`

- [ ] **Step 1: Implement NLLB data collator + training loop**

Create `gec-paper/src/gecpaper/models/nllb_gec.py` with:
- `NLLBGECCollator`: seq2seq collator, source=input, target=target, uses NLLB tokenizer with `kaz_Cyrl` lang code
- `load_gec_jsonl(path)`: loads JSONL into HF Dataset
- `train_nllb_gec(config)`: loads model (`facebook/nllb-200-distilled-600M`), splits data 95/5, runs `Seq2SeqTrainer`
- `generate_correction(model, tokenizer, text, num_beams=5)`: beam search with `forced_bos_token_id=kaz_Cyrl`

- [ ] **Step 2: Create CLI `gec-paper/scripts/train_nllb.py`**

Reads YAML config, optional `--data_path` override, calls `train_nllb_gec`.

- [ ] **Step 3: Create Round 1 config**

```yaml
# gec-paper/configs/round1_nllb_baseline.yaml
model_name: facebook/nllb-200-distilled-600M
data_path: data/synthetic.jsonl
output_dir: outputs/round1_nllb_baseline
num_train_epochs: 3
batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 3e-5
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
max_source_length: 256
max_target_length: 256
logging_steps: 50
save_steps: 500
report_to: none
```

- [ ] **Step 4: Commit**

```bash
git add gec-paper/src/gecpaper/models/nllb_gec.py gec-paper/scripts/train_nllb.py gec-paper/configs/round1_nllb_baseline.yaml
git commit -m "feat(gec-paper): add NLLB-600M GEC fine-tuning with seq2seq collator"
```

---

### Task 7: Benchmark Runner

**Files:**
- Create: `gec-paper/src/gecpaper/scoring/benchmark.py`
- Create: `gec-paper/scripts/run_scoring.py`

- [ ] **Step 1: Implement benchmark runner**

Create `gec-paper/src/gecpaper/scoring/benchmark.py` with:
- `run_assessment(predict_fn, test_data, multi_ref=False)`: runs prediction on all test examples, computes EM, CER, F0.5, GLEU, identity preservation, per-L1 breakdown. Returns `{"metrics": {...}, "results": [...]}`
- `print_metrics(metrics, model_name)`: formatted console output

- [ ] **Step 2: Create CLI `gec-paper/scripts/run_scoring.py`**

`--model`, `--test_data`, `--model_type` (nllb|causal), `--multi_ref` flag. Loads model, wraps in predict_fn, calls runner.

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/scoring/benchmark.py gec-paper/scripts/run_scoring.py
git commit -m "feat(gec-paper): add benchmark runner with per-L1 breakdown and multi-ref support"
```

---

# ROUND 2: FULL SYSTEM

---

### Task 8: Wikipedia Edit Extractor

**Files:**
- Create: `gec-paper/src/gecpaper/data/organic_wiki.py`
- Create: `gec-paper/scripts/collect_wiki_edits.py`

- [ ] **Step 1: Implement Wikipedia kk edit extraction**

Create `gec-paper/src/gecpaper/data/organic_wiki.py` with:
- `get_recent_changes(limit)`: MediaWiki API, paginated, namespace=0 (articles)
- `get_revision_content(revid)`: fetch revision text
- `_strip_markup(text)`: remove wikitext markup (links, templates, headers)
- `_extract_sentences(text)`: split into sentences, filter by length
- `extract_edit_pairs(max_revisions, max_edit_ratio=0.20)`: for each edit, diff old/new revisions, use SequenceMatcher to find replaced sentences with >80% similarity, deduplicate, output JSONL

- [ ] **Step 2: Create CLI script**

`--max_revisions` (default 5000), `--output` (default `data/organic_wiki.jsonl`)

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/data/organic_wiki.py gec-paper/scripts/collect_wiki_edits.py
git commit -m "feat(gec-paper): add Wikipedia kk edit history extractor for organic GEC data"
```

---

### Task 9: Social Media Organic Collector

**Files:**
- Create: `gec-paper/src/gecpaper/data/organic_social.py`
- Create: `gec-paper/scripts/collect_social_data.py`

- [ ] **Step 1: Implement social → GPT-4o correction pipeline**

Create `gec-paper/src/gecpaper/data/organic_social.py` with:
- `correct_texts(client, texts, model="gpt-4o", output_path=None)`: for each text, GPT-4o returns `{corrected, has_errors, error_types}`. Filters identity/invalid, outputs JSONL.

- [ ] **Step 2: Create CLI script**

`--input` (text file, one per line), `--output`, `--model`

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/data/organic_social.py gec-paper/scripts/collect_social_data.py
git commit -m "feat(gec-paper): add social media organic GEC collector via GPT-4o"
```

---

### Task 10: Data Mixer

**Files:**
- Create: `gec-paper/src/gecpaper/data/mixer.py`
- Create: `gec-paper/scripts/mix_dataset.py`

- [ ] **Step 1: Implement mixer**

Create `gec-paper/src/gecpaper/data/mixer.py` with:
- `mix_datasets(sources: dict[str, Path], identity_ratio=0.05)`: loads all JSONL sources, deduplicates by MD5(input+target), auto-classifies via `classify()` if missing `error_tag`, adds identity examples, shuffles, splits 90/5/5
- `save_splits(splits, output_dir)`: saves train/validation/test JSONL

- [ ] **Step 2: Create CLI script**

`--synthetic`, `--wiki`, `--social`, `--output_dir`, `--identity_ratio`

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/data/mixer.py gec-paper/scripts/mix_dataset.py
git commit -m "feat(gec-paper): add data mixer with dedup, auto-classification, and split"
```

---

### Task 11: Apertium-kaz Morpheme Segmenter

**Files:**
- Create: `gec-paper/src/gecpaper/morph/apertium.py`
- Create: `gec-paper/tests/test_morph.py`

- [ ] **Step 1: Write tests**

Test fallback behavior when apertium not installed, test word count preservation.

- [ ] **Step 2: Implement wrapper**

Create `gec-paper/src/gecpaper/morph/apertium.py` with:
- `is_available()`: check if `apertium` in PATH
- `segment_word(word)`: pipe through `apertium kaz-morph`, parse output into `stem|suffix` format
- `segment_text(text, fallback=True)`: segment each word, fallback to original if unavailable

- [ ] **Step 3: Run tests, commit**

```bash
git add gec-paper/src/gecpaper/morph/apertium.py gec-paper/tests/test_morph.py
git commit -m "feat(gec-paper): add apertium-kaz morpheme segmenter wrapper"
```

---

### Task 12: Edit Tag Extraction and Tagger

**Files:**
- Create: `gec-paper/src/gecpaper/models/edit_tagger.py`
- Create: `gec-paper/scripts/train_tagger.py`
- Create: `gec-paper/configs/round2_tagger.yaml`
- Create: `gec-paper/tests/test_edit_tags.py`

- [ ] **Step 1: Write tests for edit tag extraction**

Test: `$KEEP` for identical words, `$REPLACE_X` for changed words, `$DELETE` for removed words, `build_tag_vocab` returns top-K, `apply_tags` reconstructs.

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Implement edit tagger module**

Create `gec-paper/src/gecpaper/models/edit_tagger.py` with:
- `extract_edit_tags(src_words, tgt_words)`: LCS alignment → per-token tag
- `build_tag_vocab(examples, top_k=2000)`: Counter → top-K tag vocabulary
- `apply_tags(src_words, tags)`: reconstruct output from tags
- `EditTagDataset`: PyTorch dataset, tokenizes with XLM-R, maps word-level tags to subword labels
- `train_edit_tagger(config)`: loads data, builds vocab, trains `AutoModelForTokenClassification`

- [ ] **Step 4: Create CLI + config**

Config: `xlm-roberta-base`, `top_k_tags: 2000`, `epochs: 5`, `batch_size: 32`, `lr: 5e-5`

- [ ] **Step 5: Run tests — verify they pass**

- [ ] **Step 6: Commit**

```bash
git add gec-paper/src/gecpaper/models/edit_tagger.py gec-paper/scripts/train_tagger.py \
       gec-paper/configs/round2_tagger.yaml gec-paper/tests/test_edit_tags.py
git commit -m "feat(gec-paper): add data-derived edit tagger with XLM-R token classifier"
```

---

### Task 13: Dual-Model Inference Pipeline

**Files:**
- Create: `gec-paper/src/gecpaper/pipeline.py`
- Create: `gec-paper/scripts/run_pipeline.py`
- Create: `gec-paper/tests/test_pipeline.py`

- [ ] **Step 1: Write tests**

Test with mock functions: tagger-only, seq2seq-only, cascade mode.

- [ ] **Step 2: Implement `DualPipeline`**

Modes: `tagger_only`, `seq2seq_only`, `cascade`. Cascade: tagger first (high-confidence), then NLLB on result. Optional morph_fn and reranker_fn. Strips `|` from output.

- [ ] **Step 3: Create CLI `run_pipeline.py`**

Interactive mode + file mode (`--input_file` JSONL).

- [ ] **Step 4: Run tests, commit**

```bash
git add gec-paper/src/gecpaper/pipeline.py gec-paper/scripts/run_pipeline.py gec-paper/tests/test_pipeline.py
git commit -m "feat(gec-paper): add dual-model inference pipeline (tagger + NLLB cascade)"
```

---

### Task 14: Multi-Reference Benchmark Generator

**Files:**
- Create: `gec-paper/src/gecpaper/data/multi_ref.py`
- Create: `gec-paper/scripts/generate_multi_ref.py`

- [ ] **Step 1: Implement multi-ref generator**

5 strategies: minimal, conservative, moderate, fluent, alternative. For each sentence, generates 5 corrections via GPT-4o, deduplicates, keeps 2-4 unique references.

- [ ] **Step 2: Create CLI script**

`--test_data`, `--output`, `--max_sentences 700`, `--model gpt-4o`

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/data/multi_ref.py gec-paper/scripts/generate_multi_ref.py
git commit -m "feat(gec-paper): add LLM multi-reference benchmark builder (5 strategies)"
```

---

### Task 15: Per-Category Analysis and Significance Testing

**Files:**
- Create: `gec-paper/src/gecpaper/scoring/analysis.py`

- [ ] **Step 1: Implement analysis module**

- `per_category_breakdown(test_data, predictions, level="l2")`: groups results by L1 or L2, computes per-group F0.5/precision/recall/EM
- `bootstrap_significance(scores_a, scores_b, n_bootstrap=1000)`: paired bootstrap, returns p-value

- [ ] **Step 2: Commit**

```bash
git add gec-paper/src/gecpaper/scoring/analysis.py
git commit -m "feat(gec-paper): add per-category breakdown and bootstrap significance testing"
```

---

# ROUND 3: ABLATION + PAPER

---

### Task 16: Qwen-Distilled Morpheme Segmenter

**Files:**
- Create: `gec-paper/src/gecpaper/morph/segmenter.py`
- Create: `gec-paper/scripts/train_morph_segmenter.py`

- [ ] **Step 1: Implement distillation + char-level model**

- `generate_segmentation_data(wordforms, model_name)`: prompts Qwen 500M for morpheme segmentations, sanity-filters (reconstructed == original), outputs JSONL
- `CharSeq2SeqDataset`: char-level dataset with vocabulary
- `CharSegmenterModel`: small transformer encoder (~5M params), char→char with `|` insertion

- [ ] **Step 2: Create CLI `train_morph_segmenter.py`**

Two modes: `generate` (create training data from Qwen), `train` (train char-level model).

- [ ] **Step 3: Commit**

```bash
git add gec-paper/src/gecpaper/morph/segmenter.py gec-paper/scripts/train_morph_segmenter.py
git commit -m "feat(gec-paper): add Qwen-distilled char-level morpheme segmenter"
```

---

### Task 17: Remaining Configs

**Files:**
- Create: `gec-paper/configs/round2_nllb_organic.yaml`
- Create: `gec-paper/configs/round3_ablation_morph.yaml`

- [ ] **Step 1: Create configs, commit**

Round 2: full data, wandb reporting. Round 3: `morpheme_segmented: true`, max_length 300.

```bash
git add gec-paper/configs/
git commit -m "feat(gec-paper): add Round 2 and Round 3 training configs"
```

---

### Task 18: Paper LaTeX Skeleton

**Files:**
- Create: `gec-paper/paper/main.tex`

- [ ] **Step 1: Create LaTeX skeleton**

Sections: Introduction, Related Work, Kazakh Error Taxonomy, Data, Method (4 subsections), Multi-Reference Benchmark, Experiments (main results table placeholder), Ablation Studies, Analysis, Conclusion. Appendices: Full Taxonomy, Data Generation Prompts, Additional Results.

- [ ] **Step 2: Commit**

```bash
git add gec-paper/paper/main.tex
git commit -m "feat(gec-paper): add LaTeX paper skeleton"
```

---

## Execution Checklist

### Round 1 (Baseline) — Tasks 1-7
- [ ] Task 1: Project Scaffold
- [ ] Task 2: Error Taxonomy Schema
- [ ] Task 3: Taxonomy Classifier
- [ ] Task 4: Scoring Metrics
- [ ] Task 5: Synthetic Data Generator (10K)
- [ ] Task 6: NLLB-600M Fine-Tuning Script
- [ ] Task 7: Benchmark Runner

### Round 2 (Full System) — Tasks 8-15
- [ ] Task 8: Wikipedia Edit Extractor
- [ ] Task 9: Social Media Organic Collector
- [ ] Task 10: Data Mixer
- [ ] Task 11: Apertium-kaz Morpheme Segmenter
- [ ] Task 12: Edit Tag Extraction and Tagger
- [ ] Task 13: Dual-Model Inference Pipeline
- [ ] Task 14: Multi-Reference Benchmark Generator
- [ ] Task 15: Per-Category Analysis

### Round 3 (Ablation + Paper) — Tasks 16-18
- [ ] Task 16: Qwen-Distilled Morpheme Segmenter
- [ ] Task 17: Remaining Configs
- [ ] Task 18: Paper LaTeX Skeleton
