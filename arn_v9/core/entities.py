"""
Lightweight regex-based entity extraction for ARN v9.

Extracts named entities, quoted strings, code identifiers, URLs,
file paths, and numeric quantities to power entity-match retrieval.
"""

import re
from typing import List, Tuple

# Proper nouns (2+ capitalized words), excluding common sentence-start stopwords
_SENTENCE_STARTERS = {
    'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'It', 'He',
    'She', 'They', 'We', 'I', 'My', 'Our', 'Your', 'His', 'Her',
    'Its', 'Is', 'Are', 'Was', 'Were', 'Be', 'In', 'On', 'At', 'Of',
    'To', 'And', 'But', 'Or', 'For', 'With', 'By', 'From', 'So',
}

_RE_PROPER_NOUN = re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b')
_RE_QUOTED = re.compile(r'"([^"]{2,60})"')
_RE_CODE_IDENT = re.compile(r'\b([a-z_][a-z0-9_]{2,}(?:\.[a-z_][a-z0-9_]{2,})+)\b')
_RE_URL = re.compile(r'https?://[^\s]{4,}')
_RE_FILEPATH = re.compile(r'(?:^|[\s])(/[^\s]{3,}|~[^\s]{3,})', re.MULTILINE)
_RE_NUM_UNIT = re.compile(r'\b(\d+(?:\.\d+)?)\s*([A-Za-z]{1,5})\b')


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities from text. Returns [(entity_text, entity_type), ...]."""
    entities: List[Tuple[str, str]] = []
    seen: set = set()

    def add(text_val: str, etype: str):
        key = (text_val.lower(), etype)
        if key not in seen and len(text_val) >= 2:
            seen.add(key)
            entities.append((text_val, etype))

    for m in _RE_PROPER_NOUN.finditer(text):
        noun = m.group(1)
        if noun not in _SENTENCE_STARTERS:
            add(noun, 'proper_noun')

    for m in _RE_QUOTED.finditer(text):
        add(m.group(1), 'quoted')

    for m in _RE_CODE_IDENT.finditer(text):
        add(m.group(1), 'code_ident')

    for m in _RE_URL.finditer(text):
        add(m.group(0), 'url')

    for m in _RE_FILEPATH.finditer(text):
        add(m.group(1), 'filepath')

    for m in _RE_NUM_UNIT.finditer(text):
        add(f"{m.group(1)}{m.group(2)}", 'num_unit')

    return entities
