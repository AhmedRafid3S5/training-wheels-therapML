from __future__ import annotations

import functools
import pathlib

FIXTURES_PATH = pathlib.Path(__file__).resolve().parent / "fixtures"

# TinyShakespeare (Karpathy, char-rnn), 1,115,394 bytes, public domain.
TINYSHAKESPEARE_PATH = FIXTURES_PATH / "tinyshakespeare.txt"


@functools.cache
def load_tinyshakespeare() -> str:
    """Return the full TinyShakespeare corpus as a string (cached)."""
    return TINYSHAKESPEARE_PATH.read_text(encoding="utf-8")
