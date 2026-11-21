"""Catalogued multichannel Lenia creatures (data in ``patterns.json``; see its ``_source``)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

_PATH = Path(__file__).with_name("patterns.json")


@cache
def _all() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_PATH.read_text(encoding="utf-8"))
    pats: dict[str, Any] = data["patterns"]
    return pats


def load_pattern(name: str) -> dict[str, Any]:
    pats = _all()
    if name not in pats:
        raise ValueError(f"unknown multichannel Lenia pattern {name!r}")
    return pats[name]  # type: ignore[no-any-return]


def source_note() -> str:
    return str(json.loads(_PATH.read_text(encoding="utf-8"))["_source"])
