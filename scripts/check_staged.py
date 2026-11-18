"""Pre-commit gate: block staged secrets, runtime artifacts and oversized files.

Usage: uv run python scripts/check_staged.py   (exit 1 on any finding)
Install as a hook:
  printf '#!/bin/sh\nuv run python scripts/check_staged.py\n' > .git/hooks/pre-commit
"""

from __future__ import annotations

import re
import subprocess
import sys

MAX_BYTES = 1_000_000
FORBIDDEN_PATHS = re.compile(
    r"(^|/)(runs|\.alife|artifacts|scratch|checkpoints|node_modules|dist|\.venv)/"
    r"|\.(sqlite3?|db|npz|npy|parquet|zarr|mp4|ckpt|pt|pth|safetensors|onnx|pem|key)$"
    r"|(^|/)\.env($|\.)(?!example$)|(^|/)id_(rsa|ed25519)"
)
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(rb"(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
]


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def main() -> int:
    names = [
        n
        for n in git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").split(b"\0")
        if n
    ]
    problems: list[str] = []
    for raw in names:
        name = raw.decode()
        if FORBIDDEN_PATHS.search(name):
            problems.append(f"forbidden path (runtime artifact/secret): {name}")
            continue
        blob = git("show", f":{name}")
        if len(blob) > MAX_BYTES:
            problems.append(f"file larger than {MAX_BYTES} bytes: {name} ({len(blob)} bytes)")
        if name.endswith(("package-lock.json", "uv.lock")):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(blob):
                problems.append(f"possible secret ({pattern.pattern[:40]!r}) in {name}")
    for p in problems:
        print(f"[blocked] {p}", file=sys.stderr)
    print(f"checked {len(names)} staged files: {'FAIL' if problems else 'ok'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
