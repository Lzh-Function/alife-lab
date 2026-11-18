"""Hardware / software capability probe.

Every value is measured at call time. Missing tools or permissions are
reported as ``unknown`` / ``None`` rather than guessed. The controller never
imports GPU runtimes: GPUs are queried through ``nvidia-smi`` with a fixed
argument list (no shell).
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

_NVIDIA_QUERY = [
    "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version,compute_cap",
    "--format=csv,noheader,nounits",
]

OPTIONAL_BACKENDS = ("numba", "llvmlite", "jax", "jaxlib", "torch", "cupy", "mujoco", "warp-lang")
CORE_PACKAGES = ("fastapi", "starlette", "uvicorn", "pydantic", "numpy", "pyyaml", "psutil")


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return 127, "", f"{cmd[0]} not found"
    try:
        proc = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _version(cmd: list[str]) -> str | None:
    code, out, _ = _run(cmd)
    if code != 0 or not out:
        return None
    return out.splitlines()[0].strip()


def probe_gpus() -> dict[str, Any]:
    code, out, err = _run(["nvidia-smi", *_NVIDIA_QUERY])
    if code == 127:
        return {"status": "nvidia_smi_not_found", "devices": []}
    if code != 0:
        return {"status": "error", "error": err or out, "devices": []}
    devices = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue

        def _int(value: str) -> int | None:
            try:
                return int(float(value))
            except ValueError:
                return None

        devices.append(
            {
                "index": _int(parts[0]),
                "name": parts[1],
                "uuid_present": parts[2].startswith("GPU-"),
                "memory_total_mib": _int(parts[3]),
                "memory_used_mib": _int(parts[4]),
                "memory_free_mib": _int(parts[5]),
                "driver_version": parts[6],
                "compute_capability": parts[7],
            }
        )
    return {"status": "ok", "devices": devices}


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def probe_system(
    runs_root: Path | None = None, frontend_dist: Path | None = None
) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk_path = runs_root if runs_root is not None and runs_root.exists() else Path.cwd()
    disk = shutil.disk_usage(disk_path)
    gpus = probe_gpus()
    runs_writable: bool | None = None
    if runs_root is not None:
        try:
            runs_root.mkdir(parents=True, exist_ok=True)
            test = runs_root / ".write_probe"
            test.write_text("ok", encoding="utf-8")
            test.unlink()
            runs_writable = True
        except OSError:
            runs_writable = False
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "wsl": is_wsl(),
            "distribution": _os_release(),
        },
        "cpu": {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "model": _cpu_model(),
        },
        "memory": {"total_mb": vm.total // 2**20, "available_mb": vm.available // 2**20},
        "disk": {
            "path_checked": "runs_root" if disk_path == runs_root else "cwd",
            "total_gb": round(disk.total / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
            "runs_root_writable": runs_writable,
        },
        "gpu": gpus,
        "python": {"version": platform.python_version(), "implementation": sys.implementation.name},
        "tools": {
            "uv": _version(["uv", "--version"]),
            "node": _version(["node", "--version"]),
            "npm": _version(["npm", "--version"]),
            "git": _version(["git", "--version"]),
            "ffmpeg": _version(["ffmpeg", "-version"]),
        },
        "sqlite": {"library_version": sqlite3.sqlite_version},
        "packages": _package_versions(CORE_PACKAGES),
        "optional_backends": _package_versions(OPTIONAL_BACKENDS),
        "frontend": {
            "dist_present": bool(frontend_dist and (frontend_dist / "index.html").is_file()),
        },
        "env": {"ALIFE_RUNS_DIR_set": "ALIFE_RUNS_DIR" in os.environ},
    }


def _os_release() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        return None
    return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return platform.processor() or None
