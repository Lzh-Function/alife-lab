"""Build an ``nca`` experiment spec from an ``nca_train`` checkpoint (NumPy only, no JAX)."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import numpy as np

from alife_lab.core.spec import NCATrainExperiment


def nca_spec_from_training(
    train: NCATrainExperiment, payload: dict[str, Any], run_id: str, checkpoint_sha256: str
) -> dict[str, Any]:
    arrays = payload["arrays"]
    raw = b"".join(np.asarray(arrays[f"w.{k}"], dtype="<f4").tobytes() for k in ("w1", "b1", "w2"))
    tick = int(payload["meta"]["tick"])
    return {
        "schema_version": 1,
        "experiment": {
            "name": f"{train.name}_model_s{tick}"[:120],
            "mode": "exploratory",
            "substrate": "nca",
            "seed": train.seed,
            "target": train.target.model_dump(mode="json"),
            "pad": train.dynamics.pad,
            "dynamics": {
                "architecture": train.dynamics.architecture.model_dump(mode="json"),
                "weights_kind": "trained",
                "weights": {
                    "data_b64": base64.b64encode(raw).decode("ascii"),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "source": {
                        "training_run_id": run_id,
                        "optimizer_step": tick,
                        "checkpoint_sha256": checkpoint_sha256,
                        "training_experiment": train.dynamics.experiment,
                    },
                },
            },
        },
    }
