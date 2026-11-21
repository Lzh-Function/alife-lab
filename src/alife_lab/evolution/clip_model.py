"""CLIP image/text embeddings for ASAL objectives (PyTorch + open_clip, optional extra ``asal``).

Model: OpenAI CLIP ViT-B/32 (the model of the ASAL paper and ``SakanaAI/asal``), loaded with
open_clip from the Hugging Face repository ``timm/vit_base_patch32_clip_224.openai`` at a
pinned revision. Only the ``.safetensors`` file is downloaded and its SHA-256 is verified before
loading; no pickle (``torch.load``) is ever used.

Preprocessing follows ``SakanaAI/asal`` for Lenia: a single-channel field in [0, 1] is shown as
grey RGB, resized to 224 x 224 with nearest-neighbour interpolation, normalised with the CLIP
mean/std; embeddings are L2-normalised. Inference runs in ``eval()`` mode without gradients and
with deterministic algorithms requested. Imported only in workers (never in the API process).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ModelCard:
    key: str
    open_clip_name: str
    repo_id: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    quick_gelu: bool
    licence: str


MODELS = {
    "openai_vit_b32": ModelCard(
        key="openai_vit_b32",
        open_clip_name="ViT-B-32",
        repo_id="timm/vit_base_patch32_clip_224.openai",
        revision="a6f597a30f7b82c51704746581f9a4e41421e878",
        filename="open_clip_model.safetensors",
        sha256="e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31",
        size_bytes=605_143_284,
        quick_gelu=True,  # OpenAI checkpoints were trained with QuickGELU
        licence="MIT (OpenAI CLIP weights; repository card: apache-2.0)",
    )
}
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 224


def model_dir() -> Path:
    env = os.environ.get("ALIFE_MODEL_DIR")
    return Path(env) if env else Path.home() / ".cache" / "alife_lab" / "models"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_weights(card: ModelCard) -> Path:
    """Download (once) the pinned safetensors file and verify its SHA-256."""
    target = model_dir() / card.key / card.filename
    if target.is_file() and sha256_file(target) == card.sha256:
        return target
    from huggingface_hub import hf_hub_download

    staging = target.parent / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp = Path(
        hf_hub_download(
            repo_id=card.repo_id,
            filename=card.filename,
            revision=card.revision,
            local_dir=str(staging),
        )
    )
    digest = sha256_file(tmp)
    if digest != card.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{card.repo_id}/{card.filename}: SHA-256 {digest} != pinned")
    tmp.replace(target)  # only a verified file ever appears at the final path
    return target


class ClipEmbedder:
    def __init__(self, model_key: str = "openai_vit_b32", device: str = "cpu") -> None:
        import open_clip
        import torch
        from safetensors.torch import load_file

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        card = MODELS[model_key]
        self.card = card
        self.device = torch.device(device)
        path = fetch_weights(card)
        import logging

        oc_log = logging.getLogger()  # open_clip warns "initialized randomly" before we load
        level = oc_log.level
        oc_log.setLevel(logging.ERROR)
        try:
            model = open_clip.create_model(
                card.open_clip_name, pretrained=None, quick_gelu=card.quick_gelu
            )
        finally:
            oc_log.setLevel(level)
        missing, unexpected = model.load_state_dict(load_file(str(path)), strict=False)
        if unexpected or [k for k in missing if not k.endswith("position_ids")]:
            raise RuntimeError(f"CLIP weights mismatch: missing={missing} unexpected={unexpected}")
        model.eval().requires_grad_(False)
        self.model = model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(card.open_clip_name)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        self.versions = {
            "torch": torch.__version__,
            "open_clip": getattr(open_clip, "__version__", "unknown"),
            "cuda": torch.version.cuda,
            "device": str(self.device),
            "weights_sha256": card.sha256,
        }
        self._text_cache: dict[tuple[str, ...], F64] = {}

    def embed_fields(self, fields: npt.NDArray[Any]) -> F64:
        """(N, H, W) grey or (N, H, W, 3) RGB values in [0, 1] -> (N, D) unit vectors."""
        import torch

        x = torch.as_tensor(np.clip(fields, 0.0, 1.0), dtype=torch.float32, device=self.device)
        # RGB (multichannel Lenia) or grey shown as RGB
        x = x.permute(0, 3, 1, 2).contiguous() if x.ndim == 4 else x[:, None].expand(-1, 3, -1, -1)
        x = torch.nn.functional.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            z = self.model.encode_image(x).float()
        z = z / z.norm(dim=-1, keepdim=True)
        out: F64 = z.cpu().numpy().astype(np.float64)
        return out

    def embed_texts(self, prompts: list[str]) -> F64:
        import torch

        key = tuple(prompts)
        if key not in self._text_cache:
            tokens = self.tokenizer(prompts).to(self.device)
            with torch.inference_mode():
                z = self.model.encode_text(tokens).float()
            z = z / z.norm(dim=-1, keepdim=True)
            self._text_cache[key] = z.cpu().numpy().astype(np.float64)
        return self._text_cache[key]


_EMBEDDERS: dict[tuple[str, str], ClipEmbedder] = {}


def get_embedder(model_key: str, device: str) -> ClipEmbedder:
    """One embedder per (model, device) and process (workers load it once)."""
    key = (model_key, device)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = ClipEmbedder(model_key, device)
    return _EMBEDDERS[key]
