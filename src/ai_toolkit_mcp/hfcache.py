"""Reuse weights already in the local Hugging Face cache instead of letting ai-toolkit re-download them.

ai-toolkit prefers ComfyUI-format weight files under its own MODELS_PATH for many models and downloads
them there when missing, ignoring the HF cache. Pointing the config at cached files skips that.
Only meaningful when ai-toolkit runs on the same machine as this server.
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")

# Models that load only ComfyUI split files, with per-file overrides in model.model_kwargs.
# Mirrors ai-toolkit extensions_built_in/diffusion_models/ltx2/ltx2.py (COMFY_LTX25_FILES).
SPLIT_FILE_MODELS: dict[str, tuple[str, dict[str, str]]] = {
    "ltx2.5": (
        "Lightricks/LTX-2.5",
        {
            "dit": "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
            "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
            "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        },
    ),
}


def hub_cache_dir() -> Path:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_local(url: str) -> bool:
    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


def _snapshots(cache: Path, repo_id: str) -> list[Path]:
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return []
    ordered = sorted((p for p in snaps.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    main = repo_dir / "refs" / "main"
    if main.is_file():
        head = snaps / main.read_text().strip()
        ordered.sort(key=lambda p: p != head)
    return ordered


def cached_file(cache: Path, repo_id: str, rel_path: str) -> Path | None:
    for snap in _snapshots(cache, repo_id):
        if (snap / rel_path).is_file():
            return snap / rel_path
    return None


def cached_diffusers_repo(cache: Path, repo_id: str) -> Path | None:
    """A snapshot holding a complete diffusers checkout (model_index.json plus every component folder)."""
    for snap in _snapshots(cache, repo_id):
        index = snap / "model_index.json"
        if not index.is_file():
            continue
        try:
            components = [k for k, v in json.loads(index.read_text()).items() if not k.startswith("_") and isinstance(v, list) and v[0]]
        except (ValueError, OSError):
            continue
        if all((snap / c).is_dir() for c in components):
            return snap
    return None


def use_cached_weights(proc: dict, cache: Path) -> list[str]:
    """Point process[0].model at cached weights where possible. Returns notes describing what changed."""
    model = proc.setdefault("model", {})
    name = model.get("name_or_path") or ""
    notes: list[str] = []
    split = SPLIT_FILE_MODELS.get(model.get("arch", ""))
    if split:
        repo, files = split
        repo = name if REPO_ID.match(name) else repo
        kwargs = model.setdefault("model_kwargs", {})
        for component, rel in files.items():
            key = f"{component}_path"
            if kwargs.get(key):
                continue
            found = cached_file(cache, repo, rel)
            if found:
                kwargs[key] = str(found)
                notes.append(f"model.model_kwargs.{key} -> cached {found.name}")
        return notes
    if REPO_ID.match(name) and not os.path.exists(name):
        snap = cached_diffusers_repo(cache, name)
        if snap:
            model["name_or_path"] = str(snap)
            notes.append(f"model.name_or_path -> cached {name} snapshot ({snap})")
    return notes


def download_warnings(proc: dict, cache: Path) -> list[str]:
    model = proc.get("model") or {}
    name = model.get("name_or_path") or ""
    split = SPLIT_FILE_MODELS.get(model.get("arch", ""))
    if split:
        kwargs = model.get("model_kwargs") or {}
        missing = [rel.rsplit("/", 1)[-1] for comp, rel in split[1].items() if not kwargs.get(f"{comp}_path")]
        if missing:
            return [
                (
                    f"Not in the Hugging Face cache ({cache}): {', '.join(missing)}. ai-toolkit will download them "
                    "into its own models folder on first run (often tens of GB)."
                )
            ]
        return []
    if REPO_ID.match(name) and not os.path.exists(name):
        return [
            (
                f"{name} isn't fully in the Hugging Face cache ({cache}), so ai-toolkit will download it on first run "
                "(often tens of GB; many models go to ai-toolkit's own models folder rather than the HF cache)."
            )
        ]
    return []
