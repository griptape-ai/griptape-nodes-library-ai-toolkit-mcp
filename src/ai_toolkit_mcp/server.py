import logging
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from mcp.server.mcpserver import Image, MCPServer
from mcp_types import ToolAnnotations

from .client import AIToolkitClient, AIToolkitError, clean_log, parse_config
from .guide import GUIDE, TOPICS
from .models import ModelArch, apply_model, parse_model_archs
from .template import default_caption_job_config, default_job_config

mcp = MCPServer(
    "ai-toolkit",
    instructions=(
        "Drives LoRA / fine-tune training in ostris ai-toolkit through its web UI's API. "
        "Jobs created here appear in the ai-toolkit UI and run through its per-GPU queue. "
        "Call training_guide before planning a training run; it covers datasets, captions, settings, and VRAM. "
        "Typical flow: ai_toolkit_status -> create_dataset / list_datasets -> get_captions + set_captions or "
        "auto_caption_dataset -> list_models -> create_job(model=...) -> start_job -> get_job / get_job_log -> "
        "view_samples. Read the warnings create_job returns. Ask before starting or stopping training; runs can "
        "take hours of GPU time."
    ),
)

_client: AIToolkitClient | None = None

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)


def client() -> AIToolkitClient:
    global _client
    if _client is None:
        _client = AIToolkitClient()
    return _client


def _dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)


def _summary(job: dict) -> dict:
    proc = (parse_config(job).get("config", {}).get("process") or [{}])[0]
    step, total = job.get("step") or 0, job.get("total_steps")
    return {
        "id": job["id"],
        "name": job["name"],
        "status": job["status"],
        "progress": f"{step}/{total}" if total else str(step),
        "speed": job.get("speed_string") or None,
        "info": job.get("info") or None,
        "gpu_ids": job["gpu_ids"],
        "type": job.get("job_type"),
        "model": (proc.get("model") or {}).get("name_or_path"),
        "arch": (proc.get("model") or {}).get("arch"),
        "updated_at": job.get("updated_at"),
    }


def _set_path(target: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node: Any = target
    for i, key in enumerate(keys):
        idx: Any = int(key) if key.isdigit() and isinstance(node, list) else key
        if i == len(keys) - 1:
            node[idx] = value
            return
        if isinstance(node, dict) and idx not in node:
            node[idx] = {}
        node = node[idx]


def _parse_config_text(text: str) -> dict:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise AIToolkitError("config must be a YAML/JSON mapping.")
    if "config" not in data:
        if "process" not in data:
            raise AIToolkitError("config needs a top-level 'config:' (full job file) or 'process:' list.")
        data = {"job": "extension", "config": data, "meta": {"name": "[name]", "version": "1.0"}}
    return data


_archs: dict[str, ModelArch] = {}


def archs(refresh: bool = False) -> dict[str, ModelArch]:
    global _archs
    if refresh or not _archs:
        _archs = parse_model_archs(client().get("/api/model_archs"))
    return _archs


def _finalize(name: str, cfg: dict, overrides: dict | None, model: str | None = None) -> dict:
    cfg["config"]["name"] = name
    proc = cfg["config"]["process"][0]
    if model:
        known = archs() if model in archs() else archs(refresh=True)
        if model not in known:
            raise AIToolkitError(f"Unknown model '{model}'. Use list_models to see the names this install supports.")
        apply_model(proc, known, model)
    for path, value in (overrides or {}).items():
        _set_path(proc, path, value)
    settings = client().get("/api/settings")
    proc["training_folder"] = settings.get("TRAINING_FOLDER", proc.get("training_folder"))
    proc.setdefault("device", "cuda")
    proc.setdefault("logging", {"log_every": 1, "use_ui_logger": True})
    return cfg


def _norm(p: str) -> str:
    return p.replace("\\", "/").rstrip("/").lower()


def _review(cfg: dict) -> list[str]:
    """Plain-language warnings about a job config, returned to the agent with create/update results."""
    c = client()
    proc = cfg["config"]["process"][0]
    settings = c.get("/api/settings")
    warnings: list[str] = []
    trigger = proc.get("trigger_word")
    train = proc.get("train") or {}

    ds_root = _norm(settings.get("DATASETS_FOLDER", ""))
    for n, ds in enumerate(proc.get("datasets") or []):
        folder = ds.get("folder_path") or ""
        if not folder or folder == "/path/to/images/folder":
            warnings.append(f"datasets.{n}.folder_path isn't set. Use a `path` from list_datasets.")
            continue
        rel = _norm(folder)[len(ds_root) + 1 :] if ds_root and _norm(folder).startswith(ds_root + "/") else None
        if not rel or "/" in rel:
            continue
        try:
            root, files = c.list_dataset_files(rel)
        except AIToolkitError:
            warnings.append(f"datasets.{n}.folder_path '{folder}' doesn't exist.")
            continue
        if not files:
            warnings.append(f"Dataset '{rel}' is empty.")
            continue
        if len(files) < 5:
            warnings.append(f"Dataset '{rel}' has only {len(files)} files; 10+ usually trains far better.")
        caps = c.post("/api/caption/getBatch", {"imgPaths": [root + f for f in files], "ext": ds.get("caption_ext") or "txt"})
        missing = sum(1 for v in caps.get("captions", {}).values() if not v.strip())
        if missing and not (trigger or ds.get("default_caption")):
            warnings.append(
                f"{missing} of {len(files)} files in '{rel}' have no caption and there's no trigger_word or "
                "default_caption, so they train with an empty prompt. Caption them or set trigger_word."
            )
        elif missing:
            warnings.append(f"{missing} of {len(files)} files in '{rel}' have no caption.")

    if trigger and train.get("cache_text_embeddings"):
        warnings.append(
            "trigger_word is set but train.cache_text_embeddings is true, and ai-toolkit ignores trigger words "
            "when caching text embeddings. Set train.cache_text_embeddings false (use train.unload_text_encoder "
            "true to save VRAM instead) or drop the trigger word."
        )
    prompts = [s.get("prompt", "") for s in (proc.get("sample") or {}).get("samples", []) if isinstance(s, dict)]
    if trigger and prompts and not any(trigger in p or "[trigger]" in p for p in prompts):
        warnings.append("No sample prompt contains the trigger word or [trigger], so samples won't show the LoRA's effect.")

    arch = archs().get((proc.get("model") or {}).get("arch", ""))
    if arch and arch.gate_url and not settings.get("HF_TOKEN"):
        warnings.append(
            f"{arch.label} is gated: accept its license at {arch.gate_url} and save an HF token in the ai-toolkit "
            "UI's Settings, or the job fails while downloading."
        )
    if arch and arch.group == "instruction":
        warnings.append(f"{arch.label} is an edit model that needs control images; check the dataset controls in the UI.")
    return warnings


# --- overview -----------------------------------------------------------------


@mcp.tool(annotations=READ)
def ai_toolkit_status() -> str:
    """Connection check and machine overview: GPUs (VRAM, load), queues, active jobs, and data folders."""
    c = client()
    settings = c.get("/api/settings")
    gpu = c.get("/api/gpu")
    return _dump(
        {
            "ui_url": c.base_url,
            "training_folder": settings.get("TRAINING_FOLDER"),
            "datasets_folder": settings.get("DATASETS_FOLDER"),
            "hf_token_set": bool(settings.get("HF_TOKEN")),
            "gpus": [
                {
                    "index": g["index"],
                    "name": g["name"],
                    "vram_used_mb": g["memory"]["used"],
                    "vram_total_mb": g["memory"]["total"],
                    "load_pct": g["utilization"]["gpu"],
                    "temp_c": g.get("temperature"),
                }
                for g in gpu.get("gpus", [])
            ]
            or gpu.get("error"),
            "queues": c.get("/api/queue").get("queues", []),
            "active_jobs": [_summary(j) for j in c.list_jobs(active_only=True)],
        }
    )


@mcp.tool(annotations=READ)
def list_datasets() -> str:
    """List dataset folders in ai-toolkit's datasets folder, with full paths and file counts.

    Use the `path` value as `datasets.0.folder_path` when creating a job.
    """
    c = client()
    root = c.get("/api/settings").get("DATASETS_FOLDER", "")
    sep = "\\" if "\\" in root else "/"
    out = []
    for name in c.get("/api/datasets/list"):
        entry: dict[str, Any] = {"name": name, "path": f"{root.rstrip(sep)}{sep}{name}"}
        try:
            listing = c.post("/api/datasets/listImages", {"datasetName": name})
            entry["path"] = listing.get("root", "").rstrip("\\/") or entry["path"]
            entry["files"] = len(listing.get("images", []))
        except AIToolkitError:
            pass
        out.append(entry)
    return _dump(out) if out else f"No datasets in {root}."


MEDIA_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".mov", ".webm", ".mkv", ".avi", ".mp3", ".wav", ".flac", ".ogg")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
UPLOAD_BATCH = 16

IMAGE_CAPTION_PROMPT = (
    "Caption this image as if you were going to generate it with an image generator. Describe the subject, "
    "setting, composition, lighting, and style. Be decisive by stating things as they are. Do not say things "
    'like "It appears that" or "possibly". No preamble. Just get to the point.'
)


def _collect_files(paths: list[str], caption_ext: str) -> list[str]:
    exts = MEDIA_EXTS + (f".{caption_ext}",)
    found: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            found += sorted(str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        elif p.is_file():
            found.append(str(p))
        else:
            raise AIToolkitError(f"Not found on this machine: {raw}")
    return found


@mcp.tool(annotations=WRITE)
def create_dataset(name: str, paths: list[str], caption_ext: str = "txt") -> str:
    """Create a dataset (or add to an existing one) by uploading local files into ai-toolkit.

    `paths` are files or folders on the machine running this MCP server; folders are searched recursively
    for images, videos, audio, and matching caption files (e.g. photo1.jpg + photo1.txt). Files are sent
    over the UI's API, so this also works when ai-toolkit runs on another machine.
    """
    c = client()
    files = _collect_files(paths, caption_ext.lstrip("."))
    if not files:
        raise AIToolkitError("No media or caption files found in those paths.")
    uploaded: list[str] = []
    for i in range(0, len(files), UPLOAD_BATCH):
        uploaded += c.upload_files(name, files[i : i + UPLOAD_BATCH])
    root, listed = c.list_dataset_files(name)
    return _dump(
        {
            "dataset": name,
            "path": root.rstrip("\\/"),
            "uploaded_files": len(uploaded),
            "media_in_dataset": len(listed),
            "captions_uploaded": sum(1 for f in uploaded if f.lower().endswith(f".{caption_ext.lstrip('.')}")),
        }
    )


@mcp.tool(annotations=READ)
def get_captions(dataset: str, caption_ext: str = "txt", limit: int = 200) -> str:
    """Captions for each file in a dataset (empty string = uncaptioned), plus a count of missing ones."""
    c = client()
    root, files = c.list_dataset_files(dataset)
    paths = [root + f for f in files]
    captions = c.post("/api/caption/getBatch", {"imgPaths": paths, "ext": caption_ext}).get("captions", {})
    by_file = {f: captions.get(root + f, "") for f in files}
    missing = sum(1 for v in by_file.values() if not v.strip())
    shown = dict(list(by_file.items())[: max(1, limit)])
    return _dump({"dataset": dataset, "files": len(files), "uncaptioned": missing, "captions": shown})


@mcp.tool(annotations=WRITE)
def set_captions(dataset: str, captions: dict[str, str], caption_ext: str = "txt") -> str:
    """Write captions. `captions` maps a file name (as shown by get_captions) to its caption text;
    existing captions for those files are replaced."""
    c = client()
    root, files = c.list_dataset_files(dataset)
    known = set(files)
    unknown = [f for f in captions if f not in known]
    if unknown:
        raise AIToolkitError(f"Not in dataset '{dataset}': {', '.join(unknown[:10])}")
    for f, text in captions.items():
        c.post("/api/img/caption", {"imgPath": root + f, "caption": text, "ext": caption_ext})
    return f"Wrote {len(captions)} caption(s) in '{dataset}'."


@mcp.tool(annotations=WRITE)
def auto_caption_dataset(
    dataset: str,
    prompt: str | None = None,
    recaption: bool = False,
    caption_ext: str = "txt",
    gpu_ids: str = "0",
) -> str:
    """Queue ai-toolkit's built-in captioning job (Qwen3-Omni) for a dataset, like the UI's Caption button.

    Only uncaptioned files are captioned unless `recaption` is set. `prompt` overrides the captioning
    instruction; by default an image prompt is used for image-only datasets and ai-toolkit's video/audio
    prompt otherwise. Uses GPU time and downloads the captioner model on first use. Track it with get_job.
    """
    c = client()
    root, files = c.list_dataset_files(dataset)
    if not files:
        raise AIToolkitError(f"Dataset '{dataset}' has no files.")
    cfg = default_caption_job_config()
    cap = cfg["config"]["process"][0]["caption"]
    cap["path_to_caption"] = root.rstrip("\\/")
    cap["recaption"] = recaption
    cap["caption_extension"] = caption_ext.lstrip(".")
    if prompt:
        cap["caption_prompt"] = prompt
    elif all(f.lower().endswith(IMAGE_EXTS) for f in files):
        cap["caption_prompt"] = IMAGE_CAPTION_PROMPT
    try:
        job = c.post(
            "/api/jobs",
            {
                "name": f"caption_{dataset}_{uuid.uuid4().hex[:6]}",
                "gpu_ids": gpu_ids,
                "job_config": cfg,
                "job_type": "caption",
                "job_ref": cap["path_to_caption"],
            },
        )
    except AIToolkitError as e:
        if "409" in str(e):
            raise AIToolkitError(f"A caption job for '{dataset}' already exists; see list_jobs.") from e
        raise
    c.start_job(job)
    return _dump({"queued": _summary(c.resolve_job(job["id"])), "files": len(files)})


# --- jobs -----------------------------------------------------------------------


@mcp.tool(annotations=READ)
def list_jobs(active_only: bool = False) -> str:
    """List training jobs (newest first) with status, progress, and model."""
    jobs = client().list_jobs(active_only=active_only)
    return _dump([_summary(j) for j in jobs]) if jobs else "No jobs."


@mcp.tool(annotations=READ)
def get_job(job: str) -> str:
    """Status and progress of one job. `job` is a job id or name."""
    return _dump(_summary(client().resolve_job(job)))


@mcp.tool(annotations=READ)
def get_job_config(job: str) -> str:
    """Full training config of a job as YAML. Useful as a starting point for a new run."""
    return _dump(parse_config(client().resolve_job(job)))


@mcp.tool(annotations=READ)
def training_guide(topic: Literal[TOPICS] = "overview") -> str:  # type: ignore[valid-type]
    """Best practices for training LoRAs with ai-toolkit. Call this before planning a first training run.

    Topics: overview (workflow + rules), dataset, captions (incl. trigger words), models, settings
    (steps, lr, rank, sampling), vram, monitoring (judging samples, overfitting), troubleshooting.
    """
    return GUIDE[topic]


@mcp.tool(annotations=READ)
def list_models(group: str | None = None) -> str:
    """Models this ai-toolkit install can train, with their preset name for create_job(model=...).

    `group` filters by image, video, instruction (edit models), audio, llm, or experimental.
    `gate_url` means the model needs its Hugging Face license accepted plus an HF token in ai-toolkit.
    """
    out = [
        {
            "name": a.name,
            "label": a.label,
            "group": a.group,
            "base": next((s for k, (s, _) in a.defaults.items() if k.endswith("model.name_or_path") and isinstance(s, str)), None),
            **({"gate_url": a.gate_url} if a.gate_url else {}),
        }
        for a in archs(refresh=True).values()
        if not group or a.group == group
    ]
    return _dump(out) if out else f"No models in group '{group}'."


@mcp.tool(annotations=READ)
def new_job_template(model: str | None = None) -> str:
    """The default job config as YAML, optionally with a model preset applied (see list_models).
    Usually simpler to call create_job(model=..., overrides=...) directly."""
    cfg = default_job_config()
    if model:
        if model not in archs(refresh=True):
            raise AIToolkitError(f"Unknown model '{model}'. Use list_models.")
        apply_model(cfg["config"]["process"][0], archs(), model)
    return _dump(cfg)


@mcp.tool(annotations=WRITE)
def create_job(
    name: str,
    model: str | None = None,
    config: str | None = None,
    base_job: str | None = None,
    overrides: dict[str, Any] | None = None,
    gpu_ids: str = "0",
    start: bool = False,
) -> str:
    """Create a training job in the ai-toolkit UI. Returns warnings about likely problems; read them.

    Config source: the default template, `config` as YAML/JSON text (full job file or its `config:` body),
    or `base_job` (id/name) to clone an existing job (most reliable for a model trained before).
    `model` applies that model's preset from list_models, like picking it in the UI.
    `overrides` then sets values inside config.process[0] by dotted path, e.g.
    {"train.steps": 2500, "datasets.0.folder_path": "<path from list_datasets>", "trigger_word": "rufus_dog",
     "network.linear": 16, "sample.samples": [{"prompt": "[trigger] on a beach"}]}.
    `gpu_ids` is the GPU index ("0", "0,1"). `start` queues it right away, but only if there are no warnings.
    """
    c = client()
    if config and base_job:
        raise AIToolkitError("Pass either config or base_job, not both.")
    if config:
        cfg = _parse_config_text(config)
    elif base_job:
        cfg = parse_config(c.resolve_job(base_job))
    else:
        cfg = default_job_config()
    cfg = _finalize(name, cfg, overrides, model)
    warnings = _review(cfg)
    job = c.save_job(name, cfg, gpu_ids)
    started = start and not warnings
    if started:
        c.start_job(job)
        job = c.resolve_job(job["id"])
    result: dict[str, Any] = {"created": _summary(job), "started": started}
    if warnings:
        result["warnings"] = warnings
        if start:
            result["note"] = "Not started because of the warnings. Fix them with update_job, or start_job to run anyway."
    return _dump(result)


@mcp.tool(annotations=WRITE)
def update_job(
    job: str,
    model: str | None = None,
    config: str | None = None,
    overrides: dict[str, Any] | None = None,
    gpu_ids: str | None = None,
) -> str:
    """Edit a job that isn't running: switch `model` preset, replace its config (YAML/JSON), and/or apply
    dotted-path `overrides` inside config.process[0] (same format as create_job). Returns warnings."""
    c = client()
    existing = c.resolve_job(job)
    if existing["status"] in ("running", "stopping", "queued"):
        raise AIToolkitError(f"Job '{existing['name']}' is {existing['status']}; stop it before editing.")
    cfg = _parse_config_text(config) if config else parse_config(existing)
    cfg = _finalize(existing["name"], cfg, overrides, model)
    saved = c.save_job(existing["name"], cfg, gpu_ids or existing["gpu_ids"], job_id=existing["id"])
    result: dict[str, Any] = {"updated": _summary(saved)}
    if warnings := _review(cfg):
        result["warnings"] = warnings
    return _dump(result)


@mcp.tool(annotations=WRITE)
def start_job(job: str) -> str:
    """Queue a job and start its GPU queue. Resumes from the last checkpoint if one exists."""
    c = client()
    j = c.resolve_job(job)
    c.start_job(j)
    return _dump(_summary(c.resolve_job(j["id"])))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False))
def stop_job(job: str, force: bool = False) -> str:
    """Stop a job gracefully (saves a checkpoint first). `force` kills the process immediately,
    only for jobs stuck and ignoring a normal stop."""
    c = client()
    j = c.resolve_job(job)
    c.get(f"/api/jobs/{j['id']}/{'kill' if force else 'stop'}")
    return _dump(_summary(c.resolve_job(j["id"])))


@mcp.tool(annotations=WRITE)
def trigger(job: str, action: Literal["save", "sample"]) -> str:
    """Ask a running job to save a checkpoint or generate samples on its next step."""
    c = client()
    j = c.resolve_job(job)
    c.get(f"/api/jobs/{j['id']}/{action}_now")
    return f"Job '{j['name']}' will {action} on its next step."


# --- monitoring ------------------------------------------------------------------


@mcp.tool(annotations=READ)
def get_job_log(job: str, tail_lines: int = 60) -> str:
    """Last lines of a job's training log, with progress-bar noise collapsed."""
    c = client()
    j = c.resolve_job(job)
    log = c.get(f"/api/jobs/{j['id']}/log").get("log", "")
    return clean_log(log, max(1, tail_lines)) or "Log is empty (job may not have started yet)."


@mcp.tool(annotations=READ)
def get_loss(job: str, key: str = "loss", max_points: int = 30) -> str:
    """Loss curve summary: first/last/min values, a smoothed trend, and evenly spaced points."""
    c = client()
    j = c.resolve_job(job)
    data = c.get(f"/api/jobs/{j['id']}/loss", key=key, limit=20000)
    pts = [p for p in data.get("points", []) if p.get("value") is not None]
    if not pts:
        return _dump({"key": key, "available_keys": data.get("keys", []), "points": "none logged yet"})
    vals = [p["value"] for p in pts]
    window = max(1, len(vals) // 10)
    smooth = lambda i: sum(vals[max(0, i - window + 1) : i + 1]) / len(vals[max(0, i - window + 1) : i + 1])
    stride = max(1, len(pts) // max(1, max_points))
    return _dump(
        {
            "key": key,
            "available_keys": data.get("keys", []),
            "steps_logged": len(pts),
            "first": {"step": pts[0]["step"], "value": round(vals[0], 5)},
            "last": {"step": pts[-1]["step"], "value": round(vals[-1], 5)},
            "min": round(min(vals), 5),
            "smoothed_start": round(smooth(window - 1), 5),
            "smoothed_end": round(smooth(len(vals) - 1), 5),
            "points": [[p["step"], round(smooth(i), 5)] for i, p in enumerate(pts)][::stride],
        }
    )


@mcp.tool(annotations=READ)
def list_job_outputs(job: str) -> str:
    """Checkpoint (.safetensors) files and sample media produced by a job, as local paths."""
    c = client()
    j = c.resolve_job(job)
    files = c.get(f"/api/jobs/{j['id']}/files").get("files", [])
    samples = c.get(f"/api/jobs/{j['id']}/samples").get("samples", [])
    return _dump(
        {
            "checkpoints": [{"path": f["path"], "size_mb": round(f["size"] / 1e6, 1)} for f in files],
            "samples": {"count": len(samples), "latest": samples[-12:]},
        }
    )


@mcp.tool(annotations=READ)
def view_samples(job: str, count: int = 4) -> list:
    """Return the most recent sample images from a job so you can judge training progress visually."""
    c = client()
    j = c.resolve_job(job)
    samples = [
        s
        for s in c.get(f"/api/jobs/{j['id']}/samples").get("samples", [])
        if s.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not samples:
        return [f"No image samples yet for '{j['name']}'."]
    # Sample filenames start with a timestamp + step, so a name sort is chronological.
    picked = sorted(samples)[-max(1, min(count, 12)) :]
    out: list = []
    for path in picked:
        ext = path.rsplit(".", 1)[-1].lower()
        out.append(path)
        out.append(Image(data=c.image_bytes(path), format="jpeg" if ext == "jpg" else ext))
    return out


def main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    mcp.run()


if __name__ == "__main__":
    main()
