import logging
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from mcp.server.mcpserver import Image, MCPServer
from mcp_types import ToolAnnotations

from .client import AIToolkitClient, AIToolkitError, clean_log, parse_config
from .template import default_caption_job_config, default_job_config

mcp = MCPServer(
    "ai-toolkit",
    instructions=(
        "Drives LoRA / fine-tune training in ostris ai-toolkit through its web UI's API. "
        "Jobs created here appear in the ai-toolkit UI and run through its per-GPU queue. "
        "Typical flow: ai_toolkit_status -> list_datasets (or create_dataset from local files, then get_captions / "
        "set_captions or auto_caption_dataset) -> create_job (from new_job_template or "
        "an existing job via base_job) -> start_job -> poll get_job / get_job_log / get_loss -> view_samples. "
        "Prefer cloning a previous job with base_job + overrides: its model block is already known to work "
        "on this machine. Ask before starting or stopping training; runs can take hours of GPU time."
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


def _finalize(name: str, cfg: dict, overrides: dict | None) -> dict:
    cfg["config"]["name"] = name
    proc = cfg["config"]["process"][0]
    for path, value in (overrides or {}).items():
        _set_path(proc, path, value)
    settings = client().get("/api/settings")
    proc["training_folder"] = settings.get("TRAINING_FOLDER", proc.get("training_folder"))
    proc.setdefault("device", "cuda")
    proc.setdefault("logging", {"log_every": 1, "use_ui_logger": True})
    return cfg


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
def new_job_template() -> str:
    """The ai-toolkit UI's default job config (Flex.1 LoRA) as YAML, to edit and pass to create_job.

    For other models, cloning a previous job with create_job(base_job=...) is more reliable,
    or see ai-toolkit's config/examples/*.yaml for per-model settings.
    """
    return _dump(default_job_config())


@mcp.tool(annotations=WRITE)
def create_job(
    name: str,
    config: str | None = None,
    base_job: str | None = None,
    overrides: dict[str, Any] | None = None,
    gpu_ids: str = "0",
    start: bool = False,
) -> str:
    """Create a training job in the ai-toolkit UI.

    Config source (pick one): `config` as YAML/JSON text (full job file or just its `config:` body),
    or `base_job` (id/name) to clone an existing job's config. Neither = the default template.
    `overrides` sets values inside config.process[0] by dotted path, e.g.
    {"train.steps": 4000, "datasets.0.folder_path": "D:/datasets/me", "trigger_word": "ohwx",
     "network.linear": 16, "sample.samples": [{"prompt": "ohwx at the beach"}]}.
    `gpu_ids` is the GPU index (e.g. "0" or "0,1"). Set `start` to queue it immediately.
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
    job = c.save_job(name, _finalize(name, cfg, overrides), gpu_ids)
    if start:
        c.start_job(job)
        job = c.resolve_job(job["id"])
    return _dump({"created": _summary(job), "started": start})


@mcp.tool(annotations=WRITE)
def update_job(
    job: str,
    config: str | None = None,
    overrides: dict[str, Any] | None = None,
    gpu_ids: str | None = None,
) -> str:
    """Edit a job that isn't running: replace its config (YAML/JSON) and/or apply dotted-path `overrides`
    inside config.process[0] (same format as create_job)."""
    c = client()
    existing = c.resolve_job(job)
    if existing["status"] in ("running", "stopping", "queued"):
        raise AIToolkitError(f"Job '{existing['name']}' is {existing['status']}; stop it before editing.")
    cfg = _parse_config_text(config) if config else parse_config(existing)
    saved = c.save_job(
        existing["name"],
        _finalize(existing["name"], cfg, overrides),
        gpu_ids or existing["gpu_ids"],
        job_id=existing["id"],
    )
    return _dump({"updated": _summary(saved)})


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
