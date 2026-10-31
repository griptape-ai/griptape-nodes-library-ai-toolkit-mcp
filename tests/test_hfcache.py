import json

from ai_toolkit_mcp.hfcache import SPLIT_FILE_MODELS, download_warnings, is_local, use_cached_weights


def make_snapshot(cache, repo, files, ref=True, sha="abc123"):
    repo_dir = cache / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / sha
    for rel, content in files.items():
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_text(content)
    if ref:
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(sha)
    return snap


INDEX = json.dumps({"_class_name": "X", "transformer": ["diffusers", "T"], "vae": ["diffusers", "V"], "scheduler": [None, None]})


def test_full_diffusers_repo_is_reused(tmp_path):
    snap = make_snapshot(tmp_path, "org/model", {"model_index.json": INDEX, "transformer/c.json": "{}", "vae/c.json": "{}"})
    proc = {"model": {"name_or_path": "org/model", "arch": "zimage"}}
    notes = use_cached_weights(proc, tmp_path)
    assert proc["model"]["name_or_path"] == str(snap) and notes
    assert download_warnings(proc, tmp_path) == []


def test_partial_repo_is_not_reused(tmp_path):
    make_snapshot(tmp_path, "org/model", {"model_index.json": INDEX, "transformer/c.json": "{}"})
    proc = {"model": {"name_or_path": "org/model", "arch": "zimage"}}
    assert use_cached_weights(proc, tmp_path) == []
    assert proc["model"]["name_or_path"] == "org/model"
    assert "will download" in download_warnings(proc, tmp_path)[0]


def test_split_file_model_overrides(tmp_path):
    repo, files = SPLIT_FILE_MODELS["ltx2.5"]
    present = {rel: "x" for comp, rel in files.items() if comp != "audio_vae"}
    snap = make_snapshot(tmp_path, repo, present)
    proc = {"model": {"name_or_path": repo, "arch": "ltx2.5"}}
    use_cached_weights(proc, tmp_path)
    kw = proc["model"]["model_kwargs"]
    assert kw["dit_path"] == str(snap / files["dit"])
    assert "audio_vae_path" not in kw
    assert "ltx-2.5-audio-vae-bf16.safetensors" in download_warnings(proc, tmp_path)[0]


def test_local_path_left_alone(tmp_path):
    proc = {"model": {"name_or_path": str(tmp_path), "arch": "zimage"}}
    assert use_cached_weights(proc, tmp_path) == [] and download_warnings(proc, tmp_path) == []


def test_is_local():
    assert is_local("http://localhost:8675") and is_local("http://127.0.0.1:8675")
    assert not is_local("http://192.168.0.5:8675")
