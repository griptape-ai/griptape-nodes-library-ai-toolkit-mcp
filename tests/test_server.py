import json
import uuid
from urllib.parse import unquote

import httpx
import pytest
import yaml
from mcp.server.mcpserver import Image

from ai_toolkit_mcp import server
from ai_toolkit_mcp.client import AIToolkitClient, AIToolkitError, clean_log


class FakeToolkit:
    """In-memory stand-in for the ai-toolkit UI API."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, q = request.url.path, request.url.params
        self.calls.append(f"{request.method} {path}")
        if path == "/api/settings":
            return httpx.Response(200, json={"TRAINING_FOLDER": "/tk/output", "DATASETS_FOLDER": "/tk/datasets", "HF_TOKEN": "secret"})
        if path == "/api/gpu":
            return httpx.Response(200, json={"gpus": [{"index": 0, "name": "RTX 4090", "memory": {"used": 1000, "total": 24000}, "utilization": {"gpu": 5}, "temperature": 40}]})
        if path == "/api/queue":
            return httpx.Response(200, json={"queues": []})
        if path == "/api/datasets/list":
            return httpx.Response(200, json=["faces"])
        if path == "/api/datasets/listImages":
            return httpx.Response(200, json={"root": "/tk/datasets/faces/", "images": ["a.jpg", "b.jpg"]})
        if path == "/api/jobs" and request.method == "GET":
            if "id" in q:
                return httpx.Response(200, content=json.dumps(self.jobs.get(q["id"])).encode())
            jobs = list(self.jobs.values())
            if q.get("only_active") == "true":
                jobs = [j for j in jobs if j["status"] in ("running", "queued", "stopping")]
            return httpx.Response(200, json={"jobs": jobs})
        if path == "/api/jobs" and request.method == "POST":
            body = json.loads(request.content)
            job_id = body.get("id") or str(uuid.uuid4())
            job = self.jobs.get(job_id, {"id": job_id, "status": "stopped", "step": 0, "total_steps": None})
            job.update(name=body["name"], gpu_ids=body["gpu_ids"], job_config=json.dumps(body["job_config"]))
            self.jobs[job_id] = job
            return httpx.Response(200, json=job)
        parts = path.split("/")
        if path.startswith("/api/queue/") and parts[-1] == "start":
            return httpx.Response(200, json={"gpu_ids": unquote(parts[3])})
        if path.startswith("/api/img/"):
            return httpx.Response(200, content=b"\x89PNGfake")
        if path.startswith("/api/jobs/"):
            job = self.jobs[parts[3]]
            action = parts[4]
            if action == "start":
                job["status"] = "queued"
            elif action in ("stop", "kill"):
                job["status"] = "stopped"
            elif action == "log":
                return httpx.Response(200, json={"log": "\x1b[32mloading\x1b[0m\nstep 1\rstep 2\rstep 3\ndone\n"})
            elif action == "loss":
                return httpx.Response(200, json={"keys": ["loss"], "points": [{"step": i, "value": 1.0 / i} for i in range(1, 101)]})
            elif action == "samples":
                return httpx.Response(200, json={"samples": ["/tk/output/x/samples/1__000000250_0.jpg", "/tk/output/x/samples/1__000000500_0.png", "/tk/output/x/samples/clip.mp4"]})
            elif action == "files":
                return httpx.Response(200, json={"files": [{"path": "/tk/output/x/x.safetensors", "size": 170_000_000}]})
            return httpx.Response(200, json=job)
        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def fake(monkeypatch):
    fake = FakeToolkit()
    monkeypatch.setattr(server, "_client", AIToolkitClient("http://tk", token="", transport=httpx.MockTransport(fake.handler)))
    return fake


def test_status_hides_hf_token(fake):
    out = yaml.safe_load(server.ai_toolkit_status())
    assert out["hf_token_set"] is True
    assert "secret" not in server.ai_toolkit_status()
    assert out["gpus"][0]["name"] == "RTX 4090"


def test_list_datasets_uses_real_root(fake):
    out = yaml.safe_load(server.list_datasets())
    assert out == [{"name": "faces", "path": "/tk/datasets/faces", "files": 2}]


def test_create_from_template_with_overrides_and_start(fake):
    out = yaml.safe_load(
        server.create_job("me_v1", overrides={"train.steps": 500, "datasets.0.folder_path": "/tk/datasets/faces", "trigger_word": "ohwx"}, start=True)
    )
    assert out["created"]["status"] == "queued"
    job = next(iter(fake.jobs.values()))
    cfg = json.loads(job["job_config"])
    proc = cfg["config"]["process"][0]
    assert cfg["config"]["name"] == "me_v1"
    assert proc["train"]["steps"] == 500
    assert proc["datasets"][0]["folder_path"] == "/tk/datasets/faces"
    assert proc["trigger_word"] == "ohwx"
    assert proc["training_folder"] == "/tk/output"
    assert f"GET /api/jobs/{job['id']}/start" in fake.calls
    assert "GET /api/queue/0/start" in fake.calls


def test_clone_by_name_and_yaml_body(fake):
    server.create_job("base", config="process:\n  - type: diffusion_trainer\n    model: {name_or_path: x/y, arch: wan21}\n    train: {steps: 10}\n")
    out = yaml.safe_load(server.create_job("clone", base_job="BASE", overrides={"train.steps": 20}))
    assert out["created"]["arch"] == "wan21"
    cfg = yaml.safe_load(server.get_job_config("clone"))
    assert cfg["config"]["process"][0]["train"]["steps"] == 20
    assert cfg["config"]["name"] == "clone"


def test_update_refused_while_queued(fake):
    server.create_job("run", start=True)
    with pytest.raises(AIToolkitError, match="stop it"):
        server.update_job("run", overrides={"train.steps": 1})
    server.stop_job("run")
    assert yaml.safe_load(server.update_job("run", overrides={"train.steps": 1}))["updated"]["name"] == "run"


def test_unknown_job(fake):
    with pytest.raises(AIToolkitError, match="No job"):
        server.get_job("nope")


def test_log_loss_outputs_samples(fake):
    server.create_job("x")
    assert server.get_job_log("x") == "loading\nstep 3\ndone"
    loss = yaml.safe_load(server.get_loss("x", max_points=10))
    assert loss["first"]["value"] == 1.0 and loss["last"]["step"] == 100
    assert len(loss["points"]) == 10
    outputs = yaml.safe_load(server.list_job_outputs("x"))
    assert outputs["checkpoints"][0]["size_mb"] == 170.0
    shown = server.view_samples("x", count=4)
    assert [type(s) for s in shown] == [str, Image, str, Image]


def test_connection_error_message():
    def boom(request):
        raise httpx.ConnectError("refused")

    c = AIToolkitClient("http://tk", token="", transport=httpx.MockTransport(boom))
    with pytest.raises(AIToolkitError, match="Can't reach"):
        c.list_jobs()


def test_clean_log_tail():
    assert clean_log("a\nb\nc\n", 2) == "b\nc"


def test_through_mcp_layer(fake):
    import anyio
    from mcp.server.mcpserver.exceptions import ToolError

    async def run():
        server.create_job("x")
        result = await server.mcp.call_tool("view_samples", {"job": "x", "count": 2})
        assert [c.type for c in result.content] == ["text", "image", "text", "image"]
        assert result.content[1].mime_type == "image/jpeg"
        result = await server.mcp.call_tool("create_job", {"name": "y", "overrides": {"train.steps": 7}})
        assert "y" in result.content[0].text
        with pytest.raises(ToolError, match="No job with id or name 'nope'"):
            await server.mcp.call_tool("get_job", {"job": "nope"})

    anyio.run(run)


def test_auto_caption_uses_image_prompt_and_queues(fake):
    out = yaml.safe_load(server.auto_caption_dataset("faces"))
    assert out["queued"]["status"] == "queued" and out["files"] == 2
    job = next(iter(fake.jobs.values()))
    cap = json.loads(job["job_config"])["config"]["process"][0]["caption"]
    assert cap["path_to_caption"] == "/tk/datasets/faces"
    assert cap["caption_prompt"] == server.IMAGE_CAPTION_PROMPT
    assert "GET /api/queue/0/start" in fake.calls


def test_create_dataset_rejects_missing_path(fake):
    with pytest.raises(AIToolkitError, match="Not found"):
        server.create_dataset("x", ["/definitely/not/here"])


def test_set_captions_rejects_unknown_file(fake):
    with pytest.raises(AIToolkitError, match="Not in dataset"):
        server.set_captions("faces", {"zzz.jpg": "hi"})
