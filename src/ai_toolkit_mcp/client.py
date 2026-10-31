import json
import os
import re
from typing import Any
from urllib.parse import quote

import httpx
from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_URL = "http://localhost:8675"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class AIToolkitError(ToolError):
    """An anticipated failure; MCP passes its message through to the agent."""


class AIToolkitClient:
    """Thin wrapper over the ai-toolkit UI's REST API (ui/src/app/api)."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = (base_url or os.environ.get("AI_TOOLKIT_URL") or DEFAULT_URL).rstrip("/")
        token = token if token is not None else os.environ.get("AI_TOOLKIT_AUTH")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=30, transport=transport)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            resp = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise AIToolkitError(
                f"Can't reach the ai-toolkit UI at {self.base_url}. Start it with "
                "`npm run build_and_start` in ai-toolkit/ui, or set AI_TOOLKIT_URL."
            ) from e
        except httpx.HTTPError as e:
            raise AIToolkitError(f"{method} {path} to {self.base_url} failed: {e!r}") from e
        if resp.status_code == 401:
            raise AIToolkitError("ai-toolkit rejected the request: set AI_TOOLKIT_AUTH to the UI's auth token.")
        if resp.status_code >= 400:
            try:
                msg = resp.json().get("error", resp.text)
            except ValueError:
                msg = resp.text
            raise AIToolkitError(f"{method} {path} failed ({resp.status_code}): {msg}")
        return resp

    def get(self, path: str, **params) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None}).json()

    def post(self, path: str, body: dict) -> Any:
        return self._request("POST", path, json=body).json()

    def get_bytes(self, path: str) -> bytes:
        return self._request("GET", path).content

    # --- jobs ---------------------------------------------------------------

    def list_jobs(self, active_only: bool = False) -> list[dict]:
        data = self.get("/api/jobs", only_active="true" if active_only else None)
        return data.get("jobs", [])

    def resolve_job(self, job: str) -> dict:
        """Find a job by id or (case-insensitive) name."""
        found = self.get("/api/jobs", id=job)
        if found:
            return found
        matches = [j for j in self.list_jobs() if j["name"].lower() == job.lower()]
        if not matches:
            raise AIToolkitError(f"No job with id or name '{job}'.")
        return matches[0]

    def save_job(self, name: str, job_config: dict, gpu_ids: str, job_id: str | None = None) -> dict:
        body = {"name": name, "job_config": job_config, "gpu_ids": gpu_ids}
        if job_id:
            body["id"] = job_id
        return self.post("/api/jobs", body)

    def start_job(self, job: dict) -> None:
        # Mirrors the UI: queue the job, then make sure that GPU's queue is running.
        self.get(f"/api/jobs/{job['id']}/start")
        self.get(f"/api/queue/{quote(job['gpu_ids'], safe='')}/start")

    def upload_files(self, dataset: str, paths: list[str]) -> list[str]:
        files = []
        try:
            for p in paths:
                files.append(("files", (os.path.basename(p), open(p, "rb"))))
            resp = self._request(
                "POST", "/api/datasets/upload", data={"datasetName": dataset}, files=files, timeout=600
            )
        finally:
            for _, (_, fh) in files:
                fh.close()
        return resp.json().get("files", [])

    def list_dataset_files(self, dataset: str) -> tuple[str, list[str]]:
        """Returns (dataset folder with trailing separator, file paths relative to it)."""
        data = self.post("/api/datasets/listImages", {"datasetName": dataset})
        return data.get("root", ""), data.get("images", [])

    def image_bytes(self, local_path: str) -> bytes:
        return self.get_bytes(f"/api/img/{quote(local_path, safe='')}")


def parse_config(job: dict) -> dict:
    cfg = job.get("job_config")
    return json.loads(cfg) if isinstance(cfg, str) else (cfg or {})


def clean_log(text: str, tail_lines: int) -> str:
    """Collapse carriage-return progress-bar rewrites and strip ANSI codes."""
    lines = []
    for raw in _ANSI.sub("", text).split("\n"):
        line = raw.rstrip("\r").rsplit("\r", 1)[-1].rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines[-tail_lines:])
