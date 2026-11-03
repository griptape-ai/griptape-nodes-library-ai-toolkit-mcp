"""Advanced library that serves ai-toolkit requests and exposes them as MCP tools.

Each handler calls the same function the standalone `ai-toolkit-mcp` server uses for that tool, so
the two never disagree. The package lives in `../src`, which is put on `sys.path` when the library is
loaded from a checkout of this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ai_toolkit_events import (  # noqa: E402
    AIToolkitGetJobLogRequest,
    AIToolkitGetJobRequest,
    AIToolkitListDatasetsRequest,
    AIToolkitListJobsRequest,
    AIToolkitListModelsRequest,
    AIToolkitResultFailure,
    AIToolkitResultSuccess,
    AIToolkitStartJobRequest,
    AIToolkitStatusRequest,
    AIToolkitStopJobRequest,
)

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary, MCPTool  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultPayload


class AIToolkitLibrary(AdvancedNodeLibrary):
    def get_request_handlers(self) -> list[tuple[type[RequestPayload], Callable]]:
        return [
            (AIToolkitStatusRequest, self._status),
            (AIToolkitListDatasetsRequest, self._list_datasets),
            (AIToolkitListModelsRequest, self._list_models),
            (AIToolkitListJobsRequest, self._list_jobs),
            (AIToolkitGetJobRequest, self._get_job),
            (AIToolkitGetJobLogRequest, self._get_job_log),
            (AIToolkitStartJobRequest, self._start_job),
            (AIToolkitStopJobRequest, self._stop_job),
        ]

    def get_mcp_tools(self) -> list[MCPTool]:
        return [
            MCPTool(AIToolkitStatusRequest, read_only=True),
            MCPTool(AIToolkitListDatasetsRequest, read_only=True),
            MCPTool(AIToolkitListModelsRequest, read_only=True),
            MCPTool(AIToolkitListJobsRequest, read_only=True),
            MCPTool(AIToolkitGetJobRequest, read_only=True),
            MCPTool(AIToolkitGetJobLogRequest, read_only=True),
            MCPTool(AIToolkitStartJobRequest),
            MCPTool(AIToolkitStopJobRequest),
        ]

    def _status(self, request: AIToolkitStatusRequest) -> ResultPayload:  # noqa: ARG002
        return self._call("ai_toolkit_status")

    def _list_datasets(self, request: AIToolkitListDatasetsRequest) -> ResultPayload:  # noqa: ARG002
        return self._call("list_datasets")

    def _list_models(self, request: AIToolkitListModelsRequest) -> ResultPayload:
        return self._call("list_models", group=request.group)

    def _list_jobs(self, request: AIToolkitListJobsRequest) -> ResultPayload:
        return self._call("list_jobs", active_only=request.active_only)

    def _get_job(self, request: AIToolkitGetJobRequest) -> ResultPayload:
        return self._call("get_job", job=request.job)

    def _get_job_log(self, request: AIToolkitGetJobLogRequest) -> ResultPayload:
        return self._call("get_job_log", job=request.job, tail_lines=request.tail_lines)

    def _start_job(self, request: AIToolkitStartJobRequest) -> ResultPayload:
        return self._call("start_job", job=request.job)

    def _stop_job(self, request: AIToolkitStopJobRequest) -> ResultPayload:
        return self._call("stop_job", job=request.job, force=request.force)

    @staticmethod
    def _call(tool: str, **kwargs: Any) -> ResultPayload:
        """Run the standalone server's function for `tool` and wrap its text or its error."""
        try:
            from ai_toolkit_mcp import server

            output = getattr(server, tool)(**kwargs)
        except Exception as e:
            return AIToolkitResultFailure(result_details=f"Attempted to run ai-toolkit '{tool}'. Failed because: {e}")
        return AIToolkitResultSuccess(output=output, result_details=f"Ran ai-toolkit '{tool}'.")
