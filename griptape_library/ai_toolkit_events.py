"""Request and result payloads for the ai-toolkit library.

`ai_toolkit_library.py` serves these requests and exposes them as MCP tools. The request
docstrings are the tool descriptions an agent reads, so they stay short and say what to pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from griptape_nodes.retained_mode.events.base_events import (
    RequestPayload,
    ResultPayloadFailure,
    ResultPayloadSuccess,
    WorkflowNotAlteredMixin,
)
from griptape_nodes.retained_mode.events.payload_registry import PayloadRegistry


@dataclass
@PayloadRegistry.register
class AIToolkitStatusRequest(RequestPayload):
    """Connection check for ai-toolkit: GPUs, queues, active jobs, data folders, HF cache.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """


@dataclass
@PayloadRegistry.register
class AIToolkitListDatasetsRequest(RequestPayload):
    """List ai-toolkit datasets with file counts and paths.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """


@dataclass
@PayloadRegistry.register
class AIToolkitListModelsRequest(RequestPayload):
    """List the model architectures ai-toolkit can train, with their default settings.

    Args:
        group: Optional model group to filter by.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """

    group: str | None = None


@dataclass
@PayloadRegistry.register
class AIToolkitListJobsRequest(RequestPayload):
    """List training jobs, newest first, with status and progress.

    Args:
        active_only: Only list jobs that are queued or running.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """

    active_only: bool = False


@dataclass
@PayloadRegistry.register
class AIToolkitGetJobRequest(RequestPayload):
    """Get one training job's status and progress.

    Args:
        job: Job id or name.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """

    job: str


@dataclass
@PayloadRegistry.register
class AIToolkitGetJobLogRequest(RequestPayload):
    """Get the tail of a training job's log.

    Args:
        job: Job id or name.
        tail_lines: Number of lines to return from the end of the log.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """

    job: str
    tail_lines: int = 60


@dataclass
@PayloadRegistry.register
class AIToolkitStartJobRequest(RequestPayload):
    """Queue a training job. It resumes from its last checkpoint. Ask the user first, since runs can take hours of GPU time.

    Args:
        job: Job id or name.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """

    job: str


@dataclass
@PayloadRegistry.register
class AIToolkitStopJobRequest(RequestPayload):
    """Stop a training job gracefully, which saves a checkpoint.

    Args:
        job: Job id or name.
        force: Kill a hung job instead of stopping it gracefully.

    Results: AIToolkitResultSuccess | AIToolkitResultFailure
    """

    job: str
    force: bool = False


@dataclass
@PayloadRegistry.register
class AIToolkitResultSuccess(WorkflowNotAlteredMixin, ResultPayloadSuccess):
    """An ai-toolkit call succeeded.

    Args:
        output: The same text the standalone ai-toolkit MCP server returns for this call.
    """

    output: str = ""


@dataclass
@PayloadRegistry.register
class AIToolkitResultFailure(WorkflowNotAlteredMixin, ResultPayloadFailure):
    """An ai-toolkit call failed. result_details has the reason."""
