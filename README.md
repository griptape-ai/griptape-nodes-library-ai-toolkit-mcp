# ai-toolkit-mcp

An MCP server for [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit). It lets an AI agent in
[Griptape Nodes](https://www.griptapenodes.com), Claude Code, Claude Desktop, or any other MCP client
set up, launch, and monitor LoRA training runs on your own machine.

> "Train a LoRA of my dog on the `rufus` dataset, same settings as last week's run but 2000 steps.
> Check on it in a bit and show me the latest samples."

It talks to the **ai-toolkit web UI's API**, so every job it creates shows up in the UI, uses the UI's
per-GPU queue, and appears on the UI's loss graph. You can switch between chatting and clicking at any time.

## Requirements

- ai-toolkit installed, with its UI running (`cd ai-toolkit/ui && npm run build_and_start`, default
  address <http://localhost:8675>)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

You don't need to clone this repo. `uvx` fetches and runs it.

## Setup

### Griptape Nodes

1. Open **Settings → MCP Servers → + New MCP Server**
2. **Server Name/ID**: `ai-toolkit`, **Connection Type**: `Local Process (stdio)`
3. **Configuration JSON**:

   ```json
   {
     "transport": "stdio",
     "command": "uvx",
     "args": ["--from", "git+https://github.com/shhlife/ai-toolkit-mcp", "ai-toolkit-mcp"],
     "env": { "AI_TOOLKIT_URL": "http://localhost:8675" }
   }
   ```

4. Enable the `ai-toolkit` server on an **Agent** node (or use it from an MCP Task node) and start asking.

This lets you build workflows like: generate or caption a dataset with Griptape nodes, have the Agent kick
off training, then pull the finished LoRA back into a generation flow.

### Claude Code

```bash
claude mcp add ai-toolkit -- uvx --from git+https://github.com/shhlife/ai-toolkit-mcp ai-toolkit-mcp
```

### Claude Desktop (or any MCP client using `mcpServers` JSON)

```json
{
  "mcpServers": {
    "ai-toolkit": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/shhlife/ai-toolkit-mcp", "ai-toolkit-mcp"]
    }
  }
}
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `AI_TOOLKIT_URL` | `http://localhost:8675` | Where the ai-toolkit UI is running. Can point at another machine or a cloud GPU box. |
| `AI_TOOLKIT_AUTH` | unset | Set this if you started the UI with `AI_TOOLKIT_AUTH` (same token). |

## Tools

| Tool | What it does |
|---|---|
| `training_guide` | Best practices by topic: workflow, dataset, captions, models, settings, VRAM, monitoring, troubleshooting |
| `ai_toolkit_status` | Connection check: GPUs, VRAM, queues, active jobs, data folders |
| `list_datasets` | Dataset folders with full paths and file counts |
| `create_dataset` | Make a dataset (or add to one) from local files/folders, including existing caption `.txt` files |
| `get_captions` / `set_captions` | Read and write per-image captions |
| `auto_caption_dataset` | Queue ai-toolkit's built-in captioner (Qwen3-Omni) on a dataset |
| `list_jobs` / `get_job` | Jobs with status, step progress, speed, model |
| `get_job_config` | A job's full config as YAML |
| `list_models` | Every model your ai-toolkit install can train (read live from it), with preset names |
| `create_job` | New job from a model preset, YAML, or a clone of an existing job, plus dotted-path overrides like `{"train.steps": 2000}`. Returns warnings (missing captions, trigger-word mistakes, gated models without a token) and won't auto-start while there are any |
| `update_job` | Edit a stopped job, including switching its model preset |
| `start_job` / `stop_job` | Queue or stop a job (graceful stop saves a checkpoint; `force` kills it) |
| `trigger` | Ask a running job to save or sample now |
| `get_job_log` | Tail of the training log, without progress-bar noise |
| `get_loss` | Loss curve summary (start/end/min, smoothed trend) |
| `list_job_outputs` | Checkpoint files and samples |
| `view_samples` | Returns the latest sample images so the agent can see how training looks |

Read-only tools are marked as read-only, so clients with tool approval (Griptape Nodes, Claude) can
auto-approve them and only ask before anything that creates, starts, or stops a job. Nothing in this server
deletes jobs, datasets, or files (`set_captions` does overwrite the captions you give it).

`create_dataset` uploads through the UI's API rather than writing into ai-toolkit's folder directly, so it
works the same whether ai-toolkit is on your machine or a remote GPU box. ai-toolkit replaces spaces and
special characters in uploaded file names with `_`.

## How agents learn best practices

- `training_guide` returns practical advice per topic. The tool descriptions tell agents to call it
  before planning a run, so this works in every MCP client. The text lives in
  [`guide.py`](src/ai_toolkit_mcp/guide.py); edit it as you learn what works.
- Model presets come from your ai-toolkit install's own model definitions, the same ones the UI's model
  dropdown uses, so new models show up automatically when you update ai-toolkit.
- `create_job` / `update_job` check the config and return warnings in plain language.
- The server also sends usage instructions when a client connects. Claude Code reads those; Griptape
  Nodes currently doesn't pass them to its agents, which is why the guide is a tool.

## Reusing models you already downloaded

ai-toolkit keeps ComfyUI-format weights in its own models folder for many models and downloads them
there even when the model is already in your Hugging Face cache (LTX-2.5 always loads this way). When
ai-toolkit runs on the same machine as this server, `create_job` / `update_job` point the job at the
cached files instead and say so under `using_hf_cache`. Models that aren't cached get a warning before
anything starts. The cache location comes from `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `HF_HOME`.

## Tips

- `create_job(model=...)` applies the same settings the UI does when you pick a model. For edit models
  that need control images (Kontext, Qwen-Image-Edit), set one job up in the UI and have the agent clone
  it with `base_job`.
- Per-model reference configs live in ai-toolkit's `config/examples/` folder. You can paste one into
  `create_job(config=...)`.

## Development

```bash
uv sync
uv run pytest
uv run mcp dev src/ai_toolkit_mcp/server.py   # MCP Inspector
```

## License

MIT

## Griptape Nodes library (experimental)

The `griptape_library/` folder is a Griptape Nodes library that exposes a subset of the tools (status, datasets,
models, jobs, job logs, start and stop) through the engine's own MCP server. It needs an engine that supports
`AdvancedNodeLibrary.get_mcp_tools` ([griptape-nodes-engine#6035](https://github.com/griptape-ai/griptape-nodes-engine/pull/6035)).

1. In Griptape Nodes, add `griptape_library/griptape_nodes_library.json` from a checkout of this repo as a library.
2. Add an MCP server pointing at `http://localhost:8125/mcp/libraries/ai_toolkit/` (or use
   `http://localhost:8125/mcp/?libraries=ai_toolkit` for the engine's tools plus these).

The handlers call the same functions as the standalone server, so results match.
