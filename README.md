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
| `ai_toolkit_status` | Connection check: GPUs, VRAM, queues, active jobs, data folders |
| `list_datasets` | Dataset folders with full paths and file counts |
| `list_jobs` / `get_job` | Jobs with status, step progress, speed, model |
| `get_job_config` | A job's full config as YAML |
| `new_job_template` | The UI's default job config to edit |
| `create_job` | New job from YAML, from the template, or cloned from an existing job, plus dotted-path overrides like `{"train.steps": 2000}` |
| `update_job` | Edit a stopped job |
| `start_job` / `stop_job` | Queue or stop a job (graceful stop saves a checkpoint; `force` kills it) |
| `trigger` | Ask a running job to save or sample now |
| `get_job_log` | Tail of the training log, without progress-bar noise |
| `get_loss` | Loss curve summary (start/end/min, smoothed trend) |
| `list_job_outputs` | Checkpoint files and samples |
| `view_samples` | Returns the latest sample images so the agent can see how training looks |

Read-only tools are marked as read-only, so clients with tool approval (Griptape Nodes, Claude) can
auto-approve them and only ask before anything that creates, starts, or stops a job. Nothing in this server
deletes jobs or files.

## Tips

- The most reliable way to train a new model type is to set one job up in the ai-toolkit UI once, then
  have the agent clone it (`base_job`) with overrides. The UI knows each model's quirks; the agent then
  only needs to change the dataset, trigger word, steps, and so on.
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
