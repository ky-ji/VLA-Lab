# Deploy Agent

`VLA-Lab` ships with a generic, config-driven deploy agent so teams can expose their own deployment scripts without editing Python source.

## Why This Exists

Different labs have different:

- server launch scripts
- client inference scripts
- robot SDK calls
- preset pose utilities

Instead of hardcoding those into `VLA-Lab`, the deploy agent lets each team provide a local JSON config and reuse the same dashboard + API contract.

## Start an Agent

```bash
vlalab deploy-agent --config /path/to/agent.json --host 0.0.0.0 --port 9001
```

The agent exposes:

- `GET /health`
- `GET /status`
- `POST /action`
- `POST /actions/{action_name}`

`GET /status` now also includes richer dashboard-facing metadata:

- `details.config_path`
- `details.state_dir`
- `details.processes[*].configured_command/configured_cwd/configured_stdout_log/configured_stderr_log`
- `details.action_catalog`
- `details.last_actions`

This allows the dashboard to render command previews, path cards, and workflow state without hardcoding lab-specific commands into the web app.

## Config Format

Top-level fields:

- `agent_id`: stable ID for this agent
- `label`: UI label
- `role`: e.g. `model-server`, `robot-client`
- `state_dir`: optional runtime directory for pid files / logs / process metadata
- `details`: optional static details shown in the dashboard
- `processes`: managed background processes
- `actions`: action definitions exposed to the dashboard

Relative paths are resolved relative to the JSON config file.

## Process Definitions

Each process can declare:

- `command`: string or argv list
- `cwd`: optional working directory
- `env`: optional environment variables
- `shell`: optional, default `false`
- `stdout_log`: optional file path
- `stderr_log`: optional file path
- `healthcheck_url`: optional HTTP probe
- `healthcheck_timeout_sec`
- `startup_grace_sec`
- `stop_timeout_sec`

Example:

```json
{
  "processes": {
    "policy_server": {
      "command": ["python", "run_policy_server.py", "--checkpoint", "{checkpoint_path}"],
      "cwd": "./realworld_deploy/server",
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "healthcheck_url": "http://127.0.0.1:5555/health",
      "startup_grace_sec": 5
    }
  }
}
```

## Action Types

### `status`

Returns the current aggregated agent status.

```json
{
  "id": "status",
  "type": "status",
  "label": "Refresh Status"
}
```

### `managed_process`

Controls one named process.

Fields:

- `process`: process name from `processes`
- `operation`: `start`, `stop`, `restart`, or `status`

```json
{
  "id": "start_server",
  "type": "managed_process",
  "process": "policy_server",
  "operation": "start",
  "label": "Start Server"
}
```

### `command`

Runs a foreground command and returns its stdout / stderr / return code.

Fields:

- `command`: string or argv list
- `cwd`: optional working directory
- `env`: optional environment variables
- `shell`: optional, default `false`
- `timeout_sec`: optional, default `60`

```json
{
  "id": "set_joint_preset",
  "type": "command",
  "command": ["python", "move_robot.py", "--preset", "{joint_preset}"],
  "cwd": "./realworld_deploy/client",
  "timeout_sec": 90
}
```

## Parameter Templating

Action and process commands support Python-style placeholders, filled from request params:

```json
["python", "run_client.py", "--prompt", "{task_prompt}", "--run-name", "{run_name}"]
```

If a required placeholder is missing, the agent returns a `400`.

The dashboard currently sends fields like:

- `run_name`
- `task_prompt`
- `checkpoint_path`
- `extra_args`
- `preset`
- `joint_preset`

You may also call the agent directly with your own params.

## Built-In Runtime Files

For managed processes, the agent stores:

- pid files
- process metadata JSON
- stdout logs
- stderr logs

under `state_dir`.

This makes it possible to:

- detect whether a process is already running
- stop it later from the dashboard
- inspect the last rendered command and logs

## Example Configs

See:

- [Server Example](examples/deploy_agent_server.json)
- [Client Example](examples/deploy_agent_client.json)
- [RealWorld GR00T Server Config](../configs/deploy/realworld_groot_server_agent.json)
- [RealWorld GR00T Client Config](../configs/deploy/realworld_groot_client_agent.json)

These examples are intentionally generic. Teams should copy them and replace the local script paths / flags with their own deployment commands.
