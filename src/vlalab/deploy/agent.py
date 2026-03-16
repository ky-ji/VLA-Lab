"""Config-driven deploy agent for coordinating remote server/client actions."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field


class AgentActionRequest(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ManagedProcessSpec:
    name: str
    command: Union[str, List[str]]
    cwd: Optional[str] = None
    env: Dict[str, Any] = field(default_factory=dict)
    shell: bool = False
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None
    healthcheck_url: Optional[str] = None
    healthcheck_timeout_sec: float = 1.0
    startup_grace_sec: float = 0.0
    stop_timeout_sec: float = 10.0


@dataclass
class AgentActionSpec:
    name: str
    action_type: str
    label: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    config_path: Path
    agent_id: str
    label: str
    role: str
    state_dir: Path
    details: Dict[str, Any]
    actions: Dict[str, AgentActionSpec]
    processes: Dict[str, ManagedProcessSpec]


_PLACEHOLDER_PATTERN = re.compile(r"(?<!\$)\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _render_template_string(value: str, params: Dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(key)
        return str(params[key])

    return _PLACEHOLDER_PATTERN.sub(repl, value)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _coerce_action_items(raw_actions: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_actions, dict):
        return [
            {"id": key, **(value if isinstance(value, dict) else {"type": str(value)})}
            for key, value in raw_actions.items()
        ]
    if isinstance(raw_actions, list):
        return [item for item in raw_actions if isinstance(item, dict)]
    raise ValueError("`actions` must be a dict or list of dicts")


def _coerce_process_items(raw_processes: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_processes, dict):
        return [
            {"name": key, **(value if isinstance(value, dict) else {"command": value})}
            for key, value in raw_processes.items()
        ]
    if isinstance(raw_processes, list):
        return [item for item in raw_processes if isinstance(item, dict)]
    raise ValueError("`processes` must be a dict or list of dicts")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Agent config root must be a JSON object")
    return data


def load_agent_config(config_path: Union[str, Path]) -> AgentConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Deploy agent config not found: {path}")

    raw = _read_json(path)
    config_dir = path.parent
    state_dir = raw.get("state_dir")
    resolved_state_dir = (
        _resolve_path(str(state_dir), config_dir)
        if state_dir
        else (config_dir / ".vlalab_agent_state").resolve()
    )

    processes: Dict[str, ManagedProcessSpec] = {}
    for item in _coerce_process_items(raw.get("processes", {})):
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError("Each process must define a non-empty `name`")
        command = item.get("command")
        if command is None:
            raise ValueError(f"Process `{name}` is missing `command`")
        if not isinstance(command, (str, list)):
            raise ValueError(f"Process `{name}` command must be a string or list")

        processes[name] = ManagedProcessSpec(
            name=name,
            command=command,
            cwd=str(item["cwd"]) if item.get("cwd") else None,
            env=item.get("env", {}) if isinstance(item.get("env", {}), dict) else {},
            shell=bool(item.get("shell", False)),
            stdout_log=str(item["stdout_log"]) if item.get("stdout_log") else None,
            stderr_log=str(item["stderr_log"]) if item.get("stderr_log") else None,
            healthcheck_url=str(item["healthcheck_url"]) if item.get("healthcheck_url") else None,
            healthcheck_timeout_sec=float(item.get("healthcheck_timeout_sec", 1.0)),
            startup_grace_sec=float(item.get("startup_grace_sec", 0.0)),
            stop_timeout_sec=float(item.get("stop_timeout_sec", 10.0)),
        )

    actions: Dict[str, AgentActionSpec] = {}
    for item in _coerce_action_items(raw.get("actions", {})):
        action_id = str(item.get("id") or "").strip()
        if not action_id:
            raise ValueError("Each action must define a non-empty `id`")

        inferred_type = item.get("type")
        if not inferred_type:
            if item.get("process"):
                inferred_type = "managed_process"
            elif item.get("command") is not None:
                inferred_type = "command"
            elif action_id == "status":
                inferred_type = "status"
            else:
                raise ValueError(f"Action `{action_id}` is missing `type`")

        actions[action_id] = AgentActionSpec(
            name=action_id,
            action_type=str(inferred_type),
            label=str(item.get("label") or action_id.replace("_", " ").title()),
            config={key: value for key, value in item.items() if key not in {"id", "type", "label"}},
        )

    if not actions:
        raise ValueError("Deploy agent config must define at least one action")

    return AgentConfig(
        config_path=path,
        agent_id=str(raw.get("agent_id") or path.stem),
        label=str(raw.get("label") or raw.get("agent_id") or path.stem),
        role=str(raw.get("role") or "agent"),
        state_dir=resolved_state_dir,
        details=raw.get("details", {}) if isinstance(raw.get("details", {}), dict) else {},
        actions=actions,
        processes=processes,
    )


def _truncate_output(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class DeployAgentRuntime:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.lock = Lock()
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._action_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _logs_dir(self) -> Path:
        return self.config.state_dir / "logs"

    @property
    def _meta_dir(self) -> Path:
        return self.config.state_dir / "process_meta"

    @property
    def _action_dir(self) -> Path:
        return self.config.state_dir / "action_meta"

    def _pid_path(self, process_name: str) -> Path:
        return self.config.state_dir / f"{process_name}.pid"

    def _meta_path(self, process_name: str) -> Path:
        return self._meta_dir / f"{process_name}.json"

    def _action_path(self, action_name: str) -> Path:
        return self._action_dir / f"{action_name}.json"

    def _render_value(self, value: Any, params: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            try:
                return _render_template_string(value, params)
            except KeyError as exc:
                raise ValueError(f"Missing action param: {exc.args[0]}") from exc
        if isinstance(value, list):
            return [self._render_value(item, params) for item in value]
        if isinstance(value, dict):
            return {key: self._render_value(item, params) for key, item in value.items()}
        return value

    def _render_command(self, value: Union[str, List[str]], params: Dict[str, Any]) -> Union[str, List[str]]:
        rendered = self._render_value(value, params)
        if isinstance(rendered, list):
            return [str(item) for item in rendered]
        return str(rendered)

    def _resolve_runtime_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        return _resolve_path(value, self.config.config_path.parent)

    def _read_meta(self, process_name: str) -> Dict[str, Any]:
        path = self._meta_path(process_name)
        if not path.exists():
            return {}
        try:
            return _read_json(path)
        except Exception:
            return {}

    def _write_meta(self, process_name: str, payload: Dict[str, Any]) -> None:
        path = self._meta_path(process_name)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _read_action_meta(self, action_name: str) -> Dict[str, Any]:
        path = self._action_path(action_name)
        if not path.exists():
            return {}
        try:
            return _read_json(path)
        except Exception:
            return {}

    def _write_action_meta(self, action_name: str, payload: Dict[str, Any]) -> None:
        path = self._action_path(action_name)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _record_action(self, action_name: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
        self._write_action_meta(
            action_name,
            {
                "action": action_name,
                "ok": bool(result.get("ok", True)),
                "message": result.get("message"),
                "params": params,
                "result": result,
                "executed_at": datetime.now().isoformat(),
            },
        )

    def _process_pid(self, process_name: str) -> Optional[int]:
        pid_path = self._pid_path(process_name)
        if not pid_path.exists():
            return None
        try:
            return int(pid_path.read_text().strip())
        except Exception:
            pid_path.unlink(missing_ok=True)
            return None

    def _process_status(self, process_name: str, spec: ManagedProcessSpec) -> Dict[str, Any]:
        meta = self._read_meta(process_name)
        pid = self._process_pid(process_name) or int(meta.get("pid", 0) or 0)
        running = _pid_running(pid)

        if not running and self._pid_path(process_name).exists():
            self._pid_path(process_name).unlink(missing_ok=True)

        health: Optional[Dict[str, Any]] = None
        if running and spec.healthcheck_url:
            params = meta.get("params", {}) if isinstance(meta.get("params"), dict) else {}
            rendered_url = str(self._render_value(spec.healthcheck_url, params))
            health = self._probe_health(rendered_url, timeout=spec.healthcheck_timeout_sec)

        started_at = str(meta.get("started_at") or "")
        state = "running" if running else "stopped"
        if running and health and not health.get("ok", False):
            within_grace = False
            if started_at:
                try:
                    elapsed = time.time() - datetime.fromisoformat(started_at).timestamp()
                    within_grace = elapsed < spec.startup_grace_sec
                except ValueError:
                    within_grace = False
            state = "starting" if within_grace else "degraded"

        return {
            "running": running,
            "pid": pid if running else None,
            "state": state,
            "started_at": meta.get("started_at"),
            "command": meta.get("rendered_command"),
            "cwd": meta.get("cwd"),
            "health": health,
            "stdout_log": meta.get("stdout_log"),
            "stderr_log": meta.get("stderr_log"),
            "last_params": meta.get("params"),
            "configured_command": spec.command,
            "configured_cwd": str(self._resolve_runtime_path(spec.cwd)) if spec.cwd else None,
            "configured_stdout_log": str(self._resolve_runtime_path(spec.stdout_log)) if spec.stdout_log else None,
            "configured_stderr_log": str(self._resolve_runtime_path(spec.stderr_log)) if spec.stderr_log else None,
            "healthcheck_url": spec.healthcheck_url,
        }

    def _probe_health(self, url: str, timeout: float) -> Dict[str, Any]:
        try:
            req = urlrequest.Request(url, headers={"Accept": "application/json"})
            with urlrequest.urlopen(req, timeout=timeout) as response:
                raw = response.read()
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            return {"ok": True, "url": url, "payload": payload}
        except urlerror.HTTPError as exc:
            return {"ok": False, "url": url, "error": f"{exc.code} {exc.reason}"}
        except urlerror.URLError as exc:
            return {"ok": False, "url": url, "error": str(exc.reason)}
        except Exception as exc:
            return {"ok": False, "url": url, "error": str(exc)}

    def status_payload(self) -> Dict[str, Any]:
        processes = {
            name: self._process_status(name, spec)
            for name, spec in self.config.processes.items()
        }
        running = [name for name, item in processes.items() if item["state"] in {"running", "starting"}]
        degraded = [name for name, item in processes.items() if item["state"] == "degraded"]

        if degraded:
            state = "degraded"
            summary = f"Unhealthy processes: {', '.join(degraded)}"
        elif running:
            state = "running"
            summary = f"Running: {', '.join(running)}"
        else:
            state = "ready"
            summary = "Agent ready"

        action_catalog = {}
        for action_name, action in self.config.actions.items():
            entry: Dict[str, Any] = {
                "name": action.name,
                "label": action.label,
                "type": action.action_type,
            }
            if action.action_type == "managed_process":
                process_name = str(action.config.get("process") or "")
                process_spec = self.config.processes.get(process_name)
                entry.update(
                    {
                        "process": process_name,
                        "operation": action.config.get("operation"),
                        "process_spec": {
                            "command": process_spec.command if process_spec else None,
                            "cwd": process_spec.cwd if process_spec else None,
                            "shell": process_spec.shell if process_spec else False,
                            "stdout_log": process_spec.stdout_log if process_spec else None,
                            "stderr_log": process_spec.stderr_log if process_spec else None,
                            "healthcheck_url": process_spec.healthcheck_url if process_spec else None,
                        },
                    }
                )
            elif action.action_type == "command":
                entry.update(
                    {
                        "command": action.config.get("command"),
                        "cwd": action.config.get("cwd"),
                        "shell": bool(action.config.get("shell", False)),
                        "timeout_sec": action.config.get("timeout_sec"),
                    }
                )
            action_catalog[action_name] = entry

        details = {
            **self.config.details,
            "agent_id": self.config.agent_id,
            "hostname": socket.gethostname(),
            "config_path": str(self.config.config_path),
            "state_dir": str(self.config.state_dir),
            "processes": processes,
            "action_catalog": action_catalog,
            "last_actions": {
                action_name: self._read_action_meta(action_name)
                for action_name in self.config.actions.keys()
            },
        }

        return {
            "state": state,
            "summary": summary,
            "actions": list(self.config.actions.keys()),
            "details": details,
        }

    def _build_process_launch(self, spec: ManagedProcessSpec, params: Dict[str, Any]) -> Dict[str, Any]:
        command = self._render_command(spec.command, params)
        if isinstance(command, list) and spec.shell:
            command = " ".join(shlex.quote(item) for item in command)
        if isinstance(command, str) and not spec.shell:
            command = shlex.split(command)
        cwd_path = self._resolve_runtime_path(spec.cwd) if spec.cwd else None
        env_values = self._render_value(spec.env, params)
        env = os.environ.copy()
        for key, value in env_values.items():
            env[str(key)] = str(value)

        stdout_path = self._resolve_runtime_path(spec.stdout_log) if spec.stdout_log else self._logs_dir / f"{spec.name}.stdout.log"
        stderr_path = self._resolve_runtime_path(spec.stderr_log) if spec.stderr_log else self._logs_dir / f"{spec.name}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "command": command,
            "cwd": str(cwd_path) if cwd_path else None,
            "env": env,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        }

    def _start_managed_process(self, process_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = self.config.processes.get(process_name)
        if spec is None:
            raise ValueError(f"Unknown managed process: {process_name}")

        status = self._process_status(process_name, spec)
        if status["running"]:
            return {
                "ok": True,
                "message": f"{process_name} is already running",
                "details": status,
            }

        launch = self._build_process_launch(spec, params)
        command = launch["command"]
        shell = spec.shell

        with open(launch["stdout_path"], "ab") as stdout_f, open(launch["stderr_path"], "ab") as stderr_f:
            proc = subprocess.Popen(
                command,
                cwd=launch["cwd"],
                env=launch["env"],
                shell=shell,
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )

        pid = int(proc.pid)
        self._pid_path(process_name).write_text(str(pid))
        meta = {
            "pid": pid,
            "started_at": datetime.now().isoformat(),
            "params": params,
            "cwd": launch["cwd"],
            "rendered_command": command if isinstance(command, list) else command,
            "stdout_log": str(launch["stdout_path"]),
            "stderr_log": str(launch["stderr_path"]),
        }
        self._write_meta(process_name, meta)

        time.sleep(min(spec.startup_grace_sec, 0.5))
        if proc.poll() is not None:
            stderr_text = ""
            try:
                stderr_text = _truncate_output(launch["stderr_path"].read_text())
            except Exception:
                stderr_text = ""
            self._pid_path(process_name).unlink(missing_ok=True)
            raise RuntimeError(stderr_text or f"{process_name} exited immediately with code {proc.returncode}")

        return {
            "ok": True,
            "message": f"Started {process_name}",
            "details": self._process_status(process_name, spec),
        }

    def _stop_managed_process(self, process_name: str) -> Dict[str, Any]:
        spec = self.config.processes.get(process_name)
        if spec is None:
            raise ValueError(f"Unknown managed process: {process_name}")

        pid = self._process_pid(process_name)
        if pid is None or not _pid_running(pid):
            self._pid_path(process_name).unlink(missing_ok=True)
            meta = self._read_meta(process_name)
            if meta:
                meta["stopped_at"] = datetime.now().isoformat()
                self._write_meta(process_name, meta)
            return {"ok": True, "message": f"{process_name} is not running", "details": self._process_status(process_name, spec)}

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            os.kill(pid, signal.SIGTERM)

        deadline = time.time() + spec.stop_timeout_sec
        while time.time() < deadline:
            if not _pid_running(pid):
                break
            time.sleep(0.2)

        if _pid_running(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                os.kill(pid, signal.SIGKILL)

        self._pid_path(process_name).unlink(missing_ok=True)
        meta = self._read_meta(process_name)
        if meta:
            meta["stopped_at"] = datetime.now().isoformat()
            self._write_meta(process_name, meta)

        return {
            "ok": True,
            "message": f"Stopped {process_name}",
            "details": self._process_status(process_name, spec),
        }

    def _command_result(self, completed: subprocess.CompletedProcess) -> Dict[str, Any]:
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        return {
            "returncode": completed.returncode,
            "stdout": _truncate_output(stdout),
            "stderr": _truncate_output(stderr),
        }

    def _run_command_action(self, action: AgentActionSpec, params: Dict[str, Any]) -> Dict[str, Any]:
        raw_command = action.config.get("command")
        if raw_command is None:
            raise ValueError(f"Action `{action.name}` is missing `command`")

        shell = bool(action.config.get("shell", False))
        command = self._render_command(raw_command, params)
        if isinstance(command, list) and shell:
            command = " ".join(shlex.quote(item) for item in command)
        if isinstance(command, str) and not shell:
            command = shlex.split(command)
        cwd = self._resolve_runtime_path(str(action.config["cwd"])) if action.config.get("cwd") else None
        env = os.environ.copy()
        rendered_env = self._render_value(action.config.get("env", {}), params)
        if isinstance(rendered_env, dict):
            for key, value in rendered_env.items():
                env[str(key)] = str(value)

        timeout_sec = float(action.config.get("timeout_sec", 60.0))
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        result = self._command_result(completed)
        if completed.returncode != 0:
            raise RuntimeError(result["stderr"] or f"Command exited with code {completed.returncode}")
        return {
            "ok": True,
            "message": action.label,
            "details": {
                "command": command,
                "cwd": str(cwd) if cwd else None,
                **result,
            },
        }

    def execute(self, action_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        action = self.config.actions.get(action_name)
        if action is None:
            raise ValueError(f"Unknown action: {action_name}")

        with self.lock:
            if action.action_type == "status":
                payload = self.status_payload()
                result = {
                    "ok": True,
                    "message": payload["summary"],
                    **payload,
                }
                self._record_action(action_name, params, result)
                return result

            if action.action_type == "managed_process":
                process_name = str(action.config.get("process") or "")
                operation = str(action.config.get("operation") or "status")
                if not process_name:
                    raise ValueError(f"Action `{action_name}` is missing `process`")
                if operation == "start":
                    result = self._start_managed_process(process_name, params)
                    self._record_action(action_name, params, result)
                    return result
                if operation == "stop":
                    result = self._stop_managed_process(process_name)
                    self._record_action(action_name, params, result)
                    return result
                if operation == "restart":
                    self._stop_managed_process(process_name)
                    result = self._start_managed_process(process_name, params)
                    self._record_action(action_name, params, result)
                    return result
                if operation == "status":
                    spec = self.config.processes.get(process_name)
                    if spec is None:
                        raise ValueError(f"Unknown managed process: {process_name}")
                    result = {
                        "ok": True,
                        "message": action.label,
                        "details": self._process_status(process_name, spec),
                    }
                    self._record_action(action_name, params, result)
                    return result
                raise ValueError(f"Unsupported managed_process operation: {operation}")

            if action.action_type == "command":
                result = self._run_command_action(action, params)
                self._record_action(action_name, params, result)
                return result

            raise ValueError(f"Unsupported action type: {action.action_type}")


def create_agent_app(config_path: Union[str, Path]) -> FastAPI:
    config = load_agent_config(config_path)
    runtime = DeployAgentRuntime(config)
    app = FastAPI(
        title=f"{config.label} Deploy Agent",
        version="0.1.0",
        description="Config-driven remote deploy agent for VLA-Lab dashboards.",
    )
    app.state.runtime = runtime

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": "VLA-Lab Deploy Agent",
            "agent_id": config.agent_id,
            "label": config.label,
            "role": config.role,
            "status": "ok",
            "health": "/health",
            "status_url": "/status",
            "actions": list(config.actions.keys()),
            "docs": "/docs",
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return runtime.status_payload()

    @app.get("/status")
    def status() -> Dict[str, Any]:
        return runtime.status_payload()

    @app.post("/action")
    def action(payload: AgentActionRequest = Body(...)) -> Dict[str, Any]:
        try:
            return runtime.execute(payload.action, payload.params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc

    @app.post("/actions/{action_name}")
    def action_route(action_name: str, payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
        try:
            return runtime.execute(action_name, payload or {})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc

    return app


def serve_agent(config_path: Union[str, Path], host: str = "0.0.0.0", port: int = 9001, reload: bool = False) -> None:
    import uvicorn

    app = create_agent_app(config_path)
    uvicorn.run(app, host=host, port=port, reload=reload)
