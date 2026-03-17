"""SSH-backed deploy helpers for the simplified /deploy dashboard."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    DeployCommandSpec,
    DeployInputSpec,
    DeployJob,
    DeployJobLogsResponse,
    DeployJobsResponse,
    DeployOverviewResponse,
    DeployRunRequest,
    DeployRunResponse,
    DeployTargetConnection,
)

TARGET_ORDER = ("server", "client")
INPUT_ORDER = ("model_server_config_path", "inference_client_config_path")
COMMAND_ORDER = (
    "start_model_server",
    "start_robot_server",
    "set_joint_preset",
    "start_inference_client",
)
COMMAND_REQUIRED_PLACEHOLDERS = {
    "start_model_server": {"model_server_config_path"},
    "start_robot_server": set(),
    "set_joint_preset": set(),
    "start_inference_client": {"inference_client_config_path"},
}
PLACEHOLDER_PATTERN = re.compile(r"(?<!\$)\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
OUTPUT_LIMIT_CHARS = 4000
INPUT_VALUES_FILENAME = "input_values.json"

_CONTROLLER: Optional["DeployController"] = None
_CONTROLLER_SIG: Optional[Tuple[str, int]] = None
_CONTROLLER_LOCK = threading.Lock()


@dataclass(frozen=True)
class TargetConfig:
    id: str
    label: str
    ssh_host: str
    workdir: str
    shell: str = "bash -lc"


@dataclass(frozen=True)
class InputConfig:
    id: str
    label: str
    type: str
    required: bool
    default: Optional[str] = None
    options: Tuple[str, ...] = ()
    target: Optional[str] = None


@dataclass(frozen=True)
class CommandConfig:
    id: str
    label: str
    target_id: str
    background: bool
    template: str
    stdout_log: Optional[str]
    stderr_log: Optional[str]
    required_inputs: Tuple[str, ...]


@dataclass(frozen=True)
class DeployConfig:
    path: Path
    state_dir: Path
    targets: Dict[str, TargetConfig]
    inputs: Dict[str, InputConfig]
    commands: Dict[str, CommandConfig]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _trim_output(text: str, max_chars: int = OUTPUT_LIMIT_CHARS) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_config_path() -> Path:
    return (_repo_root() / "configs" / "deploy" / "dashboard.json").resolve()


def resolve_deploy_config_path(explicit_path: Optional[str] = None) -> Path:
    raw = explicit_path or os.getenv("VLALAB_DEPLOY_CONFIG") or str(_default_config_path())
    return Path(raw).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Deploy config root must be a JSON object")
    return payload


def _resolve_optional_path(value: Optional[str], base_dir: Path, default: Path) -> Path:
    if not value:
        return default.resolve()
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _ordered_placeholders(template: str) -> List[str]:
    seen: List[str] = []
    for match in PLACEHOLDER_PATTERN.finditer(str(template or "")):
        key = match.group(1)
        if key not in seen:
            seen.append(key)
    return seen


def _normalize_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Deploy config field `{field_name}` must be a non-empty string")
    return text


def load_deploy_config(config_path: Optional[str] = None) -> DeployConfig:
    path = resolve_deploy_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Deploy config not found: {path}")

    raw = _load_json(path)
    config_dir = path.parent
    state_dir = _resolve_optional_path(raw.get("state_dir"), config_dir, config_dir / ".deploy_state")

    raw_targets = raw.get("targets")
    if not isinstance(raw_targets, dict):
        raise ValueError("Deploy config must define `targets` as an object")
    raw_inputs = raw.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise ValueError("Deploy config must define `inputs` as an object")
    raw_commands = raw.get("commands")
    if not isinstance(raw_commands, dict):
        raise ValueError("Deploy config must define `commands` as an object")

    targets: Dict[str, TargetConfig] = {}
    for target_id in TARGET_ORDER:
        payload = raw_targets.get(target_id)
        if not isinstance(payload, dict):
            raise ValueError(f"Deploy config is missing target `{target_id}`")
        targets[target_id] = TargetConfig(
            id=target_id,
            label=_normalize_text(payload.get("label") or target_id.replace("_", " ").title(), f"targets.{target_id}.label"),
            ssh_host=_normalize_text(payload.get("ssh_host"), f"targets.{target_id}.ssh_host"),
            workdir=_normalize_text(payload.get("workdir"), f"targets.{target_id}.workdir"),
            shell=str(payload.get("shell") or "bash -lc").strip() or "bash -lc",
        )

    inputs: Dict[str, InputConfig] = {}
    for input_id in INPUT_ORDER:
        payload = raw_inputs.get(input_id)
        if not isinstance(payload, dict):
            raise ValueError(f"Deploy config is missing input `{input_id}`")

        input_type = _normalize_text(payload.get("type"), f"inputs.{input_id}.type")
        required = bool(payload.get("required", True))
        default = payload.get("default")
        target = payload.get("target")
        options = payload.get("options", [])
        if options is None:
            options = []
        if not isinstance(options, list):
            raise ValueError(f"Deploy config field `inputs.{input_id}.options` must be a list")
        normalized_options = tuple(str(item).strip() for item in options if str(item).strip())

        if input_type == "path":
            if target not in TARGET_ORDER:
                raise ValueError(f"Path input `{input_id}` must define `target` as `server` or `client`")
        elif input_type == "enum":
            if not normalized_options:
                raise ValueError(f"Enum input `{input_id}` must define at least one option")
            if default is None:
                default = normalized_options[0]
        else:
            raise ValueError(f"Unsupported deploy input type `{input_type}` for `{input_id}`")

        inputs[input_id] = InputConfig(
            id=input_id,
            label=_normalize_text(payload.get("label") or input_id.replace("_", " ").title(), f"inputs.{input_id}.label"),
            type=input_type,
            required=required,
            default=str(default).strip() if default is not None and str(default).strip() else None,
            options=normalized_options,
            target=str(target).strip() if target else None,
        )

    commands: Dict[str, CommandConfig] = {}
    for command_id in COMMAND_ORDER:
        payload = raw_commands.get(command_id)
        if not isinstance(payload, dict):
            raise ValueError(f"Deploy config is missing command `{command_id}`")

        template = _normalize_text(payload.get("template"), f"commands.{command_id}.template")
        placeholders = _ordered_placeholders(template)
        unknown_placeholders = [item for item in placeholders if item not in inputs]
        if unknown_placeholders:
            raise ValueError(
                f"Command `{command_id}` references unknown inputs: {', '.join(unknown_placeholders)}"
            )
        missing_required = COMMAND_REQUIRED_PLACEHOLDERS[command_id] - set(placeholders)
        if missing_required:
            raise ValueError(
                f"Command `{command_id}` must include placeholders: {', '.join(sorted(missing_required))}"
            )

        background = bool(payload.get("background", False))
        stdout_log = str(payload.get("stdout_log")).strip() if payload.get("stdout_log") else None
        stderr_log = str(payload.get("stderr_log")).strip() if payload.get("stderr_log") else None
        if background and (not stdout_log or not stderr_log):
            raise ValueError(f"Background command `{command_id}` must define both stdout_log and stderr_log")

        target_id = _normalize_text(payload.get("target"), f"commands.{command_id}.target")
        if target_id not in targets:
            raise ValueError(f"Command `{command_id}` targets unknown target `{target_id}`")

        commands[command_id] = CommandConfig(
            id=command_id,
            label=_normalize_text(payload.get("label") or command_id.replace("_", " ").title(), f"commands.{command_id}.label"),
            target_id=target_id,
            background=background,
            template=template,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            required_inputs=tuple(placeholders),
        )

    return DeployConfig(
        path=path,
        state_dir=state_dir,
        targets=targets,
        inputs=inputs,
        commands=commands,
    )


class DeployController:
    def __init__(self, config: DeployConfig):
        self.config = config
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="deploy-job")
        self._stop_event = threading.Event()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._futures: Dict[str, Future[Any]] = {}
        self._local_processes: Dict[str, subprocess.Popen[str]] = {}
        self._input_values_path = self.config.state_dir / INPUT_VALUES_FILENAME
        self._input_values: Dict[str, str] = {}
        self._jobs_dir = self.config.state_dir / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_input_values()
        self._load_jobs()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="deploy-job-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        with self._lock:
            local_job_ids = list(self._local_processes.keys())
        for job_id in local_job_ids:
            self._terminate_local_process(job_id, sig=signal.SIGTERM)
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _job_path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    def _persist_job(self, job: Dict[str, Any]) -> None:
        path = self._job_path(str(job["id"]))
        with open(path, "w") as handle:
            json.dump(job, handle, indent=2, sort_keys=True)

    def _persist_input_values(self) -> None:
        self._input_values_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._input_values_path, "w") as handle:
            json.dump(self._input_values, handle, indent=2, sort_keys=True)

    def _load_input_values(self) -> None:
        if not self._input_values_path.exists():
            self._input_values = {}
            return
        try:
            payload = _load_json(self._input_values_path)
        except Exception:
            self._input_values = {}
            return
        normalized: Dict[str, str] = {}
        for input_id in INPUT_ORDER:
            value = payload.get(input_id)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                normalized[input_id] = text
        self._input_values = normalized

    def _load_jobs(self) -> None:
        for path in sorted(self._jobs_dir.glob("*.json")):
            try:
                payload = _load_json(path)
            except Exception:
                continue
            job_id = str(payload.get("id") or "").strip()
            if not job_id:
                continue
            self._jobs[job_id] = payload

    def _public_job(self, payload: Dict[str, Any]) -> DeployJob:
        state = str(payload.get("state") or "queued")
        return DeployJob(
            id=str(payload.get("id")),
            command_id=str(payload.get("command_id")),
            target_id=str(payload.get("target_id")),
            state=state,
            remote_pid=int(payload["remote_pid"]) if payload.get("remote_pid") not in {None, ""} else None,
            stdout_log=payload.get("stdout_log"),
            stderr_log=payload.get("stderr_log"),
            last_stdout=payload.get("last_stdout"),
            last_stderr=payload.get("last_stderr"),
            submitted_at=str(payload.get("submitted_at") or _now_iso()),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            error=payload.get("error"),
            stoppable=state in {"queued", "running", "stopping"},
        )

    def list_jobs(self) -> List[DeployJob]:
        with self._lock:
            jobs = [self._public_job(item) for item in self._jobs.values()]
        return sorted(jobs, key=lambda item: item.submitted_at, reverse=True)

    def refresh_jobs(self) -> None:
        with self._lock:
            job_ids = [
                job_id
                for job_id, payload in self._jobs.items()
                if payload.get("_background") and str(payload.get("state")) in {"running", "stopping"}
            ]
        for job_id in job_ids:
            self._refresh_background_job(job_id)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(3.0):
            self.refresh_jobs()

    def _ssh_prefix(self, ssh_host: str, connect_timeout: int = 5) -> List[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            ssh_host,
        ]

    def _shell_command(self, shell: str, script: str) -> str:
        normalized_shell = str(shell or "bash -lc").strip() or "bash -lc"
        return f"{normalized_shell} {shlex.quote(script)}"

    def _shell_program(self, shell: str) -> str:
        normalized_shell = str(shell or "bash -lc").strip() or "bash -lc"
        argv = shlex.split(normalized_shell)
        return argv[0] if argv else "bash"

    def _run_ssh_script(
        self,
        target: TargetConfig,
        script: str,
        *,
        timeout: float = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        command = self._ssh_prefix(target.ssh_host) + [self._shell_command(target.shell, script)]
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )

    def _run_ssh_host(
        self,
        ssh_host: str,
        script: str,
        *,
        timeout: float = 20.0,
    ) -> subprocess.CompletedProcess[str]:
        command = self._ssh_prefix(ssh_host) + [f"bash -lc {shlex.quote(script)}"]
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )

    def _spawn_ssh_script(
        self,
        target: TargetConfig,
        script: str,
    ) -> subprocess.Popen[str]:
        command = self._ssh_prefix(target.ssh_host) + [self._shell_command(target.shell, script)]
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def probe_target(self, target_id: str) -> DeployTargetConnection:
        target = self.config.targets[target_id]
        command = self._ssh_prefix(target.ssh_host, connect_timeout=3) + ["echo __vlalab_ok__"]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
            )
        except Exception as exc:
            return DeployTargetConnection(
                id=target.id,
                label=target.label,
                connected=False,
                last_error=str(exc),
            )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode == 0 and "__vlalab_ok__" in stdout:
            return DeployTargetConnection(id=target.id, label=target.label, connected=True, last_error=None)
        message = stderr or stdout or f"ssh exited with code {result.returncode}"
        return DeployTargetConnection(
            id=target.id,
            label=target.label,
            connected=False,
            last_error=message,
        )

    def input_specs(self) -> List[DeployInputSpec]:
        merged_values = self._merge_values_with_defaults({})
        items: List[DeployInputSpec] = []
        for input_id in INPUT_ORDER:
            spec = self.config.inputs[input_id]
            items.append(
                DeployInputSpec(
                    id=spec.id,
                    label=spec.label,
                    type=spec.type,
                    required=spec.required,
                    default=spec.default,
                    current_value=merged_values.get(spec.id, spec.default),
                    options=list(spec.options),
                )
            )
        return items

    def _merge_values_with_defaults(self, values: Optional[Dict[str, Any]]) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        raw_values = values or {}
        for input_id in INPUT_ORDER:
            spec = self.config.inputs[input_id]
            raw = raw_values.get(input_id, self._input_values.get(input_id, spec.default))
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                merged[input_id] = text
        return merged

    def get_saved_values(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._merge_values_with_defaults({}))

    def save_input_values(self, values: Optional[Dict[str, Any]]) -> Dict[str, str]:
        with self._lock:
            next_values = dict(self._input_values)
            raw_values = values or {}
            for input_id in INPUT_ORDER:
                if input_id not in raw_values:
                    continue
                raw = raw_values.get(input_id)
                text = str(raw).strip() if raw is not None else ""
                if text:
                    next_values[input_id] = text
                else:
                    next_values.pop(input_id, None)
            self._input_values = next_values
            self._persist_input_values()
            return dict(self._input_values)

    def _render_template(
        self,
        template: str,
        values: Dict[str, str],
        *,
        preserve_missing: bool,
    ) -> str:
        missing: List[str] = []

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            value = values.get(key)
            if value is None or value == "":
                if preserve_missing:
                    missing.append(key)
                    return "{" + key + "}"
                raise ValueError(f"Missing required deploy input: {key}")
            return shlex.quote(str(value))

        rendered = PLACEHOLDER_PATTERN.sub(repl, template)
        if not preserve_missing and missing:
            raise ValueError(f"Missing required deploy input: {', '.join(missing)}")
        return rendered

    def command_specs(self, values: Optional[Dict[str, Any]] = None) -> List[DeployCommandSpec]:
        merged_values = self._merge_values_with_defaults(values)
        items: List[DeployCommandSpec] = []
        for command_id in COMMAND_ORDER:
            command = self.config.commands[command_id]
            rendered = self._render_template(command.template, merged_values, preserve_missing=True)
            preview = f"cd {shlex.quote(self.config.targets[command.target_id].workdir)} && {rendered}"
            items.append(
                DeployCommandSpec(
                    id=command.id,
                    label=command.label,
                    target_id=command.target_id,
                    background=command.background,
                    required_inputs=list(command.required_inputs),
                    resolved_preview=preview,
                )
            )
        return items

    def _status_path_for_job(self, command: CommandConfig, job_id: str) -> str:
        if command.stdout_log:
            return f"{command.stdout_log}.{job_id}.status"
        return f"/tmp/vlalab/{job_id}.status"

    def _launcher_path_for_job(self, command: CommandConfig, job_id: str) -> str:
        command_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", command.id).strip("._") or "deploy"
        return f"/tmp/vlalab/{command_slug}.{job_id}.launcher.sh"

    def _validate_values_for_command(self, command: CommandConfig, values: Dict[str, Any]) -> Dict[str, str]:
        merged = self._merge_values_with_defaults(values)
        missing = [item for item in command.required_inputs if item not in merged]
        if missing:
            raise ValueError(f"Missing required deploy input(s): {', '.join(missing)}")
        return merged

    def submit_command(self, payload: DeployRunRequest) -> DeployRunResponse:
        command = self.config.commands.get(payload.command_id)
        if command is None:
            raise ValueError(f"Unknown deploy command: {payload.command_id}")

        values = self._validate_values_for_command(command, payload.values)
        self.save_input_values(values)
        target_status = self.probe_target(command.target_id)
        if not target_status.connected:
            raise RuntimeError(target_status.last_error or f"Target `{command.target_id}` is not reachable")

        job_id = uuid.uuid4().hex[:12]
        status_path = self._status_path_for_job(command, job_id) if command.background else None
        job = {
            "id": job_id,
            "command_id": command.id,
            "target_id": command.target_id,
            "state": "queued",
            "remote_pid": None,
            "stdout_log": command.stdout_log,
            "stderr_log": command.stderr_log,
            "last_stdout": "",
            "last_stderr": "",
            "submitted_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "_background": command.background,
            "_ssh_host": self.config.targets[command.target_id].ssh_host,
            "_status_path": status_path,
            "_values": values,
            "_stop_requested": False,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist_job(job)
        future = self._executor.submit(self._execute_job, job_id)
        with self._lock:
            if future is not None:
                self._futures[job_id] = future
        return DeployRunResponse(ok=True, message=f"{command.label} 已提交", job=self._public_job(job))

    def _set_job_stopped(self, job_id: str, message: Optional[str] = None) -> Optional[DeployJob]:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return None
            current["state"] = "stopped"
            current["finished_at"] = current.get("finished_at") or _now_iso()
            current["error"] = message
            current["_stop_requested"] = True
            self._persist_job(current)
            return self._public_job(current)

    def _terminate_local_process(self, job_id: str, *, sig: int = signal.SIGTERM) -> bool:
        with self._lock:
            proc = self._local_processes.get(job_id)
        if proc is None or proc.poll() is not None:
            return False
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return False
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                return False
        return True

    def _stop_remote_process(self, ssh_host: str, remote_pid: int) -> None:
        pid = int(remote_pid)
        script = (
            f"kill -TERM -- -{pid} >/dev/null 2>&1 || kill -TERM {pid} >/dev/null 2>&1 || true\n"
            "sleep 1\n"
            f"kill -0 -- -{pid} >/dev/null 2>&1 && kill -KILL -- -{pid} >/dev/null 2>&1 || true\n"
            f"kill -0 {pid} >/dev/null 2>&1 && kill -KILL {pid} >/dev/null 2>&1 || true"
        )
        self._run_ssh_host(ssh_host, script, timeout=10.0)

    def stop_job(self, job_id: str) -> DeployJob:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise ValueError(f"Unknown deploy job: {job_id}")
            if str(current.get("state")) not in {"queued", "running", "stopping"}:
                raise ValueError(f"Deploy job `{job_id}` is not running")
            current["_stop_requested"] = True
            current["state"] = "stopping" if current.get("started_at") else "queued"
            self._persist_job(current)
            future = self._futures.get(job_id)
            ssh_host = str(current.get("_ssh_host") or "")
            remote_pid = current.get("remote_pid")

        if future is not None and future.cancel():
            stopped = self._set_job_stopped(job_id, "Stopped before execution")
            if stopped is not None:
                return stopped

        local_stopped = self._terminate_local_process(job_id, sig=signal.SIGTERM)
        if remote_pid not in {None, ""} and ssh_host:
            try:
                self._stop_remote_process(ssh_host, int(remote_pid))
            except Exception:
                pass

        if not local_stopped and remote_pid in {None, ""}:
            stopped = self._set_job_stopped(job_id, "Stop requested")
            if stopped is not None:
                return stopped

        with self._lock:
            latest = self._jobs.get(job_id)
            if latest is None:
                raise ValueError(f"Unknown deploy job: {job_id}")
            latest["state"] = "stopping"
            latest["error"] = "Stopping..."
            self._persist_job(latest)
            return self._public_job(latest)

    def _execute_job(self, job_id: str) -> None:
        with self._lock:
            job = dict(self._jobs.get(job_id) or {})
        if not job:
            return

        try:
            if job.get("_stop_requested"):
                self._set_job_stopped(job_id, "Stopped before execution")
                return
            command = self.config.commands[job["command_id"]]
            target = self.config.targets[job["target_id"]]
            values = dict(job.get("_values") or {})
            if command.background:
                self._start_background_job(job_id, command, target, values)
            else:
                self._run_foreground_job(job_id, command, target, values)
        except Exception as exc:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is None:
                    return
                if current.get("_stop_requested"):
                    current["state"] = "stopped"
                    current["error"] = current.get("error") or "Stopped"
                    current["finished_at"] = current.get("finished_at") or _now_iso()
                else:
                    current["state"] = "failed"
                    current["error"] = str(exc)
                    current["finished_at"] = _now_iso()
                self._persist_job(current)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    def _start_background_job(
        self,
        job_id: str,
        command: CommandConfig,
        target: TargetConfig,
        values: Dict[str, str],
    ) -> None:
        rendered_command = self._render_template(command.template, values, preserve_missing=False)
        stdout_log = str(command.stdout_log)
        stderr_log = str(command.stderr_log)
        status_path = str(self._jobs[job_id]["_status_path"])
        launcher_path = self._launcher_path_for_job(command, job_id)
        quoted_workdir = shlex.quote(target.workdir)
        quoted_stdout = shlex.quote(stdout_log)
        quoted_stderr = shlex.quote(stderr_log)
        quoted_status = shlex.quote(status_path)
        quoted_launcher = shlex.quote(launcher_path)
        mkdir_dirs = {
            str(Path(stdout_log).parent),
            str(Path(stderr_log).parent),
            str(Path(status_path).parent),
            str(Path(launcher_path).parent),
        }
        mkdir_script = "mkdir -p " + " ".join(shlex.quote(item) for item in sorted(mkdir_dirs))
        launcher_body = "\n".join(
            [
                "#!/usr/bin/env bash",
                f"trap 'rm -f {quoted_launcher}' EXIT",
                f"if ! cd {quoted_workdir}; then",
                f"  code=$?; printf '%s\\n%s\\n' \"$code\" \"$(date -Iseconds)\" > {quoted_status}; exit \"$code\"",
                "fi",
                rendered_command,
                f"code=$?; printf '%s\\n%s\\n' \"$code\" \"$(date -Iseconds)\" > {quoted_status}; exit \"$code\"",
            ]
        )
        heredoc_tag = f"__VLALAB_LAUNCHER_{job_id.upper()}__"
        remote_script = (
            f"{mkdir_script}\n"
            f"cat > {quoted_launcher} <<'{heredoc_tag}'\n"
            f"{launcher_body}\n"
            f"{heredoc_tag}\n"
            f"chmod +x {quoted_launcher}\n"
            f": > {quoted_stdout}\n"
            f": > {quoted_stderr}\n"
            f"rm -f {quoted_status}\n"
            f"nohup setsid {shlex.quote(self._shell_program(target.shell))} {quoted_launcher} "
            f">{quoted_stdout} 2>{quoted_stderr} < /dev/null & echo $!"
        )

        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                current["started_at"] = _now_iso()
                current["state"] = "running"
                self._persist_job(current)

        result = self._run_ssh_script(target, remote_script, timeout=30.0)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            raise RuntimeError(stderr or stdout or f"Failed to start `{command.id}`")
        pid_text = stdout.splitlines()[-1].strip() if stdout else ""
        if not pid_text.isdigit():
            raise RuntimeError(f"Failed to parse remote pid for `{command.id}`: {stdout or stderr}")

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            current["state"] = "running"
            current["remote_pid"] = int(pid_text)
            current["stdout_log"] = stdout_log
            current["stderr_log"] = stderr_log
            current["last_stdout"] = ""
            current["last_stderr"] = ""
            current["error"] = None
            self._persist_job(current)
            stop_requested = bool(current.get("_stop_requested"))

        if stop_requested:
            try:
                self._stop_remote_process(target.ssh_host, int(pid_text))
            except Exception:
                pass

    def _run_foreground_job(
        self,
        job_id: str,
        command: CommandConfig,
        target: TargetConfig,
        values: Dict[str, str],
    ) -> None:
        rendered_command = self._render_template(command.template, values, preserve_missing=False)
        remote_script = f"cd {shlex.quote(target.workdir)} && {rendered_command}"

        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                current["state"] = "running"
                current["started_at"] = _now_iso()
                self._persist_job(current)

        proc = self._spawn_ssh_script(target, remote_script)
        with self._lock:
            self._local_processes[job_id] = proc

        try:
            stdout_text, stderr_text = proc.communicate(timeout=300.0)
        except subprocess.TimeoutExpired:
            self._terminate_local_process(job_id, sig=signal.SIGKILL)
            stdout_text, stderr_text = proc.communicate()
            raise RuntimeError(f"Foreground command `{command.id}` timed out after 300s")
        finally:
            with self._lock:
                self._local_processes.pop(job_id, None)

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            current["last_stdout"] = _trim_output(stdout_text or "")
            current["last_stderr"] = _trim_output(stderr_text or "")
            current["finished_at"] = _now_iso()
            if current.get("_stop_requested"):
                current["state"] = "stopped"
                current["error"] = current.get("error") or "Stopped"
            elif proc.returncode == 0:
                current["state"] = "success"
                current["error"] = None
            else:
                current["state"] = "failed"
                current["error"] = _trim_output((stderr_text or stdout_text or "").strip())
            self._persist_job(current)

    def _is_remote_pid_alive(self, ssh_host: str, remote_pid: int) -> bool:
        result = self._run_ssh_host(
            ssh_host,
            f"kill -0 {int(remote_pid)} >/dev/null 2>&1",
            timeout=10.0,
        )
        return result.returncode == 0

    def _read_remote_file_tail(self, ssh_host: str, path: Optional[str], lines: int) -> str:
        if not path:
            return ""
        safe_lines = max(1, min(int(lines), 500))
        result = self._run_ssh_host(
            ssh_host,
            f"if [ -f {shlex.quote(path)} ]; then tail -n {safe_lines} {shlex.quote(path)}; fi",
            timeout=10.0,
        )
        if result.returncode != 0:
            return ""
        return _trim_output(result.stdout or "")

    def _read_remote_status(self, ssh_host: str, path: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
        if not path:
            return None, None
        result = self._run_ssh_host(
            ssh_host,
            f"if [ -f {shlex.quote(path)} ]; then cat {shlex.quote(path)}; fi",
            timeout=10.0,
        )
        if result.returncode != 0:
            return None, None
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if not lines:
            return None, None
        exit_code = int(lines[0]) if lines[0].isdigit() else None
        finished_at = lines[1] if len(lines) > 1 else None
        return exit_code, finished_at

    def _refresh_background_job(self, job_id: str) -> None:
        with self._lock:
            job = dict(self._jobs.get(job_id) or {})
        if not job:
            return

        ssh_host = str(job.get("_ssh_host") or "")
        remote_pid = job.get("remote_pid")
        if not ssh_host or remote_pid in {None, ""}:
            return

        stdout_text = self._read_remote_file_tail(ssh_host, job.get("stdout_log"), lines=60)
        stderr_text = self._read_remote_file_tail(ssh_host, job.get("stderr_log"), lines=60)

        alive = False
        try:
            alive = self._is_remote_pid_alive(ssh_host, int(remote_pid))
        except Exception:
            alive = False

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            current["last_stdout"] = stdout_text
            current["last_stderr"] = stderr_text
            if alive:
                if current.get("_stop_requested"):
                    current["state"] = "stopping"
                    current["error"] = "Stopping..."
                self._persist_job(current)
                return

        exit_code, finished_at = self._read_remote_status(ssh_host, job.get("_status_path"))
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            current["finished_at"] = finished_at or current.get("finished_at") or _now_iso()
            if current.get("_stop_requested"):
                current["state"] = "stopped"
                current["error"] = current.get("error") or "Stopped"
            elif exit_code == 0:
                current["state"] = "success"
                current["error"] = None
            else:
                current["state"] = "failed"
                if exit_code is None:
                    current["error"] = current.get("error") or "Background process exited without a status file"
                else:
                    current["error"] = current.get("error") or f"Remote exit code: {exit_code}"
            self._persist_job(current)

    def get_logs(self, job_id: str, stream: str, lines: int) -> DeployJobLogsResponse:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("`stream` must be either `stdout` or `stderr`")

        with self._lock:
            job = dict(self._jobs.get(job_id) or {})
        if not job:
            raise ValueError(f"Unknown deploy job: {job_id}")

        path_key = "stdout_log" if stream == "stdout" else "stderr_log"
        cache_key = "last_stdout" if stream == "stdout" else "last_stderr"
        content = str(job.get(cache_key) or "")
        path = job.get(path_key)

        if job.get("_background") and job.get("_ssh_host") and path:
            content = self._read_remote_file_tail(str(job["_ssh_host"]), str(path), lines)
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current[cache_key] = content
                    self._persist_job(current)

        return DeployJobLogsResponse(
            job_id=job_id,
            stream=stream,
            path=str(path) if path else None,
            content=content,
            updated_at=_now_iso(),
        )


def _controller_signature(path: Path) -> Tuple[str, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns


def _get_controller() -> DeployController:
    global _CONTROLLER, _CONTROLLER_SIG

    path = resolve_deploy_config_path()
    signature = _controller_signature(path)
    with _CONTROLLER_LOCK:
        if _CONTROLLER is None or _CONTROLLER_SIG != signature:
            if _CONTROLLER is not None:
                _CONTROLLER.close()
            _CONTROLLER = DeployController(load_deploy_config(str(path)))
            _CONTROLLER_SIG = signature
        return _CONTROLLER


def build_deploy_overview(values: Optional[Dict[str, Any]] = None) -> DeployOverviewResponse:
    controller = _get_controller()
    controller.refresh_jobs()
    return DeployOverviewResponse(
        refreshed_at=_now_iso(),
        targets=[controller.probe_target(target_id) for target_id in TARGET_ORDER],
        inputs=controller.input_specs(),
        commands=controller.command_specs(values),
        jobs=controller.list_jobs(),
    )


def run_deploy_command(payload: DeployRunRequest) -> DeployRunResponse:
    controller = _get_controller()
    return controller.submit_command(payload)


def save_deploy_input_values(values: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    controller = _get_controller()
    return controller.save_input_values(values)


def list_deploy_jobs() -> DeployJobsResponse:
    controller = _get_controller()
    controller.refresh_jobs()
    return DeployJobsResponse(
        refreshed_at=_now_iso(),
        jobs=controller.list_jobs(),
    )


def get_deploy_job_logs(job_id: str, stream: str, lines: int) -> DeployJobLogsResponse:
    controller = _get_controller()
    return controller.get_logs(job_id, stream, lines)


def stop_deploy_job(job_id: str) -> DeployJob:
    controller = _get_controller()
    return controller.stop_job(job_id)
