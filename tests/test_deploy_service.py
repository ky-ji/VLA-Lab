import copy
import json
import subprocess

import pytest

from vlalab.apps.webapi import deploy_service as ds
from vlalab.apps.webapi.models import DeployRunRequest, DeployTargetConnection


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return None

    def shutdown(self, wait=False, cancel_futures=False):
        return None


def _base_config():
    return {
        "state_dir": ".deploy_state",
        "targets": {
            "server": {
                "label": "Model Server",
                "ssh_host": "groot-gpu",
                "workdir": "/srv/model",
                "shell": "bash -lc",
            },
            "client": {
                "label": "Robot Client",
                "ssh_host": "groot-robot",
                "workdir": "/srv/client",
                "shell": "bash -lc",
            },
        },
        "inputs": {
            "model_server_config_path": {
                "label": "Model Server Config Path",
                "type": "path",
                "required": True,
                "target": "server",
            },
            "inference_client_config_path": {
                "label": "Inference Client Config Path",
                "type": "path",
                "required": True,
                "target": "client",
            },
        },
        "commands": {
            "start_model_server": {
                "label": "Start Model Server",
                "target": "server",
                "background": True,
                "template": "python server.py --config {model_server_config_path}",
                "stdout_log": "/tmp/vlalab/start_model_server.stdout.log",
                "stderr_log": "/tmp/vlalab/start_model_server.stderr.log",
            },
            "start_robot_server": {
                "label": "Start Robot Server",
                "target": "client",
                "background": True,
                "template": "python control/start_server.py",
                "stdout_log": "/tmp/vlalab/start_robot_server.stdout.log",
                "stderr_log": "/tmp/vlalab/start_robot_server.stderr.log",
            },
            "set_joint_preset": {
                "label": "Set Joint Preset",
                "target": "client",
                "background": False,
                "template": "python control/set_joint_positions.py",
            },
            "start_inference_client": {
                "label": "Start Inference Client",
                "target": "client",
                "background": True,
                "template": "python inference_client.py --config {inference_client_config_path}",
                "stdout_log": "/tmp/vlalab/start_inference_client.stdout.log",
                "stderr_log": "/tmp/vlalab/start_inference_client.stderr.log",
            },
        },
    }


def _write_config(tmp_path, payload=None):
    config = copy.deepcopy(payload or _base_config())
    path = tmp_path / "dashboard.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _reset_controller():
    controller = ds._CONTROLLER
    if controller is not None:
        controller.close()
    ds._CONTROLLER = None
    ds._CONTROLLER_SIG = None


@pytest.fixture(autouse=True)
def reset_deploy_controller():
    _reset_controller()
    yield
    _reset_controller()


def test_load_deploy_config_accepts_valid_dashboard(tmp_path):
    path = _write_config(tmp_path)

    config = ds.load_deploy_config(str(path))

    assert list(config.targets.keys()) == ["server", "client"]
    assert list(config.inputs.keys()) == [
        "model_server_config_path",
        "inference_client_config_path",
    ]
    assert list(config.commands.keys()) == [
        "start_model_server",
        "start_robot_server",
        "set_joint_preset",
        "start_inference_client",
    ]
    assert config.commands["start_model_server"].required_inputs == ("model_server_config_path",)
    assert config.commands["set_joint_preset"].required_inputs == ()


def test_load_deploy_config_rejects_missing_required_placeholder(tmp_path):
    payload = _base_config()
    payload["commands"]["start_model_server"]["template"] = "python server.py"
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="must include placeholders"):
        ds.load_deploy_config(str(path))


def test_build_deploy_overview_returns_fixed_sections(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setenv("VLALAB_DEPLOY_CONFIG", str(path))

    def fake_run(command, capture_output, text, timeout, check):
        assert command[0] == "ssh"
        return subprocess.CompletedProcess(command, 0, "__vlalab_ok__\n", "")

    monkeypatch.setattr(ds.subprocess, "run", fake_run)

    overview = ds.build_deploy_overview(
        {
            "model_server_config_path": "/remote/server.yaml",
            "inference_client_config_path": "/remote/client.yaml",
        }
    )

    assert len(overview.targets) == 2
    assert all(target.connected for target in overview.targets)
    assert len(overview.inputs) == 2
    assert len(overview.commands) == 4
    assert overview.jobs == []
    command_map = {command.id: command for command in overview.commands}
    assert "/remote/server.yaml" in command_map["start_model_server"].resolved_preview
    assert "/remote/client.yaml" in command_map["start_inference_client"].resolved_preview
    assert command_map["set_joint_preset"].resolved_preview.endswith("python control/set_joint_positions.py")


def test_submit_command_requires_missing_inputs(tmp_path):
    controller = ds.DeployController(ds.load_deploy_config(str(_write_config(tmp_path))))
    try:
        with pytest.raises(ValueError, match="Missing required deploy input"):
            controller.submit_command(
                DeployRunRequest(
                    command_id="start_model_server",
                    values={},
                )
            )
    finally:
        controller.close()


def test_background_command_records_remote_pid(tmp_path, monkeypatch):
    controller = ds.DeployController(ds.load_deploy_config(str(_write_config(tmp_path))))
    monkeypatch.setattr(controller, "_executor", ImmediateExecutor())
    monkeypatch.setattr(
        controller,
        "probe_target",
        lambda target_id: DeployTargetConnection(
            id=target_id,
            label=target_id,
            connected=True,
            last_error=None,
        ),
    )
    monkeypatch.setattr(
        controller,
        "_run_ssh_script",
        lambda target, script, timeout=60.0: subprocess.CompletedProcess([], 0, "4321\n", ""),
    )

    try:
        response = controller.submit_command(
            DeployRunRequest(
                command_id="start_model_server",
                values={"model_server_config_path": "/remote/server.yaml"},
            )
        )
        assert response.job.state == "running"
        assert response.job.remote_pid == 4321
        job = next(item for item in controller.list_jobs() if item.command_id == "start_model_server")
        assert job.command_id == "start_model_server"
        assert job.state == "running"
        assert job.remote_pid == 4321
        assert job.stdout_log == "/tmp/vlalab/start_model_server.stdout.log"
        assert job.stderr_log == "/tmp/vlalab/start_model_server.stderr.log"
    finally:
        controller.close()


class _FakePopen:
    def __init__(self, stdout="preset done\n", stderr="", returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 2468

    def communicate(self, timeout=None):
        return self._stdout, self._stderr

    def poll(self):
        return self.returncode


def test_foreground_command_records_stdout_and_success(tmp_path, monkeypatch):
    controller = ds.DeployController(ds.load_deploy_config(str(_write_config(tmp_path))))
    monkeypatch.setattr(controller, "_executor", ImmediateExecutor())
    monkeypatch.setattr(
        controller,
        "probe_target",
        lambda target_id: DeployTargetConnection(
            id=target_id,
            label=target_id,
            connected=True,
            last_error=None,
        ),
    )
    monkeypatch.setattr(
        controller,
        "_spawn_ssh_script",
        lambda target, script: _FakePopen(stdout="preset done\n", stderr="", returncode=0),
    )

    try:
        response = controller.submit_command(
            DeployRunRequest(
                command_id="set_joint_preset",
                values={},
            )
        )
        assert response.job.state == "success"
        assert response.job.last_stdout == "preset done\n"
        job = next(item for item in controller.list_jobs() if item.command_id == "set_joint_preset")
        assert job.command_id == "set_joint_preset"
        assert job.state == "success"
        assert job.remote_pid is None
        assert job.last_stdout == "preset done\n"
        assert job.error is None
    finally:
        controller.close()


def test_saved_input_values_are_exposed_in_input_specs(tmp_path):
    controller = ds.DeployController(ds.load_deploy_config(str(_write_config(tmp_path))))
    try:
        controller.save_input_values(
            {
                "model_server_config_path": "/remote/server.saved.yaml",
                "inference_client_config_path": "/remote/client.saved.yaml",
            }
        )
        specs = controller.input_specs()
        spec_map = {item.id: item for item in specs}
        assert spec_map["model_server_config_path"].current_value == "/remote/server.saved.yaml"
        assert spec_map["inference_client_config_path"].current_value == "/remote/client.saved.yaml"
    finally:
        controller.close()


def test_stop_background_job_sends_remote_kill(tmp_path, monkeypatch):
    controller = ds.DeployController(ds.load_deploy_config(str(_write_config(tmp_path))))
    calls = []

    def fake_run_ssh_host(ssh_host, script, timeout=20.0):
        calls.append((ssh_host, script))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(controller, "_run_ssh_host", fake_run_ssh_host)

    job = {
        "id": "job-stop-bg",
        "command_id": "start_model_server",
        "target_id": "server",
        "state": "running",
        "remote_pid": 4321,
        "stdout_log": "/tmp/vlalab/start_model_server.stdout.log",
        "stderr_log": "/tmp/vlalab/start_model_server.stderr.log",
        "last_stdout": "",
        "last_stderr": "",
        "submitted_at": "2026-03-17T00:00:00",
        "started_at": "2026-03-17T00:00:01",
        "finished_at": None,
        "error": None,
        "_background": True,
        "_ssh_host": "groot-gpu",
        "_status_path": "/tmp/vlalab/job-stop-bg.status",
        "_values": {"model_server_config_path": "/remote/server.yaml"},
        "_stop_requested": False,
    }

    try:
        controller._jobs[job["id"]] = job
        stopped = controller.stop_job(job["id"])
        assert stopped.state == "stopping"
        assert calls
        assert calls[0][0] == "groot-gpu"
        assert "kill -TERM" in calls[0][1]
        assert "4321" in calls[0][1]
    finally:
        controller.close()
