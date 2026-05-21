import click

from vlalab import cli


def test_resolve_launch_port_returns_requested_port_when_free(monkeypatch):
    monkeypatch.setattr(cli, "_is_port_in_use", lambda port: False)

    assert cli._resolve_launch_port(3000, "Frontend", "--port") == 3000


def test_resolve_launch_port_reuses_port_after_stopping_vlalab(monkeypatch):
    state = {"calls": 0}

    def fake_is_port_in_use(port):
        state["calls"] += 1
        return state["calls"] == 1

    monkeypatch.setattr(cli, "_is_port_in_use", fake_is_port_in_use)
    monkeypatch.setattr(cli, "_terminate_vlalab_process_on_port", lambda port, force=False: True)
    monkeypatch.setattr(cli, "_wait_for_port_release", lambda port, attempts=10, delay_s=0.5: True)

    assert cli._resolve_launch_port(8000, "API", "--api-port") == 8000


def test_resolve_launch_port_falls_back_when_taken_by_other_process(monkeypatch):
    monkeypatch.setattr(cli, "_is_port_in_use", lambda port: port in {3000, 3001})
    monkeypatch.setattr(cli, "_terminate_vlalab_process_on_port", lambda port, force=False: False)

    assert cli._resolve_launch_port(3000, "Frontend", "--port") == 3002


def test_resolve_launch_port_skips_reserved_port(monkeypatch):
    monkeypatch.setattr(cli, "_is_port_in_use", lambda port: False)

    assert cli._resolve_launch_port(3000, "Frontend", "--port", reserved_ports={3000}) == 3001


def test_resolve_launch_port_aborts_when_no_ports_available(monkeypatch):
    monkeypatch.setattr(cli, "_is_port_in_use", lambda port: True)
    monkeypatch.setattr(cli, "_terminate_vlalab_process_on_port", lambda port, force=False: False)

    try:
        cli._resolve_launch_port(9000, "API", "--api-port")
    except click.Abort:
        return

    raise AssertionError("expected click.Abort when no free ports are available")



def _make_cli_run(tmp_path):
    import json

    run_dir = tmp_path / "project" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_name": "run",
                "start_time": "2026-03-06T19:53:31",
                "model_name": "groot",
                "task_name": "Stack bowls",
                "robot_name": "franka",
                "total_steps": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "steps.jsonl").write_text(
        json.dumps(
            {
                "step_idx": 0,
                "obs": {"state": [1.0, 2.0, 3.0], "images": []},
                "action": {"values": [[0.1, 0.2, 0.3]]},
                "timing": {"inference_latency_ms": 12.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_index_build_command_creates_sidecar(tmp_path):
    from click.testing import CliRunner

    run_dir = _make_cli_run(tmp_path)
    result = CliRunner().invoke(cli.main, ["index", "build", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "available" in result.output
    assert (run_dir / "artifacts" / "vlalab_index" / "manifest.json").exists()


def test_rerun_status_command_reports_recording_path(tmp_path):
    from click.testing import CliRunner

    run_dir = _make_cli_run(tmp_path)
    result = CliRunner().invoke(cli.main, ["rerun", "status", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "recording.rrd" in result.output
