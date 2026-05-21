import json
import os
from pathlib import Path
from urllib.error import HTTPError


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_step(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _make_run(tmp_path: Path) -> tuple[Path, Path]:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "project_a" / "run_a"
    images_dir = run_dir / "artifacts" / "images"
    images_dir.mkdir(parents=True)
    _write_json(
        run_dir / "meta.json",
        {
            "run_name": "run_a",
            "start_time": "2026-03-06T19:53:31",
            "model_name": "groot",
            "task_name": "Stack bowls",
            "task_prompt": "Stack the bowls",
            "robot_name": "franka",
            "cameras": [
                {"name": "front", "resolution": [640, 480]},
                {"name": "ego", "resolution": [320, 240]},
            ],
            "inference_freq": 10,
            "action_dim": 3,
            "action_horizon": 2,
            "total_steps": 4,
            "state_keys": ["x", "y", "gripper"],
        },
    )
    steps_path = run_dir / "steps.jsonl"
    for idx in range(4):
        _append_step(
            steps_path,
            {
                "step_idx": idx,
                "prompt": "Stack the bowls",
                "obs": {
                    "state": [float(idx), float(idx + 1), float(idx) / 10.0],
                    "pose": [idx, idx + 0.1, idx + 0.2, 0, 0, 0, 1],
                    "gripper": float(idx) / 10.0,
                    "images": [
                        {
                            "camera_name": "front",
                            "path": f"artifacts/images/front_{idx:03d}.jpg",
                            "shape": [480, 640, 3],
                            "encoding": "jpeg",
                        },
                        {
                            "camera_name": "ego",
                            "path": f"artifacts/images/ego_{idx:03d}.jpg",
                            "shape": [240, 320, 3],
                            "encoding": "jpeg",
                        },
                    ],
                },
                "action": {
                    "values": [
                        [idx + 0.0, idx + 0.1, idx + 0.2],
                        [idx + 1.0, idx + 1.1, idx + 1.2],
                    ],
                    "action_dim": 3,
                    "chunk_size": 2,
                },
                "timing": {
                    "client_send": 100.0 + idx * 0.1,
                    "server_recv": 100.01 + idx * 0.1,
                    "infer_start": 100.02 + idx * 0.1,
                    "infer_end": 100.04 + idx * 0.1,
                    "send_timestamp": 100.05 + idx * 0.1,
                    "transport_latency_ms": 10 + idx,
                    "inference_latency_ms": 20 + idx,
                    "total_latency_ms": 30 + idx,
                },
            },
        )
    return runs_dir, run_dir


def test_sidecar_index_build_status_and_stale_detection(tmp_path):
    from vlalab.index.sidecar import build_index, index_status

    _, run_dir = _make_run(tmp_path)

    missing = index_status(run_dir)
    assert missing["available"] is False

    built = build_index(run_dir)
    assert built["available"] is True
    assert built["stale"] is False
    assert built["manifest"]["step_count"] == 4
    assert built["manifest"]["timelines"]["step_idx"]["type"] == "sequence"
    assert "/camera/front/image" in built["manifest"]["entities"]
    assert "/policy/action_chunk" in built["manifest"]["entities"]
    assert (run_dir / "artifacts" / "vlalab_index" / "manifest.json").exists()
    assert (run_dir / "artifacts" / "vlalab_index" / "columns.npz").exists()
    assert (run_dir / "artifacts" / "vlalab_index" / "media.json").exists()

    current = index_status(run_dir)
    assert current["available"] is True
    assert current["stale"] is False

    with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    stale = index_status(run_dir)
    assert stale["available"] is True
    assert stale["stale"] is True


def test_sidecar_replay_window_and_series_queries(tmp_path):
    from vlalab.index.sidecar import build_index, load_replay_window, load_series

    _, run_dir = _make_run(tmp_path)
    build_index(run_dir)

    window = load_replay_window(run_dir, project="project_a", run_name="run_a", center=2, radius=1)
    assert window["total_steps"] == 4
    assert window["offset"] == 1
    assert [item["step_idx"] for item in window["step_details"]] == [1, 2, 3]
    assert window["step_details"][1]["images"][0]["url"].endswith(
        "/api/runs/project_a/run_a/artifacts/artifacts/images/front_002.jpg"
    )
    assert window["step_details"][1]["action_chunk"][0] == [2.0, 2.1, 2.2]

    series = load_series(
        run_dir,
        entities=["/robot/state", "/policy/action_first", "/latency/inference_ms"],
        start=1,
        end=4,
        stride=2,
    )
    assert series["timeline"]["step_idx"] == [1, 3]
    assert series["entities"]["/robot/state"]["values"] == [[1.0, 2.0, 0.1], [3.0, 4.0, 0.3]]
    assert series["entities"]["/policy/action_first"]["values"] == [[1.0, 1.1, 1.2], [3.0, 3.1, 3.2]]
    assert series["entities"]["/latency/inference_ms"]["values"] == [21.0, 23.0]


def test_sidecar_index_api_endpoints(tmp_path, monkeypatch):
    runs_dir, _ = _make_run(tmp_path)
    monkeypatch.setenv('VLALAB_DIR', str(runs_dir))
    monkeypatch.delenv('VLALAB_DEPLOY_CONFIG', raising=False)

    from vlalab.apps.webapi import main

    status = main.run_index_status('project_a', 'run_a')
    assert status['available'] is False

    build = main.run_index_build('project_a', 'run_a')
    assert build['available'] is True

    summary = main.run_replay_summary('project_a', 'run_a')
    assert summary['summary']['total_steps'] == 4

    window = main.run_replay_window('project_a', 'run_a', center=1, radius=1)
    assert [item['step_idx'] for item in window['step_details']] == [0, 1, 2]

    series = main.run_series(
        'project_a',
        'run_a',
        entities=['/robot/state', '/latency/total_ms'],
        stride=2,
    )
    assert series['timeline']['step_idx'] == [0, 2]
    assert series['entities']['/latency/total_ms']['values'] == [30.0, 32.0]

    rerun_status = main.run_rerun_status('project_a', 'run_a')
    assert set(rerun_status) >= {'available', 'stale', 'path'}

def test_serve_rerun_recording_starts_server(tmp_path, monkeypatch):
    from vlalab.index import sidecar

    _, run_dir = _make_run(tmp_path)
    rrd_path = sidecar.rerun_file_path(run_dir)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rrd_path.write_bytes(b"rrd")

    captured = {}

    class DummyProcess:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(sidecar, "build_rerun_recording", lambda run_path, force=False: sidecar.rerun_status(run_path))
    monkeypatch.setattr(sidecar, "_rerun_cli_command", lambda: (["rerun"], {}))
    monkeypatch.setattr(sidecar, "_stop_rerun_ports", lambda web_port, recording_port: None)
    wait_calls = []

    def fake_wait(host, port, timeout=8.0, path="/"):
        wait_calls.append((host, port, path))
        return True

    monkeypatch.setattr(sidecar, "_wait_for_rerun_http", fake_wait)
    monkeypatch.setattr(sidecar.subprocess, "Popen", fake_popen)

    status = sidecar.serve_rerun_recording(run_dir)

    assert status["server"]["ready"] is True
    assert status["server"]["web_ready"] is True
    assert status["server"]["recording_ready"] is True
    assert status["server"]["pid"] == 12345
    assert status["viewer_url"] == "http://127.0.0.1:19090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A19876%2Fproxy"
    assert captured["cmd"][:2] == ["rerun", str(rrd_path)]
    assert "--web-viewer" in captured["cmd"]
    assert ("127.0.0.1", 9090, "/") in wait_calls
    assert ("127.0.0.1", 9876, "/proxy") in wait_calls
    assert (rrd_path.parent / "server.json").exists()


def test_serve_rerun_recording_reuses_ready_server(tmp_path, monkeypatch):
    from vlalab.index import sidecar

    _, run_dir = _make_run(tmp_path)
    rrd_path = sidecar.rerun_file_path(run_dir)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rrd_path.write_bytes(b"rrd")
    state = {
        "running": True,
        "ready": True,
        "web_ready": True,
        "recording_ready": True,
        "pid": os.getpid(),
        "rrd_path": str(rrd_path),
        "rrd_mtime_ns": rrd_path.stat().st_mtime_ns,
        "log_path": str(rrd_path.parent / "server.log"),
        "started_at": "2026-05-21T00:00:00Z",
        "web_viewer_url": "http://127.0.0.1:19090",
        "recording_url": "rerun+http://127.0.0.1:19876/proxy",
        "viewer_url": "http://127.0.0.1:19090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A19876%2Fproxy",
        "web_port": 9090,
        "recording_port": 9876,
    }
    (rrd_path.parent / "server.json").write_text(json.dumps(state), encoding="utf-8")

    calls = {"stop": 0, "popen": 0}

    def fake_popen(*args, **kwargs):
        calls["popen"] += 1
        raise AssertionError("ready rerun server should be reused")

    monkeypatch.setattr(sidecar, "build_rerun_recording", lambda run_path, force=False: sidecar.rerun_status(run_path))
    monkeypatch.setattr(sidecar, "_rerun_cli_command", lambda: (["rerun"], {}))
    monkeypatch.setattr(sidecar, "_stop_rerun_ports", lambda web_port, recording_port: calls.__setitem__("stop", calls["stop"] + 1))
    monkeypatch.setattr(sidecar.subprocess, "Popen", fake_popen)

    status = sidecar.serve_rerun_recording(run_dir)

    assert status["server"]["reused"] is True
    assert status["server"]["pid"] == os.getpid()
    assert status["server"]["ready"] is True
    assert calls == {"stop": 0, "popen": 0}


def test_rerun_http_readiness_accepts_client_error_from_grpc_proxy(monkeypatch):
    from vlalab.index import sidecar

    def fake_urlopen(url, timeout=0.5):
        raise HTTPError(url, 400, "Bad Request", hdrs=None, fp=None)

    monkeypatch.setattr(sidecar, "urlopen", fake_urlopen)

    assert sidecar._wait_for_rerun_http("127.0.0.1", 9876, timeout=0.01, path="/proxy") is True


def test_rerun_open_endpoint_returns_viewer_url(tmp_path, monkeypatch):
    runs_dir, _ = _make_run(tmp_path)
    monkeypatch.setenv("VLALAB_DIR", str(runs_dir))
    monkeypatch.delenv("VLALAB_DEPLOY_CONFIG", raising=False)

    from vlalab.apps.webapi import main

    monkeypatch.setattr(
        main,
        "serve_rerun_recording",
        lambda run_path, force=False: {
            "available": True,
            "stale": False,
            "path": str(run_path / "artifacts" / "rerun" / "recording.rrd"),
            "viewer_url": "http://127.0.0.1:19090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A19876%2Fproxy",
        },
    )

    opened = main.run_rerun_open("project_a", "run_a")

    assert opened["available"] is True
    assert opened["url"] == "/api/runs/project_a/run_a/rerun/file"
    assert opened["viewer_url"].startswith("http://127.0.0.1:19090/")
