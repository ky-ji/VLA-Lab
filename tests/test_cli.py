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
