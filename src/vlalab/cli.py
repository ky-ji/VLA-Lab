"""
VLA-Lab Command Line Interface

Commands:
- vlalab view: Launch the default visualization app
- vlalab serve: Launch the FastAPI backend and optional Next.js frontend
- vlalab convert: Convert old log formats to VLA-Lab run format
- vlalab init-run: Initialize a new run directory
- vlalab info: Show information about a run
- vlalab kill: Kill VLA-Lab processes on a port
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
import os
import signal
import atexit
import sys
from typing import Optional

console = Console()

# PID file location
def _get_pid_file(port: int, prefix: str = "view") -> Path:
    """Get PID file path for a given port."""
    pid_dir = Path.home() / ".vlalab"
    pid_dir.mkdir(exist_ok=True)
    return pid_dir / f"{prefix}_{port}.pid"


def _terminate_process_group(proc) -> None:
    """Terminate a process group if it is still alive."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def _cleanup_pid_file(port: int, prefix: str = "view") -> None:
    """Remove the PID file for a service if it exists."""
    _get_pid_file(port, prefix=prefix).unlink(missing_ok=True)


def _resolve_web_dir() -> Path:
    """Locate the bundled Next.js frontend."""
    candidates = [
        Path(__file__).resolve().parents[2] / "web",
        Path.cwd() / "web",
    ]
    for candidate in candidates:
        if (candidate / "package.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate the VLA-Lab web/ directory.")


def _kill_process_on_port(port: int, force: bool = False) -> bool:
    """
    Kill any process running on the specified port.
    Returns True if a process was killed.
    """
    import subprocess
    
    # Method 1: Check PID file first
    for prefix in ("view", "api", "web"):
        pid_file = _get_pid_file(port, prefix=prefix)
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            if force:
                os.kill(pid, signal.SIGKILL)
                console.print(f"[yellow]Killed previous VLA-Lab process (PID: {pid})[/yellow]")
            else:
                os.kill(pid, signal.SIGTERM)
                console.print(f"[yellow]Terminated previous VLA-Lab process (PID: {pid})[/yellow]")
            pid_file.unlink(missing_ok=True)
            return True
        except (ProcessLookupError, ValueError):
            pid_file.unlink(missing_ok=True)
        except PermissionError:
            console.print(f"[red]Permission denied to kill process[/red]")
    
    # Method 2: Use fuser/lsof to find process on port
    try:
        # Try fuser first (more reliable on Linux)
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split()
            for pid in pids:
                try:
                    pid = int(pid.strip())
                    sig = signal.SIGKILL if force else signal.SIGTERM
                    os.kill(pid, sig)
                    console.print(f"[yellow]Killed process on port {port} (PID: {pid})[/yellow]")
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
            return True
    except FileNotFoundError:
        # fuser not available, try lsof
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split('\n'):
                    try:
                        pid = int(pid.strip())
                        sig = signal.SIGKILL if force else signal.SIGTERM
                        os.kill(pid, sig)
                        console.print(f"[yellow]Killed process on port {port} (PID: {pid})[/yellow]")
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass
                return True
        except FileNotFoundError:
            pass
    
    return False


def _is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def _cleanup_on_exit(port: int, proc):
    """Cleanup function called on exit."""
    _cleanup_pid_file(port)
    _terminate_process_group(proc)


def _ensure_port_available(port: int, action_hint: str) -> None:
    """Ensure a port is free before launching a new service."""
    import time

    if not _is_port_in_use(port):
        return

    console.print(f"[yellow]Port {port} is in use, killing previous process...[/yellow]")
    _kill_process_on_port(port, force=True)
    for _ in range(10):
        time.sleep(0.5)
        if not _is_port_in_use(port):
            return

    console.print(f"[red]Port {port} still in use after cleanup.[/red]")
    console.print(f"[yellow]Try: {action_hint}[/yellow]")
    raise click.Abort()


def _missing_python_modules(modules) -> list[str]:
    """Return the list of modules that cannot be imported in the current env."""
    missing = []
    for module_name in modules:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


def _ensure_web_python_dependencies(install: bool = False) -> None:
    """Ensure FastAPI web-mode dependencies are available before launch."""
    required = ["fastapi", "uvicorn"]
    missing = _missing_python_modules(required)
    if not missing:
        return

    if install:
        console.print(
            f"[blue]Installing missing Python web dependencies: {', '.join(missing)}[/blue]"
        )
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            check=False,
        )
        if result.returncode == 0 and not _missing_python_modules(required):
            return

        console.print("[red]Automatic Python dependency installation failed.[/red]")
        raise click.Abort()

    console.print(
        "[red]Missing Python web dependencies:[/red] "
        + ", ".join(missing)
    )
    console.print(
        "[yellow]Run `vlalab view --install` or "
        f"`{sys.executable} -m pip install {' '.join(missing)}` first.[/yellow]"
    )
    raise click.Abort()


def _launch_streamlit_view(port: int, run_dir: str):
    """Launch the legacy Streamlit visualization app."""
    import subprocess
    import sys

    _ensure_port_available(port, f"vlalab view --legacy --port {port + 1}")

    app_path = Path(__file__).parent / "apps" / "streamlit" / "app.py"

    if not app_path.exists():
        import vlalab

        package_dir = Path(vlalab.__file__).parent
        app_path = package_dir / "apps" / "streamlit" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
    ]

    if run_dir:
        cmd.extend(["--", "--run-dir", run_dir])

    console.print(f"[green]Starting legacy VLA-Lab Streamlit viewer on port {port}...[/green]")

    proc = subprocess.Popen(cmd, start_new_session=True)
    pid_file = _get_pid_file(port, prefix="view")
    pid_file.write_text(str(proc.pid))
    atexit.register(lambda: _cleanup_on_exit(port, proc))

    def signal_handler(signum, frame):
        console.print("\n[yellow]Shutting down VLA-Lab viewer...[/yellow]")
        _terminate_process_group(proc)
        pid_file.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        proc.wait()
    finally:
        pid_file.unlink(missing_ok=True)


def _launch_web_services(
    host: str,
    api_port: int,
    web_port: int,
    frontend: bool,
    install: bool,
    run_dir: str,
):
    """Launch the FastAPI backend and optional Next.js frontend."""
    import subprocess
    import time

    _ensure_web_python_dependencies(install=install)

    if frontend and api_port == web_port:
        console.print("[red]api-port and web-port must be different when frontend is enabled.[/red]")
        raise click.Abort()

    _ensure_port_available(api_port, f"vlalab view --api-port {api_port + 1}")
    if frontend:
        _ensure_port_available(web_port, f"vlalab view --port {web_port + 1}")

    env = os.environ.copy()
    if run_dir:
        env["VLALAB_DIR"] = str(Path(run_dir).resolve())

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "vlalab.apps.webapi.main:app",
        "--host",
        host,
        "--port",
        str(api_port),
    ]

    console.print(f"[green]Starting VLA-Lab API on port {api_port}...[/green]")
    api_proc = subprocess.Popen(api_cmd, start_new_session=True, env=env)
    api_pid = _get_pid_file(api_port, prefix="api")
    api_pid.write_text(str(api_proc.pid))

    processes = [("API", api_proc, api_pid)]
    atexit.register(lambda: _cleanup_on_exit(api_port, api_proc))

    if frontend:
        try:
            web_dir = _resolve_web_dir()
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            _cleanup_on_exit(api_port, api_proc)
            raise click.Abort()

        frontend_env = env.copy()
        api_base_url = f"http://127.0.0.1:{api_port}"
        frontend_env["VLALAB_API_BASE_URL"] = api_base_url
        frontend_env["NEXT_PUBLIC_VLALAB_API_BASE_URL"] = api_base_url

        if install or not (web_dir / "node_modules").exists():
            console.print("[blue]Installing frontend dependencies in web/...[/blue]")
            result = subprocess.run(
                ["npm", "install", "--no-package-lock"],
                cwd=web_dir,
                env=frontend_env,
                check=False,
            )
            if result.returncode != 0:
                console.print("[red]npm install failed, stopping API process.[/red]")
                _cleanup_on_exit(api_port, api_proc)
                raise click.Abort()

        console.print(f"[green]Starting VLA-Lab frontend on port {web_port}...[/green]")
        web_cmd = ["npm", "run", "dev", "--", "-H", host, "-p", str(web_port)]
        web_proc = subprocess.Popen(
            web_cmd,
            cwd=web_dir,
            start_new_session=True,
            env=frontend_env,
        )
        web_pid = _get_pid_file(web_port, prefix="web")
        web_pid.write_text(str(web_proc.pid))
        processes.append(("Frontend", web_proc, web_pid))
        atexit.register(lambda: _cleanup_on_exit(web_port, web_proc))

    console.print(f"[green]API[/green]: http://127.0.0.1:{api_port}/api/health")
    if frontend:
        console.print(f"[green]Web[/green]: http://127.0.0.1:{web_port}")

    def cleanup_all():
        for name, proc, pid_file in reversed(processes):
            pid_file.unlink(missing_ok=True)
            _terminate_process_group(proc)

    def signal_handler(signum, frame):
        console.print("\n[yellow]Shutting down VLA-Lab web services...[/yellow]")
        cleanup_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            for name, proc, _ in processes:
                code = proc.poll()
                if code is not None:
                    console.print(f"[yellow]{name} exited with code {code}[/yellow]")
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        cleanup_all()


@click.group()
@click.version_option()
def main():
    """VLA-Lab: Track and visualize VLA real-world deployment."""
    pass


@main.command()
@click.option(
    "--port", "-p", default=None, type=int,
    help="Frontend port for the web UI, or the Streamlit port when --legacy is enabled"
)
@click.option("--api-port", default=8000, type=int, help="Port for the FastAPI backend")
@click.option("--host", default="0.0.0.0", help="Host for the web services")
@click.option(
    "--install/--no-install",
    default=False,
    help="Run npm install in web/ before starting the frontend",
)
@click.option("--legacy", is_flag=True, default=False, help="Launch the old Streamlit viewer instead of the web UI")
@click.option(
    "--run-dir", "-r", default=None, type=click.Path(exists=True),
    help="Default run directory to load"
)
def view(port: Optional[int], api_port: int, host: str, install: bool, legacy: bool, run_dir: str):
    """Launch the default VLA-Lab viewer."""
    resolved_port = port if port is not None else (8501 if legacy else 3000)

    if legacy:
        _launch_streamlit_view(port=resolved_port, run_dir=run_dir)
        return

    _launch_web_services(
        host=host,
        api_port=api_port,
        web_port=resolved_port,
        frontend=True,
        install=install,
        run_dir=run_dir,
    )


@main.command()
@click.option("--host", default="0.0.0.0", help="Host for the web services")
@click.option("--api-port", default=8000, type=int, help="Port for the FastAPI backend")
@click.option("--web-port", default=3000, type=int, help="Port for the Next.js frontend")
@click.option(
    "--frontend/--no-frontend",
    default=True,
    help="Launch the Next.js frontend alongside the API",
)
@click.option(
    "--install/--no-install",
    default=False,
    help="Run npm install in web/ before starting the frontend",
)
@click.option(
    "--run-dir", "-r", default=None, type=click.Path(exists=True),
    help="Override the VLALAB_DIR used by the API"
)
def serve(host: str, api_port: int, web_port: int, frontend: bool, install: bool, run_dir: str):
    """Launch the FastAPI backend and optional Next.js frontend."""
    _launch_web_services(
        host=host,
        api_port=api_port,
        web_port=web_port,
        frontend=frontend,
        install=install,
        run_dir=run_dir,
    )


@main.command()
@click.option(
    "--port", "-p", default=8501, type=int, help="Port to kill processes on"
)
@click.option(
    "--force", "-f", is_flag=True, default=False,
    help="Force kill (SIGKILL instead of SIGTERM)"
)
def kill(port: int, force: bool):
    """Kill VLA-Lab processes running on a port."""
    if _is_port_in_use(port):
        if _kill_process_on_port(port, force=force):
            console.print(f"[green]Successfully cleared port {port}[/green]")
        else:
            console.print(f"[red]Failed to kill process on port {port}[/red]")
            console.print(f"[yellow]Try manually: fuser -k {port}/tcp[/yellow]")
    else:
        cleaned = False
        for prefix in ("view", "api", "web"):
            pid_file = _get_pid_file(port, prefix=prefix)
            if pid_file.exists():
                pid_file.unlink()
                cleaned = True
        if cleaned:
            console.print(f"[green]Cleaned up stale PID file[/green]")
        else:
            console.print(f"[green]Port {port} is not in use[/green]")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default=None, type=click.Path(),
    help="Output run directory (default: auto-generated)"
)
@click.option(
    "--format", "-f", "input_format", 
    type=click.Choice(["dp", "groot", "auto"]), 
    default="auto",
    help="Input log format"
)
def convert(input_path: str, output: str, input_format: str):
    """Convert old log formats to VLA-Lab run format."""
    from vlalab.adapters.converter import convert_legacy_log
    
    input_path = Path(input_path)
    
    if output is None:
        output = input_path.parent / f"run_{input_path.stem}"
    
    output = Path(output)
    
    console.print(f"[blue]Converting {input_path} -> {output}[/blue]")
    
    try:
        stats = convert_legacy_log(input_path, output, input_format)
        console.print(f"[green]Converted {stats['steps']} steps, {stats['images']} images[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command("init-run")
@click.argument("run_dir", type=click.Path())
@click.option("--model", "-m", default="unknown", help="Model name/path")
@click.option("--task", "-t", default="unknown", help="Task name")
@click.option("--robot", "-r", default="unknown", help="Robot name")
def init_run(run_dir: str, model: str, task: str, robot: str):
    """Initialize a new run directory with metadata."""
    from vlalab.logging import RunLogger
    
    run_dir = Path(run_dir)
    
    logger = RunLogger(
        run_dir=run_dir,
        model_name=model,
        task_name=task,
        robot_name=robot,
    )
    
    console.print(f"[green]Initialized run directory: {run_dir}[/green]")
    console.print(f"  Model: {model}")
    console.print(f"  Task: {task}")
    console.print(f"  Robot: {robot}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
def info(run_dir: str):
    """Show information about a run."""
    from vlalab.logging.run_loader import load_run_info
    
    run_dir = Path(run_dir)
    
    try:
        info = load_run_info(run_dir)
        
        table = Table(title=f"Run: {run_dir.name}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in info.items():
            table.add_row(key, str(value))
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error loading run: {e}[/red]")


if __name__ == "__main__":
    main()
