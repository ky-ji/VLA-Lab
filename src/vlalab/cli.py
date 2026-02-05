"""
VLA-Lab Command Line Interface

Commands:
- vlalab view: Launch Streamlit visualization app
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

console = Console()

# PID file location
def _get_pid_file(port: int) -> Path:
    """Get PID file path for a given port."""
    pid_dir = Path.home() / ".vlalab"
    pid_dir.mkdir(exist_ok=True)
    return pid_dir / f"view_{port}.pid"


def _kill_process_on_port(port: int, force: bool = False) -> bool:
    """
    Kill any process running on the specified port.
    Returns True if a process was killed.
    """
    import subprocess
    
    # Method 1: Check PID file first
    pid_file = _get_pid_file(port)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is still running
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            if force:
                os.kill(pid, signal.SIGKILL)
                console.print(f"[yellow]Killed previous VLA-Lab process (PID: {pid})[/yellow]")
            else:
                os.kill(pid, signal.SIGTERM)
                console.print(f"[yellow]Terminated previous VLA-Lab process (PID: {pid})[/yellow]")
            pid_file.unlink(missing_ok=True)
            return True
        except (ProcessLookupError, ValueError):
            # Process doesn't exist, clean up stale PID file
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
    pid_file = _get_pid_file(port)
    pid_file.unlink(missing_ok=True)
    
    if proc and proc.poll() is None:
        # Process still running, terminate it
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


@click.group()
@click.version_option()
def main():
    """VLA-Lab: Track and visualize VLA real-world deployment."""
    pass


@main.command()
@click.option(
    "--port", "-p", default=8501, type=int, help="Port for Streamlit server"
)
@click.option(
    "--run-dir", "-r", default=None, type=click.Path(exists=True),
    help="Default run directory to load"
)
@click.option(
    "--kill", "-k", "kill_existing", is_flag=True, default=False,
    help="Kill existing process on the port before starting"
)
def view(port: int, run_dir: str, kill_existing: bool):
    """Launch the Streamlit visualization app."""
    import subprocess
    import sys
    import time
    
    # Check if port is in use
    if _is_port_in_use(port):
        if kill_existing:
            _kill_process_on_port(port, force=True)
            time.sleep(1)  # Wait for port to be released
        else:
            console.print(f"[red]Port {port} is already in use![/red]")
            console.print(f"[yellow]Options:[/yellow]")
            console.print(f"  1. Run with --kill flag: vlalab view --kill")
            console.print(f"  2. Use different port: vlalab view --port {port + 1}")
            console.print(f"  3. Manually kill: vlalab kill --port {port}")
            raise click.Abort()
    
    app_path = Path(__file__).parent / "apps" / "streamlit" / "app.py"
    
    if not app_path.exists():
        # Fallback to package data location
        import vlalab
        package_dir = Path(vlalab.__file__).parent
        app_path = package_dir / "apps" / "streamlit" / "app.py"
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.address", "0.0.0.0"]
    
    if run_dir:
        cmd.extend(["--", "--run-dir", run_dir])
    
    console.print(f"[green]Starting VLA-Lab viewer on port {port}...[/green]")
    
    # Start process in a new process group so we can kill all children
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,  # Creates new process group
    )
    
    # Save PID to file
    pid_file = _get_pid_file(port)
    pid_file.write_text(str(proc.pid))
    
    # Register cleanup on exit
    atexit.register(lambda: _cleanup_on_exit(port, proc))
    
    # Handle signals
    def signal_handler(signum, frame):
        console.print(f"\n[yellow]Shutting down VLA-Lab viewer...[/yellow]")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        pid_file.unlink(missing_ok=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for process
    try:
        proc.wait()
    finally:
        pid_file.unlink(missing_ok=True)


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
        # Still try to clean up PID file
        pid_file = _get_pid_file(port)
        if pid_file.exists():
            pid_file.unlink()
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
