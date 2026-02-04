"""
VLA-Lab Command Line Interface

Commands:
- vlalab view: Launch Streamlit visualization app
- vlalab convert: Convert old log formats to VLA-Lab run format
- vlalab init-run: Initialize a new run directory
- vlalab info: Show information about a run
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


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
def view(port: int, run_dir: str):
    """Launch the Streamlit visualization app."""
    import subprocess
    import sys
    
    app_path = Path(__file__).parent / "apps" / "streamlit" / "app.py"
    
    if not app_path.exists():
        # Fallback to package data location
        import vlalab
        package_dir = Path(vlalab.__file__).parent
        app_path = package_dir / "apps" / "streamlit" / "app.py"
    
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    
    if run_dir:
        cmd.extend(["--", "--run-dir", run_dir])
    
    console.print(f"[green]Starting VLA-Lab viewer on port {port}...[/green]")
    subprocess.run(cmd)


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
