"""
Typer CLI for the prompt optimization pipeline.
Commands: optimize, evaluate, report, resume
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="prompt-optimizer",
    help="Automated Prompt Optimization for Structured Extraction using Ollama.",
    add_completion=False,
)
console = Console()


@app.command()
def optimize(
    config_dir: str = typer.Option("./configs", help="Path to config directory"),
    dataset: str = typer.Option(None, help="Override dataset key from datasets.yaml"),
    model: str = typer.Option(None, help="Override model key from models.yaml"),
    optimizer: str = typer.Option(None, help="Override optimizer strategy"),
    budget: int = typer.Option(None, help="Override max LLM call budget"),
    experiment_name: str = typer.Option(None, help="Experiment name for tracking"),
):
    """Run the full prompt optimization loop."""
    from app.main import run_optimize
    run_optimize(
        config_dir=config_dir,
        dataset_override=dataset,
        model_override=model,
        optimizer_override=optimizer,
        budget_override=budget,
        experiment_name=experiment_name,
    )


@app.command()
def evaluate(
    config_dir: str = typer.Option("./configs", help="Path to config directory"),
    prompt_file: str = typer.Option(None, help="Path to a .txt file containing the prompt to evaluate"),
    split: str = typer.Option("test", help="Which split to evaluate on: val or test"),
):
    """Evaluate a specific prompt on the test set."""
    from app.main import run_evaluate
    run_evaluate(config_dir=config_dir, prompt_file=prompt_file, split=split)


@app.command()
def report(
    config_dir: str = typer.Option("./configs", help="Path to config directory"),
    experiment_id: int = typer.Option(None, help="Experiment ID to generate report for"),
):
    """Generate a report for a completed experiment."""
    from app.main import run_report
    run_report(config_dir=config_dir, experiment_id=experiment_id)


@app.command()
def resume(
    config_dir: str = typer.Option("./configs", help="Path to config directory"),
    experiment_name: str = typer.Option(None, help="Name of the experiment to resume"),
):
    """Resume an interrupted optimization run."""
    from app.main import run_resume
    run_resume(config_dir=config_dir, experiment_name=experiment_name)


@app.command()
def list_experiments(
    config_dir: str = typer.Option("./configs", help="Path to config directory"),
):
    """List all tracked experiments."""
    from app.utils.config import load_config
    from app.persistence.database import DatabaseManager

    cfg = load_config(config_dir)
    db = DatabaseManager(url=cfg.database.url)
    experiments = db.list_experiments()

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Dataset")
    table.add_column("Model")
    table.add_column("Optimizer")
    table.add_column("Status", style="green")
    table.add_column("Started")

    for exp in experiments:
        table.add_row(
            str(exp.id),
            exp.name,
            exp.dataset,
            exp.model,
            exp.optimizer,
            exp.status,
            str(exp.started_at)[:19],
        )
    console.print(table)
