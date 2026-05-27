"""
Main pipeline entry point.
Wires together all components and runs the optimization pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from app.cli import app as cli_app
from app.utils.logging_utils import get_logger, setup_root_logging

console = Console()
logger = get_logger(__name__)


def run_optimize(
    config_dir: str = "./configs",
    dataset_override: Optional[str] = None,
    model_override: Optional[str] = None,
    optimizer_override: Optional[str] = None,
    budget_override: Optional[int] = None,
    experiment_name: Optional[str] = None,
    resume_state=None,
) -> None:
    """Full optimization pipeline."""
    from app.utils.config import (
        ensure_dirs, get_dataset_config, get_model_config,
        get_optimizer_config, load_config,
    )
    from app.persistence.cache import ResponseCache
    from app.persistence.database import DatabaseManager
    from app.persistence.checkpoints import CheckpointManager
    from app.datasets.loader import DatasetLoader
    from app.datasets.splitter import DatasetSplitter
    from app.llm.ollama_client import OllamaClient
    from app.llm.prompts import build_seed_prompt
    from app.extraction.extractor import Extractor
    from app.scoring.evaluator import Evaluator
    from app.optimizer.acceptance import build_acceptance_policy
    from app.optimizer.beam_search import BeamSearchOptimizer
    from app.optimizer.population import PopulationOptimizer
    from app.reporting.report_generator import ReportGenerator
    from app.utils.hashing import hash_string

    cfg = load_config(config_dir)
    setup_root_logging(cfg.logging.level, cfg.experiment.logs_dir)
    ensure_dirs(cfg)

    # Apply overrides
    dataset_key = dataset_override or cfg.pipeline.dataset
    model_key = model_override or cfg.pipeline.model
    optimizer_key = optimizer_override or cfg.pipeline.optimizer
    exp_name = experiment_name or cfg.experiment.name

    if budget_override:
        cfg.pipeline.max_budget = budget_override

    console.print(Panel.fit(
        f"[bold cyan]Prompt Optimizer[/bold cyan]\n"
        f"Dataset: {dataset_key} | Model: {model_key} | Strategy: {optimizer_key}",
        title="Starting Optimization",
    ))

    # Check Ollama
    client = OllamaClient(
        base_url=cfg.ollama.base_url,
        timeout=cfg.ollama.timeout,
        max_retries=cfg.ollama.max_retries,
    )
    if not client.is_available():
        console.print("[bold red]ERROR: Ollama is not running. Start it with: ollama serve[/bold red]")
        sys.exit(1)

    # Load configs
    dataset_cfg = get_dataset_config(cfg, dataset_key)
    model_cfg = get_model_config(cfg, model_key)
    opt_cfg = get_optimizer_config(cfg, optimizer_key)
    opt_cfg["batch_size"] = cfg.pipeline.batch_size

    # Mutator model (higher temperature)
    mutator_key = f"{model_key}_optimizer" if f"{model_key}_optimizer" in cfg.models else model_key
    mutator_cfg = get_model_config(cfg, mutator_key)

    # Load data
    loader = DatasetLoader(seed=cfg.experiment.random_seed)
    splitter = DatasetSplitter(seed=cfg.experiment.random_seed)
    samples = loader.load(dataset_cfg)
    fields = loader.get_field_specs(dataset_cfg)
    train_split, val_split, test_split = splitter.split(
        samples, fields,
        validation_ratio=cfg.pipeline.validation_split,
        test_ratio=cfg.pipeline.test_split,
    )

    # Persistence
    cache = ResponseCache(cfg.cache.dir) if cfg.cache.enabled else None
    db = DatabaseManager(url=cfg.database.url)
    cp_manager = CheckpointManager(db)

    # Seed prompt
    seed_prompt = build_seed_prompt(
        [{"name": f.name, "type": f.type, "required": f.required} for f in fields]
    )

    # Create experiment record
    exp_id = db.create_experiment(
        name=exp_name,
        dataset=dataset_key,
        model=model_key,
        optimizer=optimizer_key,
        seed_prompt_hash=hash_string(seed_prompt)[:16],
        config={"pipeline": cfg.pipeline.model_dump(), "optimizer": opt_cfg},
    )
    db.upsert_prompt(hash_string(seed_prompt), seed_prompt, mutation_strategy="seed")

    # Evaluate seed prompt
    extractor = Extractor(client, model_cfg, cache)
    evaluator = Evaluator(fields)

    console.print("[cyan]Evaluating seed prompt on validation set...[/cyan]")
    seed_results = extractor.extract_batch(seed_prompt, val_split.samples, fields)
    seed_scored = evaluator.evaluate_batch(seed_results, {s.id: s for s in val_split.samples})
    seed_scores = evaluator.aggregate_scores(seed_scored)
    console.print(f"[green]Seed F1: {seed_scores['mean_f1']:.4f}[/green]")

    # Build optimizer
    acceptance = build_acceptance_policy(
        opt_cfg.get("acceptance", "strict"), opt_cfg, seed=cfg.experiment.random_seed
    )
    strategy = opt_cfg.get("strategy", "beam")

    if strategy == "population":
        optimizer = PopulationOptimizer(
            client=client, extractor=extractor, evaluator=evaluator,
            mutator_model_config=mutator_cfg, optimizer_config=opt_cfg,
            mutation_params=cfg.mutation_params, acceptance_policy=acceptance,
            db=db, checkpoint_manager=cp_manager, fields=fields,
            experiment_id=exp_id, seed=cfg.experiment.random_seed,
        )
    else:
        optimizer = BeamSearchOptimizer(
            client=client, extractor=extractor, evaluator=evaluator,
            mutator_model_config=mutator_cfg, optimizer_config=opt_cfg,
            mutation_params=cfg.mutation_params, acceptance_policy=acceptance,
            db=db, checkpoint_manager=cp_manager, fields=fields,
            experiment_id=exp_id, seed=cfg.experiment.random_seed,
        )

    console.print(f"[cyan]Running {strategy} optimization...[/cyan]")
    best_prompt, trajectory = optimizer.optimize(
        seed_prompt=seed_prompt,
        val_split=val_split,
        train_split=train_split,
        resume_state=resume_state,
    )

    # Evaluate best prompt on test set
    console.print("[cyan]Evaluating best prompt on test set...[/cyan]")
    test_results = extractor.extract_batch(best_prompt, test_split.samples, fields)
    test_scored = evaluator.evaluate_batch(test_results, {s.id: s for s in test_split.samples})
    test_scores = evaluator.aggregate_scores(test_scored)

    best_val_results = extractor.extract_batch(best_prompt, val_split.samples, fields)
    best_val_scored = evaluator.evaluate_batch(best_val_results, {s.id: s for s in val_split.samples})
    best_scores = evaluator.aggregate_scores(best_val_scored)

    db.mark_experiment_complete(exp_id)

    # Report
    reporter = ReportGenerator(cfg.experiment.reports_dir)
    report_path = reporter.generate(
        experiment_name=exp_name,
        seed_prompt=seed_prompt,
        best_prompt=best_prompt,
        trajectory_entries=[vars(e) for e in trajectory.entries],
        seed_scores=seed_scores,
        best_scores=best_scores,
        test_scores=test_scores,
    )

    console.print(Panel.fit(
        f"[bold green]Optimization Complete![/bold green]\n"
        f"Seed F1:  {seed_scores['mean_f1']:.4f}\n"
        f"Best F1:  {best_scores['mean_f1']:.4f}  (val)\n"
        f"Test F1:  {test_scores['mean_f1']:.4f}\n"
        f"Report:   {report_path}",
        title="Results",
    ))

    # Save best prompt to file
    best_prompt_path = Path(cfg.experiment.output_dir) / f"{exp_name}_best_prompt.txt"
    best_prompt_path.write_text(best_prompt, encoding="utf-8")
    console.print(f"Best prompt saved to: {best_prompt_path}")


def run_evaluate(
    config_dir: str = "./configs",
    prompt_file: Optional[str] = None,
    split: str = "test",
) -> None:
    """Evaluate a prompt file on val or test split."""
    from app.utils.config import ensure_dirs, get_dataset_config, get_model_config, load_config
    from app.persistence.cache import ResponseCache
    from app.datasets.loader import DatasetLoader
    from app.datasets.splitter import DatasetSplitter
    from app.llm.ollama_client import OllamaClient
    from app.llm.prompts import build_seed_prompt
    from app.extraction.extractor import Extractor
    from app.scoring.evaluator import Evaluator

    cfg = load_config(config_dir)
    ensure_dirs(cfg)

    if prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8")
    else:
        fields_raw = get_dataset_config(cfg, cfg.pipeline.dataset).get("fields_to_extract", [])
        prompt = build_seed_prompt(fields_raw)

    client = OllamaClient(base_url=cfg.ollama.base_url)
    model_cfg = get_model_config(cfg, cfg.pipeline.model)
    cache = ResponseCache(cfg.cache.dir) if cfg.cache.enabled else None
    loader = DatasetLoader(seed=cfg.experiment.random_seed)
    splitter = DatasetSplitter(seed=cfg.experiment.random_seed)
    dataset_cfg = get_dataset_config(cfg, cfg.pipeline.dataset)
    samples = loader.load(dataset_cfg)
    fields = loader.get_field_specs(dataset_cfg)
    train_split, val_split, test_split = splitter.split(samples, fields)

    target_split = test_split if split == "test" else val_split
    extractor = Extractor(client, model_cfg, cache)
    evaluator = Evaluator(fields)
    results = extractor.extract_batch(prompt, target_split.samples, fields)
    scored = evaluator.evaluate_batch(results, {s.id: s for s in target_split.samples})
    agg = evaluator.aggregate_scores(scored)

    console.print(Panel.fit(
        "\n".join(f"{k}: {v}" for k, v in agg.items()),
        title=f"Evaluation Results ({split})",
    ))


def run_report(config_dir: str = "./configs", experiment_id: Optional[int] = None) -> None:
    """Generate report for an existing experiment."""
    console.print("[yellow]Report generation from DB records coming soon. "
                  "Reports are auto-generated after optimize.[/yellow]")


def run_resume(config_dir: str = "./configs", experiment_name: Optional[str] = None) -> None:
    """Resume an interrupted optimization run."""
    from app.utils.config import load_config
    from app.persistence.database import DatabaseManager
    from app.persistence.checkpoints import CheckpointManager

    cfg = load_config(config_dir)
    db = DatabaseManager(url=cfg.database.url)
    cp = CheckpointManager(db)

    name = experiment_name or cfg.experiment.name
    exp_id = cp.find_latest_experiment_id(name)
    if exp_id is None:
        console.print(f"[red]No experiment found with name '{name}'[/red]")
        return

    state = cp.load(exp_id)
    if state is None:
        console.print(f"[red]No checkpoint found for experiment #{exp_id}[/red]")
        return

    console.print(f"[green]Resuming experiment #{exp_id} from iteration {state.iteration}[/green]")
    run_optimize(config_dir=config_dir, experiment_name=name, resume_state=state)


if __name__ == "__main__":
    cli_app()
