"""
SQLAlchemy ORM models and database initialization.
Persists experiments, prompts, mutations, scores, and LLM calls.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """Top-level experiment record."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    dataset = Column(String(128), nullable=False)
    model = Column(String(128), nullable=False)
    optimizer = Column(String(64), nullable=False)
    seed_prompt_hash = Column(String(64))
    config_snapshot = Column(Text)  # JSON of full config
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(32), default="running")  # running | completed | failed | resumed

    # Relationships
    iterations = relationship("OptimizerIteration", back_populates="experiment", cascade="all, delete-orphan")
    llm_calls = relationship("LLMCall", back_populates="experiment", cascade="all, delete-orphan")


class Prompt(Base):
    """A versioned prompt text with hash for deduplication."""

    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hash = Column(String(64), unique=True, nullable=False, index=True)
    text = Column(Text, nullable=False)
    parent_hash = Column(String(64), nullable=True)
    mutation_strategy = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class OptimizerIteration(Base):
    """One iteration of the optimization loop."""

    __tablename__ = "optimizer_iterations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    iteration_number = Column(Integer, nullable=False)
    prompt_hash = Column(String(64), nullable=False)
    mutation_strategy = Column(String(64), nullable=True)
    val_f1 = Column(Float, nullable=True)
    val_precision = Column(Float, nullable=True)
    val_recall = Column(Float, nullable=True)
    val_aggregate = Column(Float, nullable=True)
    parse_success_rate = Column(Float, nullable=True)
    accepted = Column(Boolean, default=False)
    is_best = Column(Boolean, default=False)
    per_field_scores = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="iterations")

    def set_per_field(self, scores: dict[str, float]) -> None:
        self.per_field_scores = json.dumps(scores)

    def get_per_field(self) -> dict[str, float]:
        if self.per_field_scores:
            return json.loads(self.per_field_scores)
        return {}


class LLMCall(Base):
    """Record of every LLM API call made during the experiment."""

    __tablename__ = "llm_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    call_type = Column(String(32), nullable=False)  # extract | mutate | score
    model = Column(String(128), nullable=False)
    prompt_hash = Column(String(64), nullable=True)
    sample_id = Column(String(128), nullable=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    cached = Column(Boolean, default=False)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="llm_calls")


class Checkpoint(Base):
    """Optimizer state checkpoint for resumability."""

    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, nullable=False)
    iteration_number = Column(Integer, nullable=False)
    best_prompt_hash = Column(String(64), nullable=False)
    best_score = Column(Float, nullable=False)
    state_json = Column(Text, nullable=False)  # full optimizer state
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Manages database connection, session lifecycle, and record creation.
    """

    def __init__(self, url: str = "sqlite:///./runs/experiments.db", echo: bool = False) -> None:
        import os
        # Ensure directory exists for SQLite
        if url.startswith("sqlite:///"):
            db_path = url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        self._engine = create_engine(url, echo=echo)
        self._session_factory = sessionmaker(bind=self._engine)
        Base.metadata.create_all(self._engine)
        logger.info(f"Database initialized at: {url}")

    def get_session(self) -> Session:
        """Create and return a new database session."""
        return self._session_factory()

    def create_experiment(
        self,
        name: str,
        dataset: str,
        model: str,
        optimizer: str,
        seed_prompt_hash: str,
        config: dict[str, Any],
    ) -> int:
        """Create a new experiment record and return its ID."""
        with self.get_session() as session:
            exp = Experiment(
                name=name,
                dataset=dataset,
                model=model,
                optimizer=optimizer,
                seed_prompt_hash=seed_prompt_hash,
                config_snapshot=json.dumps(config, default=str),
                status="running",
            )
            session.add(exp)
            session.commit()
            eid = exp.id
        logger.info(f"Created experiment #{eid}: {name}")
        return eid

    def upsert_prompt(
        self,
        hash_: str,
        text: str,
        parent_hash: Optional[str] = None,
        mutation_strategy: Optional[str] = None,
    ) -> None:
        """Insert a prompt if it doesn't already exist."""
        with self.get_session() as session:
            existing = session.query(Prompt).filter_by(hash=hash_).first()
            if existing is None:
                session.add(
                    Prompt(
                        hash=hash_,
                        text=text,
                        parent_hash=parent_hash,
                        mutation_strategy=mutation_strategy,
                    )
                )
                session.commit()

    def get_prompt_by_hash(self, hash_: str) -> Optional[str]:
        """Retrieve prompt text by hash."""
        with self.get_session() as session:
            p = session.query(Prompt).filter_by(hash=hash_).first()
            return p.text if p else None

    def log_iteration(
        self,
        experiment_id: int,
        iteration_number: int,
        prompt_hash: str,
        mutation_strategy: Optional[str],
        scores: dict[str, Any],
        accepted: bool,
        is_best: bool,
    ) -> int:
        """Log one optimizer iteration."""
        with self.get_session() as session:
            it = OptimizerIteration(
                experiment_id=experiment_id,
                iteration_number=iteration_number,
                prompt_hash=prompt_hash,
                mutation_strategy=mutation_strategy,
                val_f1=scores.get("mean_f1"),
                val_precision=scores.get("mean_precision"),
                val_recall=scores.get("mean_recall"),
                val_aggregate=scores.get("mean_aggregate"),
                parse_success_rate=scores.get("parse_success_rate"),
                accepted=accepted,
                is_best=is_best,
            )
            it.set_per_field(scores.get("per_field", {}))
            session.add(it)
            session.commit()
            return it.id

    def log_llm_call(
        self,
        experiment_id: int,
        call_type: str,
        model: str,
        prompt_hash: Optional[str],
        sample_id: Optional[str],
        tokens: dict[str, int],
        latency_ms: float,
        cached: bool,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Persist a single LLM call record."""
        with self.get_session() as session:
            session.add(
                LLMCall(
                    experiment_id=experiment_id,
                    call_type=call_type,
                    model=model,
                    prompt_hash=prompt_hash,
                    sample_id=sample_id,
                    prompt_tokens=tokens.get("prompt_tokens", 0),
                    completion_tokens=tokens.get("completion_tokens", 0),
                    total_tokens=tokens.get("total_tokens", 0),
                    latency_ms=latency_ms,
                    cached=cached,
                    success=success,
                    error_message=error_message,
                )
            )
            session.commit()

    def save_checkpoint(
        self,
        experiment_id: int,
        iteration: int,
        best_prompt_hash: str,
        best_score: float,
        state: dict[str, Any],
    ) -> None:
        """Save optimizer checkpoint for resumability."""
        with self.get_session() as session:
            cp = Checkpoint(
                experiment_id=experiment_id,
                iteration_number=iteration,
                best_prompt_hash=best_prompt_hash,
                best_score=best_score,
                state_json=json.dumps(state, default=str),
            )
            session.add(cp)
            session.commit()
        logger.debug(f"Checkpoint saved: iter={iteration}, score={best_score:.4f}")

    def load_latest_checkpoint(
        self, experiment_id: int
    ) -> Optional[dict[str, Any]]:
        """Load the most recent checkpoint for an experiment."""
        with self.get_session() as session:
            cp = (
                session.query(Checkpoint)
                .filter_by(experiment_id=experiment_id)
                .order_by(Checkpoint.iteration_number.desc())
                .first()
            )
            if cp is None:
                return None
            return json.loads(cp.state_json)

    def get_experiment_iterations(
        self, experiment_id: int
    ) -> list[OptimizerIteration]:
        """Retrieve all iterations for an experiment, ordered by iteration number."""
        with self.get_session() as session:
            return (
                session.query(OptimizerIteration)
                .filter_by(experiment_id=experiment_id)
                .order_by(OptimizerIteration.iteration_number)
                .all()
            )

    def mark_experiment_complete(self, experiment_id: int) -> None:
        """Mark an experiment as completed."""
        with self.get_session() as session:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            if exp:
                exp.status = "completed"
                exp.completed_at = datetime.utcnow()
                session.commit()

    def list_experiments(self) -> list[Experiment]:
        """List all experiments."""
        with self.get_session() as session:
            return session.query(Experiment).order_by(Experiment.started_at.desc()).all()
