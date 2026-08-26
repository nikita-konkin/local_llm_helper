"""SQLite-backed job queue, results, and the token-savings ledger.

The ledger is the point of the project made measurable: every offloaded call
records how many tokens the material *would* have cost in an expensive context
against how many actually came back.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import Config, get_config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


class Priority:
    """Lower runs first. EXPRESS exists so a synchronous call jumps the queue."""

    EXPRESS = 0
    HIGH = 3
    NORMAL = 5  # "second priority" work lands here
    LOW = 8


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    spec: Mapped[dict] = mapped_column(JSON, default=dict)
    tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=Priority.NORMAL, index=True)
    state: Mapped[str] = mapped_column(String(16), default=JobState.QUEUED, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    label: Mapped[str] = mapped_column(String(200), default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The project that queued this. Carried onto the savings row when the worker
    # runs it, so queued work is attributed to whoever asked for it rather than
    # to the worker process.
    client: Mapped[str | None] = mapped_column(String(260), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "tier": self.tier,
            "priority": self.priority,
            "attempts": self.attempts,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    path: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Saving(Base):
    """One offloaded call's token accounting."""

    __tablename__ = "savings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    # mcp | cli | worker are real work. bench | ab are measurement traffic and are
    # excluded from the headline figure: benchmarking re-reads the same file dozens
    # of times, which nobody would ever have done at full price. Counting it as
    # "saved" turns the ledger into a flattering fiction.
    source: Mapped[str] = mapped_column(String(16), default="mcp")
    tier: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(120), default="")
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # Which project asked. An MCP server is launched as a child of the Claude
    # session, so its working directory is that session's project root — which
    # makes "is my other project actually using this?" an answerable question
    # rather than a guess.
    client: Mapped[str | None] = mapped_column(String(260), nullable=True, index=True)
    # Tokens of raw material the expensive model did NOT have to read.
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # Tokens actually handed back.
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    local_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    local_completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)

    @property
    def saved(self) -> int:
        return max(0, self.input_tokens - self.output_tokens)


class Store:
    """All database access. Safe for one writer plus several readers (WAL)."""

    def __init__(self, cfg: Config | None = None, url: str | None = None):
        self.cfg = cfg or get_config()
        if url is None:
            self.cfg.paths.jobs_db.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{self.cfg.paths.jobs_db}"
        self.engine = create_engine(url, future=True, json_serializer=json.dumps)

        @event.listens_for(self.engine, "connect")
        def _pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")  # MCP reads never block worker writes
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.close()

        Base.metadata.create_all(self.engine)
        self._migrate()
        self._sessions = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        SQLite has no IF NOT EXISTS for columns, so ask before adding. Keeping
        this here means an existing jobs.db keeps working instead of raising
        OperationalError on the first query after an upgrade.
        """
        from sqlalchemy import text

        with self.engine.begin() as conn:
            have = {r[1] for r in conn.execute(text("PRAGMA table_info(savings)"))}
            if "client" not in have:
                conn.execute(text("ALTER TABLE savings ADD COLUMN client VARCHAR(260)"))
            have_jobs = {r[1] for r in conn.execute(text("PRAGMA table_info(jobs)"))}
            if "client" not in have_jobs:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN client VARCHAR(260)"))

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._sessions()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ----------------------------------------------------------------- jobs

    def submit(
        self,
        kind: str,
        spec: dict[str, Any],
        tier: str | None = None,
        priority: int = Priority.NORMAL,
        label: str = "",
        max_attempts: int | None = None,
        client: str | None = None,
    ) -> str:
        job = Job(
            id=str(uuid.uuid4()),
            kind=kind,
            spec=spec,
            tier=tier,
            priority=priority,
            label=label[:200],
            max_attempts=max_attempts or self.cfg.worker.max_attempts,
            client=client,
        )
        with self.session() as s:
            s.add(job)
        return job.id

    def claim(self, worker_id: str | None = None) -> Job | None:
        """Atomically take the highest-priority queued job, or return None.

        The conditional UPDATE is the lock: a second worker racing for the same
        row changes zero rows and loops around instead of double-running it.
        """
        worker_id = worker_id or str(os.getpid())
        with self.session() as s:
            candidate = s.execute(
                select(Job)
                .where(Job.state == JobState.QUEUED)
                .order_by(Job.priority.asc(), Job.created_at.asc())
                .limit(1)
            ).scalar_one_or_none()
            if candidate is None:
                return None

            changed = (
                s.query(Job)
                .filter(Job.id == candidate.id, Job.state == JobState.QUEUED)
                .update(
                    {
                        "state": JobState.RUNNING,
                        "attempts": Job.attempts + 1,
                        "started_at": utcnow(),
                        "worker_id": worker_id,
                    },
                    synchronize_session=False,
                )
            )
            if not changed:
                return None  # lost the race
            s.expire_all()
            return s.get(Job, candidate.id)

    def complete(
        self,
        job_id: str,
        summary: str,
        path: str | None = None,
        meta: dict | None = None,
    ) -> None:
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            job.state = JobState.DONE
            job.finished_at = utcnow()
            job.error = None
            s.add(Result(job_id=job_id, summary=summary, path=path, meta=meta or {}))

    #: Failures that will never succeed on a retry. A refused path stays refused
    #: and a malformed spec stays malformed, so retrying only burns the attempt
    #: budget and makes the queue look busy while achieving nothing. Seen in real
    #: use: a digest aimed outside the read roots was attempted three times.
    PERMANENT_ERRORS = (
        "GuardError",
        "FileNotFoundError",
        "PermissionError",
        "ValueError",
        "KeyError",
    )

    @classmethod
    def is_permanent(cls, error: str) -> bool:
        return any(error.startswith(name + ":") for name in cls.PERMANENT_ERRORS)

    def fail(self, job_id: str, error: str, permanent: bool | None = None) -> JobState:
        """Requeue if attempts remain, otherwise mark failed. Returns the new state.

        A permanent error skips the retries entirely.
        """
        if permanent is None:
            permanent = self.is_permanent(error)
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            job.error = error[:4000]
            if not permanent and job.attempts < job.max_attempts:
                job.state = JobState.QUEUED
                job.started_at = None
                job.worker_id = None
            else:
                job.state = JobState.FAILED
                job.finished_at = utcnow()
            return JobState(job.state)

    def cancel(self, job_id: str) -> bool:
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None or JobState(job.state).terminal:
                return False
            job.state = JobState.CANCELLED
            job.finished_at = utcnow()
            return True

    def get(self, job_id: str) -> Job | None:
        with self.session() as s:
            return s.get(Job, job_id)

    def result(self, job_id: str) -> Result | None:
        with self.session() as s:
            return s.execute(
                select(Result)
                .where(Result.job_id == job_id)
                .order_by(Result.id.desc())
                .limit(1)
            ).scalar_one_or_none()

    def list_jobs(self, state: str | None = None, limit: int = 50) -> list[Job]:
        with self.session() as s:
            q = select(Job).order_by(Job.created_at.desc()).limit(limit)
            if state:
                q = q.where(Job.state == state)
            return list(s.execute(q).scalars())

    def counts(self) -> dict[str, int]:
        with self.session() as s:
            rows = s.execute(select(Job.state, func.count()).group_by(Job.state)).all()
        return {state: n for state, n in rows}

    def prune_jobs(self, days: int, dry_run: bool = False) -> dict:
        """Delete finished jobs older than `days`, and their results.

        Nothing removed these before, so the table only ever grew and the
        dashboard filled with week-old completed work under a heading that says
        "Queue". Only terminal states go: anything queued or running is somebody
        waiting on an answer.

        Savings rows are deliberately kept. They are the ledger - the record of
        what was actually offloaded - and losing them to a cleanup would quietly
        rewrite the one number this project exists to report.
        """
        from sqlalchemy import delete

        cutoff = utcnow() - timedelta(days=days)
        terminal = (JobState.DONE, JobState.FAILED, JobState.CANCELLED)
        with self.session() as s:
            doomed = [
                r[0] for r in s.execute(
                    select(Job.id).where(
                        Job.state.in_(terminal), Job.created_at < cutoff
                    )
                ).all()
            ]
            if not doomed or dry_run:
                return {"jobs": len(doomed), "results": 0, "dry_run": dry_run}

            results = s.execute(
                delete(Result).where(Result.job_id.in_(doomed))
            ).rowcount
            jobs = s.execute(delete(Job).where(Job.id.in_(doomed))).rowcount
        return {"jobs": jobs, "results": results, "dry_run": False}

    def active_jobs(self) -> list[Job]:
        """Jobs someone is still waiting on."""
        with self.session() as s:
            return list(
                s.execute(
                    select(Job)
                    .where(Job.state.in_((JobState.QUEUED, JobState.RUNNING)))
                    .order_by(Job.priority.asc(), Job.created_at.asc())
                ).scalars()
            )

    def requeue_stale_running(self) -> int:
        """Recover jobs a killed worker left mid-flight."""
        with self.session() as s:
            return (
                s.query(Job)
                .filter(Job.state == JobState.RUNNING)
                .update(
                    {"state": JobState.QUEUED, "started_at": None, "worker_id": None},
                    synchronize_session=False,
                )
            )

    # -------------------------------------------------------------- savings

    def record_saving(self, **kw: Any) -> None:
        with self.session() as s:
            s.add(Saving(**kw))

    #: Sources that represent measurement rather than work someone needed done.
    SYNTHETIC_SOURCES = ("bench", "ab")

    def savings_summary(
        self, days: int | None = None, include_synthetic: bool = False
    ) -> dict[str, Any]:
        with self.session() as s:
            q = select(
                func.count(Saving.id),
                func.coalesce(func.sum(Saving.input_tokens), 0),
                func.coalesce(func.sum(Saving.output_tokens), 0),
                func.coalesce(func.sum(Saving.duration_s), 0.0),
            )
            if days:
                q = q.where(Saving.at >= utcnow() - timedelta(days=days))
            if not include_synthetic:
                q = q.where(Saving.source.notin_(self.SYNTHETIC_SOURCES))
            calls, tin, tout, secs = s.execute(q).one()

            by_kind_q = select(
                Saving.kind,
                func.count(Saving.id),
                func.coalesce(func.sum(Saving.input_tokens), 0),
                func.coalesce(func.sum(Saving.output_tokens), 0),
            ).group_by(Saving.kind)
            if days:
                by_kind_q = by_kind_q.where(Saving.at >= utcnow() - timedelta(days=days))
            if not include_synthetic:
                by_kind_q = by_kind_q.where(Saving.source.notin_(self.SYNTHETIC_SOURCES))
            by_source_q = select(
                Saving.source, func.count(Saving.id),
                func.coalesce(func.sum(Saving.input_tokens), 0),
            ).group_by(Saving.source)
            by_source = [
                {"source": src, "calls": c, "input_tokens": i}
                for src, c, i in s.execute(by_source_q).all()
            ]

            by_client_q = select(
                Saving.client, func.count(Saving.id),
                func.coalesce(func.sum(Saving.input_tokens), 0),
                func.coalesce(func.sum(Saving.output_tokens), 0),
                func.max(Saving.at),
            ).where(Saving.client.isnot(None)).group_by(Saving.client)
            if not include_synthetic:
                by_client_q = by_client_q.where(
                    Saving.source.notin_(self.SYNTHETIC_SOURCES)
                )
            by_client = [
                {"client": cl, "calls": c, "input_tokens": i,
                 "saved": max(0, i - o), "last": last.isoformat() if last else None}
                for cl, c, i, o, last in s.execute(by_client_q).all()
            ]

            by_kind = [
                {
                    "kind": k,
                    "calls": c,
                    "input_tokens": i,
                    "output_tokens": o,
                    "saved": max(0, i - o),
                }
                for k, c, i, o in s.execute(by_kind_q).all()
            ]

        return {
            "calls": calls,
            "input_tokens": int(tin),
            "output_tokens": int(tout),
            "tokens_saved": max(0, int(tin) - int(tout)),
            "compression": round(int(tin) / int(tout), 1) if tout else 0.0,
            "local_seconds": round(float(secs), 1),
            "by_kind": sorted(by_kind, key=lambda d: -d["saved"]),
            "by_source": sorted(by_source, key=lambda d: -d["input_tokens"]),
            "by_client": sorted(by_client, key=lambda d: -d["saved"]),
            "includes_synthetic": include_synthetic,
        }
