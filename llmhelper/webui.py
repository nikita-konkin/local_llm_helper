"""A local dashboard: status, usage, the queue, and the controls worth having.

Deliberately small. One FastAPI app, one HTML page, no build step and no
JavaScript dependencies — the machine this runs on has no npm, and a dashboard
that needs a toolchain to change one number is not lightweight.

Two decisions worth knowing about:

*Bound to localhost by default.* The controls here start processes and load
multi-gigabyte models. That is not something to expose by accident.

*``read_roots`` and ``deny_globs`` are read-only here.* They are the filesystem
security boundary. A web UI able to widen them would let anything that reaches
the page grant itself read access to the whole disk, which defeats the guard
entirely. Operational settings — tiers, budgets, retention — are editable;
the boundary is edited in ``config.toml``, deliberately and with intent.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import PROJECT_ROOT, Config, get_config, load_config
from .store import Store

app = FastAPI(title="local_llm_helper", docs_url=None, redoc_url=None)

_started = time.monotonic()

# The dashboard polls; LM Studio logs every request it receives. Without a cache
# a 5 s refresh put ~25 requests a minute into its Developer tab forever, which
# is not a dashboard reporting on a system so much as a dashboard being the
# system's main workload. Runtime probes are cached; the database is not, since
# reading SQLite costs nobody anything.
_PROBE_TTL = 20.0
_probe_cache: dict[str, Any] = {"at": 0.0, "value": None}


def _probe(force: bool = False) -> dict:
    """Runtime state, at most once per _PROBE_TTL seconds."""
    now = time.monotonic()
    if not force and _probe_cache["value"] is not None and now - _probe_cache["at"] < _PROBE_TTL:
        return _probe_cache["value"]

    rt = runtime()
    snap = rt.snapshot()
    tiers = []
    for name, t in cfg().tiers.items():
        info = rt.find(t) if snap["lms_present"] else None
        tiers.append(
            {
                "name": name,
                "model": t.model,
                "quant": t.quant,
                "context": t.context,
                "present": info is not None,
                "size_gb": info.size_gb if info else None,
                "loaded": (not t.is_embedding) and rt.is_loaded(t),
                "is_embedding": t.is_embedding,
            }
        )
    value = {"snap": snap, "tiers": tiers, "can_manage_models": rt.can_manage_models}
    _probe_cache.update(at=now, value=value)
    return value


def _invalidate() -> None:
    """After anything that changes runtime state, the next read must be fresh."""
    _probe_cache["value"] = None


def cfg() -> Config:
    return get_config()


def runtime():
    from .runtime import Runtime

    return Runtime(cfg())


def store() -> Store:
    return Store(cfg())


# --------------------------------------------------------------- worker state


def worker_pid() -> int | None:
    """The running worker's pid from its lock file, or None.

    The lock is the same one `llmh worker` takes, so the dashboard and the CLI
    always agree about whether a worker exists.
    """
    lock = Path(cfg().paths.logs) / "worker.pid"
    if not lock.is_file():
        return None
    try:
        pid = int(lock.read_text().strip() or 0)
    except (ValueError, OSError):
        return None
    if not pid:
        return None
    try:
        import psutil

        return pid if psutil.pid_exists(pid) else None
    except Exception:  # noqa: BLE001
        return None


def _spawn_detached(args: list[str]) -> int:
    """Start a long-running child that outlives this request."""
    kwargs: dict[str, Any] = {"cwd": str(PROJECT_ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs
    ).pid


# --------------------------------------------------------------------- API


@app.get("/api/status")
def api_status(fresh: bool = False) -> dict:
    probe = _probe(force=fresh)
    snap = probe["snap"]
    pid = worker_pid()

    return {
        "server_up": snap["server_up"],
        "base_url": snap["base_url"],
        "can_manage_models": probe["can_manage_models"],
        "worker_pid": pid,
        "ram_free_gb": snap["ram_free_gb"],
        "ram_total_gb": snap["ram_total_gb"],
        "vram_used_mb": snap["vram_used_mb"],
        "vram_total_mb": snap["vram_total_mb"],
        "loaded": snap["loaded"],
        "tiers": probe["tiers"],
        "probe_age_s": round(time.monotonic() - _probe_cache["at"]),
        # Which machine's ledger this is. A remote client (base_url pointing at
        # another box) keeps its OWN jobs.db, so its savings are recorded where it
        # runs, not where the model runs. Saying so here stops that being a
        # mystery: a laptop's numbers live on the laptop.
        "host": socket.gethostname(),
        "db": str(cfg().paths.jobs_db),
        "remote_model": not cfg().server.base_url.startswith(
            ("http://127.0.0.1", "http://localhost")
        ),
        "queue": store().counts(),
        "uptime_s": round(time.monotonic() - _started),
    }


@app.get("/api/connect")
def api_connect() -> dict:
    """Everything another machine needs to connect to this one."""
    from .runtime import lan_addresses

    c = cfg()
    addrs = lan_addresses()
    best = next((a["ip"] for a in addrs if a.get("likely_lan")), None)
    reachable = c.server.bind in ("0.0.0.0", "::")

    return {
        "addresses": addrs,
        "best": best,
        "ambiguous": bool(addrs and addrs[0].get("ambiguous")),
        "bind": c.server.bind,
        "port": c.server.port,
        "lan_reachable": reachable,
        # Option A: the other machine runs its own copy and borrows this GPU.
        "model_url": f"http://{best or '<this-pc>'}:{c.server.port}/v1",
        # Option B: the other machine calls this machine's MCP tools over HTTP.
        "mcp_url": f"http://{best or '<this-pc>'}:8765/mcp",
        "note": (
            "Bound to localhost only — no other machine can reach it. Set "
            'server.bind = "0.0.0.0" in config.toml and restart the server.'
            if not reachable else
            "Reachable from the LAN, provided the firewall allows the port."
        ),
    }


@app.get("/api/stats")
def api_stats(days: int | None = None, all_sources: bool = False) -> dict:
    return store().savings_summary(days=days, include_synthetic=all_sources)


@app.get("/api/activity")
def api_activity(limit: int = 15, all_sources: bool = False) -> list[dict]:
    """The last N offloaded calls, express and queued alike."""
    return store().recent_activity(limit=limit, include_synthetic=all_sources)


@app.get("/api/jobs")
def api_jobs(limit: int = 25, state: str | None = None) -> dict:
    """Work in progress and finished work, kept apart.

    A single list sorted by time reads as "the queue is full of week-old jobs"
    when in fact the queue is empty and that is history. Splitting them makes
    "is anything running?" answerable at a glance.
    """
    s = store()
    active = [j.as_dict() for j in s.active_jobs()]
    active_ids = {j["id"] for j in active}
    recent = [
        j.as_dict()
        for j in s.list_jobs(state=state, limit=limit)
        if j.id not in active_ids
    ]
    return {"active": active, "recent": recent}


@app.post("/api/jobs/prune")
def api_prune_jobs(days: int = 7) -> dict:
    """Delete finished jobs older than `days`. Savings rows are never touched."""
    return store().prune_jobs(days)


@app.get("/api/job/{job_id}")
def api_job(job_id: str) -> dict:
    s = store()
    job = s.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    res = s.result(job_id)
    return {
        "job": job.as_dict(),
        "summary": res.summary if res else None,
        "path": res.path if res else None,
        "meta": res.meta if res else None,
    }


@app.get("/api/settings")
def api_settings() -> dict:
    c = cfg()
    return {
        "editable": {
            "map_chunk_tokens": c.budgets.map_chunk_tokens,
            "map_summary_tokens": c.budgets.map_summary_tokens,
            "map_concurrency": c.budgets.map_concurrency,
            "express_max_out_tokens": c.budgets.express_max_out_tokens,
            "express_timeout_s": c.budgets.express_timeout_s,
            "max_map_chunks": c.budgets.max_map_chunks,
            "result_retention_days": c.budgets.result_retention_days,
            "poll_interval_s": c.worker.poll_interval_s,
            "max_attempts": c.worker.max_attempts,
            "fallback_tier": c.worker.fallback_tier,
        },
        # Shown so you can see the boundary, never written from here.
        "readonly": {
            "read_roots": [str(r) for r in c.paths.read_roots],
            "deny_globs": list(c.paths.deny_globs),
            "outbox": str(c.paths.outbox),
            "base_url": c.server.base_url,
            "lms_exe": str(c.server.lms_exe),
        },
        "readonly_reason": (
            "read_roots and deny_globs are the filesystem security boundary. "
            "Widening them from a web UI would let anything that reaches this page "
            "grant itself read access to the whole disk. Edit config.toml directly."
        ),
        "config_path": str(PROJECT_ROOT / "config.toml"),
        "tiers": sorted(cfg().llm_tiers),
    }


class SettingsPatch(BaseModel):
    map_chunk_tokens: int | None = None
    map_summary_tokens: int | None = None
    map_concurrency: int | None = None
    express_max_out_tokens: int | None = None
    express_timeout_s: float | None = None
    max_map_chunks: int | None = None
    result_retention_days: int | None = None
    poll_interval_s: float | None = None
    max_attempts: int | None = None
    fallback_tier: str | None = None


# Which TOML table each editable key lives in.
_TABLE = {
    "map_chunk_tokens": "budgets",
    "map_summary_tokens": "budgets",
    "map_concurrency": "budgets",
    "express_max_out_tokens": "budgets",
    "express_timeout_s": "budgets",
    "max_map_chunks": "budgets",
    "result_retention_days": "budgets",
    "poll_interval_s": "worker",
    "max_attempts": "worker",
    "fallback_tier": "worker",
}


@app.post("/api/settings")
def api_set_settings(patch: SettingsPatch) -> dict:
    """Rewrite matching keys in config.toml, in place, preserving comments.

    A line-level edit rather than a parse-and-dump: this file carries the reasons
    behind every tuned number, and a round trip through a TOML writer would throw
    all of that away.
    """
    path = PROJECT_ROOT / "config.toml"
    if not path.is_file():
        raise HTTPException(400, f"no config at {path}")

    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not changes:
        return {"changed": [], "message": "nothing to change"}

    if "fallback_tier" in changes and changes["fallback_tier"] not in cfg().tiers:
        raise HTTPException(400, f"unknown tier {changes['fallback_tier']!r}")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    table = None
    applied: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            table = stripped[1:-1]
            continue
        if "=" not in stripped or stripped.startswith("#"):
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in changes and _TABLE.get(key) == table:
            value = changes[key]
            rendered = f'"{value}"' if isinstance(value, str) else value
            pad = len(line) - len(line.lstrip())
            width = max(len(key), 22)
            lines[i] = f"{' ' * pad}{key:<{width}} = {rendered}\n"
            applied.append(key)

    missing = sorted(set(changes) - set(applied))
    if missing:
        raise HTTPException(400, f"keys not found in config.toml: {missing}")

    backup = path.with_suffix(".toml.bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("".join(lines), encoding="utf-8")

    try:
        load_config(path)  # refuse to leave a broken config behind
    except Exception as exc:  # noqa: BLE001
        path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        raise HTTPException(400, f"change rejected, config restored: {exc}") from exc

    get_config.cache_clear()
    return {
        "changed": applied,
        "note": "A running worker keeps its old settings until restarted.",
    }


# ------------------------------------------------------------------ controls


@app.post("/api/server/start")
def api_server_start() -> dict:
    rt = runtime()
    if not rt.can_manage_models:
        raise HTTPException(400, "no LM Studio CLI on this machine (remote client)")
    ok = rt.ensure_server()
    _invalidate()
    return {"server_up": ok}


@app.post("/api/server/stop")
def api_server_stop() -> dict:
    from .runtime import RuntimeError_

    try:
        ok = runtime().stop_server()
    except RuntimeError_ as exc:
        raise HTTPException(400, str(exc)) from exc
    _invalidate()
    return {"server_up": not ok, "message": "server stopped" if ok else "still answering"}


@app.post("/api/server/restart")
def api_server_restart() -> dict:
    from .runtime import RuntimeError_

    try:
        ok = runtime().restart_server()
    except RuntimeError_ as exc:
        raise HTTPException(400, str(exc)) from exc
    _invalidate()
    return {"server_up": ok, "message": "restarted" if ok else "did not come back"}


@app.post("/api/worker/restart")
def api_worker_restart() -> dict:
    """Stop the worker and start a fresh one.

    Needed after a code or settings change: a worker loads the package once and
    keeps running whatever it started with, which has quietly produced stale
    behaviour more than once.
    """
    api_worker_stop()
    time.sleep(1.0)
    return api_worker_start()


@app.post("/api/worker/start")
def api_worker_start() -> dict:
    if (pid := worker_pid()) is not None:
        return {"worker_pid": pid, "message": "already running"}
    _spawn_detached([sys.executable, "-m", "llmhelper.cli", "worker", "--quiet"])
    for _ in range(40):  # the child writes the lock file, so wait for it
        time.sleep(0.25)
        if (pid := worker_pid()) is not None:
            _invalidate()
            return {"worker_pid": pid}
    return {"worker_pid": None, "message": "started, but no lock file yet — check logs"}


@app.post("/api/worker/stop")
def api_worker_stop() -> dict:
    pid = worker_pid()
    if pid is None:
        return {"stopped": False, "message": "not running"}
    try:
        import psutil

        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=15)
        _invalidate()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"could not stop pid {pid}: {exc}") from exc
    return {"stopped": True}


@app.post("/api/model/load/{tier_name}")
def api_model_load(tier_name: str) -> dict:
    rt = runtime()
    if not rt.can_manage_models:
        raise HTTPException(400, "no LM Studio CLI on this machine (remote client)")
    try:
        tier = cfg().tier(tier_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc
    try:
        rt.ensure_loaded(tier)
    except Exception as exc:  # noqa: BLE001 - RAM refusal is a normal outcome
        raise HTTPException(400, str(exc)) from exc
    _invalidate()
    return {"loaded": [m.identifier for m in rt.loaded()]}


@app.post("/api/model/unload")
def api_model_unload() -> dict:
    rt = runtime()
    if not rt.can_manage_models:
        raise HTTPException(400, "no LM Studio CLI on this machine (remote client)")
    rt.unload_all()
    _invalidate()
    return {"loaded": []}


@app.post("/api/job/{job_id}/cancel")
def api_job_cancel(job_id: str) -> dict:
    return {"cancelled": store().cancel(job_id)}


@app.post("/api/prune")
def api_prune(days: int | None = None) -> dict:
    from .guard import iter_outbox

    c = cfg()
    cutoff = time.time() - (days or c.budgets.result_retention_days) * 86400
    removed = freed = 0
    for p in iter_outbox(c):
        try:
            if p.stat().st_mtime < cutoff:
                freed += p.stat().st_size
                p.unlink()
                removed += 1
        except OSError:
            continue
    return {"removed": removed, "freed_kb": round(freed / 1024)}


# --------------------------------------------------------------------- page


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "webui.html").read_text(encoding="utf-8")


@app.exception_handler(Exception)
def _unhandled(_request, exc: Exception):  # noqa: ANN001
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def serve(host: str = "127.0.0.1", port: int = 8770) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
