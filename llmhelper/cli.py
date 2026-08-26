"""Command line for local_llm_helper.

Everything the MCP server can do is reachable here too, so the system can be
driven, debugged and verified without an agent in the loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ConfigError, get_config
from .runner import DEFAULT_TIER, TierResolver, execute
from .store import JobState, Priority, Store

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Offload second-priority work to the local LLM.",
)
console = Console()
err = Console(stderr=True)


def _store() -> Store:
    return Store(get_config())


def _fmt(n: int) -> str:
    return f"{n:,}"


# Generation-speed thresholds for routing decisions.
#
# Measured on this machine: prefill ~4,900 tok/s, generation ~104 tok/s — prefill
# is ~47x faster, so a digest that reads 39,000 tokens and writes 900 is bound by
# the writing. Working the express-lane worst case (the 120,000-token cap in
# mcp_server.py, chunked at 24,000, ~400 output tokens per chunk plus a reduce):
#
#   100 tok/s -> ~40-55 s   comfortable synchronously
#    60 tok/s -> ~70-90 s   acceptable
#    30 tok/s -> ~3 min     queue only
#    12 tok/s -> ~6 min     matches the observed 403 s on the `work` tier
#
# Below ~10 tok/s the wall-clock cost stops being worth the tokens saved.
EXPRESS_MIN_TOK_S = 60.0
QUEUE_MIN_TOK_S = 30.0
USABLE_MIN_TOK_S = 10.0


def speed_verdict(tok_per_s: float) -> tuple[str, str]:
    """(colour, verdict) for a measured generation rate."""
    if tok_per_s >= EXPRESS_MIN_TOK_S:
        return "green", "express OK"
    if tok_per_s >= QUEUE_MIN_TOK_S:
        return "yellow", "queue only"
    if tok_per_s >= USABLE_MIN_TOK_S:
        return "yellow", "slow, queue only"
    return "red", "too slow"


# --------------------------------------------------------------------- doctor


@app.command()
def doctor(
    fix: bool = typer.Option(False, "--fix", help="Start the LM Studio server if it is down."),
) -> None:
    """Check that everything this depends on is actually working."""
    from .runtime import Runtime

    try:
        cfg = get_config()
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1)

    rt = Runtime(cfg)
    problems: list[str] = []

    if fix and not rt.server_running():
        console.print("starting LM Studio server...")
        rt.ensure_server()

    snap = rt.snapshot()
    table = Table(title="llmh doctor", show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("result")

    def row(name: str, ok: bool, detail: str) -> None:
        table.add_row(name, f"[{'green' if ok else 'red'}]{'OK' if ok else 'FAIL'}[/] {detail}")
        if not ok:
            problems.append(name)

    row("lms CLI", snap["lms_present"], snap["lms_exe"])
    row("server", snap["server_up"], snap["base_url"])

    ram_ok = snap["ram_free_gb"] > 8
    row("RAM", ram_ok, f"{snap['ram_free_gb']} GB free of {snap['ram_total_gb']} GB")

    if snap["vram_total_mb"]:
        row("GPU", True, f"{snap['vram_used_mb']} / {snap['vram_total_mb']} MB VRAM used")
    else:
        table.add_row("GPU", "[yellow]none detected[/] (CPU-only inference)")

    catalog = rt.catalog() if snap["lms_present"] else []
    resolver = TierResolver(cfg, rt)
    for name, tier in cfg.tiers.items():
        pin = f" @ {tier.quant}" if tier.quant else ""
        if not catalog:
            table.add_row(f"tier {name}", f"[yellow]unknown[/] {tier.model}{pin}")
            continue
        info = rt.find(tier)
        if info is None:
            table.add_row(
                f"tier {name}",
                f"[yellow]not downloaded[/] {tier.model}{pin}",
            )
        else:
            table.add_row(
                f"tier {name}",
                f"[green]present[/] {info.key} {info.quant} "
                f"({info.size_gb:.1f} GB, needs {rt.required_ram_gb(tier):.0f} GB free, "
                f"ctx {min(tier.context, info.max_context or tier.context):,})",
            )

    for root in cfg.paths.read_roots:
        row(f"read root {root}", Path(root).exists(), "")

    outbox_ok = True
    try:
        cfg.paths.outbox.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        outbox_ok = False
        err.print(f"outbox: {exc}")
    row("outbox", outbox_ok, str(cfg.paths.outbox))

    try:
        counts = _store().counts()
        row("job database", True, str(counts or "empty"))
    except Exception as exc:  # noqa: BLE001
        row("job database", False, str(exc))

    if snap["loaded"]:
        table.add_row("loaded now", ", ".join(snap["loaded"]))

    console.print(table)

    if not snap["server_up"]:
        console.print("\n[yellow]The server is down.[/] Start it with:  llmh doctor --fix")
    try:
        chosen = resolver.resolve("digest", None)
        console.print(f"\ndigest would run on tier [bold]{chosen.name}[/] ({chosen.model})")
    except RuntimeError as exc:
        problems.append("tier resolution")
        err.print(f"[red]{exc}[/red]")

    raise typer.Exit(1 if problems else 0)


# ---------------------------------------------------------------------- bench


@app.command()
def fetch(
    tier_name: str = typer.Argument(..., help="Tier whose model should be downloaded."),
    attempts: int = typer.Option(40, "--attempts", help="Retries; downloads resume."),
) -> None:
    """Download a tier's model, resuming until it is actually present.

    `lms get` times out on slow links and must be re-run to resume, so a large
    download otherwise needs babysitting. Presence is judged by the same
    quantisation-aware rule the resolver uses, so an existing copy in the wrong
    quantisation does not count as done.
    """
    import subprocess
    import time

    from .runtime import Runtime

    cfg = get_config()
    tier = cfg.tier(tier_name)
    rt = Runtime(cfg)

    for attempt in range(1, attempts + 1):
        if (info := rt.find(tier)) is not None:
            console.print(
                f"[green]{tier.model} present[/] — {info.quant}, {info.size_gb:.1f} GB"
            )
            raise typer.Exit(0)

        console.print(f"attempt {attempt}/{attempts}: fetching {tier.model}...")
        subprocess.run(
            [str(cfg.server.lms_exe), "get", tier.load_ref, "--gguf", "-y"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        rt.catalog(refresh=True)  # a timed-out fetch may still have completed
        time.sleep(5)

    err.print(f"[red]{tier.model} still not present after {attempts} attempt(s)[/red]")
    raise typer.Exit(1)


@app.command()
def bench(
    tier: Optional[str] = typer.Option(None, "--tier", help="Only benchmark this tier."),
    tokens: int = typer.Option(200, "--tokens", help="Tokens to generate per tier."),
) -> None:
    """Measure load time and tokens/second for each tier, and record the results."""
    from .client import LocalLLM
    from .runtime import Runtime, gpu_memory_mb

    cfg = get_config()
    rt = Runtime(cfg)
    if not rt.ensure_server():
        err.print("[red]LM Studio server did not come up[/red]")
        raise typer.Exit(1)

    llm = LocalLLM(cfg)
    names = [tier] if tier else [n for n, t in cfg.tiers.items() if not t.is_embedding]
    prompt = (
        "Explain in one paragraph why a mixture-of-experts model with 3B active "
        "parameters can be faster than a dense 9B model despite being larger."
    )
    rows = []

    for name in names:
        t = cfg.tier(name)
        console.print(f"[bold]{name}[/] — {t.model}")
        try:
            import time

            rt.unload_all()
            vram_before = (gpu_memory_mb() or (0, 0))[0]
            t0 = time.monotonic()
            rt.ensure_loaded(t)
            load_s = time.monotonic() - t0
            vram_after = (gpu_memory_mb() or (0, 0))[0]

            completion = llm.ask(prompt, t, max_tokens=tokens)
            rows.append(
                {
                    "tier": name,
                    "model": t.model,
                    "context": t.context,
                    "load_s": round(load_s, 1),
                    "tok_per_s": round(completion.tok_per_s, 1),
                    "completion_tokens": completion.completion_tokens,
                    "duration_s": round(completion.duration_s, 1),
                    "vram_delta_mb": vram_after - vram_before,
                }
            )
            console.print(
                f"  loaded in {load_s:.0f}s | {completion.tok_per_s:.1f} tok/s | "
                f"VRAM +{vram_after - vram_before} MB"
            )
        except Exception as exc:  # noqa: BLE001 - a failing tier is a result, not a crash
            console.print(f"  [red]failed:[/] {exc}")
            rows.append({"tier": name, "model": t.model, "error": str(exc)})

    cfg.paths.bench.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.bench / "results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    table = Table(title="benchmark", header_style="bold")
    for col in ("tier", "model", "load s", "tok/s", "VRAM MB", "verdict"):
        table.add_column(col)
    for r in rows:
        if "error" in r:
            table.add_row(r["tier"], r["model"], "-", f"[red]{r['error'][:40]}[/]", "-", "[red]unusable[/]")
        else:
            colour, verdict = speed_verdict(r["tok_per_s"])
            r["verdict"] = verdict
            table.add_row(
                r["tier"], r["model"], str(r["load_s"]), str(r["tok_per_s"]),
                str(r["vram_delta_mb"]), f"[{colour}]{verdict}[/]",
            )
    console.print(table)
    console.print(
        "\n[dim]Thresholds are derived from measurement, not taste: prefill runs at "
        "~4,900 tok/s and generation at ~100 tok/s on this box, so a 120,000-token "
        "express call (the cap in mcp_server.py) is generation-bound. At 60 tok/s that "
        "call takes ~90 s; at 30 tok/s, ~3 min; at 12 tok/s, ~6 min.[/dim]"
    )
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    console.print(f"written to {out}")


# --------------------------------------------------------------- express lane


def _run_now(kind: str, spec: dict, tier: str | None) -> None:
    from .guard import GuardError

    cfg = get_config()
    store = _store()
    try:
        with console.status(f"running {kind} locally..."):
            result, chosen = execute(
                kind=kind,
                spec=spec,
                store=store,
                cfg=cfg,
                tier_name=tier,
                source="cli",
                on_progress=lambda m: console.print(f"  [dim]{m}[/dim]"),
            )
    except (GuardError, FileNotFoundError, ValueError) as exc:
        # A bad path or a malformed request is a user error, not a crash.
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(Panel(result.summary, title=f"{kind} — {chosen.name}", border_style="green"))
    console.print(
        f"[dim]{_fmt(result.input_tokens)} source tokens -> {_fmt(result.output_tokens)} returned "
        f"({result.compression}x, {result.duration_s:.0f}s)[/dim]"
    )
    if result.outbox_path:
        console.print(f"[dim]full detail: {result.outbox_path}[/dim]")


@app.command()
def digest(
    path: list[str] = typer.Option(..., "--path", "-p", help="File, directory or glob. Repeatable."),
    question: str = typer.Option(..., "--question", "-q"),
    tier: Optional[str] = typer.Option(None, "--tier"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
) -> None:
    """Answer a question about files without reading them yourself."""
    spec = {"paths": list(path), "question": question}
    if max_tokens:
        spec["max_out_tokens"] = max_tokens
    _run_now("digest", spec, tier)


@app.command()
def grep(
    path: list[str] = typer.Option(..., "--path", "-p"),
    query: str = typer.Option(..., "--query", "-q"),
    tier: Optional[str] = typer.Option(None, "--tier"),
    max_hits: int = typer.Option(40, "--max-hits"),
) -> None:
    """Search, then return only the hits that actually answer the question."""
    _run_now("grep", {"paths": list(path), "query": query, "max_hits": max_hits}, tier)


@app.command()
def ask(
    prompt: str = typer.Argument(...),
    tier: Optional[str] = typer.Option(None, "--tier"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
) -> None:
    """One-shot generation on the local model."""
    spec = {"prompt": prompt}
    if max_tokens:
        spec["max_out_tokens"] = max_tokens
    _run_now("ask", spec, tier)


@app.command()
def transform(
    path: list[str] = typer.Option(..., "--path", "-p"),
    instruction: str = typer.Option(..., "--instruction", "-i"),
    suffix: str = typer.Option(".md", "--suffix", help="Extension for output files."),
    tier: Optional[str] = typer.Option(None, "--tier"),
) -> None:
    """Apply an instruction to each file; results land in the outbox."""
    _run_now(
        "transform", {"paths": list(path), "instruction": instruction, "suffix": suffix}, tier
    )


@app.command()
def review(
    path: list[str] = typer.Option([], "--path", "-p"),
    diff_file: Optional[Path] = typer.Option(None, "--diff", help="File containing a diff."),
    focus: Optional[str] = typer.Option(None, "--focus"),
    tier: Optional[str] = typer.Option(None, "--tier"),
) -> None:
    """First-pass defect review. Candidates only — verify before acting."""
    spec: dict = {"focus": focus or ""}
    if diff_file:
        spec["diff"] = diff_file.read_text(encoding="utf-8")
    if path:
        spec["paths"] = list(path)
    if not spec.get("diff") and not spec.get("paths"):
        err.print("[red]give --path or --diff[/red]")
        raise typer.Exit(1)
    _run_now("review", spec, tier)


@app.command()
def transcribe(
    path: str = typer.Argument(..., help="Audio or video file."),
    model_size: str = typer.Option("small", "--model-size", help="tiny|base|small|medium|large"),
    language: Optional[str] = typer.Option(None, "--language"),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Keep the raw transcript only."),
    tier: Optional[str] = typer.Option(None, "--tier"),
) -> None:
    """Transcribe audio locally, then tidy the transcript with the local model."""
    _run_now(
        "transcribe",
        {
            "path": path,
            "model_size": model_size,
            "language": language,
            "cleanup": not no_cleanup,
        },
        tier,
    )


# ----------------------------------------------------------------- queue lane


@app.command()
def ab(
    tiers: str = typer.Option(..., "--tiers", help="Comma-separated tiers to compare."),
    path: list[str] = typer.Option(..., "--path", "-p"),
    question: str = typer.Option(..., "--question", "-q"),
    kind: str = typer.Option("digest", "--kind"),
) -> None:
    """Run the same job on several tiers and print the answers side by side.

    Model comparisons are worth almost nothing on generic benchmarks and quite a
    lot on the job you actually run, so this replays one real question and shows
    what each tier costs and says. Judging which answer is *right* is left to you
    — that part does not automate honestly.
    """
    from .runtime import Runtime

    cfg = get_config()
    store = _store()
    rt = Runtime(cfg)
    names = [t.strip() for t in tiers.split(",") if t.strip()]
    rows = []

    for name in names:
        try:
            tier = cfg.tier(name)
        except ConfigError as exc:
            err.print(f"[red]{exc}[/red]")
            continue
        if rt.find(tier) is None:
            err.print(f"[yellow]{name}: model not downloaded ({tier.model}) — skipped[/]")
            continue

        console.print(f"\n[bold]{name}[/] — {tier.model}")
        spec = {"paths": list(path), "question": question}
        try:
            result, _ = execute(
                kind=kind, spec=spec, store=store, cfg=cfg, tier_name=name,
                source="ab", runtime=rt,  # measurement, not work anyone needed
                on_progress=lambda m: None,
            )
        except Exception as exc:  # noqa: BLE001 - a failing tier is a result
            err.print(f"  [red]failed: {exc}[/red]")
            rows.append({"tier": name, "model": tier.model, "error": str(exc)[:80]})
            continue

        rows.append({
            "tier": name,
            "model": tier.model,
            "seconds": round(result.duration_s, 1),
            "out_tokens": result.output_tokens,
            "compression": result.compression,
            "answer": result.summary,
        })
        console.print(Panel(result.summary, border_style="cyan"))
        console.print(f"[dim]{result.duration_s:.0f}s  {result.output_tokens} tokens out[/dim]")

    table = Table(title="A/B comparison", header_style="bold")
    for col in ("tier", "model", "seconds", "out tokens", "compression"):
        table.add_column(col)
    for r in rows:
        if "error" in r:
            table.add_row(r["tier"], r["model"], "[red]failed[/]", r["error"][:30], "-")
        else:
            table.add_row(
                r["tier"], r["model"], str(r["seconds"]),
                str(r["out_tokens"]), f"{r['compression']}x",
            )
    console.print(table)

    cfg.paths.bench.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.bench / "ab.json"
    out.write_text(json.dumps({"question": question, "paths": list(path), "results": rows}, indent=2), encoding="utf-8")
    console.print(f"[dim]written to {out}[/dim]")


@app.command()
def submit(
    kind: str = typer.Argument(..., help="digest | grep | ask"),
    path: list[str] = typer.Option([], "--path", "-p"),
    question: Optional[str] = typer.Option(None, "--question", "-q"),
    prompt: Optional[str] = typer.Option(None, "--prompt"),
    spec_json: Optional[str] = typer.Option(None, "--spec", help="Raw JSON spec."),
    tier: Optional[str] = typer.Option(None, "--tier"),
    priority: int = typer.Option(Priority.NORMAL, "--priority", help="0 highest, 8 lowest."),
) -> None:
    """Queue a job for the background worker."""
    spec: dict = json.loads(spec_json) if spec_json else {}
    if path:
        spec["paths"] = list(path)
    if question:
        spec["question"] = question
        spec.setdefault("query", question)
    if prompt:
        spec["prompt"] = prompt

    label = question or prompt or ", ".join(path) or kind
    job_id = _store().submit(kind, spec, tier=tier, priority=priority, label=label)
    console.print(f"queued [bold]{job_id}[/] ({kind}, priority {priority})")
    console.print(f"[dim]llmh status {job_id[:8]}[/dim]")


def _resolve_id(store: Store, prefix: str) -> str:
    if len(prefix) >= 36:
        return prefix
    matches = [j.id for j in store.list_jobs(limit=500) if j.id.startswith(prefix)]
    if not matches:
        err.print(f"[red]no job starting with {prefix!r}[/red]")
        raise typer.Exit(1)
    if len(matches) > 1:
        err.print(f"[red]{prefix!r} is ambiguous ({len(matches)} jobs)[/red]")
        raise typer.Exit(1)
    return matches[0]


@app.command()
def status(
    job_id: Optional[str] = typer.Argument(None),
    state: Optional[str] = typer.Option(None, "--state"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Show queue state, or one job in detail."""
    store = _store()
    if job_id:
        job = store.get(_resolve_id(store, job_id))
        console.print_json(json.dumps(job.as_dict(), indent=2))
        return

    counts = store.counts()
    console.print(" ".join(f"[bold]{k}[/]={v}" for k, v in counts.items()) or "queue empty")

    table = Table(header_style="bold")
    for col in ("id", "kind", "state", "pri", "label", "created"):
        table.add_column(col)
    for job in store.list_jobs(state=state, limit=limit):
        colour = {"done": "green", "failed": "red", "running": "yellow"}.get(job.state, "white")
        table.add_row(
            job.id[:8],
            job.kind,
            f"[{colour}]{job.state}[/]",
            str(job.priority),
            (job.label or "")[:50],
            job.created_at.strftime("%m-%d %H:%M") if job.created_at else "",
        )
    console.print(table)


@app.command()
def result(
    job_id: str = typer.Argument(...),
    full: bool = typer.Option(False, "--full", help="Print the whole outbox file."),
) -> None:
    """Fetch a finished job's result."""
    store = _store()
    resolved = _resolve_id(store, job_id)
    job = store.get(resolved)
    res = store.result(resolved)

    if res is None:
        console.print(f"[yellow]{job.state}[/] — no result yet")
        if job.error:
            console.print(f"[red]{job.error}[/red]")
        raise typer.Exit(0 if job.state != JobState.FAILED else 1)

    if full and res.path and Path(res.path).is_file():
        console.print(Path(res.path).read_text(encoding="utf-8"))
    else:
        console.print(Panel(res.summary, title=f"{job.kind} — {job.label[:60]}", border_style="green"))
        if res.meta:
            # Only the headline numbers: dumping raw meta buries the answer under
            # a list of every skipped file.
            interesting = ("files", "chunks", "strategy", "tokens_saved", "duration_s", "tier")
            line = "  ".join(
                f"{k}={res.meta[k]}" for k in interesting if k in res.meta
            )
            skipped = res.meta.get("skipped") or []
            if skipped:
                line += f"  skipped={len(skipped)}"
            console.print(f"[dim]{line}[/dim]")
        if res.path:
            console.print(f"[dim]full detail: {res.path}[/dim]")


@app.command()
def cancel(job_id: str = typer.Argument(...)) -> None:
    """Cancel a queued job."""
    store = _store()
    ok = store.cancel(_resolve_id(store, job_id))
    console.print("cancelled" if ok else "[yellow]already finished[/]")


@app.command()
def worker(
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    once: bool = typer.Option(False, "--once", help="Drain the queue and exit."),
) -> None:
    """Run the background worker."""
    from .lock import AlreadyRunning, single_instance
    from .worker import Worker, setup_logging

    cfg = get_config()
    setup_logging(cfg, verbose=False)
    w = Worker(cfg, on_event=lambda m: console.print(f"[dim]{m}[/dim]") if verbose else None)

    from .runtime import Runtime

    if not Runtime(cfg).ensure_server():
        err.print("[red]LM Studio server did not come up — run `llmh doctor --fix`[/red]")
        raise typer.Exit(1)

    if once:
        n = 0
        while w.run_one():
            n += 1
        console.print(f"drained {n} job(s)")
        return

    try:
        with single_instance(cfg, "worker"):
            w.run_forever()
    except AlreadyRunning as exc:
        err.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("stopped")


# ---------------------------------------------------------------------- stats


@app.command()
def stats(
    days: Optional[int] = typer.Option(None, "--days"),
    all_sources: bool = typer.Option(
        False, "--all-sources", help="Include bench/ab measurement traffic."
    ),
) -> None:
    """How many tokens the local model has kept out of an expensive context.

    Measurement traffic (`llmh bench`, `llmh ab`) is excluded by default. It reads
    the same file dozens of times, which nobody would have done at full price, so
    counting it would flatter the number into meaninglessness.
    """
    store = _store()
    s = store.savings_summary(days=days, include_synthetic=all_sources)
    window = f"last {days} day(s)" if days else "all time"

    console.print(
        Panel(
            f"[bold green]{_fmt(s['tokens_saved'])}[/] tokens kept out of context\n"
            f"{_fmt(s['input_tokens'])} read locally -> {_fmt(s['output_tokens'])} returned "
            f"({s['compression']}x)\n"
            f"{s['calls']} call(s), {s['local_seconds']:.0f}s of local compute",
            title=f"savings — {window}",
            border_style="green",
        )
    )
    if s.get("by_source"):
        synthetic = sum(
            r["input_tokens"] for r in s["by_source"]
            if r["source"] in ("bench", "ab")
        )
        if synthetic and not all_sources:
            console.print(
                f"[dim]excluded {_fmt(synthetic)} tokens of measurement traffic "
                f"(bench/ab); --all-sources to include[/dim]"
            )
    if s.get("by_client"):
        console.print("\n[bold]which project called[/]")
        t = Table(header_style="bold")
        for col in ("project", "calls", "read", "saved", "last used"):
            t.add_column(col)
        for row in s["by_client"]:
            t.add_row(
                row["client"], str(row["calls"]), _fmt(row["input_tokens"]),
                _fmt(row["saved"]), (row["last"] or "")[:16].replace("T", " "),
            )
        console.print(t)
    elif not all_sources:
        console.print(
            "\n[dim]No MCP calls recorded yet from any project. Availability is not "
            "use: `claude mcp get local-llm` shows it is reachable, this table shows "
            "whether anything actually called it.[/dim]"
        )

    if s["by_kind"]:
        table = Table(header_style="bold")
        for col in ("kind", "calls", "read", "returned", "saved"):
            table.add_column(col)
        for row in s["by_kind"]:
            table.add_row(
                row["kind"],
                str(row["calls"]),
                _fmt(row["input_tokens"]),
                _fmt(row["output_tokens"]),
                _fmt(row["saved"]),
            )
        console.print(table)


@app.command()
def prune(
    days: Optional[int] = typer.Option(None, "--days", help="Defaults to config retention."),
    dry_run: bool = typer.Option(False, "--dry-run"),
    jobs: bool = typer.Option(True, "--jobs/--no-jobs", help="Also drop finished job rows."),
) -> None:
    """Delete old outbox files, and finished jobs of the same age.

    Job rows were never cleaned up, so the table only grew and the dashboard
    filled with week-old completed work. Only terminal jobs go; anything queued
    or running is somebody waiting. The savings ledger is never touched - it is
    the record of what was offloaded, and a cleanup that quietly shrank it would
    rewrite the one number worth reporting.
    """
    import time

    cfg = get_config()
    from .guard import iter_outbox

    cutoff = time.time() - (days or cfg.budgets.result_retention_days) * 86400
    removed = 0
    freed = 0
    for path in iter_outbox(cfg):
        if path.stat().st_mtime < cutoff:
            freed += path.stat().st_size
            removed += 1
            if not dry_run:
                path.unlink()
    verb = "would remove" if dry_run else "removed"
    console.print(f"{verb} {removed} outbox file(s), {freed / 1024:.0f} KB")

    if jobs:
        r = _store().prune_jobs(days or cfg.budgets.result_retention_days, dry_run=dry_run)
        console.print(
            f"{verb} {r['jobs']} finished job(s)"
            + (f" and {r['results']} stored result(s)" if r["results"] else "")
        )
        console.print("[dim]the savings ledger is unaffected[/dim]")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8770, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Serve the dashboard: status, usage, the queue and settings.

    Bound to localhost by default. The controls here start processes and load
    multi-gigabyte models, which is not something to expose to a network by
    accident — pass --host 0.0.0.0 only deliberately, and read the exposure note
    in the README first.
    """
    from .lock import AlreadyRunning, single_instance
    from .webui import serve

    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/"
    console.print(f"dashboard on [bold]{url}[/]")
    if host == "0.0.0.0":
        console.print(
            "[yellow]Bound to all interfaces. There is no authentication: anything "
            "that reaches this port can start the worker, load models and change "
            "settings.[/yellow]"
        )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        # Two dashboards are worse than confusing: one binds the port and the
        # other silently does not, so edits appear to have no effect while a
        # stale process keeps answering. That cost real debugging time here.
        with single_instance(get_config(), "webui"):
            serve(host=host, port=port)
    except AlreadyRunning as exc:
        err.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("stopped")


# ------------------------------------------------------------------- server

server_app = typer.Typer(
    no_args_is_help=True, help="Start, stop and inspect the LM Studio server."
)
app.add_typer(server_app, name="server")


def _runtime():
    from .runtime import Runtime

    return Runtime(get_config())


@server_app.command("start")
def server_start() -> None:
    """Start the model server if it is not already answering."""
    ok = _runtime().ensure_server()
    console.print("[green]server up[/]" if ok else "[red]server did not come up[/]")
    raise typer.Exit(0 if ok else 1)


@server_app.command("stop")
def server_stop() -> None:
    """Stop the model server, unloading any resident model first."""
    from .runtime import RuntimeError_

    try:
        ok = _runtime().stop_server()
    except RuntimeError_ as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print("[green]server stopped[/]" if ok else "[yellow]still answering[/]")
    raise typer.Exit(0 if ok else 1)


@server_app.command("restart")
def server_restart() -> None:
    """Stop then start. The resident model is unloaded on the way through."""
    from .runtime import RuntimeError_

    try:
        ok = _runtime().restart_server()
    except RuntimeError_ as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print("[green]server restarted[/]" if ok else "[red]did not come back[/]")
    raise typer.Exit(0 if ok else 1)


@server_app.command("status")
def server_status_cmd() -> None:
    """Whether the server answers, and what LM Studio thinks."""
    rt = _runtime()
    st = rt.server_status()
    console.print(
        f"answering: [{'green' if st['answering'] else 'red'}]{st['answering']}[/]"
    )
    if st["lms_says"]:
        console.print(f"lms says : {st['lms_says']}")
    loaded = [m.identifier for m in rt.loaded()] if rt.can_manage_models else []
    console.print(f"loaded   : {', '.join(loaded) if loaded else 'nothing'}")


@server_app.command("restart-worker")
def server_restart_worker() -> None:
    """Stop the worker and start a fresh one.

    Needed after changing this code or config: a worker loads the package once
    and keeps running whatever it started with.
    """
    import subprocess
    import sys
    import time

    from .lock import holder, pid_alive

    cfg = get_config()
    pid = holder(cfg, "worker")
    if pid:
        try:
            import psutil

            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=20)
            console.print(f"stopped worker pid {pid}")
        except Exception as exc:  # noqa: BLE001
            err.print(f"[red]could not stop pid {pid}: {exc}[/red]")
            raise typer.Exit(1)
    else:
        console.print("no worker was running")

    subprocess.Popen(
        [sys.executable, "-m", "llmhelper.cli", "worker", "--quiet"],
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        time.sleep(0.25)
        if (new := holder(cfg, "worker")) and pid_alive(new):
            console.print(f"[green]worker running as pid {new}[/]")
            return
    err.print("[yellow]worker started but took no lock — check logs/worker.log[/yellow]")


@app.command()
def connect() -> None:
    """Show what another machine needs to reach this one."""
    from .runtime import lan_addresses

    cfg = get_config()
    addrs = lan_addresses()
    best = next((a["ip"] for a in addrs if a.get("likely_lan")), None)
    reachable = cfg.server.bind in ("0.0.0.0", "::")

    console.print(
        Panel(
            f"Model API   [bold]http://{best or '<this-pc>'}:{cfg.server.port}/v1[/]\n"
            f"            for a machine running its own copy of this package\n\n"
            f"MCP / HTTP  [bold]http://{best or '<this-pc>'}:8765/mcp[/]\n"
            f"            for a machine calling this PC's tools "
            f"(start it with autostart/start-mcp-http.cmd)",
            title="connect another machine",
            border_style="green" if reachable else "yellow",
        )
    )

    if not reachable:
        err.print(
            f"[yellow]LM Studio is bound to {cfg.server.bind} — localhost only. "
            'Set server.bind = "0.0.0.0" in config.toml and run `llmh server '
            "restart`.[/yellow]"
        )

    table = Table(header_style="bold")
    for col in ("address", "interface", ""):
        table.add_column(col)
    for a in addrs:
        if not a["usable"]:
            continue
        tag = (
            "[green]likely[/]" if a.get("likely_lan")
            else ("[dim]virtual[/]" if a["virtual"] else "")
        )
        table.add_row(a["ip"], a["interface"], tag)
    console.print(table)

    if addrs and addrs[0].get("ambiguous"):
        console.print(
            "[dim]Several addresses are plausible. Pick the one on the same subnet "
            "as the machine you are connecting from — the guess is only a guess.[/dim]"
        )
    console.print(
        "[dim]The firewall must also allow the port. A Public network profile "
        "blocks inbound by default.[/dim]"
    )


def main() -> None:
    try:
        app()
    except ConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
