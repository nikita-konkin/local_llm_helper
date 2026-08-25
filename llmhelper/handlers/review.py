"""First-pass code review on the local model.

Deliberately scoped to what a small model is actually good at: mechanical defects
that are visible within a single file — unhandled errors, resource leaks, obvious
off-by-ones, contradictions between a docstring and its code. It is a filter that
narrows where a careful reader looks, not a replacement for one.

The prompt pushes hard against the failure mode of small reviewer models, which
is inventing plausible-sounding issues to seem useful. Findings must quote the
line they are about, so anything hallucinated is cheap to disprove.
"""

from __future__ import annotations

from typing import Any

from ..chunk import chunk_files, count_tokens
from ..config import Tier
from ..guard import expand
from .base import DATA_RULE, Ctx, HandlerResult, fence

NONE_FOUND = "NO_ISSUES"

SYSTEM = f"""You review code for defects. You are strict, literal and quiet.

{DATA_RULE}

Report ONLY problems you can point at in the given lines:
- unhandled errors, swallowed exceptions, resources never closed
- off-by-one and boundary mistakes, wrong comparison operators
- values used before assignment, unreachable or dead code
- a docstring or comment that contradicts what the code does
- obviously incorrect logic

Do NOT report: style, formatting, naming, missing type hints, "consider adding a
test", or anything you would phrase as a suggestion rather than a defect.

Format each finding on its own, exactly:
[<line>] <severity: high|medium|low> — <what is wrong> — <the code, quoted>

If the excerpt has no defects, reply with exactly: {NONE_FOUND}
Inventing a finding is worse than reporting none."""

SUMMARY_SYSTEM = f"""You consolidate code-review findings.

{DATA_RULE}

Merge duplicates, drop anything that is a style preference rather than a defect,
and order by severity. Keep the file:line references exactly as given. Do not add
findings that are not in the input."""


class ReviewHandler:
    kind = "review"

    def run(self, spec: dict[str, Any], ctx: Ctx, tier: Tier) -> HandlerResult:
        diff = (spec.get("diff") or "").strip()
        paths = spec.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        focus = (spec.get("focus") or "").strip()
        if not diff and not paths:
            raise ValueError("review needs spec['paths'] or spec['diff']")

        focus_line = f"\nPay particular attention to: {focus}" if focus else ""
        findings: list[str] = []
        source_tokens = 0
        skipped: list[str] = []
        units = 0

        if diff:
            source_tokens = count_tokens(diff)
            units = 1
            ctx.progress("reviewing supplied diff")
            found = self._review_one(ctx, tier, diff, "the diff", focus_line)
            if found:
                findings.append(found)
        else:
            files, skipped = expand(paths, ctx.cfg)
            if not files:
                raise FileNotFoundError(
                    f"no readable files matched {paths!r}. " + "; ".join(skipped[:3])
                )
            chunks, notes, source_tokens = chunk_files(
                files,
                max_tokens=ctx.cfg.budgets.map_chunk_tokens,
                max_chunks=ctx.cfg.budgets.max_map_chunks,
            )
            skipped += notes
            units = len(chunks)
            for i, c in enumerate(chunks, start=1):
                ctx.progress(f"review {i}/{len(chunks)} — {c.citation}")
                body = "\n".join(
                    f"{c.start_line + n}: {line}" for n, line in enumerate(c.text.splitlines())
                )
                found = self._review_one(
                    ctx, tier, body, f"{c.path.name} lines {c.start_line}-{c.end_line}", focus_line
                )
                if found:
                    findings.append(f"{c.path}:\n{found}")

        if not findings:
            summary = f"No defects found in {units} unit(s) reviewed."
            return HandlerResult(
                summary=summary,
                detail=summary,
                meta={"findings": 0, "units": units, "skipped": skipped[:20]},
                input_tokens=source_tokens,
                output_tokens=count_tokens(summary),
                local_prompt_tokens=ctx.local_prompt_tokens,
                local_completion_tokens=ctx.local_completion_tokens,
                duration_s=ctx.elapsed,
            )

        raw = "\n\n".join(findings)
        ctx.progress(f"consolidating {len(findings)} finding block(s)")
        consolidated = ctx.generate(
            f"Findings from a first-pass review:\n{fence(raw)}\n\n"
            f"Consolidate them into a single ordered list.",
            tier,
            system=SUMMARY_SYSTEM,
            max_tokens=ctx.cfg.budgets.express_max_out_tokens,
        ).text

        header = (
            f"First-pass local review of {units} unit(s). "
            "These are candidates from a small model — verify each before acting.\n\n"
        )
        summary = header + consolidated

        return HandlerResult(
            summary=summary,
            detail=f"{header}## Consolidated\n\n{consolidated}\n\n## Raw findings\n\n{raw}",
            meta={"findings": len(findings), "units": units, "skipped": skipped[:20]},
            input_tokens=source_tokens,
            output_tokens=count_tokens(summary),
            local_prompt_tokens=ctx.local_prompt_tokens,
            local_completion_tokens=ctx.local_completion_tokens,
            duration_s=ctx.elapsed,
        )

    def _review_one(self, ctx: Ctx, tier: Tier, body: str, what: str, focus_line: str) -> str:
        text = ctx.generate(
            f"Review {what}.{focus_line}\n\n{fence(body)}",
            tier,
            system=SYSTEM,
            max_tokens=ctx.cfg.budgets.map_summary_tokens * 2,
            reasoning_effort="none",
        ).text.strip()
        if not text or text.upper().startswith(NONE_FOUND):
            return ""
        return text
