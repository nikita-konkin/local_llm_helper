"""Search that returns the hits that matter, not every hit.

A plain grep for a common word across a repo can return thousands of lines, and
reading them is exactly the kind of bulk cost this project exists to avoid. So:

  1. ask the local model for search patterns implied by the question
  2. scan for them in pure Python (guarded reads, no external tool to go missing)
  3. hand each hit's neighbourhood back to the local model to judge relevance
  4. return only the excerpts that survive, with file:line

Step 3 is what separates this from grep: 'where is the retry limit configured'
finds the line that sets it, not every line containing the word retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chunk import count_tokens
from ..config import Config, Tier
from ..guard import expand, read_text
from .base import DATA_RULE, Ctx, HandlerResult, fence

PATTERN_SYSTEM = """You turn a search request into literal search patterns.

Output 5 to 10 patterns, one per line, nothing else. No numbering, no explanation.

Rules, in order of importance:
1. Prefer SHORT ROOT WORDS over invented compound identifiers. For "maximum retry
   attempts" emit `retry`, `attempt`, `max_attempt` — NOT `maxRetryAttempts`.
   Long guessed names almost never exist verbatim; roots almost always do.
2. Emit the same concept in more than one naming convention when you are unsure
   which the codebase uses: `max_attempts`, `maxAttempts`, `max-attempts`.
3. Never emit ordinary English words that carry no meaning in code ("where",
   "the", "maximum" on its own, "number", "value").

Patterns are matched case-insensitively as substrings."""

# Words that would match everywhere and tell us nothing, so they are never used
# as a fallback pattern.
_STOPWORDS = frozenset(
    """where what which when how does done make does used using with from that this
       there here maximum minimum number value values thing things code file files
       function functions does about into over under been have has was were will
       should would could same each every some many much most least more less
       set get put run job jobs""".split()
)

TRIAGE_SYSTEM = f"""You judge whether a code or text excerpt actually answers a
search request.

{DATA_RULE}

Reply with exactly one of:
- "NO" if the excerpt does not answer the request.
- "YES: <one line saying what it shows>" if it does.

Be strict. A passing mention of a word is not an answer."""


def _is_yes(verdict: str) -> bool:
    """Was the triage verdict affirmative?

    The model is asked for "YES: ..." or "NO", and returns "**YES**: ...",
    "`YES`" or "Yes -" often enough that a bare startswith() check silently
    discards good hits. Strip decoration before deciding.
    """
    return verdict.strip().lstrip("*_`#>- ").upper().startswith("YES")


@dataclass
class Hit:
    path: Path
    line_no: int
    line: str
    context: str
    # How many distinct query patterns matched this line; used to rank within a
    # file before the hit budget is shared out.
    score: int = 1

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.line_no}"


class GrepHandler:
    kind = "grep"

    def run(self, spec: dict[str, Any], ctx: Ctx, tier: Tier) -> HandlerResult:
        paths = spec.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        query = (spec.get("query") or "").strip()
        if not paths or not query:
            raise ValueError("grep needs spec['paths'] and spec['query']")

        max_hits = int(spec.get("max_hits") or 40)
        context_lines = int(spec.get("context_lines") or 4)
        explicit = spec.get("patterns")

        files, skipped = expand(paths, ctx.cfg)
        if not files:
            raise FileNotFoundError(
                f"no readable files matched {paths!r}. " + ("; ".join(skipped) or "")
            )

        patterns = explicit or self._derive_patterns(ctx, tier, query)
        ctx.progress(f"scanning {len(files)} file(s) for {len(patterns)} pattern(s)")

        hits, scanned_tokens, unreadable, total_hits = self._scan(
            files, patterns, context_lines, max_hits, ctx.cfg
        )
        if not hits:
            return HandlerResult(
                summary=(
                    f"No matches for {query!r} in {len(files)} file(s).\n"
                    f"Patterns tried: {', '.join(patterns)}"
                ),
                detail="",
                meta={
                    "files": len(files),
                    "patterns": patterns,
                    "hits": 0,
                    "unreadable": unreadable[:20],
                },
                input_tokens=scanned_tokens,
                output_tokens=40,
                local_prompt_tokens=ctx.local_prompt_tokens,
                local_completion_tokens=ctx.local_completion_tokens,
                duration_s=ctx.elapsed,
            )

        ctx.progress(f"{len(hits)} raw hit(s); triaging")
        kept: list[tuple[Hit, str]] = []
        for i, hit in enumerate(hits, start=1):
            ctx.progress(f"triage {i}/{len(hits)} — {hit.citation}")
            verdict = ctx.generate(
                f"Search request: {query}\n\n"
                f"Excerpt from {hit.path} around line {hit.line_no}:\n"
                f"{fence(hit.context)}\n\n"
                f"Does this answer the request?",
                tier,
                system=TRIAGE_SYSTEM,
                max_tokens=120,
                reasoning_effort="none",  # a YES/NO verdict has no room for thinking
            ).text.strip()
            if _is_yes(verdict):
                kept.append((hit, verdict.split(":", 1)[-1].strip(" *`:")))

        if not kept:
            # Do NOT report this as "nothing answers the question". The triage pass
            # is a small model judging excerpts one at a time, and it over-rejects:
            # observed on tier fast, 40 real matches were all refused while the
            # answer sat in one of them. The hits are already in hand, so hand them
            # over unranked rather than turning a triage miss into a confident
            # negative. A search tool that silently drops the right answer is worse
            # than one that returns too much.
            shown = hits[:15]
            listing = "\n".join(
                f"- {h.citation}: {h.line.strip()[:160]}" for h in shown
            )
            more = f"\n(+{len(hits) - len(shown)} more)" if len(hits) > len(shown) else ""
            summary = (
                f"{len(hits)} textual match(es) for {query!r}. Relevance triage on "
                f"tier '{tier.name}' kept none of them - which is at least as likely "
                f"to be a triage miss as a real absence, so the raw matches follow "
                f"unranked. If this search must not miss, re-run with tier=\"deep\".\n\n"
                f"{listing}{more}\n\nPatterns tried: {', '.join(patterns)}"
            )
            detail = "\n\n".join(
                f"### {h.citation}\n```\n{h.context}\n```" for h in shown
            )
        else:
            lines = [f"{len(kept)} relevant match(es) for: {query}\n"]
            for hit, why in kept:
                lines.append(f"**{hit.citation}** — {why}")
                lines.append(f"```\n{hit.line.strip()[:300]}\n```")
            summary = "\n".join(lines)
            detail = "\n\n".join(
                f"### {hit.citation} — {why}\n```\n{hit.context}\n```" for hit, why in kept
            )

        return HandlerResult(
            summary=summary,
            detail=detail,
            meta={
                "files": len(files),
                "patterns": patterns,
                "raw_hits": len(hits),
                "total_hits": total_hits,
                "kept": len(kept),
                "skipped": skipped[:20],
                "unreadable": unreadable[:20],
            },
            input_tokens=scanned_tokens,
            output_tokens=count_tokens(summary),
            local_prompt_tokens=ctx.local_prompt_tokens,
            local_completion_tokens=ctx.local_completion_tokens,
            duration_s=ctx.elapsed,
        )

    # ------------------------------------------------------------- internals

    def _derive_patterns(self, ctx: Ctx, tier: Tier, query: str) -> list[str]:
        raw = ctx.generate(
            f"Search request: {query}",
            tier,
            system=PATTERN_SYSTEM,
            max_tokens=200,
            reasoning_effort="none",
        ).text
        patterns: list[str] = []
        for line in raw.splitlines():
            p = line.strip().strip("`-*0123456789. ").strip()
            if p and len(p) > 2 and p not in patterns:
                patterns.append(p)

        # Backstop against a bad generation: the query's own content words, minus
        # the ones that would match every line in the corpus.
        seen = {p.lower() for p in patterns}
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query):
            low = word.lower()
            if low in _STOPWORDS or low in seen or len(patterns) >= 12:
                continue
            patterns.append(word)
            seen.add(low)
        return patterns[:12]

    def _scan(
        self,
        files: list[Path],
        patterns: list[str],
        context_lines: int,
        max_hits: int,
        cfg: Config | None = None,
    ) -> tuple[list[Hit], int, list[str], int]:
        """Collect matches, giving every file a share of the budget.

        This used to stop walking the moment max_hits was reached, which meant the
        cap silently truncated the corpus in directory order: a broad pattern that
        matched often in cli.py exhausted all 40 slots before the walk ever opened
        guard.py, and the answer was reported as absent from files that were never
        read. Alphabetical position is not relevance.

        Now every file is scanned, and the budget is filled round-robin, so a file
        with one good match is represented before any file contributes a second.
        """
        compiled = []
        for pat in patterns:
            try:
                compiled.append(re.compile(pat, re.IGNORECASE))
            except re.error:
                compiled.append(re.compile(re.escape(pat), re.IGNORECASE))

        # Bounds the work on a pathological file without biasing between files.
        per_file_cap = max(max_hits, 200)

        found: list[list[Hit]] = []
        unreadable: list[str] = []
        scanned_tokens = 0
        for path in files:
            try:
                text = read_text(path, cfg=cfg)
            except Exception as exc:  # noqa: BLE001 - one bad file is not fatal
                # Recorded, not swallowed. A file that could not be read is a file
                # the answer might have been in, and a search that quietly narrows
                # its own corpus is the same failure as a triage that quietly drops
                # the right hit.
                unreadable.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            scanned_tokens += count_tokens(text)
            lines = text.splitlines()
            claimed: set[int] = set()
            in_file: list[Hit] = []
            for i, line in enumerate(lines):
                if len(in_file) >= per_file_cap:
                    break
                if i in claimed:
                    continue
                score = sum(1 for rx in compiled if rx.search(line))
                if not score:
                    continue
                lo = max(0, i - context_lines)
                hi = min(len(lines), i + context_lines + 1)
                claimed.update(range(lo, hi))  # don't re-report the same neighbourhood
                in_file.append(
                    Hit(
                        path=path,
                        line_no=i + 1,
                        line=line,
                        score=score,
                        context="\n".join(
                            f"{n + 1:>6}: {lines[n]}" for n in range(lo, hi)
                        ),
                    )
                )
            if in_file:
                # A line matching three of the query's patterns is a better bet
                # than one matching a single common word. Offer each file's
                # strongest lines first, so the round-robin below spends the
                # budget on candidates rather than on whatever came first.
                in_file.sort(key=lambda h: (-h.score, h.line_no))
                found.append(in_file)

        total = sum(len(f) for f in found)
        hits: list[Hit] = []
        depth = 0
        while len(hits) < max_hits and any(len(f) > depth for f in found):
            for f in found:
                if depth < len(f) and len(hits) < max_hits:
                    hits.append(f[depth])
            depth += 1

        hits.sort(key=lambda h: (str(h.path), h.line_no))
        return hits, scanned_tokens, unreadable, total
