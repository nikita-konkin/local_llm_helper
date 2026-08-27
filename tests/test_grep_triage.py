"""Search must never turn a triage miss into a confident negative.

Observed on tier fast: a search for where read roots are enforced produced 40
real textual matches, the triage pass rejected every one, and the tool reported
that none of them answered the question — while the answer sat in guard.py:95,
inside a hit the caller was never shown. The hits went into `detail`; the express
lane returns `summary`. The evidence was in hand and the summary contradicted it.
"""

from __future__ import annotations

from llmhelper.handlers.grep import GrepHandler, _is_yes


def test_decorated_affirmatives_are_not_read_as_rejections():
    """A bare startswith("YES") silently discarded formatted verdicts."""
    for verdict in (
        "YES: sets the limit",
        "**YES**: here",
        "`YES`",
        "Yes - this one",
        "  yes: indented",
        "> YES: quoted",
    ):
        assert _is_yes(verdict), verdict


def test_rejections_stay_rejections():
    for verdict in ("NO", "**NO**", "no, a passing mention", "No."):
        assert not _is_yes(verdict), verdict


def test_an_empty_verdict_is_not_affirmative():
    assert not _is_yes("")
    assert not _is_yes("   ")


class _Reply:
    def __init__(self, text: str):
        self.text = text


class _RejectingCtx:
    """A triage pass that refuses everything, which is the observed failure."""

    local_prompt_tokens = 0
    local_completion_tokens = 0
    elapsed = 0.1

    def __init__(self, cfg):
        self.cfg = cfg

    def progress(self, *a, **k):
        pass

    def generate(self, prompt, tier, **kw):
        system = kw.get("system") or ""
        if "search patterns" in system.lower():
            return _Reply("read_roots\nGuardError")
        return _Reply("NO")


def test_zero_kept_still_returns_the_raw_hits(sandbox):
    root = sandbox.paths.read_roots[0]
    (root / "project" / "guard.py").write_text(
        "def resolve_read(path):\n"
        "    if not any(under(t, r) for r in cfg.paths.read_roots):\n"
        "        raise GuardError('refusing to read outside the allowed roots')\n",
        encoding="utf-8",
    )

    res = GrepHandler().run(
        {"paths": [str(root)], "query": "where are read roots enforced"},
        _RejectingCtx(sandbox),
        sandbox.tiers["fast"],
    )

    assert res.meta["raw_hits"] > 0
    assert res.meta["kept"] == 0
    assert "guard.py" in res.summary, "hits must reach the summary, not only detail"
    assert "read_roots" in res.summary
    assert "deep" in res.summary, "should point at the tier that does not miss"
    assert "none of them actually answer" not in res.summary.lower()


def test_the_summary_does_not_claim_absence(sandbox):
    """The wording matters: this is what a caller acts on."""
    root = sandbox.paths.read_roots[0]
    (root / "project" / "conf.py").write_text(
        "read_roots = ['/tmp']\n", encoding="utf-8"
    )

    res = GrepHandler().run(
        {"paths": [str(root)], "query": "read roots"},
        _RejectingCtx(sandbox),
        sandbox.tiers["fast"],
    )
    low = res.summary.lower()
    assert "triage" in low, "the caller must be told why nothing was ranked"
    assert "as likely" in low or "miss" in low
