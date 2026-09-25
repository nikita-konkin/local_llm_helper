"""Find downloaded models that no tier uses, describe them, and add one as a tier.

A model downloaded in LM Studio is invisible to this project until a
``[tiers.*]`` block names it: jobs are routed by tier, never by whatever happens
to be on disk. That is deliberate - a 32 GB model silently picked up would spill
off the card and make every job an order of magnitude slower - but it meant that
adding a model was a matter of hand-writing TOML and knowing which of the tier
settings matter on a 16 GB card. This module does that part.

*Descriptions are computed, not generated.* The obvious move is to ask the local
model to describe each candidate. It is also the wrong one here: the facts are
sizes and ratios, and the one thing measured about the small models is that
numbers they *compute* rather than copy have been wrong. Rules over LM Studio's
own metadata are exact and cost nothing.

*Suggested settings encode what this project measured*, chiefly that forcing a
GPU layer count onto a model that does not fit collapses throughput (0.5 tok/s
against 17.5 with auto-fit), and that parallel slots split the context and
multiply the KV cache. Every suggestion is editable before it is written.

*Model metadata is untrusted.* A model's display name is whatever its uploader
chose, and it ends up in config.toml as a comment. It is flattened to one line
and stripped of control characters, so a name containing a newline cannot write
config of its own. Values go through a string encoder, never interpolation.

*Writing is append-only and validated.* The new block goes at the end of the
file, so every hand-written comment above it survives byte for byte. The result
must parse, and the new tier must actually resolve to the chosen model - a tier
that silently matches nothing would look added and never run. Either failure
restores the original file.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .config import Config, Tier, config_path, invalidate, load_config
from .runtime import ModelInfo

# Bare TOML keys, and short enough to type as tier="..." in a tool call.
TIER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

# What the card needs beyond the weights for a model to count as fitting: the
# CUDA context, the desktop, and a KV cache for a useful context window.
FIT_HEADROOM_GB = 4.0
# Below this much spare, a model loads but only with a small context and one slot.
TIGHT_HEADROOM_GB = 0.5

# Settings a caller may override. Everything else in a tier block is derived.
OVERRIDABLE = {
    "context": int,
    "parallel": int,
    "gpu_ratio": float,
    "min_free_ram_gb": float,
    "temperature": float,
    "reasoning_effort": str,
}

_MOE = re.compile(r"(\d+(?:\.\d+)?)B-A(\d+(?:\.\d+)?)B", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


class ModelsError(ValueError):
    """The requested model cannot be added as a tier, with a reason to show."""


@dataclass
class Candidate:
    """One downloaded model, as someone choosing a tier needs to see it."""

    key: str
    display_name: str
    params: str
    arch: str
    quant: str
    size_gb: float
    max_context: int
    kind: str
    vision: bool
    tool_use: bool
    moe: bool
    fit: str                      # "fits" | "tight" | "spills" | "unknown" | "n/a"
    description: str
    tiers: list[str] = field(default_factory=list)   # tiers already using it
    suggested_name: str = ""
    suggested: dict[str, Any] = field(default_factory=dict)

    @property
    def addable(self) -> bool:
        return self.kind == "llm"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["addable"] = self.addable
        return d


# ----------------------------------------------------------------- analysis


def is_moe(info: ModelInfo) -> bool:
    return bool(_MOE.search(info.params or ""))


def fit(info: ModelInfo, vram_gb: float | None) -> str:
    if info.kind != "llm":
        return "n/a"
    if not vram_gb or not info.size_gb:
        return "unknown"
    if info.size_gb <= vram_gb - FIT_HEADROOM_GB:
        return "fits"
    if info.size_gb <= vram_gb - TIGHT_HEADROOM_GB:
        return "tight"
    return "spills"


def _clean(text: str) -> str:
    """One line, no control characters: safe to place in a TOML comment."""
    return _CONTROL.sub(" ", text or "").strip()


def describe(info: ModelInfo, vram_gb: float | None) -> str:
    """Two or three plain sentences, from metadata alone."""
    if info.kind != "llm":
        return "Embedding model: turns text into vectors. Not usable as a chat tier."

    name = _clean(info.display_name) or info.key
    shape = " ".join(p for p in (info.params, info.arch) if p)
    moe = _MOE.search(info.params or "")
    if moe:
        shape += f", mixture-of-experts ({moe.group(2)}B active)"
    parts = [f"{name}: {shape or 'unknown size'}, {info.quant or '?'}, {info.size_gb:g} GB."]

    verdict = fit(info, vram_gb)
    if verdict == "fits":
        parts.append(
            f"Fits the {vram_gb:g} GB card with room for context, so it runs entirely on the GPU."
        )
    elif verdict == "tight":
        parts.append(
            f"Only just fits the {vram_gb:g} GB card: it needs one slot and a modest "
            "context to stay on the GPU."
        )
    elif verdict == "spills":
        ratio = info.size_gb / vram_gb
        if moe:
            parts.append(
                f"{ratio:.1f}x the card, but only a few experts run per token, so it "
                "tolerates spilling into system RAM far better than a dense model."
            )
        else:
            parts.append(
                f"{ratio:.1f}x the card and dense: most layers will run from system RAM, "
                "so expect it to be several times slower than a model that fits."
            )

    # A high-precision copy that spills is usually the wrong download rather than
    # the wrong model: the same weights at 4 bits might fit.
    if verdict in ("tight", "spills") and info.quant_bits >= 8 and vram_gb:
        four_bit = info.size_gb * 4.8 / (info.quant_bits + 0.5)
        if four_bit <= vram_gb - TIGHT_HEADROOM_GB:
            parts.append(
                f"A 4-bit copy of the same model would be roughly {four_bit:.0f} GB "
                "and could fit the card."
            )

    extras = []
    if info.vision:
        extras.append("vision")
    if info.tool_use:
        extras.append("tool use")
    if info.max_context:
        extras.append(f"context up to {info.max_context // 1024}k")
    if extras:
        parts.append(extras[0].capitalize() + "".join(f", {e}" for e in extras[1:]) + ".")
    return " ".join(parts)


def suggest_settings(info: ModelInfo, vram_gb: float | None) -> dict[str, Any]:
    """Tier settings that are safe on this card, following what was measured."""
    verdict = fit(info, vram_gb)
    ceiling = info.max_context or 32768
    if verdict == "fits":
        context, parallel, floor = 32768, 0, 8.0
    elif verdict == "tight":
        # The deep tier's measurement: ctx 16384 at parallel=1 kept a 14 GB model
        # on a 16 GB card; ctx 32768 pushed layers back onto the CPU.
        context, parallel, floor = 16384, 1, 12.0
    else:
        context, parallel, floor = 8192, 1, 8.0

    return {
        "quant": info.quant,
        "context": min(context, ceiling),
        # -1 = let llama.cpp fit it. Forcing a layer count onto a model that does
        # not fit was measured at 0.5 tok/s against 17.5 with auto-fit.
        "gpu_ratio": -1.0,
        "parallel": parallel,
        # A floor only: Runtime.required_ram_gb already insists on size x headroom.
        "min_free_ram_gb": max(floor, float(math.ceil(info.size_gb * 1.15)))
        if verdict == "spills"
        else floor,
        "temperature": 0.2,
        # Every LLM tier here disables thinking: an extraction call with a small
        # budget can otherwise spend all of it reasoning and return nothing.
        "reasoning_effort": "none",
    }


def suggest_name(info: ModelInfo, taken: set[str]) -> str:
    base = info.key.split("@", 1)[0].rsplit("/", 1)[-1].lower()
    words = [w for w in re.split(r"[^a-z0-9]+", base) if w]
    name = "-".join(words[:4])[:32].strip("-") or "model"
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f"{name}-{n}", n + 1
    return candidate


def vram_total_gb() -> float | None:
    from .runtime import gpu_memory_mb

    mem = gpu_memory_mb()
    return round(mem[1] / 1024, 1) if mem else None


# --------------------------------------------------------------------- scan


def scan(cfg: Config, runtime: Any, vram_gb: float | None = None, refresh: bool = True) -> list[Candidate]:
    """Every downloaded model, marked with the tiers (if any) that use it.

    ``vram_gb`` is looked up when not given; pass it in tests.
    """
    if vram_gb is None:
        vram_gb = vram_total_gb()
    taken = set(cfg.tiers)
    out: list[Candidate] = []
    for info in runtime.catalog(refresh=refresh):
        users = [name for name, tier in cfg.tiers.items() if info.matches(tier)]
        out.append(
            Candidate(
                key=info.key,
                display_name=_clean(info.display_name),
                params=info.params,
                arch=info.arch,
                quant=info.quant,
                size_gb=info.size_gb,
                max_context=info.max_context,
                kind=info.kind,
                vision=info.vision,
                tool_use=info.tool_use,
                moe=is_moe(info),
                fit=fit(info, vram_gb),
                description=describe(info, vram_gb),
                tiers=users,
                suggested_name="" if users or info.kind != "llm" else suggest_name(info, taken),
                suggested=suggest_settings(info, vram_gb) if info.kind == "llm" else {},
            )
        )
    # Unused models first: they are the reason anyone opens this list.
    out.sort(key=lambda c: (bool(c.tiers), not c.addable, c.size_gb))
    return out


# ---------------------------------------------------------------------- add


def _lookup(runtime: Any, key: str) -> ModelInfo:
    wanted = key.strip().lower()
    catalog = runtime.catalog(refresh=True)
    for info in catalog:
        if info.key.lower() == wanted or info.key.split("@", 1)[0].lower() == wanted:
            return info
    known = ", ".join(m.key for m in catalog) or "none"
    raise ModelsError(f"{key!r} is not a downloaded model. On disk: {known}")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    # JSON string escaping is a subset of TOML basic-string escaping.
    return json.dumps(str(value))


def render_block(name: str, info: ModelInfo, settings: dict[str, Any], description: str, today: date | None = None) -> str:
    today = today or date.today()
    comment = textwrap.wrap(_clean(description), width=78) or [""]
    lines = [f"# --- added by `llmh models add` on {today.isoformat()} ---"]
    lines += [f"# {c}" for c in comment]
    lines.append(f"[tiers.{name}]")
    rows = {"model": info.key, **settings}
    if not rows.get("parallel"):
        rows.pop("parallel", None)  # 0 means "leave LM Studio's default alone"
    width = max(len(k) for k in rows)
    lines += [f"{k:<{width}} = {_toml_value(v)}" for k, v in rows.items()]
    return "\n".join(lines) + "\n"


@dataclass
class AddResult:
    name: str
    model: str
    path: Path
    block: str
    fit: str


def add_tier(
    key: str,
    name: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    runtime: Any = None,
    cfg_path: Path | None = None,
    vram_gb: float | None = None,
    dry_run: bool = False,
    today: date | None = None,
) -> AddResult:
    """Append a ``[tiers.<name>]`` block for a downloaded model.

    ``dry_run`` builds and validates the block without writing it, so a caller
    can show exactly what will be added before asking to confirm.
    """
    path = Path(cfg_path) if cfg_path else config_path()
    cfg = load_config(path)
    if runtime is None:
        from .runtime import Runtime

        runtime = Runtime(cfg)
    if vram_gb is None:
        vram_gb = vram_total_gb()

    info = _lookup(runtime, key)
    if info.kind != "llm":
        raise ModelsError(f"{info.key} is an {info.kind} model; tiers are for chat models")

    name = (name or "").strip() or suggest_name(info, set(cfg.tiers))
    if not TIER_NAME.match(name):
        raise ModelsError(
            f"tier name {name!r} must be 1-40 letters, digits, '-' or '_'"
        )
    if name in cfg.tiers:
        raise ModelsError(f"a tier named {name!r} already exists")

    settings = suggest_settings(info, vram_gb)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in OVERRIDABLE:
            raise ModelsError(f"{k!r} cannot be set here; allowed: {', '.join(OVERRIDABLE)}")
        try:
            settings[k] = OVERRIDABLE[k](v)
        except (TypeError, ValueError):
            raise ModelsError(f"{k}={v!r} is not a valid {OVERRIDABLE[k].__name__}") from None
    if settings["context"] < 512:
        raise ModelsError(f"context={settings['context']} is too small to be useful")
    if info.max_context and settings["context"] > info.max_context:
        raise ModelsError(
            f"context={settings['context']} is over this model's maximum of {info.max_context}"
        )
    if settings.get("parallel", 0) < 0:
        raise ModelsError("parallel cannot be negative")

    block = render_block(name, info, settings, describe(info, vram_gb), today=today)
    result = AddResult(name=name, model=info.key, path=path, block=block, fit=fit(info, vram_gb))

    original = path.read_text(encoding="utf-8")
    updated = original.rstrip("\n") + "\n\n" + block
    if dry_run:
        _validate(updated, path, name, info)
        return result

    backup = path.with_suffix(".toml.bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    try:
        _validate_file(path, name, info)
    except Exception:
        path.write_text(original, encoding="utf-8")
        raise
    invalidate()
    return result


def _check(cfg: Config, name: str, info: ModelInfo) -> None:
    tier: Tier = cfg.tier(name)
    if not info.matches(tier):
        raise ModelsError(
            f"the new tier would not resolve to {info.key} "
            f"(model {tier.model!r}, quant {tier.quant!r}); nothing was changed"
        )


def _validate_file(path: Path, name: str, info: ModelInfo) -> None:
    _check(load_config(path), name, info)


def _validate(text: str, path: Path, name: str, info: ModelInfo) -> None:
    """Validate candidate config text without touching the real file."""
    import tempfile

    # Same directory, so relative paths inside the config resolve identically.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".toml", dir=path.parent, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        _validate_file(tmp, name, info)
    finally:
        tmp.unlink(missing_ok=True)
