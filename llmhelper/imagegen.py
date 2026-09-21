"""Local image generation, on the same one card the LLM runs on.

This is the first capability here that is not a context saving. Everything else
in the project exists to keep tokens out of an expensive model's context; this
keeps an image out of a paid image API. The ledger reflects that honestly - an
image job records zero tokens saved rather than inventing an equivalence.

Three things about this machine shaped the design:

*One 16 GB card, already occupied.* SDXL in fp16 is about 7 GB of weights and the
`fast` LLM tier is 6.1 GB resident in LM Studio. Both at once leaves under 3 GB
for activations, the CUDA context and the desktop, which is where allocation
failures live. So by default the LLM is unloaded first and the pipeline runs
wholly on the GPU. Set `offload = true` instead to keep the LLM resident and page
SDXL submodules on and off the card: peak VRAM drops to roughly 4 GB and the two
coexist, at maybe half the speed.

*The weights are big and the import is slow.* torch and diffusers together cost
seconds and hundreds of megabytes to import, and the MCP server imports every
handler at startup. Nothing here is imported at module scope; a session that
never generates an image pays nothing.

*The card is shared.* Callers must hold the GPU lease. The handler runs inside
`runner.execute`, which already takes it, so this module does not take it itself -
two places claiming the same lease is how re-entrancy bugs start.

FLUX.1-schnell was the first choice and turned out to be gated behind an HF
token. SDXL base is OpenRAIL++-M, ungated, and native at 1024x1024.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Config, get_config
from .guard import resolve_write, slug

# Latent-space models work in multiples of 8; SDXL was trained at 1024x1024 and
# degrades visibly away from about a megapixel.
SIZE_STEP = 8
MIN_SIZE = 512
MAX_SIZE = 1536

# Deliberately short. A long boilerplate negative prompt is cargo cult - these
# are the terms that actually move flat vector work away from its failure modes.
DEFAULT_NEGATIVE = (
    "photo, photograph, 3d render, gradient, drop shadow, glow, bevel, texture, "
    "noise, grain, text, letters, words, watermark, signature, blurry, jpeg artifacts"
)


class ImageGenError(RuntimeError):
    """Generation could not run, with a reason worth showing the caller."""


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 7.0
    seed: int | None = None
    count: int = 1
    label: str = ""

    def validated(self, images_cfg: Any) -> "ImageRequest":
        """Reject what the pipeline would only fail on later, or silently round."""
        if not self.prompt.strip():
            raise ValueError("image generation needs a non-empty prompt")
        for name, value in (("width", self.width), ("height", self.height)):
            if not MIN_SIZE <= value <= MAX_SIZE:
                raise ValueError(f"{name}={value} is outside {MIN_SIZE}-{MAX_SIZE}")
            if value % SIZE_STEP:
                raise ValueError(f"{name}={value} must be a multiple of {SIZE_STEP}")
        if not 1 <= self.steps <= 150:
            raise ValueError(f"steps={self.steps} is outside 1-150")
        if not 0.0 <= self.guidance <= 30.0:
            raise ValueError(f"guidance={self.guidance} is outside 0-30")
        limit = int(getattr(images_cfg, "max_count", 4))
        if not 1 <= self.count <= limit:
            raise ValueError(f"count={self.count} is outside 1-{limit}")
        return self


@dataclass
class GeneratedImage:
    path: Path
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    duration_s: float = 0.0


@dataclass
class Availability:
    """Why image generation can or cannot run, in terms a caller can act on."""

    ok: bool
    reason: str = ""
    device: str = ""
    weights_cached: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# One pipeline per process, keyed by what would force a rebuild. Loading SDXL
# takes ~20 s from a warm page cache; a second generation must not pay it again.
_PIPE: Any = None
_PIPE_KEY: tuple | None = None


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise ImageGenError(
            "torch is not importable in this environment. Install it with CUDA "
            "support from the pytorch cu128 index."
        ) from exc
    return torch


def weights_present(model: str) -> bool:
    """True if the snapshot is already in the HF cache, without touching the network."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - any failure here means "not usable offline"
        return False


def availability(cfg: Config | None = None) -> Availability:
    """Can this machine generate an image right now, and if not, why not.

    Called by `llmh doctor` and by the handler before it commits to anything
    expensive, so every failure mode gets named once, here, rather than arriving
    as a traceback from somewhere inside diffusers.
    """
    cfg = cfg or get_config()
    images = cfg.images
    if not images.enabled:
        return Availability(
            False, "image generation is disabled in config.toml ([images] enabled = false)"
        )

    try:
        import diffusers  # noqa: F401
    except ImportError:
        return Availability(False, "diffusers is not installed in this environment")
    try:
        torch = _torch()
    except ImageGenError as exc:
        return Availability(False, str(exc))

    cached = weights_present(images.model)
    if not torch.cuda.is_available():
        # CPU generation is technically possible and practically useless: minutes
        # per image. Refuse rather than appear to hang.
        return Availability(
            False,
            "CUDA is not available to torch, and SDXL on CPU takes minutes per image",
            device="cpu",
            weights_cached=cached,
        )

    name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    if not cached:
        return Availability(
            False,
            f"the weights for {images.model} are not in the local HuggingFace cache; "
            f"run `llmh image --fetch` to download them",
            device=name,
            weights_cached=False,
            details={"vram_gb": vram_gb},
        )
    return Availability(
        True,
        device=name,
        weights_cached=True,
        details={"vram_gb": vram_gb, "model": images.model, "offload": images.offload},
    )


def _load_pipeline(cfg: Config, on_progress: Callable[[str], None]) -> Any:
    global _PIPE, _PIPE_KEY
    images = cfg.images
    key = (images.model, images.offload)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE

    torch = _torch()
    from diffusers import StableDiffusionXLPipeline

    on_progress(f"loading {images.model} (fp16)")
    started = time.monotonic()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        images.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        add_watermarker=False,
    )
    if images.offload:
        # Moves submodules on and off the GPU around each call. Must NOT be
        # combined with pipe.to("cuda") - accelerate installs hooks that a manual
        # .to() then fights with, and the symptom is a device-mismatch error
        # several layers deep.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    _PIPE, _PIPE_KEY = pipe, key
    on_progress(f"pipeline ready in {time.monotonic() - started:.0f}s")
    return pipe


def _tokens_over_limit(pipe: Any, text: str) -> int:
    """How many CLIP tokens past the 77-token window `text` runs, or 0.

    Asks the pipeline's own tokenizer rather than guessing from word count, so
    the number matches what actually gets dropped.
    """
    if not text:
        return 0
    try:
        tokenizer = pipe.tokenizer
        limit = tokenizer.model_max_length
        n = len(tokenizer(text, truncation=False).input_ids)
        return max(0, n - limit)
    except Exception:  # noqa: BLE001 - a warning that fails must not fail the job
        return 0


def release() -> None:
    """Drop the pipeline and give the VRAM back.

    Called when a process is done generating, for the same reason the worker
    unloads the LLM: a card holding 7 GB of weights nobody is using is a card the
    next job has to work around.
    """
    global _PIPE, _PIPE_KEY
    if _PIPE is None:
        return
    _PIPE, _PIPE_KEY = None, None
    try:
        import gc

        gc.collect()
        _torch().cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort; a failed cleanup is not an error
        pass


def generate(
    req: ImageRequest,
    cfg: Config | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[GeneratedImage]:
    """Render `req` and write the PNGs into the outbox.

    Callers must already hold the GPU lease. Returns one entry per image, each
    carrying the seed that produced it so any single result can be reproduced.
    """
    cfg = cfg or get_config()
    say = on_progress or (lambda _m: None)
    req = req.validated(cfg.images)

    avail = availability(cfg)
    if not avail.ok:
        raise ImageGenError(avail.reason)

    if not cfg.images.offload:
        # One card. The LLM is 6 GB of it, and nothing in this job needs the LLM.
        try:
            from .runtime import Runtime

            say("unloading the local LLM to free the card")
            Runtime(cfg).unload_all()
        except Exception as exc:  # noqa: BLE001 - not fatal, just tighter
            say(f"could not unload the LLM ({type(exc).__name__}); continuing")

    torch = _torch()
    pipe = _load_pipeline(cfg, say)

    for label, text in (("prompt", req.prompt), ("negative prompt", req.negative_prompt)):
        over = _tokens_over_limit(pipe, text)
        if over:
            # diffusers only logs this to stderr, which the MCP caller never sees.
            # Silently dropping the tail of a prompt is the kind of quiet input
            # narrowing that produces a confident wrong answer - say it out loud.
            say(f"warning: the {label} is {over} token(s) over CLIP limit; the tail was ignored")

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stem = slug(req.label or req.prompt, 40)
    out: list[GeneratedImage] = []

    for n in range(req.count):
        # An explicit seed applies to the first image; the rest walk forward from
        # it, so a batch is reproducible as a batch and each frame individually.
        seed = (req.seed + n) if req.seed is not None else random.randrange(2**31 - 1)
        say(
            f"generating {n + 1}/{req.count} - {req.width}x{req.height}, "
            f"{req.steps} steps, seed {seed}"
        )
        started = time.monotonic()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        try:
            image = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                generator=generator,
            ).images[0]
        except torch.cuda.OutOfMemoryError as exc:
            release()
            raise ImageGenError(
                f"out of VRAM at {req.width}x{req.height}. Set images.offload = true "
                f"in config.toml, or generate smaller. ({exc})"
            ) from exc

        target = resolve_write(f"images/{day}/{stem}-{seed}.png", cfg)
        image.save(target)
        out.append(
            GeneratedImage(
                path=target,
                seed=seed,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance=req.guidance,
                duration_s=round(time.monotonic() - started, 1),
            )
        )
        say(f"  wrote {target.name} in {out[-1].duration_s:.0f}s")

    return out
