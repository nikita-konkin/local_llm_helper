"""Generate images locally instead of calling a paid image API.

The odd one out among the handlers, in a way worth being explicit about: it
saves no tokens. Every other kind absorbs raw material and hands back something
smaller, and the ledger measures exactly that. An image job absorbs a sentence
and produces a PNG, so it records ``input_tokens=0`` and claims no compression.

Inventing an equivalence here - "a 1024x1024 image is worth N tokens" - would put
a made-up number into the one table in this project that is supposed to be
measured. The savings ledger stays honest and the image is its own reward.

What it does share with the rest: the GPU lease, the outbox sandbox, the queue.
A batch of eight at 50 steps belongs on the worker, and gets there by the same
``local_submit`` route as anything else.
"""

from __future__ import annotations

from typing import Any

from ..chunk import count_tokens
from ..config import Tier
from ..imagegen import DEFAULT_NEGATIVE, ImageRequest, generate
from .base import Ctx, HandlerResult


class ImageHandler:
    kind = "image"

    def run(self, spec: dict[str, Any], ctx: Ctx, tier: Tier) -> HandlerResult:
        cfg_images = ctx.cfg.images
        prompt = (spec.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("image needs spec['prompt']")

        # An explicitly passed None means "no negative prompt", which is different
        # from omitting the key and getting the default.
        negative = spec["negative_prompt"] if "negative_prompt" in spec else DEFAULT_NEGATIVE

        req = ImageRequest(
            prompt=prompt,
            negative_prompt=negative or "",
            width=int(spec.get("width", cfg_images.width)),
            height=int(spec.get("height", cfg_images.height)),
            steps=int(spec.get("steps", cfg_images.steps)),
            guidance=float(spec.get("guidance", cfg_images.guidance)),
            seed=None if spec.get("seed") is None else int(spec["seed"]),
            count=int(spec.get("count", 1)),
            label=str(spec.get("label") or ""),
        )

        images = generate(req, cfg=ctx.cfg, on_progress=ctx.progress)

        listing = "\n".join(
            f"- {img.path}  (seed {img.seed}, {img.duration_s:.0f}s)" for img in images
        )
        summary = (
            f"Generated {len(images)} image(s) with {cfg_images.model} at "
            f"{req.width}x{req.height}, {req.steps} steps, guidance {req.guidance}.\n"
            f"{listing}\n\n"
            "Re-run any one of them exactly by passing its seed."
        )

        detail = "\n".join(
            [
                f"# Image generation - {req.label or prompt[:60]}\n",
                f"- model: {cfg_images.model}",
                f"- size: {req.width}x{req.height}",
                f"- steps: {req.steps}, guidance: {req.guidance}",
                f"- offload: {cfg_images.offload}",
                "",
                "## Prompt\n",
                prompt,
                "",
                "## Negative prompt\n",
                req.negative_prompt or "(none)",
                "",
                "## Files\n",
                listing,
            ]
        )

        return HandlerResult(
            summary=summary,
            detail=detail,
            meta={
                "model": cfg_images.model,
                "paths": [str(img.path) for img in images],
                "seeds": [img.seed for img in images],
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "guidance": req.guidance,
            },
            # Deliberately zero. See the module docstring.
            input_tokens=0,
            output_tokens=count_tokens(summary),
            duration_s=ctx.elapsed,
        )
