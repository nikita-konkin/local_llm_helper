"""One-shot generation with no file access.

The cheap case: small text work that does not need the corpus. Kept as a handler
so it flows through the same accounting and the same queue as everything else.
"""

from __future__ import annotations

from typing import Any

from ..chunk import count_tokens
from ..config import Tier
from .base import DATA_RULE, Ctx, HandlerResult

SYSTEM = f"""You are a careful assistant doing focused, self-contained work.
Answer directly and concisely. If the request is ambiguous, state the assumption
you made and answer anyway rather than asking a question back.

{DATA_RULE}"""


class AskHandler:
    kind = "ask"

    def run(self, spec: dict[str, Any], ctx: Ctx, tier: Tier) -> HandlerResult:
        prompt = (spec.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("ask needs spec['prompt']")

        max_out = int(spec.get("max_out_tokens") or ctx.cfg.budgets.express_max_out_tokens)
        completion = ctx.generate(
            prompt,
            tier,
            system=spec.get("system") or SYSTEM,
            max_tokens=max_out,
        )

        text = completion.text
        if completion.truncated:
            text += f"\n\n[cut off at the {max_out}-token limit]"

        return HandlerResult(
            summary=text,
            detail=text,
            meta={"model": completion.model, "tok_per_s": round(completion.tok_per_s, 1)},
            # No corpus was read, so the only thing avoided is the prompt itself.
            input_tokens=count_tokens(prompt),
            output_tokens=count_tokens(text),
            local_prompt_tokens=ctx.local_prompt_tokens,
            local_completion_tokens=ctx.local_completion_tokens,
            duration_s=ctx.elapsed,
        )
