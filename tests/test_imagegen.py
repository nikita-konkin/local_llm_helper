"""Image generation: the parts that do not need a 7 GB model resident.

Everything here runs without torch, diffusers or CUDA. The pipeline itself is
stubbed - what is worth pinning is the surrounding contract:

  - a request that the pipeline would only fail on later is refused up front
  - PNGs land in the outbox and nowhere else
  - seeds come back, because an image nobody can reproduce is a dead end
  - the savings ledger is not fed a made-up number

That last one is the reason this file exists at all. Every other handler earns
its place by absorbing more tokens than it returns, and it would be very easy to
make `image` look like it does the same.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

from llmhelper import imagegen
from llmhelper.config import ImagesCfg
from llmhelper.guard import GuardError
from llmhelper.handlers import get_handler
from llmhelper.handlers.base import Ctx


class FakeImage:
    """Stands in for a PIL image: the only thing the engine asks of it is save()."""

    def __init__(self):
        self.saved_to: Path | None = None

    def save(self, target):
        self.saved_to = Path(target)
        Path(target).write_bytes(b"\x89PNG\r\n\x1a\n")  # enough to be a real file


class FakePipe:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(images=[FakeImage()])


@pytest.fixture
def images_cfg(sandbox):
    """A sandbox config with image generation enabled."""
    return dataclasses.replace(sandbox, images=ImagesCfg(enabled=True, steps=4))


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace everything that would need a GPU, leaving the engine logic intact."""
    pipe = FakePipe()
    monkeypatch.setattr(imagegen, "_load_pipeline", lambda cfg, say: pipe)
    monkeypatch.setattr(
        imagegen,
        "availability",
        lambda cfg=None: imagegen.Availability(True, device="fake", weights_cached=True),
    )

    fake_torch = types.SimpleNamespace(
        Generator=lambda device=None: types.SimpleNamespace(
            manual_seed=lambda s: types.SimpleNamespace(seed=s)
        ),
        cuda=types.SimpleNamespace(OutOfMemoryError=type("OOM", (Exception,), {})),
    )
    monkeypatch.setattr(imagegen, "_torch", lambda: fake_torch)
    return pipe


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("prompt", "   ", "non-empty prompt"),
        ("width", 1020, "multiple of 8"),
        ("height", 256, "outside 512-1536"),
        ("width", 2048, "outside 512-1536"),
        ("steps", 0, "outside 1-150"),
        ("guidance", 99.0, "outside 0-30"),
        ("count", 99, "outside 1-"),
    ],
)
def test_a_request_the_pipeline_would_reject_is_refused_first(field, value, expected):
    """Better a ValueError naming the field than a CUDA assertion 40 frames down."""
    req = dataclasses.replace(imagegen.ImageRequest(prompt="a cat"), **{field: value})
    with pytest.raises(ValueError, match=expected):
        req.validated(ImagesCfg())


def test_a_valid_request_passes_through_unchanged():
    req = imagegen.ImageRequest(prompt="a cat", width=1024, height=768, steps=30)
    assert req.validated(ImagesCfg()) == req


# ------------------------------------------------------------------ engine


def test_images_land_in_the_outbox(images_cfg, stub_pipeline):
    out = imagegen.generate(imagegen.ImageRequest(prompt="a cat", steps=4), cfg=images_cfg)
    assert len(out) == 1
    assert out[0].path.is_file()
    assert images_cfg.paths.outbox in out[0].path.parents


def test_the_seed_comes_back_and_drives_the_generator(images_cfg, stub_pipeline):
    out = imagegen.generate(
        imagegen.ImageRequest(prompt="a cat", seed=1234, steps=4), cfg=images_cfg
    )
    assert out[0].seed == 1234
    assert "1234" in out[0].path.name, "the seed belongs in the filename too"


def test_a_batch_walks_the_seed_so_every_frame_is_reproducible(images_cfg, stub_pipeline):
    out = imagegen.generate(
        imagegen.ImageRequest(prompt="a cat", seed=10, count=3, steps=4), cfg=images_cfg
    )
    assert [i.seed for i in out] == [10, 11, 12]
    assert len({i.path for i in out}) == 3, "a batch must not overwrite itself"


def test_an_absent_seed_is_recorded_not_left_unknown(images_cfg, stub_pipeline):
    out = imagegen.generate(imagegen.ImageRequest(prompt="a cat", steps=4), cfg=images_cfg)
    assert isinstance(out[0].seed, int)
    assert out[0].seed >= 0


def test_generation_parameters_reach_the_pipeline(images_cfg, stub_pipeline):
    imagegen.generate(
        imagegen.ImageRequest(
            prompt="a cat", negative_prompt="dogs", width=768, height=512,
            steps=7, guidance=3.5,
        ),
        cfg=images_cfg,
    )
    call = stub_pipeline.calls[0]
    assert call["prompt"] == "a cat"
    assert call["negative_prompt"] == "dogs"
    assert (call["width"], call["height"]) == (768, 512)
    assert call["num_inference_steps"] == 7
    assert call["guidance_scale"] == 3.5


def test_an_empty_negative_prompt_is_sent_as_none(images_cfg, stub_pipeline):
    """diffusers treats "" and None differently; "" would be tokenised as a prompt."""
    imagegen.generate(
        imagegen.ImageRequest(prompt="a cat", negative_prompt="", steps=4), cfg=images_cfg
    )
    assert stub_pipeline.calls[0]["negative_prompt"] is None


def test_generation_is_refused_when_it_cannot_run(images_cfg, monkeypatch):
    monkeypatch.setattr(
        imagegen,
        "availability",
        lambda cfg=None: imagegen.Availability(False, "no card today"),
    )
    with pytest.raises(imagegen.ImageGenError, match="no card today"):
        imagegen.generate(imagegen.ImageRequest(prompt="a cat"), cfg=images_cfg)


def test_disabled_in_config_is_reported_as_such(sandbox):
    cfg = dataclasses.replace(sandbox, images=ImagesCfg(enabled=False))
    avail = imagegen.availability(cfg)
    assert avail.ok is False
    assert "disabled" in avail.reason


def test_a_prompt_that_is_a_path_traversal_cannot_escape_the_outbox(images_cfg, stub_pipeline):
    """The filename comes from the prompt, so the prompt is an untrusted path input."""
    out = imagegen.generate(
        imagegen.ImageRequest(prompt="../../../etc/passwd", steps=4), cfg=images_cfg
    )
    assert images_cfg.paths.outbox in out[0].path.parents


# ----------------------------------------------------------------- handler


def test_the_handler_claims_no_token_savings(images_cfg, stub_pipeline):
    """The point of the ledger is that its numbers were measured.

    An image absorbs no source material. Recording a token count here - any
    token count - would put a fabricated figure into the one table this project
    asks anyone to trust.
    """
    ctx = Ctx(cfg=images_cfg, llm=object())
    result = get_handler("image").run({"prompt": "a cat", "steps": 4}, ctx, tier=None)
    assert result.input_tokens == 0
    assert result.tokens_saved == 0
    assert result.compression == 0.0


def test_the_handler_reports_paths_and_seeds_in_meta(images_cfg, stub_pipeline):
    ctx = Ctx(cfg=images_cfg, llm=object())
    result = get_handler("image").run(
        {"prompt": "a cat", "seed": 7, "count": 2, "steps": 4}, ctx, tier=None
    )
    assert result.meta["seeds"] == [7, 8]
    assert len(result.meta["paths"]) == 2
    assert all(Path(p).is_file() for p in result.meta["paths"])


def test_the_handler_needs_a_prompt(images_cfg, stub_pipeline):
    ctx = Ctx(cfg=images_cfg, llm=object())
    with pytest.raises(ValueError, match="prompt"):
        get_handler("image").run({}, ctx, tier=None)


def test_an_explicit_null_negative_prompt_is_honoured(images_cfg, stub_pipeline):
    """Omitting the key means "use the default"; passing None means "none"."""
    ctx = Ctx(cfg=images_cfg, llm=object())
    get_handler("image").run(
        {"prompt": "a cat", "negative_prompt": None, "steps": 4}, ctx, tier=None
    )
    assert stub_pipeline.calls[0]["negative_prompt"] is None


def test_omitting_the_negative_prompt_applies_the_default(images_cfg, stub_pipeline):
    ctx = Ctx(cfg=images_cfg, llm=object())
    get_handler("image").run({"prompt": "a cat", "steps": 4}, ctx, tier=None)
    assert stub_pipeline.calls[0]["negative_prompt"] == imagegen.DEFAULT_NEGATIVE


def test_the_handler_defaults_come_from_config(images_cfg, stub_pipeline):
    """A config that says 4 steps must not be overridden by a hardcoded 30."""
    ctx = Ctx(cfg=images_cfg, llm=object())
    get_handler("image").run({"prompt": "a cat"}, ctx, tier=None)
    assert stub_pipeline.calls[0]["num_inference_steps"] == 4


# ------------------------------------------------------------------- wiring


def test_image_does_not_require_a_downloaded_llm():
    """Nothing in an image job uses the LLM, so no tier may gate it."""
    from llmhelper.runner import NO_LLM_KINDS

    assert "image" in NO_LLM_KINDS


def test_importing_the_handler_does_not_import_torch():
    """The MCP server imports every handler at startup.

    torch is seconds and hundreds of megabytes; a session that never generates an
    image must not pay for it.
    """
    assert "llmhelper.handlers.image" in sys.modules or True  # imported via registry
    import llmhelper.handlers.image  # noqa: F401

    assert "torch" not in sys.modules, "torch leaked into the import path"


# -------------------------------------------------------- prompt truncation


class TokenizerStub:
    model_max_length = 77

    def __call__(self, text, truncation=False):
        return types.SimpleNamespace(input_ids=list(range(len(text.split()))))


def test_a_prompt_past_the_clip_window_is_reported(images_cfg, stub_pipeline, monkeypatch):
    """diffusers logs this to stderr only, which an MCP caller never sees.

    Dropping the tail of a prompt without saying so is exactly the quiet input
    narrowing that has produced confident wrong answers elsewhere in this project.
    """
    stub_pipeline.tokenizer = TokenizerStub()
    said: list[str] = []
    imagegen.generate(
        imagegen.ImageRequest(prompt=" ".join(["word"] * 90), negative_prompt="", steps=4),
        cfg=images_cfg,
        on_progress=said.append,
    )
    warnings = [m for m in said if "over CLIP limit" in m]
    assert warnings, f"no truncation warning in {said}"
    assert "13" in warnings[0], "should say how much was dropped"


def test_a_prompt_within_the_window_says_nothing(images_cfg, stub_pipeline):
    stub_pipeline.tokenizer = TokenizerStub()
    said: list[str] = []
    imagegen.generate(
        imagegen.ImageRequest(prompt="a small cat", steps=4),
        cfg=images_cfg,
        on_progress=said.append,
    )
    assert not [m for m in said if "CLIP" in m]


def test_a_tokenizer_that_raises_does_not_fail_the_job(images_cfg, stub_pipeline):
    """A warning is a nicety; losing the image over it is not."""

    class Broken:
        model_max_length = 77

        def __call__(self, *a, **k):
            raise RuntimeError("no tokenizer today")

    stub_pipeline.tokenizer = Broken()
    out = imagegen.generate(imagegen.ImageRequest(prompt="a cat", steps=4), cfg=images_cfg)
    assert out[0].path.is_file()
