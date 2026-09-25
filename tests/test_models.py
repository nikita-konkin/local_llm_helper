"""Scanning downloaded models and adding one as a tier.

The catalogue is faked; the config file is the sandbox's real one on disk,
because the thing most worth pinning is what happens to that file: comments
survive, the result parses, the new tier actually resolves to the chosen model,
and any failure leaves the original exactly as it was.
"""

from __future__ import annotations

import tomllib
from datetime import date

import pytest

from llmhelper import models
from llmhelper.config import load_config
from llmhelper.models import ModelsError, add_tier, describe, fit, scan, suggest_settings
from llmhelper.runtime import ModelInfo

VRAM = 15.9

SMALL = ModelInfo(key="acme/small-9b", quant="Q4_K_M", size_gb=6.1, max_context=262144,
                  display_name="Small 9B", params="9B", arch="qwen35", quant_bits=4,
                  vision=True, tool_use=True)
TIGHT = ModelInfo(key="tight-27b", quant="IQ4_XS", size_gb=13.3, max_context=262144,
                  params="27B", arch="qwen35", quant_bits=4)
DENSE_BIG = ModelInfo(key="big-27b-q8", quant="Q8_0", size_gb=29.9, max_context=262144,
                      params="27B", arch="qwen35", quant_bits=8)
MOE = ModelInfo(key="acme/moe-35b-a3b", quant="Q4_K_M", size_gb=20.6, max_context=262144,
                params="35B-A3B", arch="qwen35moe", quant_bits=4)
EMBED = ModelInfo(key="nomic-embed", size_gb=0.08, kind="embedding")
IN_USE = ModelInfo(key="test/model", quant="Q4_K_M", size_gb=5.0, max_context=32768)


class FakeRuntime:
    def __init__(self, *infos):
        self.infos = list(infos)

    def catalog(self, refresh: bool = False):
        return self.infos


@pytest.fixture
def cfg_file(sandbox, tmp_path):
    return tmp_path / "config.toml"


# ----------------------------------------------------------------- analysis


@pytest.mark.parametrize(
    "info,expected",
    [(SMALL, "fits"), (TIGHT, "tight"), (DENSE_BIG, "spills"), (MOE, "spills"), (EMBED, "n/a")],
)
def test_fit_matches_what_was_measured_on_this_card(info, expected):
    assert fit(info, VRAM) == expected


def test_fit_is_unknown_without_a_card_size():
    assert fit(SMALL, None) == "unknown"


def test_a_dense_spill_is_described_as_slow():
    text = describe(DENSE_BIG, VRAM)
    assert "dense" in text and "system RAM" in text


def test_a_moe_spill_is_described_as_tolerable():
    text = describe(MOE, VRAM)
    assert "mixture-of-experts" in text and "3B active" in text


def test_embeddings_are_not_offered_as_chat_tiers():
    assert "Not usable as a chat tier" in describe(EMBED, VRAM)


def test_capabilities_are_listed():
    text = describe(SMALL, VRAM)
    assert "Vision" in text and "tool use" in text and "256k" in text


def test_nothing_forces_a_gpu_layer_count():
    """Forcing layers onto a model that does not fit was measured at 0.5 tok/s."""
    for info in (SMALL, TIGHT, DENSE_BIG, MOE):
        assert suggest_settings(info, VRAM)["gpu_ratio"] == -1.0


def test_models_that_do_not_fit_get_one_slot():
    """Parallel slots split the context and multiply the KV cache."""
    assert suggest_settings(TIGHT, VRAM)["parallel"] == 1
    assert suggest_settings(DENSE_BIG, VRAM)["parallel"] == 1
    assert suggest_settings(SMALL, VRAM)["parallel"] == 0


def test_suggested_context_never_exceeds_the_model():
    tiny = ModelInfo(key="tiny", size_gb=1.0, max_context=4096, quant="Q4_0")
    assert suggest_settings(tiny, VRAM)["context"] == 4096


# ---------------------------------------------------------------------- scan


def test_scan_marks_models_already_in_a_tier(sandbox):
    cands = {c.key: c for c in scan(sandbox, FakeRuntime(IN_USE, SMALL), vram_gb=VRAM)}
    assert cands["test/model"].tiers == ["fast"]
    assert cands["test/model"].suggested_name == ""
    assert cands["acme/small-9b"].tiers == []
    assert cands["acme/small-9b"].suggested_name


def test_scan_lists_unused_models_first(sandbox):
    cands = scan(sandbox, FakeRuntime(IN_USE, EMBED, SMALL), vram_gb=VRAM)
    assert cands[0].key == "acme/small-9b"


def test_suggested_names_avoid_existing_tiers():
    info = ModelInfo(key="fast", size_gb=1.0, quant="Q4_0")
    assert models.suggest_name(info, {"fast"}) == "fast-2"


# ----------------------------------------------------------------------- add


def test_adding_appends_a_tier_that_resolves_to_the_model(sandbox, cfg_file):
    before = cfg_file.read_text(encoding="utf-8")
    done = add_tier("acme/small-9b", "small", runtime=FakeRuntime(SMALL),
                    cfg_path=cfg_file, vram_gb=VRAM)
    after = cfg_file.read_text(encoding="utf-8")

    assert after.startswith(before.rstrip("\n")), "everything above the block must survive"
    cfg = load_config(cfg_file)
    assert SMALL.matches(cfg.tier("small"))
    assert cfg.tier("small").quant == "Q4_K_M"
    assert done.name == "small"


def test_hand_written_comments_survive(sandbox, cfg_file):
    text = cfg_file.read_text(encoding="utf-8")
    cfg_file.write_text("# measured 2026-08-24: keep this\n" + text, encoding="utf-8")
    add_tier("acme/small-9b", "small", runtime=FakeRuntime(SMALL), cfg_path=cfg_file, vram_gb=VRAM)
    assert cfg_file.read_text(encoding="utf-8").startswith("# measured 2026-08-24: keep this\n")


def test_a_dry_run_writes_nothing(sandbox, cfg_file):
    before = cfg_file.read_text(encoding="utf-8")
    done = add_tier("acme/small-9b", "small", runtime=FakeRuntime(SMALL),
                    cfg_path=cfg_file, vram_gb=VRAM, dry_run=True)
    assert "[tiers.small]" in done.block
    assert cfg_file.read_text(encoding="utf-8") == before


def test_a_duplicate_tier_name_is_refused(sandbox, cfg_file):
    with pytest.raises(ModelsError, match="already exists"):
        add_tier("acme/small-9b", "fast", runtime=FakeRuntime(SMALL), cfg_path=cfg_file, vram_gb=VRAM)


@pytest.mark.parametrize("bad", ["has space", "dotted.name", "x" * 41, "quo\"te", "a]b"])
def test_a_name_that_is_not_a_bare_toml_key_is_refused(sandbox, cfg_file, bad):
    with pytest.raises(ModelsError, match="tier name"):
        add_tier("acme/small-9b", bad, runtime=FakeRuntime(SMALL), cfg_path=cfg_file, vram_gb=VRAM)


def test_an_embedding_model_is_refused(sandbox, cfg_file):
    with pytest.raises(ModelsError, match="embedding"):
        add_tier("nomic-embed", "emb", runtime=FakeRuntime(EMBED), cfg_path=cfg_file, vram_gb=VRAM)


def test_a_model_that_is_not_downloaded_is_refused(sandbox, cfg_file):
    with pytest.raises(ModelsError, match="not a downloaded model"):
        add_tier("nope/missing", "x", runtime=FakeRuntime(SMALL), cfg_path=cfg_file, vram_gb=VRAM)


def test_context_past_the_model_maximum_is_refused(sandbox, cfg_file):
    with pytest.raises(ModelsError, match="maximum"):
        add_tier("acme/small-9b", "s", {"context": 10_000_000}, runtime=FakeRuntime(SMALL),
                 cfg_path=cfg_file, vram_gb=VRAM)


def test_only_whitelisted_settings_can_be_overridden(sandbox, cfg_file):
    with pytest.raises(ModelsError, match="cannot be set"):
        add_tier("acme/small-9b", "s", {"model": "something/else"}, runtime=FakeRuntime(SMALL),
                 cfg_path=cfg_file, vram_gb=VRAM)


def test_overrides_are_applied(sandbox, cfg_file):
    add_tier("acme/small-9b", "s", {"context": 4096, "parallel": 2}, runtime=FakeRuntime(SMALL),
             cfg_path=cfg_file, vram_gb=VRAM)
    tier = load_config(cfg_file).tier("s")
    assert (tier.context, tier.parallel) == (4096, 2)


def test_a_failed_validation_restores_the_original(sandbox, cfg_file, monkeypatch):
    """Whatever goes wrong after the write, the file must end up as it started."""
    before = cfg_file.read_text(encoding="utf-8")

    def boom(*a, **k):
        raise ModelsError("simulated: tier does not resolve")

    monkeypatch.setattr(models, "_validate_file", boom)
    with pytest.raises(ModelsError, match="simulated"):
        add_tier("acme/small-9b", "s", runtime=FakeRuntime(SMALL), cfg_path=cfg_file, vram_gb=VRAM)
    assert cfg_file.read_text(encoding="utf-8") == before


def test_a_hostile_display_name_cannot_write_config(sandbox, cfg_file):
    """A model's display name is chosen by whoever uploaded it."""
    evil = ModelInfo(
        key="evil/model", quant="Q4_0", size_gb=2.0, max_context=8192,
        display_name='Nice\n[paths]\nread_roots = ["C:/"]\n#',
    )
    add_tier("evil/model", "evil", runtime=FakeRuntime(evil), cfg_path=cfg_file,
             vram_gb=VRAM, today=date(2026, 9, 25))
    raw = tomllib.loads(cfg_file.read_text(encoding="utf-8"))
    assert raw["paths"]["read_roots"] != ["C:/"]
    assert load_config(cfg_file).paths.read_roots == sandbox.paths.read_roots


def test_the_model_key_is_encoded_not_interpolated(sandbox, cfg_file):
    odd = ModelInfo(key='odd"key\\x', quant="Q4_0", size_gb=2.0, max_context=8192)
    add_tier(odd.key, "odd", runtime=FakeRuntime(odd), cfg_path=cfg_file, vram_gb=VRAM)
    assert load_config(cfg_file).tier("odd").model == 'odd"key\\x'
