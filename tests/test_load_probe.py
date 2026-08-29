"""CPU and GPU load, as reported to the dashboard.

psutil.cpu_percent(interval=None) keeps its previous sample in process-global
state shared by every caller, and read a flat 0.0 intermittently because of it.
The rate is computed from our own baseline instead, so nothing outside this
module can reset it.
"""

from __future__ import annotations

import time

from llmhelper.runtime import cpu_load, gpu_stats


def _sample() -> dict:
    """Two reads far enough apart for a rate to exist."""
    cpu_load()
    time.sleep(0.15)
    return cpu_load()


def test_cpu_load_reports_a_plausible_percentage():
    out = _sample()

    assert out["cpu_cores"] >= 1
    assert out["cpu_pct"] is not None
    assert 0.0 <= out["cpu_pct"] <= 100.0


def test_cpu_load_is_a_rate_not_a_running_total():
    """Two consecutive reads must both be percentages, not an accumulation."""
    first = _sample()["cpu_pct"]
    second = _sample()["cpu_pct"]
    assert first is not None and second is not None
    assert second <= 100.0


def test_the_first_ever_sample_has_no_baseline_to_subtract():
    """None, not a fabricated zero: the UI shows a dash rather than 'idle'."""
    from llmhelper import runtime

    saved = dict(runtime._cpu_prev)
    try:
        runtime._cpu_prev["busy"] = None
        runtime._cpu_prev["total"] = None
        assert cpu_load()["cpu_pct"] is None
    finally:
        runtime._cpu_prev.update(saved)


def test_gpu_stats_is_absent_or_complete():
    """No GPU is a valid answer; a half-filled dict is not."""
    st = gpu_stats()
    if st is None:
        return
    for key in ("vram_used_mb", "vram_total_mb", "gpu_pct", "gpu_temp_c", "gpu_power_w"):
        assert key in st, f"{key} missing"
    if st["gpu_pct"] is not None:
        assert 0 <= st["gpu_pct"] <= 100
    if st["vram_total_mb"] is not None:
        assert st["vram_total_mb"] > 0
        assert st["vram_used_mb"] <= st["vram_total_mb"]


def test_gpu_memory_mb_still_answers_the_old_shape():
    """ensure_loaded's RAM guard depends on this tuple."""
    from llmhelper.runtime import gpu_memory_mb

    mem = gpu_memory_mb()
    assert mem is None or (len(mem) == 2 and all(isinstance(x, int) for x in mem))
