"""Typed access to config.toml.

Paths in the file may be relative; everything here hands back absolute paths so
no caller has to care what the working directory is.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


class ConfigError(RuntimeError):
    """config.toml is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Tier:
    """One model configuration the worker can route a job to."""

    name: str
    model: str
    context: int
    min_free_ram_gb: float
    temperature: float = 0.2
    # Fraction of layers to put on the GPU, or -1 to let llama.cpp fit it itself.
    #
    # Do NOT set this to "everything" for a model that does not comfortably fit.
    # Forcing a layer count makes llama.cpp emit `common_fit_params: failed to fit
    # params to free device memory: n_gpu_layers already set by user` and then
    # thrash instead of adjusting. Measured on Qwen3.8-27B IQ4_XS (13.3 GB on a
    # 16.3 GB card): forced-max 0.5 tok/s, auto-fit 17.5, ratio 0.9 -> 24.4.
    # A 49x swing from this one setting.
    gpu_ratio: float = -1.0
    kind: str = "llm"  # "llm" | "embedding"
    quant: str = ""    # e.g. "Q4_K_M"; empty means any quantisation will do
    # Multiplier over the model's on-disk size used to demand free RAM, covering
    # the KV cache and allocator slack. A tier with a small context needs less,
    # so a large model can be made loadable by lowering this *and* its context
    # together — lowering min_free_ram_gb alone does nothing, because the
    # requirement is the larger of the two.
    ram_headroom: float = 1.15
    # Max concurrent predictions. llama.cpp divides the context between slots AND
    # allocates KV cache for all of them, so parallel=4 on an 8192 context gives
    # each request only 2048 tokens while paying for 32768. For a big model on a
    # small card, parallel=1 buys back both the context and the VRAM.
    parallel: int = 0  # 0 = leave LM Studio's default alone
    # Reasoning models bill thinking against max_tokens, so an extraction task
    # with a small budget can spend all of it thinking and return nothing.
    # "none" disables it. Empty string means "do not send the parameter".
    reasoning_effort: str = ""

    @property
    def is_embedding(self) -> bool:
        return self.kind == "embedding"

    @property
    def base_key(self) -> str:
        """Model key without publisher prefix or ``@quant`` suffix.

        LM Studio keys are quantisation-agnostic, so matching on this alone
        cannot tell a 21 GB Q4 copy from a 71 GB BF16 one — hence ``quant``.
        """
        key = self.model.split("@", 1)[0]
        return key.rsplit("/", 1)[-1].lower()

    @property
    def load_ref(self) -> str:
        """What to hand ``lms load``.

        The bare model key, deliberately. ``lms load`` rejects the ``key@quant``
        form even when the catalogue lists exactly that string as a variant; it
        resolves the quantisation itself via ``selectedVariant``. ``quant`` is
        therefore used for *matching* the catalogue (so a 21 GB Q4 is not
        confused with a 71 GB BF16), never for loading.
        """
        return self.model.split("@", 1)[0]


@dataclass(frozen=True)
class ServerCfg:
    base_url: str
    lms_exe: Path
    startup_timeout_s: float
    request_timeout_s: float
    # What to pass `lms server start --bind`. LM Studio does not remember this
    # across an explicit stop, so a restart silently dropped a 0.0.0.0 binding
    # back to localhost and cut off every other machine on the network. Keeping
    # it in config means a restart cannot quietly change who can reach the model.
    bind: str = "127.0.0.1"
    port: int = 1234

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"


@dataclass(frozen=True)
class PathsCfg:
    outbox: Path
    jobs_db: Path
    logs: Path
    bench: Path
    read_roots: tuple[Path, ...]
    deny_globs: tuple[str, ...]


@dataclass(frozen=True)
class BudgetsCfg:
    express_max_out_tokens: int
    express_timeout_s: float
    # The queue exists for work too long to wait on, so it must not inherit the
    # synchronous cap. A deep-tier digest ran past express_timeout_s and was
    # killed three times over, once per retry, on a lane whose whole purpose is
    # that nobody is waiting.
    queue_timeout_s: float
    # How long a synchronous call waits for the GPU when the worker holds it.
    # One card cannot hold two tiers, so the lanes take turns.
    gpu_wait_s: float
    map_chunk_tokens: int
    map_overlap_tokens: int
    map_summary_tokens: int
    max_map_chunks: int
    map_concurrency: int
    result_retention_days: int


@dataclass(frozen=True)
class WorkerCfg:
    poll_interval_s: float
    max_attempts: int
    fallback_tier: str
    # Seconds of idle before the worker releases the resident model. 0 keeps it
    # loaded indefinitely, which is what a remote client needs: it cannot load
    # models itself, so once this unloads, the next request JIT-loads at LM
    # Studio's default context instead of the tier's - quietly shrinking the
    # window a laptop gets from 131072 to 8192.
    idle_unload_after_s: float = 900.0


@dataclass(frozen=True)
class ImagesCfg:
    """Local image generation. Off by default: the weights are a 7 GB download.

    ``offload`` is the one setting worth understanding. There is a single 16 GB
    card and the LLM already occupies 6 GB of it:

      offload = false  unload the LLM, put SDXL entirely on the GPU. Fastest.
      offload = true   keep the LLM resident, page SDXL submodules across the
                       PCIe bus per step. ~4 GB peak, roughly half the speed.

    False is the default because an image job has no use for the LLM, and a
    remote client can reload it on its next request. Flip it when something is
    mid-conversation with the model and must not be interrupted.
    """

    enabled: bool = False
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    offload: bool = False
    steps: int = 30
    guidance: float = 7.0
    width: int = 1024
    height: int = 1024
    # A batch shares one pipeline load, so the marginal image is cheap - but each
    # one is still seconds of a card someone else may be waiting for.
    max_count: int = 4


@dataclass(frozen=True)
class Config:
    server: ServerCfg
    paths: PathsCfg
    budgets: BudgetsCfg
    worker: WorkerCfg
    tiers: dict[str, Tier] = field(default_factory=dict)
    images: ImagesCfg = field(default_factory=ImagesCfg)

    def tier(self, name: str) -> Tier:
        try:
            return self.tiers[name]
        except KeyError:
            known = ", ".join(sorted(self.tiers)) or "<none>"
            raise ConfigError(f"unknown tier {name!r}; configured tiers: {known}") from None

    @property
    def llm_tiers(self) -> dict[str, Tier]:
        return {n: t for n, t in self.tiers.items() if not t.is_embedding}


def _abs(value: str) -> Path:
    """Resolve a config path against the project root when it is relative."""
    p = Path(value.strip())
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return Path(os.path.normpath(p))


def _require(table: dict, key: str, where: str):
    if key not in table:
        raise ConfigError(f"missing required key {key!r} in [{where}]")
    return table[key]


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate config.toml. Raises ConfigError on any problem."""
    cfg_path = Path(path) if path else Path(os.environ.get("LLMHELPER_CONFIG", DEFAULT_CONFIG))
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    try:
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{cfg_path} is not valid TOML: {exc}") from exc

    srv = raw.get("server", {})
    server = ServerCfg(
        base_url=_require(srv, "base_url", "server"),
        lms_exe=Path(_require(srv, "lms_exe", "server")),
        startup_timeout_s=float(srv.get("startup_timeout_s", 240)),
        request_timeout_s=float(srv.get("request_timeout_s", 900)),
        bind=str(srv.get("bind", "127.0.0.1")),
        port=int(srv.get("port", 1234)),
    )

    pth = raw.get("paths", {})
    paths = PathsCfg(
        outbox=_abs(pth.get("outbox", "results")),
        jobs_db=_abs(pth.get("jobs_db", "jobs/llmhelper.db")),
        logs=_abs(pth.get("logs", "logs")),
        bench=_abs(pth.get("bench", "bench")),
        read_roots=tuple(_abs(r) for r in pth.get("read_roots", [])),
        deny_globs=tuple(pth.get("deny_globs", [])),
    )
    if not paths.read_roots:
        raise ConfigError("paths.read_roots is empty — the local model could not read anything")

    bud = raw.get("budgets", {})
    budgets = BudgetsCfg(
        express_max_out_tokens=int(bud.get("express_max_out_tokens", 1200)),
        express_timeout_s=float(bud.get("express_timeout_s", 420)),
        queue_timeout_s=float(bud.get("queue_timeout_s", 3600)),
        gpu_wait_s=float(bud.get("gpu_wait_s", 180)),
        map_chunk_tokens=int(bud.get("map_chunk_tokens", 6000)),
        map_overlap_tokens=int(bud.get("map_overlap_tokens", 200)),
        map_summary_tokens=int(bud.get("map_summary_tokens", 400)),
        max_map_chunks=int(bud.get("max_map_chunks", 400)),
        map_concurrency=max(1, int(bud.get("map_concurrency", 1))),
        result_retention_days=int(bud.get("result_retention_days", 30)),
    )
    if budgets.map_overlap_tokens >= budgets.map_chunk_tokens:
        raise ConfigError("budgets.map_overlap_tokens must be smaller than map_chunk_tokens")

    wrk = raw.get("worker", {})
    worker = WorkerCfg(
        poll_interval_s=float(wrk.get("poll_interval_s", 2.0)),
        max_attempts=int(wrk.get("max_attempts", 3)),
        fallback_tier=wrk.get("fallback_tier", "fast"),
        idle_unload_after_s=float(wrk.get("idle_unload_after_s", 900.0)),
    )

    img = raw.get("images", {})
    images = ImagesCfg(
        enabled=bool(img.get("enabled", False)),
        model=str(img.get("model", ImagesCfg.model)),
        offload=bool(img.get("offload", False)),
        steps=int(img.get("steps", 30)),
        guidance=float(img.get("guidance", 7.0)),
        width=int(img.get("width", 1024)),
        height=int(img.get("height", 1024)),
        max_count=max(1, int(img.get("max_count", 4))),
    )

    tiers: dict[str, Tier] = {}
    for name, t in raw.get("tiers", {}).items():
        tiers[name] = Tier(
            name=name,
            model=_require(t, "model", f"tiers.{name}"),
            context=int(t.get("context", 8192)),
            min_free_ram_gb=float(t.get("min_free_ram_gb", 4)),
            temperature=float(t.get("temperature", 0.2)),
            gpu_ratio=float(t.get("gpu_ratio", -1.0)),
            kind=t.get("kind", "llm"),
            quant=str(t.get("quant", "")),
            ram_headroom=float(t.get("ram_headroom", 1.15)),
            parallel=int(t.get("parallel", 0)),
            reasoning_effort=str(t.get("reasoning_effort", "")),
        )
    if not tiers:
        raise ConfigError("no [tiers.*] configured")
    if worker.fallback_tier not in tiers:
        raise ConfigError(f"worker.fallback_tier {worker.fallback_tier!r} is not a configured tier")

    return Config(
        server=server,
        paths=paths,
        budgets=budgets,
        worker=worker,
        tiers=tiers,
        images=images,
    )


_cache: tuple[tuple, Config] | None = None


def _stamp(path: Path) -> tuple:
    """Cheap identity for the config file: path, mtime, size."""
    try:
        st = path.stat()
        return (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), 0, 0)


def config_path() -> Path:
    return Path(os.environ.get("LLMHELPER_CONFIG", DEFAULT_CONFIG))


def invalidate() -> None:
    """Force the next get_config() to re-read from disk."""
    global _cache
    _cache = None


def get_config() -> Config:
    """Process-wide config, re-read whenever config.toml changes on disk.

    This used to be @lru_cache(maxsize=1), which is wrong for the way the MCP
    server actually runs: it is a long-lived child of a Claude session, so it
    snapshots the config at startup and then serves that snapshot for the rest
    of the session. Adding a read root therefore had no effect on any session
    already running - the path stayed refused for hours, and looked like a
    permanent limitation of the tool rather than a stale process.

    Stat-per-call is roughly a microsecond against calls that take seconds.
    """
    global _cache
    path = config_path()
    stamp = _stamp(path)
    if _cache is not None and _cache[0] == stamp:
        return _cache[1]
    try:
        cfg = load_config(path)
    except ConfigError:
        # A half-saved or malformed file must not take down every running
        # server. Keep serving the last good config; the next write is another
        # mtime change, so a fix is picked up without a restart.
        if _cache is not None:
            _cache = (stamp, _cache[1])
            return _cache[1]
        raise
    _cache = (stamp, cfg)
    return cfg
