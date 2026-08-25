"""Control of the LM Studio server and which model is resident.

Two jobs:

  * make sure the OpenAI-compatible server is actually up
  * load exactly one model at a time, with an explicit context length

The second one matters more than it looks. LM Studio will JIT-load a model when
a request names it, but it does so at the app's default context length (4096
here), which is far too small for digesting large files. Loading explicitly is
the only way to get the context we need.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass

import psutil

from .client import LocalLLM
from .config import Config, Tier, get_config

GB = 1024**3


class RuntimeError_(RuntimeError):
    """Server or model-loading failure."""


@dataclass
class LoadedModel:
    identifier: str
    context: int = 0
    # llama.cpp serves several requests at once by splitting the context window
    # between slots. With parallel=4 a 8192-token context gives each concurrent
    # request only 2048 tokens — the single most surprising cause of
    # "Context size has been exceeded" when prompts look comfortably small.
    parallel: int = 1


@dataclass
class ModelInfo:
    """One model as LM Studio has it on disk."""

    key: str
    quant: str = ""
    size_gb: float = 0.0
    max_context: int = 0
    variants: tuple[str, ...] = ()
    kind: str = "llm"
    # Where this entry came from. "disk" knows quantisations; "server" is a bare
    # id from /v1/models on a remote host, where quantisation is unknowable and
    # there is nothing to disambiguate anyway.
    origin: str = "disk"

    @property
    def base_key(self) -> str:
        return self.key.split("@", 1)[0].rsplit("/", 1)[-1].lower()

    def matches(self, tier: Tier) -> bool:
        """Does this model satisfy the tier, quantisation included?

        Quantisation matters only when several copies of one model could be on
        disk — the whole point is never to load a 66 GB BF16 when the tier asked
        for a 20 GB Q4. A remote server reports one id per model and no quant, so
        insisting on a quant there matches nothing and refuses every tier.
        """
        if self.base_key != tier.base_key:
            return False
        if self.origin == "server":
            return True
        if not tier.quant:
            return True
        wanted = tier.quant.lower()
        if self.quant and self.quant.lower() == wanted:
            return True
        # A multi-variant entry counts if it carries the quant we asked for.
        return any(wanted == v.split("@", 1)[-1].lower() for v in self.variants)


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / GB


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / GB


def gpu_memory_mb() -> tuple[int, int] | None:
    """(used, total) VRAM in MB from nvidia-smi, or None if unavailable."""
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(","))
        return used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class Runtime:
    """Wraps ``lms.exe`` plus a health probe."""

    def __init__(self, cfg: Config | None = None, llm: LocalLLM | None = None):
        self.cfg = cfg or get_config()
        self.llm = llm or LocalLLM(self.cfg)
        self.lms = self.cfg.server.lms_exe
        self._catalog: list[ModelInfo] | None = None

    # ------------------------------------------------------------ lms shell

    @property
    def can_manage_models(self) -> bool:
        """True when LM Studio's CLI is on this machine.

        False on a remote client — a laptop pointing `base_url` at another box's
        LM Studio. There the server owns model residency; this process can send
        requests but must not try to load, unload or enumerate anything locally.
        """
        return self.lms.is_file()

    def _lms(self, *args: str, timeout: float = 120.0, check: bool = True) -> str:
        if not self.lms.is_file():
            if not check:
                return ""  # optional query on a remote client: absence is an answer
            raise RuntimeError_(f"lms CLI not found at {self.lms} — is LM Studio installed?")
        try:
            proc = subprocess.run(
                [str(self.lms), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,  # lms must never block waiting on a prompt
                # lms writes UTF-8 spinner glyphs; the Windows default codepage
                # cannot decode them and would raise mid-command.
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError_(f"`lms {' '.join(args)}` timed out after {timeout}s") from exc
        if check and proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-600:]
            raise RuntimeError_(f"`lms {' '.join(args)}` failed ({proc.returncode}): {tail}")
        return proc.stdout or ""

    # -------------------------------------------------------------- server

    def server_running(self) -> bool:
        return self.llm.healthy()

    def ensure_server(self, wait_s: float | None = None) -> bool:
        """Start the server if needed. Returns True once it answers."""
        if self.server_running():
            return True
        self._lms("server", "start", check=False, timeout=120)
        deadline = time.monotonic() + (wait_s or 60.0)
        while time.monotonic() < deadline:
            if self.server_running():
                return True
            time.sleep(1.5)
        return False

    def stop_server(self) -> bool:
        """Stop the LM Studio server. Returns True once it stops answering.

        Unloads first: a model left resident holds its VRAM and RAM even with the
        server down, which is rarely what someone stopping a server wants.
        """
        if not self.can_manage_models:
            raise RuntimeError_("no LM Studio CLI on this machine (remote client)")
        try:
            self.unload_all()
        except RuntimeError_:
            pass  # stopping matters more than tidying
        self._lms("server", "stop", check=False, timeout=120)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self.server_running():
                return True
            time.sleep(1.0)
        return False

    def restart_server(self, wait_s: float | None = None) -> bool:
        """Stop then start. Useful after changing an LM Studio setting."""
        self.stop_server()
        time.sleep(1.5)
        return self.ensure_server(wait_s=wait_s)

    def server_status(self) -> dict:
        """What `lms server status` reports, plus our own probe.

        The two can disagree: the CLI knows whether LM Studio thinks a server is
        running, the probe knows whether it actually answers.
        """
        text = self._lms("server", "status", check=False, timeout=60).strip()
        return {
            "answering": self.server_running(),
            "lms_says": text.splitlines()[0] if text else "",
        }

    # -------------------------------------------------------------- models

    def catalog(self, refresh: bool = False) -> list[ModelInfo]:
        """Everything on disk, with quantisation and size."""
        if self._catalog is not None and not refresh:
            return self._catalog

        import json

        out = self._lms("ls", "--json", check=False, timeout=90)
        models: list[ModelInfo] = []
        try:
            for e in json.loads(out):
                key = e.get("modelKey") or e.get("path") or ""
                if not key:
                    continue
                models.append(
                    ModelInfo(
                        key=key,
                        quant=((e.get("quantization") or {}).get("name") or ""),
                        size_gb=round(float(e.get("sizeBytes") or 0) / GB, 2),
                        max_context=int(e.get("maxContextLength") or 0),
                        variants=tuple(e.get("variants") or ()),
                        kind=e.get("type") or "llm",
                    )
                )
        except (ValueError, TypeError):
            # lms without --json: fall back to bare ids from the server.
            try:
                models = [ModelInfo(key=m, origin="server") for m in self.llm.models()]
            except Exception:  # noqa: BLE001 - diagnostics must never be the thing that fails
                models = []

        self._catalog = models
        return models

    def downloaded(self) -> list[str]:
        """Model keys present on disk (whether loaded or not)."""
        return [m.key for m in self.catalog()]

    def find(self, tier: Tier) -> ModelInfo | None:
        """The on-disk model satisfying this tier, or None.

        This is the single matching rule. `doctor` and the tier resolver both go
        through it so they can never report different answers.
        """
        for m in self.catalog():
            if m.matches(tier):
                return m
        return None

    def required_ram_gb(self, tier: Tier) -> float:
        """RAM to insist on before loading.

        Trusting ``min_free_ram_gb`` alone means a typo in config.toml can OOM the
        machine, so when the catalog knows the file size we take the larger of the
        two, plus headroom for the KV cache.
        """
        info = self.find(tier)
        if info is None or not info.size_gb:
            return tier.min_free_ram_gb
        return max(tier.min_free_ram_gb, info.size_gb * tier.ram_headroom)

    def loaded(self) -> list[LoadedModel]:
        out = self._lms("ps", "--json", check=False, timeout=60)
        try:
            import json

            return [
                LoadedModel(
                    identifier=e.get("identifier") or e.get("modelKey", ""),
                    context=int(e.get("contextLength", 0) or 0),
                    parallel=max(1, int(e.get("parallel", 1) or 1)),
                )
                for e in json.loads(out)
            ]
        except (ValueError, TypeError):
            return []

    def is_loaded(self, tier: Tier) -> bool:
        want = tier.base_key
        return any(
            m.identifier.split("@", 1)[0].rsplit("/", 1)[-1].lower() == want for m in self.loaded()
        )

    def parallel_slots(self, tier: Tier) -> int:
        """How many ways the loaded model splits its context, 1 if unknown."""
        want = tier.base_key
        for m in self.loaded():
            if m.identifier.split("@", 1)[0].rsplit("/", 1)[-1].lower() == want:
                return max(1, m.parallel)
        return 1

    def unload_all(self) -> None:
        self._lms("unload", "--all", check=False, timeout=120)

    def ensure_loaded(self, tier: Tier, exclusive: bool = True) -> None:
        """Make `tier` the resident model at its configured context length.

        Refuses rather than thrashing swap when RAM headroom is short — an OOM
        during a 59 GB load takes the desktop down with it.
        """
        if tier.is_embedding:
            # Small enough to sit alongside an LLM; JIT loading is fine.
            return
        if not self.can_manage_models:
            # Remote client: the machine running LM Studio decides what is
            # resident, and JIT-loads on demand. Trying to manage it from here
            # would fail on a missing CLI and achieve nothing if it did not.
            return
        if self.is_loaded(tier):
            return

        info = self.find(tier)
        if info is None:
            raise RuntimeError_(
                f"tier {tier.name!r} wants {tier.model}"
                + (f" @ {tier.quant}" if tier.quant else "")
                + " but no such model is downloaded. Run `lms get` or `llmh doctor`."
            )

        if exclusive:
            self.unload_all()
            time.sleep(1.0)

        needed = self.required_ram_gb(tier)
        free = free_ram_gb()
        if free < needed:
            raise RuntimeError_(
                f"tier {tier.name!r} ({info.size_gb:.0f} GB model) needs {needed:.0f} GB "
                f"free RAM, only {free:.1f} GB available. "
                "Close something, or route this job to a smaller tier."
            )

        context = tier.context
        if info.max_context and context > info.max_context:
            context = info.max_context

        args = ["load", tier.load_ref, "--context-length", str(context), "-y"]
        if tier.gpu_ratio >= 0:
            args += ["--gpu", str(tier.gpu_ratio)]
        if tier.parallel > 0:
            args += ["--parallel", str(tier.parallel)]
        # Otherwise pass no --gpu at all: llama.cpp fits the split to free VRAM.
        # Forcing it is catastrophic for a model near the card's limit.
        self._lms(*args, timeout=self.cfg.server.startup_timeout_s)

    def snapshot(self) -> dict:
        """Everything `llmh doctor` wants to report, in one call."""
        vram = gpu_memory_mb()
        return {
            "server_up": self.server_running(),
            "base_url": self.cfg.server.base_url,
            "lms_exe": str(self.lms),
            "lms_present": self.lms.is_file(),
            "ram_free_gb": round(free_ram_gb(), 1),
            "ram_total_gb": round(total_ram_gb(), 1),
            "vram_used_mb": vram[0] if vram else None,
            "vram_total_mb": vram[1] if vram else None,
            "loaded": [m.identifier for m in self.loaded()],
        }
