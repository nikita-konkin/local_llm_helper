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
import threading
from typing import Any
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


def lan_addresses() -> list[dict]:
    """This machine's IPv4 addresses with interface names, most likely LAN first.

    Ordering is the whole point. A developer box can easily have ten interfaces —
    VPN tunnels, VirtualBox host-only networks, Bluetooth, disconnected Wi-Fi
    adapters — and only one is the address to give another machine on the desk.

    Note what does *not* work here: asking the OS which address it would route a
    public packet out of. On this machine that answers with the VPN tunnel, which
    is exactly wrong for reaching a laptop on the same physical network. Interface
    names are cruder but honest, so virtual adapters are labelled and demoted
    rather than silently recommended.
    """
    import ipaddress
    import socket

    # Substrings that mark an interface as not-the-LAN.
    VIRTUAL = (
        "virtualbox", "vmware", "hyper-v", "vethernet", "docker", "wsl",
        "tun", "tap", "vpn", "openvpn", "wireguard", "tailscale", "zerotier",
        "bluetooth", "loopback", "teredo", "isatap",
    )

    out: list[dict] = []
    try:
        import psutil

        for name, addrs in psutil.net_if_addrs().items():
            low = name.lower()
            virtual = any(v in low for v in VIRTUAL)
            for a in addrs:
                # Compare the enum, not its string form: str() of psutil's family
                # gives the numeric value ("-1" for AF_LINK), not the enum name.
                if getattr(a, "family", None) != socket.AF_INET:
                    continue
                try:
                    addr = ipaddress.IPv4Address(a.address)
                except ValueError:
                    continue
                if addr.is_loopback:
                    continue
                out.append(
                    {
                        "ip": a.address,
                        "interface": name,
                        "virtual": virtual,
                        # 169.254.x means the adapter never got a DHCP lease:
                        # present in every listing, never reachable.
                        "usable": not addr.is_link_local,
                        "private": addr.is_private,
                    }
                )
    except Exception:  # noqa: BLE001 - address discovery must never break a page
        return []

    def subnet_rank(ip: str) -> int:
        """Prefer the ranges a home or office LAN actually uses.

        192.168/16 first: overwhelmingly what consumer routers hand out. 10/8
        next. 172.16/12 last of the private ranges — technically a LAN range, but
        in practice usually Docker, Hyper-V or a corporate tunnel, which is
        exactly the address that will not reach a laptop on the same desk.
        """
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        return 2

    # Interfaces literally named for physical hardware beat generic ones like
    # "Local Area Connection 2", which Windows also hands to virtual adapters.
    def name_rank(name: str) -> int:
        low = name.lower()
        return 0 if low.startswith(("ethernet", "wi-fi", "wlan", "en", "eth")) else 1

    out.sort(
        key=lambda d: (
            not d["usable"],
            d["virtual"],
            not d["private"],
            subnet_rank(d["ip"]),
            name_rank(d["interface"]),
            d["ip"],
        )
    )

    # "Likely" is as strong a claim as this deserves. With several plausible
    # candidates the honest answer is to show them and let a human pick the one
    # on the same subnet as the machine they are connecting from.
    plausible = [d for d in out if d["usable"] and not d["virtual"] and d["private"]]
    for d in out:
        d["likely_lan"] = bool(plausible) and d is plausible[0]
        d["ambiguous"] = len(plausible) > 1
    return out


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / GB


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / GB


# CPU use is a rate, so it needs two samples. psutil.cpu_percent(interval=None)
# keeps that previous sample in process-global state shared by every caller,
# which made it read a flat 0.0 intermittently. Own the arithmetic instead: our
# own baseline cannot be reset by anyone else, and a genuinely idle machine then
# reads 0.3 rather than exactly zero, which is the difference between a number
# and a bug that looks like one.
_cpu_prev: dict[str, Any] = {"busy": None, "total": None}
_cpu_lock = threading.Lock()


def _cpu_sample() -> tuple[float, float]:
    t = psutil.cpu_times()
    total = float(sum(t))
    return total - float(t.idle), total


def cpu_load() -> dict:
    """System-wide CPU use since the previous call, and the cores it spans."""
    with _cpu_lock:
        busy, total = _cpu_sample()
        prev_busy, prev_total = _cpu_prev["busy"], _cpu_prev["total"]
        _cpu_prev["busy"], _cpu_prev["total"] = busy, total

        pct = None
        if prev_busy is not None:
            d_total = total - prev_total
            if d_total > 0:
                pct = round(100.0 * (busy - prev_busy) / d_total, 1)

    return {
        "cpu_pct": pct,  # None until there are two samples to compare
        "cpu_cores": psutil.cpu_count(logical=True),
    }


# Establish the baseline at import so the first request has something to subtract.
cpu_load()


def gpu_stats() -> dict | None:
    """Utilisation, temperature, power and VRAM in one nvidia-smi call.

    One call rather than one per field: nvidia-smi costs tens of milliseconds of
    process startup, and the dashboard polls this on every refresh.
    """
    smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    fields = "memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        used, total, util, temp, power = (x.strip() for x in out.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None

    def num(raw):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None  # nvidia-smi prints "[N/A]" for unsupported fields

    return {
        "vram_used_mb": num(used),
        "vram_total_mb": num(total),
        "gpu_pct": num(util),
        "gpu_temp_c": num(temp),
        "gpu_power_w": num(power),
    }


def gpu_memory_mb() -> tuple[int, int] | None:
    """(used, total) VRAM in MB, or None if nvidia-smi is unavailable."""
    st = gpu_stats()
    if not st or st["vram_used_mb"] is None or st["vram_total_mb"] is None:
        return None
    return st["vram_used_mb"], st["vram_total_mb"]


class Runtime:
    """Wraps ``lms.exe`` plus a health probe."""

    def __init__(self, cfg: Config | None = None, llm: LocalLLM | None = None):
        self.cfg = cfg or get_config()
        self.llm = llm or LocalLLM(self.cfg)
        self.lms = self.cfg.server.lms_exe
        self._catalog: list[ModelInfo] | None = None
        # None until we have tried to run it; False once a spawn has failed.
        self._lms_usable: bool | None = None
        # Why it failed, in words worth showing someone.
        self.lms_error: str | None = None

    # ------------------------------------------------------------ lms shell

    @property
    def can_manage_models(self) -> bool:
        """True when LM Studio's CLI is on this machine.

        False on a remote client — a laptop pointing `base_url` at another box's
        LM Studio. There the server owns model residency; this process can send
        requests but must not try to load, unload or enumerate anything locally.
        """
        # Present is not the same as runnable. Seen for real: an LM Studio
        # self-update left lms.exe as 120 MB of zeros. is_file() still said yes,
        # so every call went on to CreateProcess and died with WinError 1392,
        # and the dashboard 500'd on a machine that could otherwise have carried
        # on serving from the ledger. Once a spawn fails we stop claiming we can
        # manage models and degrade to the same path a remote client uses.
        return self.lms.is_file() and self._lms_usable is not False

    def _lms(self, *args: str, timeout: float = 120.0, check: bool = True) -> str:
        if not self.can_manage_models:
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
        except OSError as exc:
            # The file is there and the OS refuses to execute it: a damaged or
            # half-written binary. Latch it so the rest of the session takes the
            # degraded path instead of paying for a failing spawn every request.
            self._lms_usable = False
            self.lms_error = (
                f"{self.lms} exists but cannot be run ({exc.__class__.__name__}: {exc}). "
                "The binary is damaged - open LM Studio and reinstall its CLI."
            )
            if not check:
                return ""
            raise RuntimeError_(self.lms_error) from exc
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
        self._lms(
            "server", "start",
            "--bind", self.cfg.server.bind,
            "--port", str(self.cfg.server.port),
            check=False, timeout=120,
        )
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
        # Ask for the model list first. A dict literal is built top to bottom, so
        # reading lms_present before this ran reported a healthy CLI in the same
        # response that had just failed to run it.
        loaded = [m.identifier for m in self.loaded()]
        return {
            "server_up": self.server_running(),
            "base_url": self.cfg.server.base_url,
            "lms_exe": str(self.lms),
            "lms_present": self.can_manage_models,
            "lms_error": self.lms_error,
            "ram_free_gb": round(free_ram_gb(), 1),
            "ram_total_gb": round(total_ram_gb(), 1),
            "vram_used_mb": vram[0] if vram else None,
            "vram_total_mb": vram[1] if vram else None,
            "loaded": loaded,
        }
