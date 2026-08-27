# local_llm_helper

A **context firewall**. A local LLM on this machine reads bulk material — large
files, logs, whole directories — and hands back a short, cited answer, so the raw
bytes never enter an expensive model's context.

It is also a job queue, so long work can run while you are away. But the queue is
the second half. The first half is the point: paying ~200 tokens for an answer
that would otherwise have cost ~35,000.

## Measured on this machine

Real runs during build, not projections:

| Job | Source | Returned | Ratio | Time |
|---|---|---|---|---|
| Digest a 119 KB LM Studio log | 40,057 tok | 242 tok | **165×** | 35 s |
| Digest the whole `llmhelper` package (19 files) | 35,748 tok | 208 tok | **172×** | 38 s |
| Semantic grep across the package | 20,377 tok | 88 tok | **232×** | 7 s |
| Digest one config file | 871 tok | 167 tok | 5× | 2 s |

Answers were spot-checked against the sources and were accurate, including exact
byte counts and line citations. The last row is the honest counter-example: on
small inputs there is nothing to save, and the tool is not worth calling.

Hardware: Ryzen 9 9950X3D, 93.6 GB RAM, RTX 5080 16 GB. Inference runs on the GPU
via the CUDA-12 llama.cpp engine — verified by sampling during generation: **93 %
peak GPU utilisation, 71 W idle to 295 W under load**.

**VRAM is the ceiling, not model quality.** Measured with `llmh bench`:

| Tier | Model | Size | tok/s | VRAM | Verdict |
|---|---|---|---|---|---|
| `fast` | qwen3.5-9b Q4_K_M | 6.1 GB | **100.5** | 8.0 GB | fits the card |
| `work` | qwen3.6-35b-a3b Q4_K_M | 20.6 GB | **12.6** | 14.1 GB | spills to CPU |
| ~~`heavy`~~ | gpt-oss-120b MXFP4 | 59 GB | **4.2** | 15.4 GB | **removed** — see below |

Anything above roughly 10 GB falls off the 16 GB card and collapses to ~12 tok/s.
One real query — "is this data read as images or arrays?" over 30,960 tokens of source —
took **403 s on `work` and 22 s on `fast`, with the same answer**. Every handler
therefore defaults to `fast`; `work` has to be asked for by name.

## Install

Requires [LM Studio](https://lmstudio.ai) with at least one GGUF model, and
Python 3.11+. Windows paths are shown; macOS and Linux differ only in the
interpreter path (`.venv/bin/python`).

```bash
git clone <this repo> local_llm_helper && cd local_llm_helper
python -m venv .venv
.venv\Scripts\pip install mcp httpx sqlalchemy typer rich tiktoken psutil
copy config.toml.example config.toml
```

Edit two things in `config.toml`:

| Setting | What it is |
|---|---|
| `server.lms_exe` | where LM Studio's CLI lives. Omit on a remote client — see below |
| `paths.read_roots` | **the only directories the local model may read.** This is the security boundary, not a convenience setting |

Then make the package importable from any working directory, so the MCP server
starts regardless of which project launched it:

```bash
echo %CD% > .venv\Lib\site-packages\llmhelper.pth
```

Check it:

```bash
.venv\Scripts\python -m llmhelper.cli doctor --fix
.venv\Scripts\python -m pytest tests -q
```

`doctor` starts LM Studio's server if it is down and reports which tiers have a
model on disk. The tests need no configuration and no model — they run on a
temporary config and should pass on a fresh clone.

## Register it with Claude Code

```bash
claude mcp add local-llm --scope user -- "<abs path>\.venv\Scripts\python.exe" -m llmhelper.mcp_server
```

`--scope user` makes it available in every project. There is no CLI-only route:
the same object can be written by hand into a project's `.mcp.json` or under
`mcpServers` in `~/.claude.json`. See `laptop/README-laptop.md` for connecting a
second machine over HTTP, and for the security caveats that come with it.

**Availability is not use.** Registering the server does not make an agent reach
for it — that is what a `CLAUDE.md` is for. Measured here: six connected sessions
over twelve hours made zero calls until instructions existed telling them when to
prefer `local_digest` over reading a file, and just as importantly when not to.
There is a worked example in `docs/CLAUDE.md.example`.

Put the repo directory on `PATH` to use `llmh` from anywhere.

## Tools exposed to Claude

**Express lane** — synchronous, capped at 120,000 source tokens per call:

| Tool | Use it instead of |
|---|---|
| `local_digest(paths, question)` | reading a large file or directory yourself |
| `local_grep(paths, query)` | a grep whose hits you would have to read |
| `local_review(paths \| diff)` | a first read-through hunting for defects |
| `local_ask(prompt)` | a small self-contained generation |

**Queue lane** — returns immediately:

`local_submit(kind, spec, priority)` → `local_status(job_id)` → `local_result(job_id, mode)`

`local_result` defaults to `mode="summary"` deliberately: pulling the full text
back would undo the saving the job just made.

`local_stats()` reports the ledger.

### What it can read

Two things decide whether a path is readable: it must sit inside
`paths.read_roots` and outside `deny_globs`, and its format must be one the
helper can turn into text.

| Format | Handling |
|---|---|
| `.txt` `.md` `.py` `.log` `.json` `.csv` and other plain text | read directly, with a UTF-8 / cp1251 / cp1252 / latin-1 fallback chain |
| `.pdf` | text layer extracted per page, prefixed `[page N]` so citations stay findable |
| `.docx` | paragraphs and tables |
| `.xlsx` `.xlsm` | every sheet as TSV, prefixed `[sheet name]` |
| `.pptx` | slide text, prefixed `[slide N]` — needs `pip install python-pptx` |
| `.html` `.htm` | markup, scripts and styles stripped |
| `.gguf` `.png` `.zip` `.exe` … | refused as binary |

PDF extraction uses **pdfplumber** where it is installed and falls back to
**pypdf**. pdfplumber positions glyphs rather than reading the content stream in
order, which matters more than it sounds: on a letter-spaced LaTeX title page
pypdf produces `PONTIFICI A UNIVERSI DAD`, which no model can match against
anything. Measured on a 992-page book the two cost 12 s and 66 s, and next to the
minutes of inference that follow, accuracy is the better buy.

A **scanned** PDF has no text layer. It is reported as needing OCR rather than
returned as an empty document, which would read as "the file is blank".

`.doc`, `.rtf` and `.epub` are not supported — convert them first.

### Changing the read roots

`config.toml` is re-read whenever it changes on disk, so adding a root takes
effect in sessions that are already running. This was not always true: `get_config()
` was cached for the life of the process, and since the MCP server is a long-lived
child of a Claude session, a newly added root stayed refused for hours and looked
like a permanent limitation of the tool.

**Code changes still need a restart.** Only the configuration reloads.

### What the ledger does and does not count

`llmh stats` excludes measurement traffic (`llmh bench`, `llmh ab`) by default, and
that exclusion matters more than it sounds. During this project's own construction
the raw figure read **1,120,355 tokens saved** — of which **1,066,872 was the same
119 KB log re-read about twenty times while benchmarking models.** Nobody would
ever have read it twenty times at full price, so counting it as "saved" made the
number meaningless. Genuine offloaded work was **62,106 tokens**.

`--all-sources` shows the unfiltered total. Prefer the default: a metric that
flatters itself is worse than no metric, because it gets quoted.

## CLI

```bash
llmh doctor --fix                      # health check; starts the server if down
llmh fetch work                        # download a tier's model, resuming until done
llmh bench                             # tokens/sec and VRAM per tier
llmh ab --tiers a,b -p <path> -q "..."  # same question on several tiers, side by side
llmh digest -p <path> -q "<question>"  # the main one
llmh grep -p <path> -q "<question>"
llmh review -p <path> [--focus ...]
llmh transform -p <path> -i "<instruction>"
llmh transcribe <audio file>
llmh submit digest -p <path> -q "..."  # queue it instead
llmh worker                            # run the background worker
llmh stats                             # tokens kept out of context
llmh prune                             # delete old outbox files and finished jobs
llmh ui                                # dashboard: status, usage, queue, settings
llmh server start|stop|restart|status  # the LM Studio model server
llmh server restart-worker             # after changing code or settings
llmh connect                           # addresses another machine should use
```

Quote glob patterns, or your shell will expand them before `llmh` sees them.

## Dashboard

```bash
llmh ui
```

Serves a single page on `http://127.0.0.1:8770` — status, the savings ledger, the
per-project table, the queue with failure reasons, and editable settings. No build
step and no JavaScript dependencies.

Controls: start/stop/restart the model server, start/stop/restart the worker,
load or unload a tier, prune old results. Stopping the server unloads the resident
model first — leaving one loaded would hold its VRAM with nothing to serve it.

**Only one dashboard runs at a time**, held by `logs/webui.pid`, for the same
reason as the worker: two of them means one binds the port and the other silently
does not, so a code change appears to do nothing while a stale process keeps
answering. That cost real debugging time before the lock existed.

Two deliberate limits:

- **Bound to localhost.** The controls start processes and load multi-gigabyte
  models. `--host 0.0.0.0` works and warns; there is no authentication.
- **`read_roots` and `deny_globs` are read-only here.** They are the filesystem
  security boundary. A UI able to widen them would let anything reaching the page
  grant itself the whole disk. They are shown so you can see the boundary, and
  edited in `config.toml` deliberately.

Settings writes edit `config.toml` line by line rather than round-tripping it
through a TOML writer, so the comments explaining every tuned number survive. The
file is backed up and restored if the result fails to parse. A running worker
keeps its old settings until restarted.

**The dashboard polls, and LM Studio logs every request it receives.** At a 5 s
refresh that put ~25 requests a minute into its Developer tab permanently — the
dashboard becoming the system's main workload. Runtime probes are cached for 20 s
and the page refreshes every 15 s, which brought it to ~4/min; controls force a
fresh read so an action's effect still shows immediately.

### Queue versus history

The dashboard separates them, because one list sorted by time reads as "the
queue is full of week-old jobs" when the queue is in fact empty and that is
history. **Queue** shows only what is waiting or running; **Recently finished**
shows what is done, with an age. Nothing in the first is the normal state.

Finished jobs were never cleaned up, so the table only grew. `llmh prune` now
removes terminal jobs of the same age as the outbox files it deletes. Queued and
running jobs are never touched, and neither is the savings ledger: it is the
record of what was actually offloaded, and a cleanup that quietly shrank it would
rewrite the one number worth reporting.

### Whose numbers are these?

The ledger is **per machine**. A second machine running its own copy — pointing
`base_url` at this one's LM Studio to borrow the GPU — keeps its own `jobs.db`, so
its savings appear in `llmh stats` *there*, not here. The dashboard names the host
it is reporting on for exactly this reason.

Only the HTTP-server setup (`--http`, where a remote Claude Code calls in) records
into this machine's ledger, because the handler runs here.

## Model tiers

The worker keeps **one** model resident and unloads before switching.

| Tier | Model | Size | Role |
|---|---|---|---|
| `fast` | `qwen/qwen3.5-9b` Q4_K_M | 6.1 GB | **Default for everything.** 100 tok/s, 131 K context, 12.2/16.3 GB VRAM. |
| `gemma` | `gemma-4-12b-it-qat` Q4_0 | 6.7 GB | Opt-in. 76 tok/s. Middle ground. |
| `work` | `qwen/qwen3.6-35b-a3b` Q4_K_M | 20.6 GB | Opt-in. 27.9 tok/s. |
| `deep` | `qwen3.8-27b` IQ4_XS | 13.3 GB | **Accuracy tier**, opt-in. 19 tok/s — queue it. |
| `embed` | `nomic-embed-text-v1.5` | 84 MB | Reserved for a future corpus index. |

The `heavy` tier (gpt-oss-120b) was **removed after measuring it**. It does load —
217 s — but runs at 4.2 tok/s and leaves **0.6 GB of free RAM**, close enough to the
edge to destabilise the desktop. A tier 24x slower than `fast` that risks the machine
is not worth routing to. The 59 GB file is still on disk; `config.toml` records how to
reclaim it or restore the tier.

The 66 GB BF16 `Qwen3.6-35B-A3B` was **deleted** on 2026-08-24. The LM Studio log
from 2026-04-30 recorded it failing with `cudaMalloc failed: out of memory` on a
68 GB CUDA buffer — it could not run on a 16 GB card. The Q4_K_M replacement (with
its own vision projector) is installed. Presence checks are quantisation-aware, so
a BF16 copy never counts as "already have" a Q4 tier.

**Runtime**: updated 2026-08-24 from llama.cpp engine 2.13.0 (b8733) to **2.29.1**
(b9888), which is what allows post-April-2026 model architectures to load. The 2.13.0
engines remain installed — `lms runtime select` reverts.

### Choosing a model: measured, 2026-08-24

Every candidate answered the same question about a 119 KB log, scored against
`grep`-verified ground truth. Each cited line number was checked against the file.

| Tier | Model | Cited lines | All verified? | Byte→GiB maths | Seconds |
|---|---|---|---|---|---|
| `fast` | qwen3.5-9b | 3 | yes | — | **27** |
| `gemma` | gemma-4-12b-it-qat | 6 | yes | **wrong** (64.60) | 50 |
| `deep` | qwen3.8-27b IQ4_XS | **12** | **yes, all 12** | **correct** (63.6) | 506 |
| — | ornith-1.5-9b | 0 | **claimed "None recorded"** | — | deleted |

`deep` is the accuracy tier and it earns the name: it found lines 250, 288, 879,
880 and 1142–1148 where `fast` found three, every citation checked out, and it was
the only model to convert 68348529152 bytes to GiB correctly — the other two both
echoed a file size from elsewhere in the log. It is also 19x slower, so it belongs
on the queue, not the express lane.

`fast` remains the default: correct, tersest, and quick enough to call synchronously.
Reach for `--tier deep` when being right matters more than being quick.

Reproduce with `llmh ab --tiers fast,gemma,deep -p <path> -q "..."`.

### Traps this project hit, and what fixed them

Every one of these produced plausible-looking output rather than an obvious error,
which is what made them expensive.

**Citations were silently wrong.** The map pass handed the model an unnumbered
excerpt while asking it to cite line numbers, so it counted from 1 *inside the
chunk*. A chunk starting at line 1142 produced citations like `[3]` — precise
looking, pointing hundreds of lines away. Lines are now prefixed with their real
file number before the model sees them. A wrong citation is worse than none: it
invites trust it has not earned. Pinned by `test_numbered_*`.

**Reasoning models bill thinking against `max_tokens`.** `qwen3.5-9b` spent 191
tokens reasoning to answer one word. An extraction call capped at 400 tokens
returned *nothing* while reporting success. Extraction steps now send
`reasoning_effort: "none"`, and the client raises instead of returning an empty
answer. Pinned by `tests/test_client.py`.

**Forcing all layers onto the GPU is catastrophic near the VRAM limit.** `--gpu max`
makes llama.cpp emit `failed to fit params to free device memory: n_gpu_layers
already set by user` and then thrash. On Qwen3.8-27B: forced 0.5 tok/s, auto-fit
17.5, ratio 0.9 → 24.4. **A 49x swing from one flag** — and it was quietly
halving the `work` tier too (12.6 → 27.9 once fixed).

**Three separate things shrink the usable context**, and missing any of them gives
`Context size has been exceeded` on prompts that look small:

1. line-number prefixes inflate a chunk ~25% (the chunker now counts them itself);
2. we budget in tiktoken cl100k, the model counts in its own tokenizer — measured
   **1.27x more** on log text, so `TOKENIZER_SAFETY` is a currency conversion, not
   a fudge factor;
3. llama.cpp **divides the context between parallel slots**: `parallel=4` on an
   8192 context leaves each request 2048, and allocates KV cache four times over.
   Setting `parallel = 1` on the `deep` tier is what made it usable at all.

An oversized chunk now halves itself and retries rather than failing.

**An empty response is not "nothing relevant".** The handler used to treat both
alike, so a broken run looked like a valid empty answer. It now raises, and partial
chunk failures are reported as a coverage warning attached to the result.

**A security check must not depend on the network.** `Path.resolve()` on a UNC
path can consult the network to resolve the host, so the guard's rejection of
`\server\share\...` was nondeterministic — it flaked exactly once across many
runs. UNC paths are now normalised textually and never reach the resolver. A test
asserts the refusal takes under a second, because a slow refusal is the symptom.

**`lms load` rejects the `model@quant` form** even though `lms ls --json` lists
exactly that string as a variant. Loads use the bare model key; `quant` in
`config.toml` is only used to match the catalogue, so a 20 GB Q4 is never confused
with a 66 GB BF16.

### The queue inherited the timeout for work someone is waiting on

`digest.py` and `ask.py` passed `budgets.express_timeout_s` straight to the
client, and the worker runs those same handler objects. So a deep-tier digest
submitted *because* it was too slow to wait for was killed at 420 s — three
times, once per retry. `[server].request_timeout_s = 900` never reached it.

The lane is now a property of the context. `Ctx` carries it, `Ctx.timeout_s`
resolves it, `generate()` applies it, and no handler names a timeout at all.
`execute()` derives the lane from `source`, which already distinguished the
caller. Queued work gets `budgets.queue_timeout_s`, default 3600.

### Two lanes, one card

The express lane and the worker each called `ensure_loaded` independently, so
both tiers went resident together: `qwen3.5-9b` (6.55 GB) beside `qwen3.8-27b`
(14.25 GB), **20.8 GB on a 16 GB card**. LM Studio's JIT loading is what makes
this possible — a request for a model that is not loaded pulls it in without
going through `ensure_loaded` at all, so the one-model-at-a-time rule never sees
it. The consequences were a deep load aborting with *"Engine protocol startup was
aborted"*, and a synchronous grep taking 875 s instead of 40 because it was
competing for a card that had already spilled into system RAM.

`lock.gpu_lease` now serialises them. The worker waits as long as it takes; a
synchronous call waits `budgets.gpu_wait_s` (default 180) and then reports that
the GPU is busy and names the holder, which is more useful than crawling.
Measured after the fix: a queued deep digest finished in 87 s on the first
attempt, an express call made during it waited and reported
`waiting for the GPU, held by queue:digest (pid 50872)`, and one model stayed
resident throughout.

### A search that lost the answer three ways at once

A search for where read roots are enforced returned 40 matches on `fast` and
reported that none of them answered it. The answer was at `guard.py:95`. Three
independent defects stacked up:

1. **The hit cap truncated the corpus in directory order.** `_scan` stopped
   walking at `max_hits`, so broad patterns matching often in `cli.py` used all 40
   slots before `guard.py` was opened. The answer was reported absent from a file
   that was never read. Fixed by scanning every file and sharing the budget
   round-robin, ranking each file's hits by how many distinct patterns they match.
2. **Zero survivors was reported as a confident negative.** The raw hits went into
   `detail`, but the express lane returns `summary` — so the evidence was in the
   result and the summary contradicted it. Now the hits come back unranked with
   the triage miss named as the likely cause.
3. **`startswith("YES")` read `**YES**: ...` as a rejection.**

The general lesson, which cost time twice: *a component that silently narrows its
own input produces a confident wrong answer.* The same shape appeared in `_scan`
swallowing unreadable files, and in the size check below.

### Sizing a document by its bytes

The express lane estimated tokens as `file size / 4`. That is fine for text and
meaningless for a document: a 635 KB PDF holding 8,463 tokens of prose was sized
at 158,725 and refused as too large to run synchronously. Extractable files are
now extracted and counted, with the result cached so the handler does not repeat
the work.

## Using it from other projects and other machines

### In another Claude Code project on this PC

**If it is already registered at user scope, there is nothing to do** — every
project on the machine has it. Verified from an unrelated project directory:
`claude mcp list` reports it connected with no `.mcp.json` present there. The
sections below are for sharing it with someone else, or registering it somewhere
it is not already global.

The CLI is a convenience wrapper, not the only route. All three scopes are plain
JSON you can edit by hand:

| Scope | File | Applies to |
|---|---|---|
| **user** | `~/.claude.json` → top-level `mcpServers` | every project (this is where the setup command wrote) |
| **project** | `<project>/.mcp.json` → `mcpServers` | that repo, shared via git |
| **local** | `~/.claude.json` → `projects["<abs path>"].mcpServers` | that project, just you |

Via the CLI:

```bash
claude mcp add local-llm --scope user -- "D:\local_llm_helper\.venv\Scripts\python.exe" -m llmhelper.mcp_server
```

Or drop a `.mcp.json` in any project root — this is the whole file, and it is what
this repo already ships:

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "D:\local_llm_helper\.venv\Scripts\python.exe",
      "args": ["-m", "llmhelper.mcp_server"]
    }
  }
}
```

Copy that into any project and it works there — no CLI involved. Backslashes must
be doubled: it is JSON. Note that a **project-scoped server prompts for trust on
first use** — Claude Code asks before running an MCP server a repo defines, since
the repo could have come from anywhere. User scope does not ask, because you
registered it yourself. An `env` object can be added alongside `args` if a server
needs environment variables.

Prefer `.mcp.json` for hand-editing. `~/.claude.json` also holds conversation
history and per-project state (60 KB+ here), so a bad edit costs more than a
mistyped `.mcp.json`, which affects one project and is obvious.

### Checking another project is actually using it

Availability and use are different questions, and they have different answers.

**Is it reachable there?** From that project's directory:

```bash
claude mcp get local-llm
```

Prints the scope (`User config (available in all your projects)`) and
`Status: ✔ Connected`. `claude mcp list` does the same for every server at once.
Inside a running session, `/mcp` shows the same thing. A session already running
will not pick up a registration change — start a new one.

**Did anything actually call it?** That is what `llmh stats` answers:

```
which project called
+----------------------------------------------------------------+
| project              | calls | read | saved | last used        |
|----------------------+-------+------+-------+------------------|
| D:
adar-analysis | 1     | 5    | 4     | 2026-08-24 21:07 |
| D:	hesis          | 1     | 5    | 4     | 2026-08-24 21:07 |
+----------------------------------------------------------------+
```

Every MCP call records the working directory it was launched from, which for a
Claude Code session is that session's project root. A project missing from this
table has the server available and has never used it — usually because the work
there has not been bulky enough to be worth offloading, which is the correct
outcome, not a fault.

Which projects it can actually *help* with is decided by `paths.read_roots` in
`config.toml`, not by the registration. That currently covers `D:\`, so every
project on that drive works. Add a root to reach anything else.

### From a laptop on your LAN

Two different setups, and the right one depends on **where the files are**.

**The two `.mcp.json` files in this repo are not interchangeable.** The root one is
stdio with an absolute `D:\...` path and works only on this Windows PC; copied to a
Mac it fails in milliseconds with `ENOENT`. A stdio server can only be spawned by a
session on the same machine as the executable. For anything else use
`laptop/mcp-over-http.json`. Full instructions, macOS paths included, are in
`laptop/README-laptop.md`.

Run `llmh connect` (or open the dashboard) for the exact URLs and this machine's
addresses. It flags whether the server is actually reachable from the LAN, because
that is where this most often goes wrong.

**`server.bind` lives in `config.toml` for a reason.** LM Studio does not remember
a bind address across an explicit stop, so `llmh server restart` silently dropped a
`0.0.0.0` binding back to `127.0.0.1` and cut off every other machine — a restart
quietly changing who can reach the model. The setting is now passed on every start.

**A — laptop's own files, this PC's GPU.** The laptop runs `llmhelper`; only the
model call crosses the network. Use this when the files you want digested live on
the laptop. On macOS the interpreter is `.venv/bin/python`, not
`.venv\Scripts\python.exe`.

On this PC, publish the model server:

```bash
"C:/Users/<you>\.lmstudio\bin\lms.exe" server start --bind 0.0.0.0 --port 1234
```

On the laptop, install this package and point `config.toml` here:

```toml
[server]
base_url = "http://DESKTOP-LAN-IP:1234/v1"
```

The laptop needs its own `paths.read_roots` and outbox. `lms_exe` will not exist
there, and that is handled: without LM Studio's CLI the client enters **remote
mode** — it does not attempt to load, unload or enumerate models, and leaves
residency to the machine actually running the server. Load the model you want on
this PC first.

Two things had to change for that to work at all, both found by simulating a
laptop rather than assuming:

- every call routed through `ensure_loaded`, which shelled out to `lms.exe` and
  raised on a machine without it;
- tier matching demanded a quantisation, but a remote server reports bare ids via
  `/v1/models` with no quant, so every tier resolved to nothing and the resolver
  refused the job. Quant matching is a local-disk concern — telling a 20 GB Q4
  from a 66 GB BF16 — and there is nothing to disambiguate remotely.

Verified by running a full digest with `lms_exe` pointed at a non-existent path:
it selected a tier, answered correctly with citations checked against the file,
and never touched a local binary.

**B — this PC's files, from the laptop.** Run the MCP server itself over HTTP, so
the laptop's Claude Code calls it and digests read *this* machine's disk.

On this PC:

```bash
"D:\local_llm_helper\.venv\Scripts\python.exe" -m llmhelper.mcp_server --http --host 0.0.0.0 --port 8765
```

On the laptop — **no CLI required**. Drop this `.mcp.json` into a project folder;
a ready-made copy is in `laptop/`:

```json
{
  "mcpServers": {
    "local-llm": {
      "type": "http",
      "url": "http://DESKTOP-LAN-IP:8765/mcp"
    }
  }
}
```

That shape was not guessed: the CLI was made to write an HTTP entry on the desktop
and the result read back. `claude mcp add --transport http local-llm <url>`
produces exactly this, so the file and the command are interchangeable.

For all projects on the laptop rather than one, the same object goes under the
top-level `mcpServers` key in its `~/.claude.json`. Prefer the per-project file —
`~/.claude.json` also carries conversation history, so a bad edit costs more.
Restart Claude Code either way; a running session ignores new servers.

The server was verified answering MCP `initialize` over HTTP on both `127.0.0.1`
and `DESKTOP-LAN-IP`.

### When every port times out but ping works

This is the failure to expect, and it looks nothing like a bug in this project.
Symptom, measured from a laptop on the same LAN:

| check | result |
|---|---|
| `ping 192.168.50.32` | 2/2 replies, 2.9 ms |
| TCP 1234, 8765 | timeout |
| TCP 445, 135, 3389 | **also timeout** |
| another host on the LAN, TCP 443 | connects fine |

Core Windows services timing out alongside yours is the tell: nothing
application-specific is wrong. The adapter is on the Windows **Public** firewall
profile, whose default inbound action is Block, and Windows *drops* inbound TCP
rather than refusing it - hence timeouts rather than connection-refused, while
ICMP still answers.

Confirm from the serving machine: `Test-NetConnection <own-LAN-IP> -Port 1234`
succeeds locally while the same test fails from anywhere else. That proves the
service is healthy and the block is inbound-from-network.

Fix it with the script, from an **elevated** PowerShell:

```bash
& 'N:\local_llm_helper\autostart\allow-lan.ps1'
```

It adds one scoped rule per port, verifies each is listening and not bound to
localhost, and takes `-Remove` to undo. Switching the adapter to "Private" also
works and is one click, but opens file sharing and remote desktop along with it -
two scoped rules are the smaller change.

**The firewall will block it until you say otherwise.** Both interfaces here are on
the **Public** profile, where Windows blocks inbound by default. In an
administrator shell:

```bash
New-NetFirewallRule -DisplayName "local_llm_helper MCP" -Direction Inbound -Protocol TCP -LocalPort 8765 -RemoteAddress 192.168.1.0/24 -Action Allow
```

`-RemoteAddress` confines it to the local subnet. Do not widen it to `Any` — there
is no authentication on this server.

### Before you expose anything

Option B hands **read access to everything under `paths.read_roots`** to anyone who
can reach port 8765. The guard still holds — reads stay inside the allowlist and
writes inside the outbox — but the audience changes from one local process to a
whole network. So:

- Bind to the LAN address on a network you control. Never a public interface, and
  never a port forwarded through your router: there is no authentication.
- Trim `paths.read_roots` to what the laptop actually needs before starting it.
- Windows Firewall prompts on first bind. Allow it for **private** networks only.
- Neither option encrypts anything; file content crosses the LAN in the clear.

Option A is the safer default. It shares GPU capacity, not filesystem access.

### When the handshake works but tool calls fail

A distinct failure, and not a network one: `initialize` succeeds and returns a
session id, then the first `tools/call` reports the server as unreachable. The
give-away is the timing — a handshake is one fast round trip, a tool call takes
seconds while the local model works.

The default streamable-HTTP transport returns tool results over an **SSE stream**,
holding the connection open for the whole call. That window is what an idle
timeout or a VPN adapter closes, and the client reports it as a connection
failure moments after a successful handshake. The server is served with
`json_response=True` instead: one plain JSON response, no held-open stream, and
nothing lost since these tools are request/response anyway. `--sse` restores
streaming for debugging.

Only one HTTP server runs at a time, held by `logs/mcp-http.pid`. Without that a
second copy fails to bind with `WinError 10048` inside a log file nobody reads
while the first keeps serving, so a restart looks like it happened and did not.

### The two ports are not interchangeable

| Port | Service | Needed by |
|---|---|---|
| **1234** | LM Studio's model API | a machine running its own copy of this package (option A) |
| **8765** | this package's MCP server over HTTP | a machine calling *this* PC's tools and reading its files (option B) |

1234 belongs to LM Studio and cannot host the MCP server as well. If your other
machine only borrows the GPU, 1234 is the only port it needs and 8765 can stay
closed.

A firewall rule alone is never enough: **something has to be listening**. Port
8765 has no listener unless `autostart/start-mcp-http.cmd` is running, and an open
port with nothing behind it times out exactly like a blocked one.

### Autostart (installed)

Two entries are installed in the Startup folder
(`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`), both launching hidden
via `autostart\run-hidden.vbs`:

| Entry | Runs | Notes |
|---|---|---|
| `LocalLLMHelper-Server.cmd` | `autostart\start-server.cmd` | LM Studio server, **bound to 127.0.0.1** |
| `LocalLLMHelper-Worker.cmd` | `autostart\start-worker.cmd` | drains the queue; waits up to 150 s for the server first |
| `LocalLLMHelper-McpHttp.cmd` | `autostart\start-mcp-http.cmd` | **exposes MCP on 0.0.0.0:8765** - see the warning below |

**The MCP HTTP entry exposes this machine's files to the LAN at every logon.**
Anything that reaches port 8765 can read everything under `paths.read_roots`,
with no authentication. That is a deliberate choice, not a default: delete
`LocalLLMHelper-McpHttp.cmd` from the Startup folder to stop it, and narrow
`read_roots` to what the other machine actually needs.

Both were tested by running them directly: the server came up on port 1234 and the
worker picked up a queued job and completed it.

The server binds to localhost, not the LAN. An unattended task listening on the
network with no authentication is a different security posture than one you start
by hand. For laptop access after a reboot, edit `BIND` in
`autostart\start-server.cmd` to `0.0.0.0` — and read the exposure warning above
first.

Startup-folder entries were used because `schtasks /sc onlogon` needs elevation.
For real Task Scheduler entries, run these in an **administrator** shell:

```bash
schtasks /create /tn LocalLLMHelper-Server /tr "D:\local_llm_helper\autostart\start-server.cmd" /sc onlogon /rl limited /f
```

```bash
schtasks /create /tn LocalLLMHelper-Worker /tr "D:\local_llm_helper\autostart\start-worker.cmd" /sc onlogon /rl limited /f
```

Then delete the two Startup-folder entries so they do not run twice.

### Keeping it running manually

Nothing starts by itself. For a persistent setup, run both at logon via Task
Scheduler (an ordinary logon task is enough):

```bash
"C:/Users/<you>\.lmstudio\bin\lms.exe" server start --bind 0.0.0.0
```

```bash
"D:\local_llm_helper\.venv\Scripts\python.exe" -m llmhelper.cli worker
```

The worker is what drains queued jobs. Without it, `local_submit` accepts work that
never runs.

**Only one worker runs at a time.** A second `llmh worker` refuses, pointing at
`logs/worker.pid`. Several workers do not corrupt the queue — claiming a job is
atomic — but they fight over which model is resident, and every unload/load cycle
costs tens of seconds. Repeated autostart launches quietly accumulated four during
development. Delete the pid file only if it is genuinely stale.

Counting worker processes is misleading on Windows: `.venv\Scripts\python.exe` is
a shim that re-execs `C:\Python\Python312\python.exe`, so **one** worker shows as
two processes, a parent and its child. Killing either takes down both.

**Restart the worker after changing this code.** It is a long-running process that
loads the package once, so a running worker keeps executing whatever it started
with. This bit during development: queued jobs lost their project attribution
because the worker predated the change that added it, while jobs submitted after
it recorded the field correctly. Symptom to recognise — the database has the new
value but the worker's output does not.

```bash
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'llmhelper.cli worker' } | Stop-Process -Force"
```

## Safety boundaries

1. **Filesystem.** `guard.py` is the only chokepoint. Reads are confined to
   `paths.read_roots` and refused for `paths.deny_globs` (`.git`, `.venv`,
   `__pycache__`, `.ssh`, `.env`, `*.key`, `*.pem`, `.claude`). Writes resolve
   under `results/` or raise. Both checks run **after** `Path.resolve()`, so `..`,
   symlinks, junctions and UNC paths cannot escape. 23 escape attempts are pinned
   in `tests/test_guard.py`.
2. **Local model output is untrusted data.** It comes from files on disk by way of
   a small model, so it is a prompt-injection surface. File content is wrapped in
   an explicit fence, the system prompts say not to obey instructions inside it,
   and every MCP result carries a banner saying the same to the caller.
3. **Resources.** Required RAM is `max(configured, model size × 1.15)` — a typo in
   `config.toml` cannot OOM the machine. A single worker serialises loads. Verified:
   the `heavy` tier refuses with a clear message when headroom is short.

Nothing the local model produces touches your projects. Moving output out of
`results/` is always a human decision.

## Verifying it still works

```bash
llmh doctor                                    # must be green
.venv/Scripts/python.exe -m pytest tests/ -q   # 55 tests
.venv/Scripts/python.exe tests/mcp_smoke.py    # speaks real MCP over stdio
llmh bench                                     # tok/s per tier
```

## Layout

```
llmhelper/
  config.py    tiers, budgets, path allowlist   guard.py    the sandbox
  client.py    OpenAI-compatible HTTP           runtime.py  lms + model residency
  store.py     queue, results, ledger           chunk.py    tokenising + chunking
  worker.py    the polling loop                 runner.py   the one execution path
  cli.py       command line                     mcp_server.py  tools for Claude
  handlers/    digest grep ask review transform transcribe
jobs/  results/  logs/  bench/  config.toml
```

## Known limits

- Map-phase concurrency is 4 (`budgets.map_concurrency`). Measured gain on a
  19-file digest was 56 s → 38 s — real but sublinear, because the reduce step is
  serial and the parallel slots share one GPU.
- `local_index` / `local_query` (embedding-backed corpus search) are planned but
  not built — the `embed` tier is configured and unused.
- Token counts use tiktoken `cl100k_base` as a proxy for what a frontier model
  would have paid. Close enough to steer decisions, not an exact bill.
- Nothing runs from the queue unless `llmh worker` is running.
- Scanned PDFs need OCR first; there is no OCR step in the pipeline.
- `.doc`, `.rtf` and `.epub` have no extractor. `.pptx` needs `python-pptx`.
- Editing the package's code requires restarting the MCP server, the worker and
  the dashboard. Only `config.toml` is picked up live.
