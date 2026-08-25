# Connecting a laptop — macOS or Linux

**There are two `.mcp.json`-shaped files in this repo. They are not interchangeable.**

| File | For | Transport |
|---|---|---|
| `../.mcp.json` | the Windows desktop only | stdio, absolute `D:\...` path |
| `mcp-over-http.json` (this folder) | any other machine | http |

Copying the first one to a Mac fails in milliseconds with
`ENOENT: Executable not found in $PATH: "D:\local_llm_helper\.venv\Scripts\python.exe"`.
A drive letter is not a path on macOS, and a macOS venv interpreter is
`.venv/bin/python`, never `.venv\Scripts\python.exe`. That is expected, not a
misconfiguration — a stdio server can only ever be spawned by a session on the
same machine as the executable.

---

## Option B — use the desktop's files from the laptop

Save `mcp-over-http.json` as `.mcp.json` in a project folder on the laptop:

```bash
cp mcp-over-http.json ~/some-project/.mcp.json
```

No CLI needed. For every project instead of one, put the same `mcpServers` object
under the top-level key in `~/.claude.json`. Restart Claude Code either way.

Two things must be true on the desktop, and neither is on by default:

1. **The HTTP server is running** — `D:\local_llm_helper\autostart\start-mcp-http.cmd`
2. **The firewall allows the port.** Both desktop interfaces are on the Windows
   **Public** profile, where inbound is blocked. In an administrator shell there:

   ```
   New-NetFirewallRule -DisplayName "local_llm_helper MCP" -Direction Inbound -Protocol TCP -LocalPort 8765 -RemoteAddress 192.168.1.0/24 -Action Allow
   ```

   Keep `-RemoteAddress` scoped to the subnet. The server has no authentication,
   and whatever reaches the port can read everything under `paths.read_roots`.

This gives the laptop tools that read the **desktop's** disk — `D:\`,
`C:/Users/<you>\Downloads`, `C:/Users/<you>\.lmstudio\server-logs`. Not the laptop's own
files.

---

## Option A — use the laptop's own files, on the desktop's GPU

Install the package on the laptop. Only the model call crosses the network, and no
filesystem is exposed, which makes this the safer option.

```bash
git clone <this repo> ~/local_llm_helper && cd ~/local_llm_helper
python3 -m venv .venv
.venv/bin/pip install mcp httpx sqlalchemy typer rich tiktoken psutil
echo "$PWD" > .venv/lib/python3*/site-packages/llmhelper.pth
```

Then edit `config.toml` for the laptop:

```toml
[server]
base_url = "http://DESKTOP-LAN-IP:1234/v1"
lms_exe  = "/usr/local/bin/lms"          # need not exist; see below

[paths]
read_roots = ["/Users/you/projects"]      # the laptop's own files
```

Register it as a stdio server on the laptop — note the macOS interpreter path:

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "/Users/you/local_llm_helper/.venv/bin/python",
      "args": ["-m", "llmhelper.mcp_server"]
    }
  }
}
```

The desktop must publish LM Studio to the LAN for this
(`lms server start --bind 0.0.0.0`, plus a firewall rule for port 1234), and have
the model you want already loaded.

**`lms_exe` not existing is fine and handled.** Without LM Studio's CLI the client
runs in remote mode: it does not try to load, unload or enumerate models, and
leaves residency to whichever machine actually runs the server. `llmh doctor` will
report the CLI as missing — that is accurate, not a failure. Verified by running a
full digest with the CLI path pointed at a file that does not exist: it selected a
tier, produced a correctly cited answer, and never touched a local binary.
