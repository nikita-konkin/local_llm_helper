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

## Test before wiring it in

```bash
./test-connection.sh 192.168.50.32
```

Checks reachability, both ports, the MCP handshake and — the part that actually
matters — a real tool call. Claude Code stops retrying a server that failed
earlier in a session, so proving the path with curl first means a retry is spent
on something already known to work rather than on a hypothesis.

The handshake is not evidence. It is one fast round trip and survives almost
anything; a tool call runs for seconds while the model works, and that is the
part which fails. Any test that stops at "handshake succeeded" proves nothing.

Nor is echoing back a string you sent. Several MCP error shapes include the
arguments in the message, so a test that greps its own prompt in the response
passes on failure. The check is `"isError":false` from the envelope **plus** a
token that appears nowhere in the request - the model's answer to 6x7 - so it
can only be satisfied by inference having actually run.

**None of this needs a CLI.** If Claude Code still refuses the server after the
test passes, quit it completely and reopen: a new session re-reads the config and
retries. To register by hand, `<project>/.mcp.json` is the whole file.

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

## Option A - the laptop's own files, the desktop's GPU  (recommended)

Your files stay here and are read here. Only the model call crosses the network,
so no filesystem is exposed. This is what you want when the work is on this
machine: option B hands you tools that read the *desktop's* disk, where a path
from a repo on this laptop does not exist at all.

Get a copy of the repo onto this machine, then:

```bash
cd ~/local_llm_helper
./laptop/setup-option-a.sh 192.168.50.32
```

It checks Python, confirms the desktop is reachable and lists its models, builds
a venv, writes `config.toml`, and finishes by running a real generation against
the desktop's GPU. A pass means the whole chain works, not that files were
created.

Then edit one line in `config.toml`:

```toml
read_roots = ["/Users/you/projects"]
```

That is the security boundary: the only directories the local model may read.

Finally save this as `.mcp.json` in a project folder, and restart Claude Code:

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

This is a stdio server, so Claude Code launches it here and `local_digest` reads
this machine's filesystem. That is the whole point of option A.

### Two things that will surprise you

**`lms_exe` points at a file that does not exist, deliberately.** Without LM
Studio's CLI the client runs in *remote mode*: it does not try to load, unload or
enumerate models, and leaves residency to the machine actually serving. `llmh
doctor` reporting the CLI as missing is accurate, not a fault.

**The tier declares an 8192 context, not 131072.** This machine cannot load a
model, so if nothing is resident on the desktop LM Studio JIT-loads at *its* own
default, which is 8192. Claiming more here would size chunks for a window you
may not get, and it surfaces as "Context size has been exceeded" rather than
anything that points at the cause. For the bigger window, load the model on the
desktop first and raise this number to match.
