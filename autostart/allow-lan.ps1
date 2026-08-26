<#
    Open exactly the ports another machine on your LAN needs, and nothing else.

    RUN FROM AN ELEVATED POWERSHELL. Firewall changes require administrator.

    Why this is needed: this PC's adapter is on the Windows "Public" firewall
    profile, whose default inbound action is Block. Windows drops inbound TCP
    silently, so from another machine every port times out - 445, 3389 and 135
    included - while ping still replies. That pattern is the signature of the
    profile block, not of any one service being down.

    What this does NOT do: it does not switch the adapter to "Private". That
    would also work, and would also expose file sharing, remote desktop and
    everything else Windows opens on a trusted network. Two scoped rules are a
    much smaller change.

    NOTE: this file is deliberately plain ASCII. Windows PowerShell 5.1 reads
    .ps1 as ANSI unless there is a BOM, so a stray typographic dash becomes
    mojibake and the script fails to parse.
#>

[CmdletBinding()]
param(
    # The subnet allowed to connect. Keep this tight. "Any" would expose these
    # ports on every network this machine ever joins, including hotel Wi-Fi.
    [string]$Subnet = "192.168.50.0/24",

    # 1234 = LM Studio model API (a laptop running its own copy borrows the GPU)
    # 8765 = MCP over HTTP (a laptop calling this PC's tools, reading its files)
    [int[]]$Ports = @(1234, 8765),

    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$admin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $admin) {
    Write-Host "This needs an elevated PowerShell." -ForegroundColor Red
    Write-Host "  Start menu, type 'powershell', press Ctrl+Shift+Enter, then run:"
    Write-Host "  & '$PSCommandPath'"
    exit 1
}

foreach ($port in $Ports) {
    $name = "local_llm_helper $port"

    $existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Remove-NetFirewallRule -DisplayName $name
        Write-Host "removed existing rule: $name" -ForegroundColor DarkGray
    }
    if ($Remove) { continue }

    New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP `
        -LocalPort $port -RemoteAddress $Subnet -Action Allow -Profile Any `
        -Description "Allow $Subnet to reach local_llm_helper on TCP $port." | Out-Null

    Write-Host "allowed TCP $port from $Subnet" -ForegroundColor Green
}

if ($Remove) {
    Write-Host ""
    Write-Host "Rules removed. Those ports are closed to the network again."
    exit 0
}

Write-Host ""
Write-Host "--- verifying ---"
foreach ($port in $Ports) {
    $listening = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $listening) {
        Write-Host ("  TCP {0}: rule added, but nothing is listening." -f $port) -ForegroundColor Yellow
        if ($port -eq 1234) { Write-Host "    start it with:  llmh server start" }
        if ($port -eq 8765) { Write-Host "    start it with:  autostart\start-mcp-http.cmd" }
        continue
    }
    $addr = ($listening | Select-Object -First 1).LocalAddress
    if ($addr -eq "127.0.0.1") {
        Write-Host ("  TCP {0}: bound to 127.0.0.1 only, so the LAN still cannot reach it." -f $port) -ForegroundColor Yellow
        Write-Host '    set  bind = "0.0.0.0"  under [server] in config.toml, then:  llmh server restart'
    } else {
        Write-Host ("  TCP {0}: listening on {1} - good" -f $port, $addr) -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Now test from the other machine:"
Write-Host "  Test-NetConnection 192.168.50.32 -Port 1234     (Windows)"
Write-Host "  nc -vz 192.168.50.32 1234                       (macOS / Linux)"
Write-Host ""
Write-Host "To undo:  & '$PSCommandPath' -Remove"
