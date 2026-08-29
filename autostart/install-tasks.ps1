# Registers the helper's background processes with Task Scheduler.
#
# The existing Startup-folder entries run at *logon* - nothing starts until
# somebody signs in. This registers real scheduled tasks that fire at *boot*.
#
# Boot-time tasks use an S4U principal ("Service For User"): the task runs as you,
# whether or not you are logged on, and Windows does not store your password to do
# it. The trade-off is a restricted token - no network shares, no DPAPI-encrypted
# secrets. Everything here reads local files and talks to localhost, so that is
# fine, but see the note about LM Studio at the bottom.
#
# Must run elevated. Registering a task for another session's user is a privileged
# operation and there is no way around that.
#
#   powershell -ExecutionPolicy Bypass -File autostart\install-tasks.ps1
#   powershell -ExecutionPolicy Bypass -File autostart\install-tasks.ps1 -Trigger Logon
#   powershell -ExecutionPolicy Bypass -File autostart\install-tasks.ps1 -Remove

[CmdletBinding()]
param(
    # Boot  = AtStartup, runs with no user logged on (S4U).
    # Logon = AtLogOn, the same timing as the Startup-folder entries it replaces.
    [ValidateSet('Boot', 'Logon')]
    [string]$Trigger = 'Boot',

    # Serve MCP over HTTP on 0.0.0.0:8765 at boot. Off by default: it hands read
    # access to everything under paths.read_roots to anything that reaches the
    # port, with no authentication.
    [switch]$IncludeMcpHttp,

    # The dashboard on 127.0.0.1:8770.
    [switch]$SkipDashboard,

    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$prefix = 'LocalLLMHelper'
$root = Split-Path -Parent $PSScriptRoot

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "This needs an elevated shell." -ForegroundColor Yellow
    Write-Host "Right-click Windows Terminal or PowerShell, 'Run as administrator', then:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 1
}

# --------------------------------------------------------------------- remove

$names = @("$prefix-Server", "$prefix-Worker", "$prefix-Dashboard", "$prefix-McpHttp")

if ($Remove) {
    foreach ($n in $names) {
        $existing = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
        if ($existing) {
            Unregister-ScheduledTask -TaskName $n -Confirm:$false
            Write-Host "  removed $n"
        }
    }
    Write-Host "Done. Startup-folder entries, if any, were left alone." -ForegroundColor Green
    exit 0
}

# --------------------------------------------------------------------- checks

$tasks = @(
    @{ Name = "$prefix-Server";    Script = 'start-server.cmd';   Delay = $null   },
    @{ Name = "$prefix-Worker";    Script = 'start-worker.cmd';   Delay = 'PT30S' }
)
if (-not $SkipDashboard) {
    $tasks += @{ Name = "$prefix-Dashboard"; Script = 'start-ui.cmd'; Delay = 'PT45S' }
}
if ($IncludeMcpHttp) {
    $tasks += @{ Name = "$prefix-McpHttp"; Script = 'start-mcp-http.cmd'; Delay = 'PT45S' }
}

foreach ($t in $tasks) {
    $p = Join-Path $PSScriptRoot $t.Script
    if (-not (Test-Path $p)) { throw "missing launcher: $p" }
}

$vbs = Join-Path $PSScriptRoot 'run-hidden.vbs'
if (-not (Test-Path $vbs)) { throw "missing launcher: $vbs" }

# --------------------------------------------------------------------- install

$user = "$env:USERDOMAIN\$env:USERNAME"

if ($Trigger -eq 'Boot') {
    # S4U: runs whether or not the user is logged on, with no stored password.
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
} else {
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
}

# Never stop these for running long, and never kill one mid-job on battery.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

foreach ($t in $tasks) {
    $cmd = Join-Path $PSScriptRoot $t.Script

    # wscript runs the .cmd with no console window, the same way the Startup
    # entries did, so a boot does not flash three black windows.
    $action = New-ScheduledTaskAction -Execute 'wscript.exe' `
        -Argument "`"$vbs`" `"$cmd`"" -WorkingDirectory $root

    if ($Trigger -eq 'Boot') {
        $trg = New-ScheduledTaskTrigger -AtStartup
    } else {
        $trg = New-ScheduledTaskTrigger -AtLogOn -User $user
    }
    # The worker and dashboard race the server otherwise; start-worker.cmd also
    # polls for it, but staggering keeps three model-server probes off one boot.
    if ($t.Delay) { $trg.Delay = $t.Delay }

    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trg `
        -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "  registered $($t.Name)  ($Trigger$(if ($t.Delay) { " +$($t.Delay)" }))"
}

# ------------------------------------------------------- retire the duplicates

$startup = [Environment]::GetFolderPath('Startup')
$stale = Get-ChildItem -Path $startup -Filter "$prefix-*" -ErrorAction SilentlyContinue
if ($stale) {
    Write-Host ""
    Write-Host "Startup-folder entries found - these would start a second copy:" -ForegroundColor Yellow
    foreach ($f in $stale) {
        Remove-Item $f.FullName -Force
        Write-Host "  removed $($f.Name)"
    }
    Write-Host "  (the singleton locks would have caught it, but noisily)"
}

Write-Host ""
Write-Host "Installed. Verify with:" -ForegroundColor Green
Write-Host "  Get-ScheduledTask -TaskName '$prefix-*' | Select TaskName,State"
Write-Host "  Start-ScheduledTask -TaskName '$prefix-Server'    # test without rebooting"
Write-Host ""
if ($Trigger -eq 'Boot') {
    Write-Host "Note: LM Studio's server is started by lms.exe from your user profile." -ForegroundColor Yellow
    Write-Host "S4U gives a restricted token, so if the server does not come up at boot"
    Write-Host "before you sign in, re-run with -Trigger Logon. Check it after a reboot:"
    Write-Host "  llmh doctor"
}
