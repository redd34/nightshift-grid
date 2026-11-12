<#
.SYNOPSIS
  Install the Nightshift Grid worker on Windows as a per-user "at logon" scheduled task.

.DESCRIPTION
  Why a scheduled task in the user's session and NOT a Windows service: services run in
  session 0, which has no desktop and no input queue. GetLastInputInfo there never sees the
  interactive user's keyboard/mouse, so the worker would believe the machine is always idle and
  never yield. A task triggered "at logon" for this user runs inside the user's session, sees
  their input, and stops when they log off.

  Idempotent: re-running updates code, venv and the task. Supports -WhatIf (dry run: prints
  every action, changes nothing) and -Uninstall [-Purge].

.EXAMPLE
  .\install_worker.ps1 -Coordinator http://10.0.0.5:8000 -Model qwen3:4b -TokenFile C:\tmp\worker.token -WhatIf
.EXAMPLE
  .\install_worker.ps1 -Uninstall -Purge
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Coordinator = $env:NIGHTSHIFT_COORDINATOR,
    [string]$WorkerId = $env:NIGHTSHIFT_WORKER_ID,
    [string]$Model = $(if ($env:NIGHTSHIFT_MODEL) { $env:NIGHTSHIFT_MODEL } else { 'qwen3:4b' }),
    [ValidateSet('', '1', '2', '3')][string]$Tier = '',
    [string]$TokenFile,                       # file to read the token from (copied, never on argv)
    [switch]$TokenStdin,                      # read the token from the pipeline / stdin
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'nightshift'),
    [double]$NativeIdleSeconds = 60,
    [double]$CpuBusyPercent = -1,              # -1 = worker default (65); 0 disables
    [double]$MinFreeRamGb = -1,                # -1 = worker default (2.0); 0 disables
    [switch]$AllowNoIdleSensor,               # run even if GetLastInputInfo can't see input (fail-open)
    [switch]$AllowBattery,
    [string]$Ollama = 'http://127.0.0.1:11434',
    [string]$Python,                          # python.exe >= 3.11 (default: py launcher / PATH)
    [switch]$PullModel,
    [switch]$NoStart,
    [switch]$Uninstall,
    [switch]$Purge
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$TaskName  = 'NightshiftWorker'
$SrcDir    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # repo root
$ConfigDir = Join-Path $env:APPDATA 'nightshift'
$TokenPath = Join-Path $ConfigDir 'worker.token'
$VenvPy    = Join-Path $InstallDir '.venv\Scripts\python.exe'
$VenvPyw   = Join-Path $InstallDir '.venv\Scripts\pythonw.exe'
$Launcher  = Join-Path $InstallDir 'run_worker.pyw'
$LogFile   = Join-Path $InstallDir 'worker.log'
$Me        = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

function Info($m) { Write-Host "==> $m" }

# ------------------------------------------------------------------ uninstall
if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($TaskName, 'Stop and unregister scheduled task')) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        }
    }
    if (Test-Path $InstallDir) {
        if ($PSCmdlet.ShouldProcess($InstallDir, 'Remove install dir')) { Remove-Item -Recurse -Force $InstallDir }
    }
    if ($Purge -and (Test-Path $ConfigDir)) {
        if ($PSCmdlet.ShouldProcess($ConfigDir, 'Remove config dir (token)')) { Remove-Item -Recurse -Force $ConfigDir }
    } elseif (-not $Purge) { Info "Token kept at $TokenPath (use -Purge to delete it)" }
    Info 'Done.'
    return
}

# ------------------------------------------------------------------ validate
if (-not $Coordinator) { throw '-Coordinator is required (e.g. http://10.0.0.5:8000)' }
if ($Coordinator -notmatch '^https?://') { throw '-Coordinator must start with http:// or https://' }
$Coordinator = $Coordinator.TrimEnd('/')
if (-not $WorkerId) {
    $h = ($env:COMPUTERNAME).ToLowerInvariant() -replace '[^a-z0-9-]', '-' -replace '-+', '-'
    $WorkerId = 'node-' + $h.Trim('-')
}
if ($WorkerId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw "Invalid worker id '$WorkerId'" }
if (-not (Test-Path (Join-Path $SrcDir 'worker_daemon.py'))) { throw "worker_daemon.py not found in $SrcDir" }

# ------------------------------------------------------------------ python >= 3.11
function Test-Py([string]$exe, [string[]]$pre = @()) {
    try { & $exe @pre -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>$null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
}
$PyExe = $null; $PyPre = @()
if ($Python) {
    if (-not (Test-Py $Python)) { throw "$Python is not Python >= 3.11" }
    $PyExe = $Python
} elseif ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-Py 'py' @('-3'))) {
    $PyExe = 'py'; $PyPre = @('-3')
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-Py 'python')) {
    $PyExe = 'python'
} else { throw 'Python >= 3.11 not found (winget install Python.Python.3.12, or pass -Python)' }
Info "Using $(& $PyExe @PyPre --version 2>&1)"

# ------------------------------------------------------------------ token (user-only ACL)
$NewToken = $null
if ($TokenStdin) { $NewToken = [Console]::In.ReadLine() }
elseif ($TokenFile) { $NewToken = (Get-Content -Raw -LiteralPath $TokenFile) }
if ($NewToken) { $NewToken = $NewToken.Trim() }
if (-not $NewToken -and -not (Test-Path $TokenPath)) {
    if ($WhatIfPreference) { Write-Warning "No token given and $TokenPath missing (a real install would fail here)" }
    else { throw 'No token: pass -TokenFile PATH or -TokenStdin (generate with: python -c "import secrets;print(secrets.token_urlsafe(32))")' }
}

function Set-UserOnlyAcl([string]$path, [bool]$isDir) {
    $acl = New-Object System.Security.AccessControl.FileSecurity
    if ($isDir) { $acl = New-Object System.Security.AccessControl.DirectorySecurity }
    $acl.SetAccessRuleProtection($true, $false)   # drop inherited ACEs (Users, Administrators...)
    $inherit = if ($isDir) { 'ContainerInherit,ObjectInherit' } else { 'None' }
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($Me, 'FullControl', $inherit, 'None', 'Allow')
    $acl.AddAccessRule($rule)
    Set-Acl -LiteralPath $path -AclObject $acl
}

if ($NewToken) {
    if ($PSCmdlet.ShouldProcess($TokenPath, "Write token ($($NewToken.Length) chars) with user-only ACL")) {
        New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
        Set-UserOnlyAcl $ConfigDir $true
        Set-Content -LiteralPath $TokenPath -Value $NewToken -NoNewline -Encoding ascii
        Set-UserOnlyAcl $TokenPath $false
    }
} else { Info "Keeping existing token $TokenPath" }

# ------------------------------------------------------------------ code + venv
if ($PSCmdlet.ShouldProcess($InstallDir, 'Copy worker code and create venv + pip install')) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Get-ChildItem -Path $SrcDir -Filter '*.py' | Copy-Item -Destination $InstallDir -Force
    Copy-Item (Join-Path $SrcDir 'requirements.txt') $InstallDir -Force
    if (-not (Test-Path $VenvPy) -or -not (Test-Py $VenvPy)) { & $PyExe @PyPre -m venv --clear (Join-Path $InstallDir '.venv') }
    & $VenvPy -m pip install --quiet --disable-pip-version-check --upgrade pip
    & $VenvPy -m pip install --quiet --disable-pip-version-check -r (Join-Path $InstallDir 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

# ------------------------------------------------------------------ ollama + model
try {
    $tags = Invoke-RestMethod -Uri "$($Ollama.TrimEnd('/'))/api/tags" -TimeoutSec 3
    $names = @($tags.models | ForEach-Object { $_.name; $_.name -replace ':latest$', '' })
    if ($names -contains $Model) { Info "Model $Model is available" }
    elseif ($PullModel) { if ($PSCmdlet.ShouldProcess($Model, 'ollama pull')) { & ollama pull $Model } }
    else { Write-Warning "Model $Model is not pulled. Run: ollama pull $Model  (or re-run with -PullModel)" }
} catch {
    Write-Warning "Ollama not reachable at $Ollama. Install from https://ollama.com, start it, then: ollama pull $Model"
}

# ------------------------------------------------------------------ launcher
# pythonw.exe has no console window (none pops up at logon) but also no stdout, so this tiny
# launcher points stdout/stderr at the log file and runs the worker in-process. The worker's
# exit code becomes the task's result, which drives Task Scheduler's restart-on-failure.
$argv = @('--id', $WorkerId, '--coordinator', $Coordinator, '--model', $Model,
          '--ollama', $Ollama, '--token-file', $TokenPath)
if ($Tier) { $argv += @('--tier', $Tier) }
if ($NativeIdleSeconds -gt 0) { $argv += @('--native-idle-seconds', "$NativeIdleSeconds") }
if ($CpuBusyPercent -ge 0) { $argv += @('--max-host-cpu-pct', "$CpuBusyPercent") }
if ($MinFreeRamGb -ge 0) { $argv += @('--min-free-ram-gb', "$MinFreeRamGb") }
if ($AllowNoIdleSensor) { $argv += '--allow-no-idle-sensor' }
if ($AllowBattery) { $argv += '--allow-battery' }
$argvJson = ConvertTo-Json -Compress -InputObject @($argv)
$daemonJson = ConvertTo-Json -Compress -InputObject (Join-Path $InstallDir 'worker_daemon.py')
$logJson = ConvertTo-Json -Compress -InputObject $LogFile
$launcherBody = @"
# Generated by deploy/windows/install_worker.ps1 - do not edit; re-run the installer instead.
import json, os, runpy, sys
daemon = json.loads(r'''$daemonJson''')
log = open(json.loads(r'''$logJson'''), 'a', buffering=1, encoding='utf-8')
sys.stdout = sys.stderr = log
os.chdir(os.path.dirname(daemon))
sys.path.insert(0, os.path.dirname(daemon))
sys.argv = [daemon] + json.loads(r'''$argvJson''')
runpy.run_path(daemon, run_name='__main__')
"@
if ($PSCmdlet.ShouldProcess($Launcher, 'Write launcher')) {
    Set-Content -LiteralPath $Launcher -Value $launcherBody -Encoding utf8
} else { Write-Host "---- $Launcher ----`n$launcherBody`n----" }

# ------------------------------------------------------------------ scheduled task
$action    = New-ScheduledTaskAction -Execute $VenvPyw -Argument "`"$Launcher`"" -WorkingDirectory $InstallDir
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $Me
$principal = New-ScheduledTaskPrincipal -UserId $Me -LogonType Interactive -RunLevel Limited
# Priority 7 = BELOW_NORMAL. Restart every 1 min on failure (a 401 exit retries slowly, no hot loop).
# Battery policy is the worker's own (--allow-battery), so Task Scheduler must not second-guess it.
$settings  = New-ScheduledTaskSettingsSet -Priority 7 -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
               -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
               -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
if ($PSCmdlet.ShouldProcess($TaskName, "Register at-logon task for $Me ($VenvPyw `"$Launcher`")")) {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Description 'Nightshift Grid preemptible worker (harvests idle cycles; yields to the user)' -Force | Out-Null
    if (-not $NoStart) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Start-ScheduledTask -TaskName $TaskName
    }
}

Info "Worker '$WorkerId' -> $Coordinator"
Info "Logs:   Get-Content -Wait '$LogFile'"
Info "Status: Get-ScheduledTask $TaskName | Get-ScheduledTaskInfo"
Info "Chaos:  New-Item `"`$env:TEMP\preempt_$WorkerId`"   (Remove-Item it to release)"
