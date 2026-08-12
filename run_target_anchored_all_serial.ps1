param(
    [string]$OutputRoot = "J:\nlp\CD-C3DA\runs\reproducible",
    [string]$Cuda = "0",
    [int]$MonitorSeconds = 120
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StepRunner = Join-Path $ProjectRoot "run_target_anchored_pipeline.ps1"
$MonitorRoot = Join-Path $OutputRoot "_target_anchor_serial_monitor"
New-Item -ItemType Directory -Force -Path $MonitorRoot | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ControllerLog = Join-Path $MonitorRoot "controller-$Timestamp.log"
$HealthPath = Join-Path $MonitorRoot "health.json"
$PidPath = Join-Path $MonitorRoot "controller_pid.txt"
[System.IO.File]::WriteAllText($PidPath, [string]$PID, [Text.UTF8Encoding]::new($false))

$RecipeIds = @{
    1 = "laptop14_to_rest15_target_anchor_step1_v1"
    2 = "laptop14_to_rest15_target_anchor_step2_gap_v1"
    3 = "laptop14_to_rest15_target_anchor_step3_tiered_v1"
}
$RunIds = @{
    1 = "laptop14-rest15-target-anchor-step1-seed1000-v1"
    2 = "laptop14-rest15-target-anchor-step2-gap-seed1000-v1"
    3 = "laptop14-rest15-target-anchor-step3-tiered-seed1000-v1"
}

function Write-ControllerEvent([string]$Message) {
    $Line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -LiteralPath $ControllerLog -Value $Line -Encoding UTF8
}

function Write-Health([int]$Step, [System.Diagnostics.Process]$Process, [string]$State) {
    $RunRoot = Join-Path (Join-Path $OutputRoot $RecipeIds[$Step]) $RunIds[$Step]
    $StageStatusPath = Join-Path $RunRoot "stage_status.json"
    $ActiveStage = $null
    $CompletedStages = 0
    if (Test-Path -LiteralPath $StageStatusPath) {
        try {
            $StageStatus = Get-Content -Raw -LiteralPath $StageStatusPath | ConvertFrom-Json
            foreach ($Property in $StageStatus.PSObject.Properties) {
                if ($Property.Value.status -eq "completed") { $CompletedStages += 1 }
                if ($Property.Value.status -eq "running") { $ActiveStage = $Property.Name }
            }
        } catch {
            $ActiveStage = "status-read-retry"
        }
    }
    $Gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    $Health = [ordered]@{
        checked_at = (Get-Date).ToString("o")
        controller_pid = $PID
        child_pid = $Process.Id
        child_has_exited = $Process.HasExited
        state = $State
        step = $Step
        run_id = $RunIds[$Step]
        active_stage = $ActiveStage
        completed_stages = $CompletedStages
        gpu_util_memory_mib = $Gpu
        run_root = $RunRoot
        controller_log = $ControllerLog
    }
    $Temporary = "$HealthPath.tmp"
    [System.IO.File]::WriteAllText(
        $Temporary,
        ($Health | ConvertTo-Json -Depth 4),
        [Text.UTF8Encoding]::new($false)
    )
    Move-Item -Force -LiteralPath $Temporary -Destination $HealthPath
}

Write-ControllerEvent "serial experiment controller started; pid=$PID"
foreach ($Step in 1, 2, 3) {
    $Stdout = Join-Path $MonitorRoot "step$Step-$Timestamp.stdout.log"
    $Stderr = Join-Path $MonitorRoot "step$Step-$Timestamp.stderr.log"
    $Arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $StepRunner,
        "-Step", [string]$Step, "-OutputRoot", $OutputRoot, "-Cuda", $Cuda
    )
    Write-ControllerEvent "starting step=$Step run_id=$($RunIds[$Step])"
    $Child = Start-Process -FilePath "powershell.exe" -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    while (-not $Child.HasExited) {
        Write-Health -Step $Step -Process $Child -State "running"
        Start-Sleep -Seconds $MonitorSeconds
        $Child.Refresh()
    }
    Write-Health -Step $Step -Process $Child -State "exited"
    Write-ControllerEvent "finished step=$Step exit_code=$($Child.ExitCode) stdout=$Stdout stderr=$Stderr"
    if ($Child.ExitCode -ne 0) {
        Write-ControllerEvent "serial experiment controller stopped after failure"
        exit $Child.ExitCode
    }
}
Write-ControllerEvent "all three steps completed"
exit 0
