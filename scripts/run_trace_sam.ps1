param(
    [ValidateSet("dry", "full")]
    [string]$Preset = "dry",
    [string]$Device = "cuda",
    [int]$EvalSteps = 0,
    [string]$Config = "",
    [switch]$SkipSrPretrain,
    [switch]$SkipSrTopology,
    [switch]$SkipSrMetric,
    [switch]$SkipSrFid,
    [switch]$SkipJoint,
    [switch]$SkipEval,
    [switch]$SkipAug,
    [switch]$SkipAugRecognition
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONUNBUFFERED = "1"

if (-not (Test-Path -LiteralPath $Python)) {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCmd) {
        $Python = $PythonCmd.Source
    } else {
        throw "Virtual environment not found and python is not on PATH: $Python"
    }
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    if ($Preset -eq "dry") {
        $Config = Join-Path $Root "configs\demo_cpu.yaml"
    } else {
        $Config = Join-Path $Root "configs\paper_trace_sam_sr.yaml"
    }
}

$ExtraArgs = @()
if ($SkipSrPretrain) { $ExtraArgs += "--skip_sr_pretrain" }
if ($SkipSrTopology) { $ExtraArgs += "--skip_sr_topology" }
if ($SkipSrMetric) { $ExtraArgs += "--skip_sr_metric" }
if ($SkipSrFid) { $ExtraArgs += "--skip_sr_fid" }
if ($SkipJoint) { $ExtraArgs += "--skip_joint" }
if ($SkipEval) { $ExtraArgs += "--skip_eval" }
if ($SkipAug) { $ExtraArgs += "--skip_aug" }
if ($SkipAugRecognition) { $ExtraArgs += "--skip_aug_recognition" }
if ($EvalSteps -gt 0) {
    $ExtraArgs += "--eval_steps"
    $ExtraArgs += "$EvalSteps"
}

if ($Preset -eq "dry") {
    Write-Host ""
    Write-Host "================== TRACE-SAM-SR dry workflow =================="
    $Manifest = Join-Path $Root "demo_data\manifest.csv"
    if (-not (Test-Path -LiteralPath $Manifest)) {
        & $Python (Join-Path $Root "tools\make_demo_dataset.py") --out (Join-Path $Root "demo_data") --image-size 64 --overwrite
    }
    & $Python -m trace_sam.scripts.validate_trace_data --config $Config
    & $Python -m trace_sam.scripts.generate_trace_aug_patches --config $Config --dry_run
    & $Python -m trace_sam.scripts.run_full_pipeline --config $Config --device $Device --dry_run $ExtraArgs
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "================== TRACE-SAM-SR full workflow =================="
& $Python -m trace_sam.scripts.run_full_pipeline --config $Config --device $Device $ExtraArgs
