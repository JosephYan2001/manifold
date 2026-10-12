param(
    [string]$Python = "python",
    [ValidateSet("an_base", "an_direction64", "an_batch64")]
    [string[]]$Runs = @("an_base", "an_direction64", "an_batch64"),
    [switch]$DryRun,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$sourceRoot = "manifold_project/experiments/results/navigation_diagnosis_20261010_2m_s40_v1"
$targetRoot = "manifold_project/experiments/results/navigation_diagnosis_20261012_4m_s40_v1"

foreach ($run in $Runs) {
    $runArguments = @(
        "-m", "manifold_project.experiments",
        "--budget", "4000000",
        "--device", "cuda",
        "--plot", "--plot-every", "5",
        "--allow-duplicate-openmp"
    )
    if ($Resume) {
        $runArguments += @("--resume", "$targetRoot/$run")
    } else {
        $runArguments += @("--resume", "$sourceRoot/$run", "--output", "$targetRoot/$run")
    }
    if ($DryRun) { $runArguments += "--dry-run" }
    & $Python @runArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Extension failed for $run (exit code $LASTEXITCODE)."
    }
}
