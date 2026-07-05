param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [string[]]$Strategies = @("random", "entropy", "quota_entropy", "quota_red_entropy", "debt_entropy", "red_entropy", "qfair"),

    [int[]]$Seeds = @(),

    [string]$Conda = "D:\conda\Scripts\conda.exe",

    [string]$LogDir = "tmp\matrix-runs"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force $LogDir | Out-Null
$summary = Join-Path $LogDir "matrix_runs.csv"
"seed,strategy,start_time,end_time,exit_code,log_file" | Set-Content -Path $summary -Encoding UTF8

$seedList = @($null)
if ($Seeds.Count -gt 0) {
    $seedList = $Seeds
}

foreach ($seed in $seedList) {
    foreach ($strategy in $Strategies) {
        $seedLabel = "config"
        if ($null -ne $seed) {
            $seedLabel = "seed$seed"
        }
        $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $logFile = Join-Path $LogDir ("run_" + $seedLabel + "_" + $strategy + ".log")
        "=== START $seedLabel $strategy $start ===" | Tee-Object -FilePath $logFile

        $cmdArgs = @("run", "-n", "te-fal", "python", ".\scripts\run_no_go.py", "--config", $Config, "--strategy", $strategy)
        if ($null -ne $seed) {
            $cmdArgs += @("--seed", [string]$seed)
        }
        & $Conda @cmdArgs 2>&1 | Tee-Object -FilePath $logFile -Append
        $exitCode = $LASTEXITCODE

        $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "=== END $seedLabel $strategy $end exit=$exitCode ===" | Tee-Object -FilePath $logFile -Append
        "$seedLabel,$strategy,$start,$end,$exitCode,$logFile" | Add-Content -Path $summary -Encoding UTF8

        if ($exitCode -ne 0) {
            throw "Strategy $strategy failed for $seedLabel with exit code $exitCode"
        }
    }
}
