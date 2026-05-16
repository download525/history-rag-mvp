$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $projectDir "..\venv\Scripts\Activate.ps1"

Set-Location $projectDir

if (-not (Test-Path $venvActivate)) {
    Write-Host "Не найдено виртуальное окружение: $venvActivate"
    Write-Host "Активируй свое окружение вручную и запусти: python web_app.py"
    exit 1
}

. $venvActivate
python web_app.py
