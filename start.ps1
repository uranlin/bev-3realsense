param([ValidateSet('demo','realsense','full')][string]$Mode = 'demo')
$ErrorActionPreference = 'Stop'
$bevPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $bevPython)) {
    throw '請先依 README 建立 .venv 並安裝 requirements-core.txt。'
}
$previousMode = $env:BEV_MODE
Push-Location $PSScriptRoot
try {
    $env:BEV_MODE = $Mode
    & $bevPython server.py
} finally {
    $env:BEV_MODE = $previousMode
    Pop-Location
}
