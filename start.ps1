param([ValidateRange(1,65535)][int]$Port = 8010, [string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$taskBackendDir = $PSScriptRoot
$taskProjectRoot = Split-Path -Parent $taskBackendDir
if (-not $PythonPath) { $PythonPath = Join-Path $taskProjectRoot '.venv-api\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw '项目 Python 环境不存在。请先运行根目录 setup-local.ps1，或用 -PythonPath 指定已安装锁定依赖的 Python 3.11 x64。'
}
$taskPython = (Resolve-Path -LiteralPath $PythonPath).Path
$taskPythonVersion = & $taskPython -B -c 'import sys,struct; print(sys.version_info.major,sys.version_info.minor,struct.calcsize(chr(80))*8)'
if ($LASTEXITCODE -ne 0 -or $taskPythonVersion -ne '3 11 64') { throw '后端需要 Python 3.11 x64；请使用项目虚拟环境。' }
$taskEnvFile = Join-Path $taskBackendDir '.env'
if (Test-Path -LiteralPath $taskEnvFile) {
    foreach ($taskLine in Get-Content -LiteralPath $taskEnvFile) {
        if ($taskLine -match '^\s*([A-Z][A-Z0-9_]*)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim(), 'Process')
        }
    }
}
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $taskProjectRoot
& $taskPython -m uvicorn backend.app:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
