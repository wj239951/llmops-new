# 本机原生部署启动脚本（替代 docker compose up -d）
# 用法：在 LLMOPS_NEW 目录执行  powershell -ExecutionPolicy Bypass -File .\start.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend  = Join-Path $root "llmops-backend"
$frontend = Join-Path $root "llmops-frontend"
$node     = "C:\node-v22.19.0-win-x64\node.exe"

function Test-Port([int]$port) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
}

if (Test-Port 8000) {
    Write-Host "backend  :8000 已在运行，跳过"
} else {
    Start-Process -FilePath (Join-Path $backend ".venv\Scripts\uvicorn.exe") `
        -ArgumentList "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $backend -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root "backend.log") `
        -RedirectStandardError  (Join-Path $root "backend.err.log")
    Write-Host "backend  启动中 -> http://127.0.0.1:8000/docs"
}

if (Test-Port 3000) {
    Write-Host "frontend :3000 已在运行，跳过"
} else {
    Start-Process -FilePath $node `
        -ArgumentList "node_modules\vite\bin\vite.js", "preview" `
        -WorkingDirectory $frontend -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root "frontend.log") `
        -RedirectStandardError  (Join-Path $root "frontend.err.log")
    Write-Host "frontend 启动中 -> http://127.0.0.1:3000"
}

Start-Sleep -Seconds 6
Write-Host "`n监听状态："
8000, 3000 | ForEach-Object {
    $p = $_
    if (Test-Port $p) { Write-Host ("  :{0}  LISTENING" -f $p) } else { Write-Host ("  :{0}  未起来，看 {1} 日志" -f $p, $root) }
}
