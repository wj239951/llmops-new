# 停止本机原生部署的前后端进程（替代 docker compose down；数据卷不受影响）
$ports = 8000, 3000
foreach ($p in $ports) {
    $conns = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
    if (-not $conns) { Write-Host "  :$p 未在运行"; continue }
    foreach ($c in $conns) {
        $procId = $c.OwningProcess
        $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Host "  :$p 已停止 (PID $procId, $name)"
    }
}
Write-Host "MySQL 与 Ollama 是宿主机服务，未被本脚本停止。"
