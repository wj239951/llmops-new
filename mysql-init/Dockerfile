# 把数据库备份烘进镜像，而不是用 bind mount 挂进去。
# 原因：Docker Desktop 对 U 盘(exFAT)上的单文件挂载会静默挂成空目录，
# initdb 读到目录 → 导入被跳过 → 容器仍显示 healthy，但库里根本没有 llmops_db。
# build 上下文由 docker client 读取再传给守护进程，任何盘符都能用，所以这条路稳。
FROM mysql:8.0

COPY llmops_db.sql /docker-entrypoint-initdb.d/01-llmops_db.sql
