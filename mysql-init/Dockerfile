# 把数据库备份烘进镜像，而不是用 bind mount 挂进去。
# 原因：Docker Desktop 对 U 盘(exFAT)上的单文件挂载会静默挂成空目录，
# initdb 读到目录 → 导入被跳过 → 容器仍显示 healthy，但库里根本没有 llmops_db。
# build 上下文由 docker client 读取再传给守护进程，任何盘符都能用，所以这条路稳。
FROM mysql:8.0

COPY llmops_db.sql /docker-entrypoint-initdb.d/01-llmops_db.sql
# 知识库扩充种子（33 篇厂内规程，2026-11-17）。initdb 按文件名顺序执行，
# 必须在 01 建库建表之后；语句是幂等的（ON DUPLICATE KEY UPDATE），重复导入不会重复。
COPY 02_knowledge_expand.sql /docker-entrypoint-initdb.d/02_knowledge_expand.sql
# 第二期语料：28 台设备 × 4 类文档 + 28 份大修方案 + 案例/制度，共 152 篇，
# 与第一期合计 196 篇、约 11.5 万字，切分后 338 条向量。同样幂等。
COPY 03_knowledge_bank.sql /docker-entrypoint-initdb.d/03_knowledge_bank.sql
