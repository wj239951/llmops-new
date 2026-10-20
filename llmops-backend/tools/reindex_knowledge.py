# 把 MySQL 里的知识库文档重新灌进 Chroma 向量库。
# 迁移包只带了数据库、没带 chroma_db，所以首次启动后 RAG 检索是 0 命中，
# 必须跑一次本脚本重建向量（需要 embedding 通道可用：Ollama bge-m3 或硅基流动 Key）。
# 用法：docker compose exec backend python tools/reindex_knowledge.py
import os
import sys

# 以脚本方式运行时，Python 只把 tools/ 放进 import 路径，找不到 app 包；
# 这里显式把项目根（/app）加进去，命令就不用依赖 -m 或 PYTHONPATH。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.knowledge import KnowledgeDocument
from app.services.rag_service import RAGService, vectorstore


def main():
    db = SessionLocal()
    try:
        docs = db.query(KnowledgeDocument).order_by(KnowledgeDocument.id).all()
        print(f"待重新向量化：{len(docs)} 篇文档")
        if not docs:
            print("MySQL 知识库里没有文档，无需重建。")
            return

        svc = RAGService()
        total = 0
        for doc in docs:
            # 先按文档 id 清掉残留向量，再重新写入，避免重复跑产生脏数据
            vectorstore._collection.delete(where={"id": doc.id})
            svc._sync_to_chroma(doc)
            n = len(vectorstore._collection.get(where={"id": doc.id})["ids"])
            total += n
            print(f"  id={doc.id} 《{doc.title}》 -> {n} 个片段")
        print(f"完成，共写入 {total} 条向量。")
    finally:
        db.close()


if __name__ == "__main__":
    main()
