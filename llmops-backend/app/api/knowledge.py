# 知识库文档 API 路由。
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Response
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.knowledge import KnowledgeCreate, KnowledgeOut, KnowledgePage, KnowledgeUpdate
from app.services.rag_service import RAGService
from app.services.file_parser import parse_file

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
service = RAGService()

@router.post("", response_model=KnowledgeOut)
def create_document(data: KnowledgeCreate, db: Session = Depends(get_db)):
    return service.add_document(db, data.title, data.content, data.source)

@router.post("/upload", response_model=KnowledgeOut)
async def upload_file(
    file: UploadFile = File(...),
    title: str | None = None,
    db: Session = Depends(get_db)
):
    """上传文件，解析内容后存入知识库"""
    try:
        content = await parse_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件解析失败: {str(e)}")

    # 用文件名作为默认标题
    doc_title = title or file.filename
    return service.add_document(db, doc_title, content, source=file.filename)

@router.get("", response_model=list[KnowledgeOut])
def list_documents(
    response: Response,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(100, ge=1, le=500, description="每页条数"),
    db: Session = Depends(get_db),
):
    """分页返回文档；总数放在 X-Total-Count 响应头，列表结构保持不变，旧前端无需改动。"""
    total = service.count_documents(db)
    rows = service.list_documents(db, offset=(page - 1) * page_size, limit=page_size)
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Page"] = str(page)
    response.headers["X-Page-Size"] = str(page_size)
    return rows

@router.get("/paged", response_model=KnowledgePage)
def list_documents_paged(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """分页对象版列表：items + total，前端表格直接用。"""
    total = service.count_documents(db)
    rows = service.list_documents(db, offset=(page - 1) * page_size, limit=page_size)
    return {"items": rows, "total": total, "page": page, "page_size": page_size}


@router.put("/{knowledge_id}", response_model=KnowledgeOut)
def update_document(knowledge_id: int, data: KnowledgeUpdate, db: Session = Depends(get_db)):
    return service.update_document(db, knowledge_id, data.title, data.content, data.source)
@router.delete("/{knowledge_id}") #删除操作没有实体返回，故不使用 response_model
def delete_document(knowledge_id: int, db: Session = Depends(get_db)):
    return service.delete_document(db, knowledge_id)
