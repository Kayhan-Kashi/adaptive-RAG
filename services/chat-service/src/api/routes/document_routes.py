# src/api/routes/document_routes.py
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi_injector import Injected
from sqlmodel import Session
from starlette import status
from typing import List

from api.schemas.document_schemas import UploadDocumentResponse
from database.sqlite_session import get_session
from services.document_service import DocumentService

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: Session = Depends(get_session),
    service: DocumentService = Injected(DocumentService),
):
    """Upload a single document"""
    document = await service.upload_document(
        session=db,
        file=file,
        user_id=user_id
    )
    
    return UploadDocumentResponse(
        id=document.id,
        name=document.name,
        category=document.category,
        created_at=document.created_at,
        message="Document uploaded successfully"
    )