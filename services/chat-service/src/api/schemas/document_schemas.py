# src/api/schemas/document_schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uuid


class UploadDocumentResponse(BaseModel):
    id: uuid.UUID
    name: str
    category: str
    created_at: datetime
    message: str


class DocumentListResponse(BaseModel):
    id: uuid.UUID
    name: str
    category: str
    path: str
    created_at: datetime


class DeleteDocumentResponse(BaseModel):
    message: str
    document_id: str