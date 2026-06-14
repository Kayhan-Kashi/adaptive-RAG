    # src/services/document_service.py
import uuid
import os
import shutil
from fastapi import HTTPException, UploadFile
from injector import inject
from sqlmodel import Session, select
from datetime import datetime
from typing import List, Optional
import logging

from database.models import UploadedFile
from common.kafka.producer import KafkaProducer #type: ignore
from common.events.document_uploaded import DocumentUploadedEvent #type: ignore

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document uploads"""
    @inject
    def __init__(self, kafka_producer: Optional[KafkaProducer] = None):
        self.upload_dir = os.getenv('UPLOAD_DIR', './uploads')
        self.kafka_producer = kafka_producer
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def upload_document(
        self,
        session: Session,
        file: UploadFile,
        user_id: str,
        category: str = "general"
    ) -> UploadedFile:
        """Upload a document file"""
        try:
            # Validate file type
            allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
                )
            
            # Generate unique filename
            file_id = uuid.uuid4()
            safe_filename = f"{file_id}{file_ext}"
            file_path = os.path.join(self.upload_dir, safe_filename)
            
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Save to database
            document = UploadedFile(
                id=file_id,
                name=file.filename,
                path=file_path,
                category=category,
                user_id=uuid.UUID(user_id)
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            
            logger.info(f"✅ Document uploaded: {file.filename} by user {user_id}")
            
            doc_event = DocumentUploadedEvent(
                document_id=str(document.id),
                filetype=file_ext,
                filename=document.name,
                # user_id=user_id
            )
              
            self.kafka_producer.produce(event=doc_event, key=str(document.id))
            logger.info(f"✅ Kafka event published successfully for document: {doc_event.event_id}")
            
            return document
            
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error uploading document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    
    async def upload_multiple_documents(
        self,
        session: Session,
        files: List[UploadFile],
        user_id: str,
        category: str = "general"
    ) -> List[UploadedFile]:
        """Upload multiple documents"""
        uploaded_files = []
        errors = []
        
        for file in files:
            try:
                document = await self.upload_document(session, file, user_id, category)
                uploaded_files.append(document)
            except Exception as e:
                errors.append({"file": file.filename, "error": str(e)})
        
        if errors:
            logger.warning(f"Some files failed to upload: {errors}")
        
        return uploaded_files
    
    def get_user_documents(
        self,
        session: Session,
        user_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[UploadedFile]:
        """Get all documents for a user"""
        try:
            documents = session.exec(
                select(UploadedFile)
                .where(UploadedFile.user_id == uuid.UUID(user_id))
                .order_by(UploadedFile.created_at.desc())
                .offset(skip)
                .limit(limit)
            ).all()
            
            logger.info(f"📋 Retrieved {len(documents)} documents for user {user_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")
    
    def get_document_by_id(
        self,
        session: Session,
        document_id: str
    ) -> Optional[UploadedFile]:
        """Get document by ID"""
        try:
            document = session.get(UploadedFile, uuid.UUID(document_id))
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            return document
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")
    
    def delete_document(
        self,
        session: Session,
        document_id: str,
        user_id: str
    ) -> dict:
        """Delete a document"""
        try:
            document = session.get(UploadedFile, uuid.UUID(document_id))
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Check ownership
            if str(document.user_id) != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Delete file from disk
            if os.path.exists(document.path):
                os.remove(document.path)
            
            # Delete from database
            session.delete(document)
            session.commit()
            
            logger.info(f"🗑️ Document deleted: {document.name} by user {user_id}")
            
            return {"message": "Document deleted successfully", "document_id": document_id}
            
        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")