from typing import List, Optional, Tuple
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from injector import inject
import logging
import os
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class DocumentLoader:
    @inject
    def __init__(self):
        self._documents: List[Document] = []
        self._file_path: Optional[str] = None
        self._images: List[str] = []  # Store image paths
        self._image_paths: List[str] = []  # Alias for images
    
    def load(self, file_path: str, encoding: str = "utf-8") -> 'DocumentLoader':
        """Load a text document"""
        try:
            loader = TextLoader(file_path, encoding=encoding)
            self._documents = loader.load()
            self._file_path = file_path
            logger.info(f"✅ Loaded text document from {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load document: {e}")
            self._documents = []
        return self
    
    def load_pdf(self, file_path: str, output_dir: str = "pages", dpi: int = 300) -> 'DocumentLoader':
        """
        Load a PDF file, extract images and text
        """
        doc = None
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"❌ PDF file not found: {file_path}")
                self._documents = []
                return self
            
            logger.info(f"📖 Loading PDF from: {file_path} (size: {os.path.getsize(file_path)} bytes)")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Open PDF
            doc = fitz.open(file_path)
            logger.info(f"📖 PDF opened: {len(doc)} pages")
            
            self._images = []
            self._image_paths = []
            extracted_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text from page
                page_text = page.get_text()
                extracted_text.append(page_text)
                
                # Extract image as PNG
                pix = page.get_pixmap(dpi=dpi)
                img_path = f"{output_dir}/page_{page_num + 1}.png"
                pix.save(img_path)
                self._images.append(img_path)
                self._image_paths.append(img_path)
                
                logger.debug(f"📄 Extracted page {page_num + 1}")
            
            # Combine all text
            full_text = "\n\n".join(extracted_text)
            
            # Create document with metadata
            self._documents = [
                Document(
                    page_content=full_text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                        "total_pages": len(doc),
                        "image_paths": self._images
                    }
                )
            ]
            
            self._file_path = file_path
            
            logger.info(f"✅ Loaded PDF from {file_path} - {len(doc)} pages, {len(self._images)} images")
            logger.info(f"📝 Extracted text: {len(full_text)} characters")
            
        except fitz.fitz.FileDataError as e:
            logger.error(f"❌ PDF file is corrupted or invalid: {e}")
            self._documents = []
            self._images = []
        except Exception as e:
            logger.error(f"❌ Failed to load PDF: {e}")
            logger.exception("Detailed traceback:")
            self._documents = []
            self._images = []
        finally:
            if doc is not None:
                doc.close()
        
        return self
    def load_pdf_with_text_only(self, file_path: str) -> 'DocumentLoader':
        """Load only text from PDF (no image extraction)"""
        try:
            doc = fitz.open(file_path)
            extracted_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                extracted_text.append(page_text)
            
            full_text = "\n\n".join(extracted_text)
            
            self._documents = [
                Document(
                    page_content=full_text,
                    metadata={
                        "source": file_path,
                        "type": "pdf",
                        "total_pages": len(doc)
                    }
                )
            ]
            
            self._file_path = file_path
            doc.close()
            
            logger.info(f"✅ Loaded PDF text from {file_path} - {len(doc)} pages")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PDF text: {e}")
            self._documents = []
        return self
    
    @property
    def documents(self) -> List[Document]:
        """Get loaded documents"""
        return self._documents
    
    @property
    def text(self) -> str:
        """Get combined text content of all documents"""
        return "\n\n".join([doc.page_content for doc in self._documents])
    
    @property
    def images(self) -> List[str]:
        """Get list of extracted image paths (from PDF)"""
        return self._images
    
    @property
    def image_paths(self) -> List[str]:
        """Alias for images"""
        return self._images
    
    @property
    def is_loaded(self) -> bool:
        """Check if document is loaded"""
        return len(self._documents) > 0
    
    @property
    def metadata(self) -> dict:
        """Get metadata of the loaded document"""
        if self._documents:
            return self._documents[0].metadata
        return {}