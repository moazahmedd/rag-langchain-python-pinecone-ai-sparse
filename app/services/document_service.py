from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
from config.settings import Settings
from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService
from .vector_store_service import VectorStoreService
import os
from pathlib import Path

class DocumentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunking_service = ChunkingService(settings)
        self.embedding_service = EmbeddingService(settings)
        self.vector_store_service = VectorStoreService(settings, self.embedding_service)
        
        # Ensure data directory exists
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """
        Ensure the data directory exists
        """
        data_dir = Path(self.settings.PDF_PATH).parent
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found at {data_dir}. "
                "Please create the 'data' directory in the project root."
            )

    def process_and_upload_document(self):
        """
        Process PDF document and upload to vector store
        """

        print("document_service: Processing and uploading document...")
        # Check if PDF exists
        if not os.path.exists(self.settings.PDF_PATH):
            raise FileNotFoundError(
                f"PDF file not found at {self.settings.PDF_PATH}. "
                "Please ensure the PDF file is in the data directory."
            )

        # Load PDF
        loader = PyPDFLoader(self.settings.PDF_PATH)
        documents = loader.load()
        
        # Split into chunks
        chunks = self.chunking_service.chunk_documents(documents)
        
        # Prepare vectors for upload
        vectors = self.embedding_service.prepare_vectors_for_upload(
            chunks,
            self.settings.PINECONE_NAMESPACE
        )
        
        # Upload to vector store
        self.vector_store_service.upload_vectors(vectors)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents
        """
        return self.vector_store_service.similarity_search(query, k=k)

    def delete_document(self):
        """
        Delete all vectors in the namespace
        """
        self.vector_store_service.delete_namespace() 