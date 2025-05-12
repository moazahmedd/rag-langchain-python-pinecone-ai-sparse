from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from typing import List, Dict, Any
import math
from config.settings import Settings
from .embedding_service import EmbeddingService

class VectorStoreService:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=embedding_service.embeddings,
            namespace=settings.PINECONE_NAMESPACE
        )

    def upload_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 50):
        """
        Upload vectors to Pinecone in batches
        """

        print("vector_store_service: Uploading vectors to Pinecone...")
        total_batches = math.ceil(len(vectors) / batch_size)
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            # Upload batch to Pinecone
            self.index.upsert(
                vectors=batch,
                namespace=self.settings.PINECONE_NAMESPACE
            )
            
            print(f"Uploaded batch {current_batch} of {total_batches}")

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents using the query
        """
        print("vector_store_service: Searching for similar documents...")
        return self.vectorstore.similarity_search(query, k=k)

    def delete_namespace(self):
        """
        Delete all vectors in the namespace
        """
        print("vector_store_service: Deleting namespace...")
        self.index.delete(delete_all=True, namespace=self.settings.PINECONE_NAMESPACE) 