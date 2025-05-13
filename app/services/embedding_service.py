from langchain.schema import Document
from typing import List, Dict, Any
from config.settings import Settings
from rank_bm25 import BM25Okapi # type: ignore
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Initialize BM25 model
        self.bm25 = None
        self.tokenized_corpus = []
        self.vocabulary = {}
        self.next_index = 0
        self.stop_words = set(stopwords.words('english'))

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by:
        1. Converting to lowercase
        2. Removing special characters
        3. Removing stopwords
        4. Tokenizing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens

    def _get_or_create_index(self, token: str) -> int:
        """Get or create an index for a token in the vocabulary"""
        if token not in self.vocabulary:
            self.vocabulary[token] = self.next_index
            self.next_index += 1
        return self.vocabulary[token]

    def _convert_to_sparse_format(self, scores: List[float], tokens: List[str]) -> Dict[str, List]:
        """Convert BM25 scores to Pinecone's required sparse vector format"""
        # Create a dictionary to accumulate scores for each index
        index_scores = {}
        
        # Get scores for each token
        for token in tokens:
            # Get score for this token against the corpus
            token_scores = self.bm25.get_scores([token])
            # Use the maximum score for this token
            score = max(token_scores) if token_scores is not None and len(token_scores) > 0 else 0
            
            # Normalize score to be between 0 and 1
            normalized_score = (score + 1) / 2  # Shift and scale to [0,1] range
            
            if normalized_score > 0.1:  # Include scores above threshold
                index = self._get_or_create_index(token)
                index_scores[index] = normalized_score
        
        # If no scores above threshold, use default scoring
        if not index_scores:
            for i, token in enumerate(tokens):
                index = self._get_or_create_index(token)
                index_scores[index] = 1.0 / (i + 1)  # Decreasing weights
        
        # Convert to lists, ensuring indices are unique and sorted
        indices = sorted(index_scores.keys())
        values = [float(index_scores[idx]) for idx in indices]  # Ensure values are float
            
        return {
            "indices": indices,
            "values": values
        }

    def get_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, List]]:
        """
        Generate BM25 sparse embeddings for a list of texts
        Returns sparse vectors in Pinecone's required format: {'indices': List[int], 'values': List[float]}
        """
        if not texts:
            return []

        print(f"Generating sparse embeddings for {len(texts)} texts")
        
        # Preprocess all texts
        self.tokenized_corpus = [self._preprocess_text(text) for text in texts]
        print(f"Tokenized corpus size: {len(self.tokenized_corpus)}")
        
        # Create BM25 model
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Get sparse vectors for each text
        sparse_vectors = []
        for text in texts:
            # Preprocess query text
            query_tokens = self._preprocess_text(text)
            print(f"Query tokens: {query_tokens}")
            
            if not query_tokens:
                print("Warning: No tokens after preprocessing")
                continue
                
            # Convert to Pinecone's required format
            sparse_vector = self._convert_to_sparse_format([], query_tokens)
            # print(f"Sparse vector: {sparse_vector}")
            
            if sparse_vector["indices"]:  # Only add if we have valid indices
                sparse_vectors.append(sparse_vector)
            else:
                print("Warning: No valid indices in sparse vector")
        
        if not sparse_vectors:
            print("Warning: No valid sparse vectors generated")
            # Return a default sparse vector with a single token
            return [{"indices": [0], "values": [1.0]}]
            
        return sparse_vectors

    def search(self, query: str, documents: List[Document], k: int = 30) -> List[Document]:
        """
        Search documents using BM25 scoring
        """
        # Preprocess all documents
        doc_texts = [doc.page_content for doc in documents]
        tokenized_docs = [self._preprocess_text(text) for text in doc_texts]
        
        # Create BM25 model
        bm25 = BM25Okapi(tokenized_docs)
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        
        # Get document scores
        doc_scores = bm25.get_scores(query_tokens)
        
        # Get top k documents
        top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
        
        # Create result documents with scores
        results = []
        for idx in top_k_indices:
            if doc_scores[idx] > 0:  # Only include documents with positive scores
                doc = documents[idx]
                doc.metadata['score'] = float(doc_scores[idx])
                results.append(doc)
        
        return results

    def prepare_vectors_for_upload(
        self, 
        documents: List[Document], 
        namespace: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for vector store upload by generating sparse embeddings
        """
        print("embedding_service: Preparing vectors for upload...")
        texts = [doc.page_content for doc in documents]
        
        # Generate sparse embeddings
        sparse_embeddings = self.get_sparse_embeddings(texts)
        
        vectors = []
        for i, (doc, sparse_emb) in enumerate(zip(documents, sparse_embeddings)):
            # Ensure sparse_values is in the correct format
            sparse_values = {
                "indices": sparse_emb["indices"],
                "values": sparse_emb["values"]
            }
            
            vector = {
                "id": f"{namespace}#chunk{i+1}",
                "sparse_values": sparse_values,  # Use the properly formatted sparse_values
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", None),
                    "page_label": doc.metadata.get("page_label", None),
                }
            }
            vectors.append(vector)
            
        return vectors 