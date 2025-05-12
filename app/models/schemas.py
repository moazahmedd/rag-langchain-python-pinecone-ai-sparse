from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class Source(BaseModel):
    page: int = Field(..., description="Page number in the document")
    # content: str = Field(..., description="Content snippet from the document")
    id: str = Field(..., description="Unique identifier for the source")

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask about the document"
    )
    k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve (1-10)"
    )

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v.strip()

class QueryResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer to the query")
    sources: List[Source] = Field(..., description="List of sources used to generate the answer")
    # timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")

class DocumentResponse(BaseModel):
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")
    document_id: Optional[str] = Field(None, description="ID of the processed document")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the error") 