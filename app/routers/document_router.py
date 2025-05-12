from fastapi import APIRouter, HTTPException, status
from services.document_service import DocumentService
from config.settings import Settings
from models.schemas import DocumentResponse, ErrorResponse
from typing import Union

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    },
)

# Initialize services
settings = Settings()
document_service = DocumentService(settings)

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Document successfully processed and uploaded"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def upload_document() -> Union[DocumentResponse, ErrorResponse]:
    """
    Upload and process a PDF document.
    
    The document should be placed in the configured path before calling this endpoint.
    """
    try:

        print("document_router: Uploading document...")
        document_service.process_and_upload_document()
        return DocumentResponse(
            message="Document processed and uploaded successfully",
            # document_id=f"doc_{settings.PINECONE_NAMESPACE}"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found in the configured path"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Documents successfully deleted"},
        404: {"description": "No documents found to delete"},
        500: {"description": "Internal server error"}
    }
)
async def delete_document() -> Union[DocumentResponse, ErrorResponse]:
    """
    Delete all documents from the vector store.
    
    This will remove all vectors from the configured namespace.
    """
    try:
        document_service.delete_document()
        return DocumentResponse(message="Document deleted successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 