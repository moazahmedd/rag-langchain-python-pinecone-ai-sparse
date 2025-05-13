from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    
    # Server settings
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # Pinecone settings
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX")
    # PINECONE_NAMESPACE: str = "think-and-grow-rich"
    
    # Document settings
    PDF_PATH: str = str(DATA_DIR / "Think-And-Grow-Rich.pdf")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Model settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.3
    
    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    settings = Settings()
    return settings 