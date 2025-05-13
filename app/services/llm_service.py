from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from typing import List, Dict, Any
from config.settings import Settings

class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOpenAI(
            model_name=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Define prompt template using ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant that provides detailed, confident, and comprehensive answers based on the provided context. Your goal is to extract maximum value from the available information.

Guidelines for your responses:
1. Start directly with your analysis - avoid phrases like "The context does not explicitly detail..."
2. Use all relevant information from the context to form a complete answer
3. Make reasonable inferences based on the available information
4. If information is partial, explain what we know and what we can infer
5. Only say "I don't know" if the context is completely irrelevant to the question

For comparative questions:
1. Analyze both sides of the comparison
2. Provide specific examples from the context
3. Explain the relationship between the concepts
4. Draw conclusions based on the evidence

Structure your response with:
- A clear introduction
- Detailed analysis with examples
- Supporting evidence from the context
- A conclusion that synthesizes the information

Ensure your answer is well-structured and answer in under 50 words."""),
            ("human", """Context: {context}

Question: {question}

Please provide a comprehensive analysis based on the context above.""")
        ])
        
        # Create chain using LCEL
        self.chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Wrap chain to include input in output
        self.outer_chain = RunnablePassthrough().assign(text=self.chain)

    def get_structured_answer(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Get a structured answer with sources from the LLM
        """
        print("llm_service: Getting structured answer...")
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        sources = []
        
        # Format sources with required fields
        for doc in documents:
            source = {
                "page": doc.metadata.get("page", "N/A"),
                "id": doc.metadata.get("id", "N/A")  # Get ID from metadata
            }
            sources.append(source)

        # Run the chain
        response = self.outer_chain.invoke({
            "question": query,
            "context": context,
        })

        return {
            "answer": response["text"],
            "sources": sources
        } 