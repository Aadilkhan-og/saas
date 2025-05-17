import os
import logging
from typing import List, Dict, Any, Optional

# Try to import Google AI Python SDK
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger("keyword_research.gemini_client")

class GeminiClient:
    """
    A wrapper for the Google Gemini API to use similar interface as langchain_openai.ChatOpenAI
    """
    
    def __init__(self, api_key=None, model="gemini-2.0-flash"):
        """
        Initialize the Gemini client
        
        Args:
            api_key (str, optional): Google API key. If not provided, tries to get from GOOGLE_API_KEY env variable
            model (str, optional): Model name to use. Defaults to "gemini-2.0-flash"
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("Google API key not found. Set GOOGLE_API_KEY environment variable.")
            raise ValueError("Google API key not found")
        
        self.model = model
        
        if not GEMINI_AVAILABLE:
            logger.error("Google AI Python SDK not installed. Run 'pip install google-generativeai langchain-google-genai'")
            raise ImportError("Required packages not installed: google-generativeai, langchain-google-genai")
        
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Initialize the LangChain compatible model
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.7
            )
            
            logger.info(f"Gemini client initialized with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            raise

    def get_langchain_llm(self):
        """
        Get the LangChain compatible LLM instance
        
        Returns:
            ChatGoogleGenerativeAI: LangChain compatible LLM
        """
        return self.llm
    
    @classmethod
    def is_available(cls):
        """
        Check if Gemini is available
        
        Returns:
            bool: True if Gemini is available, False otherwise
        """
        return GEMINI_AVAILABLE
