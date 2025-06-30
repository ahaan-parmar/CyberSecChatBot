import os
import logging
from typing import Dict, List
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for the cybersecurity chatbot"""
      # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cybersecurity_knowledge")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Data Sources
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    CVE_DATA_PATH = os.getenv("CVE_DATA_PATH", "./data/cve_data.json")
    OWASP_DATA_PATH = os.getenv("OWASP_DATA_PATH", "./data/owasp_top10.json")
    MITRE_DATA_PATH = os.getenv("MITRE_DATA_PATH", "./data/mitre_attack.json")
    EXPLOIT_DATA_PATH = os.getenv("EXPLOIT_DATA_PATH", "./data/exploit_payloads.json")
      # Retrieval Configuration
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/chatbot.log")
    
    # UI Configuration
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
    
    # Security Settings
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Cache Settings
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    @classmethod
    def get_data_sources(cls) -> Dict[str, str]:
        """Get all data source paths"""
        return {
            "cve": cls.CVE_DATA_PATH,
            "owasp": cls.OWASP_DATA_PATH,
            "mitre": cls.MITRE_DATA_PATH,
            "exploits": cls.EXPLOIT_DATA_PATH
        }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check required API keys
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            errors.append("At least one LLM API key is required (OpenAI or Anthropic)")
        
        # Check data directory
        if not cls.DATA_DIR.exists():
            errors.append(f"Data directory does not exist: {cls.DATA_DIR}")
        
        # Check vector store path
        vector_store_dir = Path(cls.VECTOR_STORE_PATH).parent
        if not vector_store_dir.exists():
            try:
                vector_store_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create vector store directory: {e}")
        
        # Check log directory
        log_dir = Path(cls.LOG_FILE).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory: {e}")
        
        return errors
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        log_dir = Path(cls.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler()
            ]
        )

# Prompt Templates
class PromptTemplates:
    """Collection of prompt templates for the chatbot"""
    
    SYSTEM_PROMPT = """You are a cybersecurity expert assistant. Your role is to provide accurate, helpful, and actionable information about cybersecurity topics including vulnerabilities, threats, defenses, and best practices.

Use the following context from cybersecurity knowledge bases to answer questions:
- CVE (Common Vulnerabilities and Exposures) database
- OWASP Top 10 security risks
- MITRE ATT&CK framework
- Exploit techniques and countermeasures

Guidelines:
1. Provide accurate and up-to-date information
2. Include specific CVE IDs, MITRE technique IDs, or OWASP categories when relevant
3. Offer practical mitigation strategies
4. Cite severity levels and risk assessments when available
5. Be clear about the scope and limitations of your knowledge
6. Always prioritize defensive and ethical security practices

Context: {context}

Question: {input}

Answer:"""

    FOLLOW_UP_PROMPT = """Based on the previous conversation and the following context, provide a comprehensive follow-up answer:

Previous Context: {previous_context}
New Context: {context}
Question: {question}

Answer:"""

    CLARIFICATION_PROMPT = """The user's question needs clarification. Based on the context provided, ask specific questions to better understand what they're looking for:

Context: {context}
User Question: {question}

Clarifying questions:"""

# Model Configurations
MODEL_CONFIGS = {
    "gpt-3.5-turbo": {
        "provider": "openai",
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 1.0
    },
    "gpt-4": {
        "provider": "openai", 
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 1.0
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "max_tokens": 2048,
        "temperature": 0.1
    },    "gemini-1.5-flash": {
        "provider": "gemini",
        "max_tokens": 2048,
        "temperature": 0.1
    },
    "gemini-1.5-pro": {
        "provider": "gemini",
        "max_tokens": 4096,
        "temperature": 0.1
    },
    "gemini-2.0-flash": {
        "provider": "gemini",
        "max_tokens": 8192,
        "temperature": 0.1
    }
}

# Data Source Schemas
DATA_SCHEMAS = {
    "cve": {
        "required_fields": ["id", "description", "cvss_score", "severity"],
        "optional_fields": ["published_date", "affected_products", "references"]
    },
    "owasp": {
        "required_fields": ["rank", "category", "description"],
        "optional_fields": ["impact", "prevention", "examples"]
    },
    "mitre": {
        "required_fields": ["id", "name", "tactic", "description"],
        "optional_fields": ["detection", "mitigation", "platforms"]
    },
    "exploits": {
        "required_fields": ["name", "type", "description"],
        "optional_fields": ["payload", "risk_level", "countermeasures", "target"]
    }
}
