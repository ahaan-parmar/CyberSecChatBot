import logging
from typing import Dict, List, Optional, Any
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.base import BaseCallbackHandler
import google.generativeai as genai
from utils.config import Config, PromptTemplates, MODEL_CONFIGS
from src.vector_store import CybersecurityVectorStore

logger = logging.getLogger(__name__)

class CybersecurityCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for logging and monitoring"""
    
    def __init__(self):
        self.query_count = 0
        self.retrieved_docs = []
    
    def on_retriever_start(self, query: str, **kwargs):
        logger.debug(f"Starting retrieval for query: {query}")
    
    def on_retriever_end(self, documents: List[Document], **kwargs):
        self.retrieved_docs = documents
        logger.debug(f"Retrieved {len(documents)} documents")
    
    def on_llm_start(self, prompts: List[str], **kwargs):
        self.query_count += 1
        logger.debug(f"LLM query #{self.query_count} started")
    
    def on_llm_end(self, response, **kwargs):
        logger.debug(f"LLM query #{self.query_count} completed")

class CybersecurityRAGChain:
    """RAG chain for cybersecurity question answering"""
    
    def __init__(self, vector_store: CybersecurityVectorStore = None):
        self.vector_store = vector_store or CybersecurityVectorStore()
        self.llm = self._initialize_llm()
        self.retriever = self._initialize_retriever()
        self.callback_handler = CybersecurityCallbackHandler()
        self.chain = self._create_rag_chain()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        model_name = Config.LLM_MODEL
        model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["gpt-3.5-turbo"])
        
        # Try to initialize with available API keys
        api_keys_checked = []
        
        try:
            if model_config["provider"] == "openai" and Config.OPENAI_API_KEY:
                api_keys_checked.append("OpenAI")
                return ChatOpenAI(
                    model_name=model_name,
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    openai_api_key=Config.OPENAI_API_KEY
                )
            elif model_config["provider"] == "anthropic" and Config.ANTHROPIC_API_KEY:
                # Import and configure Anthropic model
                from langchain_community.chat_models import ChatAnthropic
                api_keys_checked.append("Anthropic")
                return ChatAnthropic(
                    model=model_name,
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    anthropic_api_key=Config.ANTHROPIC_API_KEY
                )
            elif model_config["provider"] == "gemini" and Config.GEMINI_API_KEY:
                # Configure Gemini model
                api_keys_checked.append("Gemini")
                genai.configure(api_key=Config.GEMINI_API_KEY)
                return self._create_gemini_wrapper(model_name)
            
            # Try fallback options
            if Config.GEMINI_API_KEY and "Gemini" not in api_keys_checked:
                logger.info("Falling back to Gemini 1.5 Flash")
                genai.configure(api_key=Config.GEMINI_API_KEY)
                return self._create_gemini_wrapper("gemini-1.5-flash")
            
            if Config.OPENAI_API_KEY and "OpenAI" not in api_keys_checked:
                logger.info("Falling back to OpenAI GPT-3.5-turbo")
                return ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    openai_api_key=Config.OPENAI_API_KEY
                )
            
            if Config.ANTHROPIC_API_KEY and "Anthropic" not in api_keys_checked:
                logger.info("Falling back to Anthropic Claude")
                from langchain_community.chat_models import ChatAnthropic
                return ChatAnthropic(
                    model="claude-3-sonnet",
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    anthropic_api_key=Config.ANTHROPIC_API_KEY
                )
            
            # If no API keys are available, create a mock LLM for demo purposes
            logger.warning("No API keys configured. Using mock LLM for demonstration.")
            return self._create_mock_llm()
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Final fallback to mock LLM
            logger.warning("Falling back to mock LLM due to initialization error")
            return self._create_mock_llm()
    
    def _initialize_retriever(self) -> BaseRetriever:
        """Initialize document retriever"""
        retriever = self.vector_store.get_retriever({
            "k": Config.RETRIEVAL_K,
            "search_type": "similarity"
        })
        
        if not retriever:
            raise ValueError("Failed to initialize retriever")
        
        return retriever
    
    def _create_rag_chain(self):
        """Create the RAG chain"""
        try:
            # Create prompt template
            prompt = ChatPromptTemplate.from_template(PromptTemplates.SYSTEM_PROMPT)
            
            # Create document chain
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt
            )
            
            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever=self.retriever,
                combine_docs_chain=document_chain
            )
            
            return retrieval_chain
            
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            # Fallback to basic RetrievalQA
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Process a cybersecurity question"""
        try:
            # Validate input
            if not question or len(question.strip()) == 0:
                return {
                    "answer": "Please provide a valid question.",
                    "source_documents": [],
                    "confidence": 0.0
                }
            
            if len(question) > Config.MAX_QUERY_LENGTH:
                return {
                    "answer": f"Question too long. Please limit to {Config.MAX_QUERY_LENGTH} characters.",
                    "source_documents": [],
                    "confidence": 0.0
                }
            
            # Process query
            logger.info(f"Processing query: {question[:100]}...")
              # Get context from retriever
            retrieved_docs = self.retriever.invoke(question)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or asking about CVEs, OWASP vulnerabilities, MITRE techniques, or exploit methods.",
                    "source_documents": [],
                    "confidence": 0.0
                }
            
            # Execute chain
            result = self.chain.invoke({
                "input": question,
                **kwargs
            })
            
            # Process result
            if isinstance(result, dict):
                answer = result.get("answer", "I couldn't generate an answer.")
                source_docs = result.get("context", retrieved_docs)
            else:
                answer = str(result)
                source_docs = retrieved_docs
            
            # Calculate confidence based on relevance
            confidence = self._calculate_confidence(question, source_docs)
            
            return {
                "answer": answer,
                "source_documents": source_docs,
                "confidence": confidence,
                "query_metadata": {
                    "retrieved_docs_count": len(retrieved_docs),
                    "sources": list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "source_documents": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_confidence(self, question: str, documents: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents"""
        if not documents:
            return 0.0
        
        # Simple confidence calculation based on:
        # 1. Number of retrieved documents
        # 2. Source diversity
        # 3. Content relevance (simplified)
        
        doc_count_score = min(len(documents) / Config.RETRIEVAL_K, 1.0)
        
        # Source diversity
        sources = set([doc.metadata.get("source", "Unknown") for doc in documents])
        source_diversity_score = min(len(sources) / 4, 1.0)  # Max 4 sources
        
        # Simple keyword overlap (basic relevance)
        question_words = set(question.lower().split())
        content_words = set()
        for doc in documents:
            content_words.update(doc.page_content.lower().split())
        
        overlap = len(question_words.intersection(content_words))
        relevance_score = min(overlap / max(len(question_words), 1), 1.0)
        
        # Weighted average
        confidence = (
            doc_count_score * 0.3 +
            source_diversity_score * 0.3 +
            relevance_score * 0.4
        )
        
        return round(confidence, 2)
    
    def query_with_filters(self, question: str, source_filter: str = None, doc_type_filter: str = None) -> Dict[str, Any]:
        """Query with metadata filters"""
        # Update retriever with filters
        filter_dict = {}
        if source_filter:
            filter_dict["source"] = source_filter
        if doc_type_filter:
            filter_dict["doc_type"] = doc_type_filter
        
        if filter_dict:
            # Create filtered retriever
            filtered_retriever = self.vector_store.get_retriever({
                "k": Config.RETRIEVAL_K,
                "filter": filter_dict
            })
            
            # Temporarily replace retriever
            original_retriever = self.retriever
            self.retriever = filtered_retriever
            
            try:
                result = self.query(question)
                result["applied_filters"] = filter_dict
                return result
            finally:
                # Restore original retriever
                self.retriever = original_retriever
        else:
            return self.query(question)
    
    def get_similar_questions(self, question: str, k: int = 3) -> List[str]:
        """Get similar questions that might be relevant"""
        try:
            # This is a simplified implementation
            # In a real system, you might maintain a database of common questions
            retrieved_docs = self.retriever.invoke(question)
            
            # Extract potential question patterns from content
            similar_questions = []
            
            for doc in retrieved_docs[:k]:
                source = doc.metadata.get("source", "")
                if source == "CVE":
                    cve_id = doc.metadata.get("cve_id", "")
                    if cve_id:
                        similar_questions.append(f"Tell me more about {cve_id}")
                elif source == "OWASP":
                    category = doc.metadata.get("category", "")
                    if category:
                        similar_questions.append(f"How to prevent {category}?")
                elif source == "MITRE":
                    technique_id = doc.metadata.get("technique_id", "")
                    if technique_id:
                        similar_questions.append(f"How to detect {technique_id}?")
            
            return similar_questions[:k]
            
        except Exception as e:
            logger.error(f"Error getting similar questions: {e}")
            return []
    
    def explain_sources(self, documents: List[Document]) -> str:
        """Generate explanation of the sources used"""
        if not documents:
            return "No sources were found for this query."
        
        source_counts = {}
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        explanations = []
        for source, count in source_counts.items():
            if source == "CVE":
                explanations.append(f"{count} CVE record(s) from the vulnerability database")
            elif source == "OWASP":
                explanations.append(f"{count} OWASP guideline(s) on web application security")
            elif source == "MITRE":
                explanations.append(f"{count} MITRE ATT&CK technique(s) describing adversary behavior")
            elif source == "Exploits":
                explanations.append(f"{count} exploit technique(s) and countermeasures")
            else:
                explanations.append(f"{count} document(s) from {source}")
        
        return "This answer is based on: " + ", ".join(explanations) + "."

    def _create_gemini_wrapper(self, model_name: str):
        """Create a wrapper for Gemini model to work with LangChain"""
        from langchain.llms.base import LLM
        from pydantic import Field
        
        class GeminiLLM(LLM):
            model_name: str = Field(default="gemini-1.5-flash")
            temperature: float = Field(default=0.1)
            max_tokens: int = Field(default=2048)
            
            def __init__(self, model_name: str = "gemini-1.5-flash", **kwargs):
                super().__init__(model_name=model_name, **kwargs)
            
            @property
            def _llm_type(self) -> str:
                return "gemini"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                try:
                    model = genai.GenerativeModel(self.model_name)
                    
                    generation_config = genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )
                    
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    return response.text
                except Exception as e:
                    logger.error(f"Error calling Gemini API: {e}")
                    return f"Error: Could not generate response from Gemini API - {str(e)}"
        
        return GeminiLLM(model_name=model_name)
    
    def _create_mock_llm(self):
        """Create a mock LLM for demonstration when no API keys are available"""
        from langchain.llms.base import LLM
        from pydantic import Field
        
        class MockLLM(LLM):
            model_name: str = Field(default="mock-llm")
            temperature: float = Field(default=0.1)
            max_tokens: int = Field(default=2048)
            
            @property
            def _llm_type(self) -> str:
                return "mock"
            
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                """Generate a mock response based on the context and question"""
                # Extract question from prompt
                if "Question:" in prompt:
                    question = prompt.split("Question:")[-1].strip()
                else:
                    question = prompt
                
                # Generate mock response based on question content
                question_lower = question.lower()
                
                if "cve" in question_lower or "vulnerability" in question_lower:
                    return """Based on the cybersecurity knowledge base, this appears to be a Common Vulnerability and Exposure (CVE) related query. 

CVE is a dictionary of publicly known information security vulnerabilities and exposures. Each CVE entry contains:
- A unique identifier (CVE-ID)
- A description of the vulnerability
- References to related vulnerability reports and advisories

To get specific information about a CVE, please provide the CVE ID (e.g., CVE-2023-1234) or describe the vulnerability you're interested in.

For the most up-to-date information, you can also check the official CVE database at https://cve.mitre.org/"""
                
                elif "owasp" in question_lower or "top 10" in question_lower:
                    return """The OWASP Top 10 is a standard awareness document for developers and web application security. It represents a broad consensus about the most critical security risks to web applications.

The current OWASP Top 10 (2021) includes:
1. Broken Access Control
2. Cryptographic Failures
3. Injection
4. Insecure Design
5. Security Misconfiguration
6. Vulnerable and Outdated Components
7. Identification and Authentication Failures
8. Software and Data Integrity Failures
9. Security Logging and Monitoring Failures
10. Server-Side Request Forgery

Each category includes specific vulnerabilities and mitigation strategies. Would you like to know more about any specific category?"""
                
                elif "mitre" in question_lower or "attack" in question_lower:
                    return """MITRE ATT&CK is a globally-accessible knowledge base of adversary tactics and techniques based on real-world observations. It provides a common taxonomy for describing cyber adversary behavior.

The framework is organized into:
- Tactics: The "why" of an attack technique
- Techniques: The "how" of an attack technique
- Sub-techniques: More specific descriptions of techniques

Key tactic categories include:
- Initial Access
- Execution
- Persistence
- Privilege Escalation
- Defense Evasion
- Credential Access
- Discovery
- Lateral Movement
- Collection
- Command and Control
- Exfiltration
- Impact

Would you like to explore any specific tactic or technique?"""
                
                elif "sql injection" in question_lower or "injection" in question_lower:
                    return """SQL Injection is a code injection technique that exploits vulnerabilities in an application's software by inserting malicious SQL statements into entry fields for execution.

**How it works:**
- Attackers insert malicious SQL code into input fields
- The application executes the malicious code
- This can lead to unauthorized data access, modification, or deletion

**Prevention:**
- Use parameterized queries/prepared statements
- Input validation and sanitization
- Least privilege database accounts
- Web Application Firewalls (WAF)
- Regular security testing

**Example of vulnerable code:**
```sql
SELECT * FROM users WHERE username = '" + userInput + "'
```

**Secure version:**
```sql
SELECT * FROM users WHERE username = ?
```

This is a critical vulnerability that should be addressed immediately in any web application."""
                
                elif "xss" in question_lower or "cross-site scripting" in question_lower:
                    return """Cross-Site Scripting (XSS) is a security vulnerability that allows attackers to inject malicious scripts into web pages viewed by other users.

**Types of XSS:**
1. **Stored XSS**: Malicious script is permanently stored on the server
2. **Reflected XSS**: Malicious script is reflected off the web server
3. **DOM-based XSS**: Malicious script modifies the DOM environment

**Prevention:**
- Input validation and output encoding
- Content Security Policy (CSP)
- HttpOnly cookies
- Regular security testing
- Framework security features

**Example of vulnerable code:**
```html
<div>Hello, <?php echo $_GET['name']; ?></div>
```

**Secure version:**
```html
<div>Hello, <?php echo htmlspecialchars($_GET['name']); ?></div>
```

XSS attacks can lead to session hijacking, defacement, and other security breaches."""
                
                else:
                    return """I'm a cybersecurity assistant designed to help with questions about:
- CVE vulnerabilities and exposures
- OWASP security guidelines
- MITRE ATT&CK framework
- Exploit techniques and countermeasures
- General cybersecurity best practices

Please ask me about specific vulnerabilities, security frameworks, attack techniques, or cybersecurity concepts. For example:
- "Tell me about CVE-2023-1234"
- "What is SQL injection?"
- "Explain the OWASP Top 10"
- "How do privilege escalation attacks work?"

Note: This is a demonstration mode. For full functionality, please configure an API key in your .env file."""
        
        return MockLLM()

def main():
    """Main function to test the RAG chain"""
    Config.setup_logging()
    
    # Initialize RAG chain
    print("Initializing RAG chain...")
    rag_chain = CybersecurityRAGChain()
    
    # Test queries
    test_queries = [
        "What is SQL injection?",
        "Tell me about CVE-2023-1234",
        "How to prevent XSS attacks?",
        "What is privilege escalation?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        result = rag_chain.query(query)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {result.get('query_metadata', {}).get('sources', [])}")
        
        if result['source_documents']:
            print(f"Source explanation: {rag_chain.explain_sources(result['source_documents'])}")

if __name__ == "__main__":
    main()
