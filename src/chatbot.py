import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.rag_chain import CybersecurityRAGChain
from src.vector_store import CybersecurityVectorStore
from utils.config import Config

logger = logging.getLogger(__name__)

class CybersecurityChatbot:
    """Main cybersecurity chatbot interface"""
    
    def __init__(self):
        self.rag_chain = CybersecurityRAGChain()
        self.conversation_history = []
        self.session_stats = {
            "queries_processed": 0,
            "session_start": datetime.now(),
            "average_response_time": 0.0,
            "sources_used": set()
        }
        logger.info("Cybersecurity chatbot initialized")
    
    def chat(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main chat interface"""
        start_time = time.time()
        
        try:
            # Validate input
            if not user_input or not user_input.strip():
                return self._create_response(
                    "Please provide a question about cybersecurity.",
                    response_type="validation_error"
                )
            
            # Check for special commands
            if user_input.startswith("/"):
                return self._handle_command(user_input)
            
            # Process the query
            result = self.rag_chain.query(user_input)
            
            # Update session statistics
            response_time = time.time() - start_time
            self._update_session_stats(result, response_time)
            
            # Add to conversation history
            self._add_to_history(user_input, result)
            
            # Create response
            response = self._create_response(
                result["answer"],
                source_documents=result.get("source_documents", []),
                confidence=result.get("confidence", 0.0),
                response_time=response_time,
                sources=result.get("query_metadata", {}).get("sources", [])
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return self._create_response(
                "I encountered an error while processing your question. Please try again or rephrase your question.",
                response_type="error",
                error=str(e)
            )
    
    def _handle_command(self, command: str) -> Dict[str, Any]:
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == "/help":
            return self._get_help_response()
        elif command == "/stats":
            return self._get_stats_response()
        elif command == "/sources":
            return self._get_sources_response()
        elif command == "/clear":
            return self._clear_history()
        elif command == "/examples":
            return self._get_examples_response()
        else:
            return self._create_response(
                "Unknown command. Type '/help' for available commands.",
                response_type="command_error"
            )
    
    def _get_help_response(self) -> Dict[str, Any]:
        """Get help information"""
        help_text = """
🛡️ **Cybersecurity Chatbot Help**

**Available Commands:**
- `/help` - Show this help message
- `/stats` - Show session statistics
- `/sources` - List available data sources
- `/clear` - Clear conversation history
- `/examples` - Show example queries

**What I can help with:**
- CVE vulnerability information
- OWASP security guidelines
- MITRE ATT&CK techniques
- Exploit methods and countermeasures
- General cybersecurity best practices

**Example Questions:**
- "What is SQL injection and how to prevent it?"
- "Tell me about CVE-2023-1234"
- "How do privilege escalation attacks work?"
- "What are the OWASP Top 10 vulnerabilities?"
- "How to detect lateral movement?"

**Tips:**
- Be specific in your questions for better results
- Ask about specific CVE IDs, MITRE technique IDs, or OWASP categories
- Use technical terms related to cybersecurity
        """
        
        return self._create_response(help_text, response_type="help")
    
    def _get_stats_response(self) -> Dict[str, Any]:
        """Get session statistics"""
        uptime = datetime.now() - self.session_stats["session_start"]
        
        stats_text = f"""
📊 **Session Statistics**

- **Queries Processed:** {self.session_stats["queries_processed"]}
- **Session Duration:** {str(uptime).split('.')[0]}
- **Average Response Time:** {self.session_stats["average_response_time"]:.2f}s
- **Sources Accessed:** {len(self.session_stats["sources_used"])}
- **Conversation History:** {len(self.conversation_history)} exchanges

**Sources Used This Session:**
{', '.join(sorted(self.session_stats["sources_used"])) if self.session_stats["sources_used"] else "None yet"}
        """
        
        return self._create_response(stats_text, response_type="stats")
    
    def _get_sources_response(self) -> Dict[str, Any]:
        """Get available data sources"""
        try:
            vector_stats = self.rag_chain.vector_store.get_statistics()
            metadata = vector_stats.get("sources", {})
            
            sources_text = "📚 **Available Data Sources:**\n\n"
            
            for source, count in metadata.items():
                if source == "CVE":
                    sources_text += f"🔍 **CVE Database:** {count} vulnerability records\n"
                elif source == "OWASP":
                    sources_text += f"🛡️ **OWASP Guidelines:** {count} security categories\n"
                elif source == "MITRE":
                    sources_text += f"⚔️ **MITRE ATT&CK:** {count} attack techniques\n"
                elif source == "Exploits":
                    sources_text += f"💥 **Exploit Database:** {count} exploit techniques\n"
                else:
                    sources_text += f"📄 **{source}:** {count} documents\n"
            
            total_docs = vector_stats.get("total_documents", 0)
            sources_text += f"\n**Total Knowledge Base:** {total_docs} documents"
            
        except Exception as e:
            sources_text = f"Error retrieving source information: {e}"
        
        return self._create_response(sources_text, response_type="sources")
    
    def _clear_history(self) -> Dict[str, Any]:
        """Clear conversation history"""
        self.conversation_history.clear()
        return self._create_response(
            "✅ Conversation history cleared.",
            response_type="system"
        )
    
    def _get_examples_response(self) -> Dict[str, Any]:
        """Get example queries"""
        examples_text = """
💡 **Example Queries to Try:**

**CVE Vulnerabilities:**
- "Tell me about CVE-2023-1234"
- "What are the most critical CVEs this year?"
- "Show me SQL injection vulnerabilities"

**OWASP Security:**
- "What is cross-site scripting (XSS)?"
- "How to prevent injection attacks?"
- "Explain the OWASP Top 10"

**MITRE ATT&CK:**
- "What is privilege escalation?"
- "How do attackers perform lateral movement?"
- "Explain persistence techniques"

**Exploit Techniques:**
- "How do buffer overflow attacks work?"
- "What are common web application exploits?"
- "How to defend against social engineering?"

**General Security:**
- "How to secure a web application?"
- "What is zero-day vulnerability?"
- "Best practices for network security"
        """
        
        return self._create_response(examples_text, response_type="examples")
    
    def _create_response(
        self, 
        answer: str, 
        source_documents: List = None,
        confidence: float = None,
        response_time: float = None,
        sources: List[str] = None,
        response_type: str = "answer",
        error: str = None
    ) -> Dict[str, Any]:
        """Create standardized response"""
        response = {
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "response_type": response_type
        }
        
        if source_documents:
            response["source_documents"] = source_documents
            response["source_explanation"] = self.rag_chain.explain_sources(source_documents)
        
        if confidence is not None:
            response["confidence"] = confidence
            response["confidence_level"] = self._get_confidence_level(confidence)
        
        if response_time is not None:
            response["response_time"] = round(response_time, 2)
        
        if sources:
            response["sources_used"] = sources
        
        if error:
            response["error"] = error
        
        return response
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _add_to_history(self, user_input: str, result: Dict[str, Any]):
        """Add exchange to conversation history"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": result["answer"],
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("query_metadata", {}).get("sources", [])
        }
        
        self.conversation_history.append(history_entry)
        
        # Keep only last 50 exchanges to manage memory
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def _update_session_stats(self, result: Dict[str, Any], response_time: float):
        """Update session statistics"""
        self.session_stats["queries_processed"] += 1
        
        # Update average response time
        current_avg = self.session_stats["average_response_time"]
        queries_count = self.session_stats["queries_processed"]
        new_avg = ((current_avg * (queries_count - 1)) + response_time) / queries_count
        self.session_stats["average_response_time"] = new_avg
        
        # Track sources used
        sources = result.get("query_metadata", {}).get("sources", [])
        self.session_stats["sources_used"].update(sources)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation history"""
        if format.lower() == "json":
            import json
            return json.dumps(self.conversation_history, indent=2)
        elif format.lower() == "txt":
            text_lines = []
            for entry in self.conversation_history:
                text_lines.append(f"[{entry['timestamp']}]")
                text_lines.append(f"User: {entry['user_input']}")
                text_lines.append(f"Bot: {entry['response']}")
                text_lines.append(f"Confidence: {entry['confidence']}")
                text_lines.append(f"Sources: {', '.join(entry['sources'])}")
                text_lines.append("-" * 50)
            return "\n".join(text_lines)
        else:
            raise ValueError("Supported formats: 'json', 'txt'")
    
    def suggest_follow_up_questions(self, last_response: Dict[str, Any]) -> List[str]:
        """Suggest follow-up questions based on the last response"""
        sources = last_response.get("sources_used", [])
        suggestions = []
        
        if "CVE" in sources:
            suggestions.append("Can you show me similar vulnerabilities?")
            suggestions.append("What are the mitigation strategies?")
        
        if "OWASP" in sources:
            suggestions.append("How can I test for this vulnerability?")
            suggestions.append("What tools can help prevent this?")
        
        if "MITRE" in sources:
            suggestions.append("What are the detection methods?")
            suggestions.append("How do attackers typically use this technique?")
        
        if "Exploits" in sources:
            suggestions.append("What are the countermeasures?")
            suggestions.append("How can I protect against this exploit?")
        
        # Add general suggestions
        suggestions.extend([
            "Can you explain this in more detail?",
            "What are the real-world examples?",
            "How serious is this threat?"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions

def main():
    """Main function to test the chatbot"""
    Config.setup_logging()
    
    # Initialize chatbot
    print("🛡️ Cybersecurity Chatbot")
    print("Type '/help' for commands or ask any cybersecurity question.")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    chatbot = CybersecurityChatbot()
    
    while True:
        try:
            user_input = input("\n🔍 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using the Cybersecurity Chatbot!")
                break
            
            if not user_input:
                continue
            
            # Get response
            response = chatbot.chat(user_input)
            
            # Display response
            print(f"\n🤖 Bot: {response['answer']}")
            
            if response.get("confidence"):
                print(f"📊 Confidence: {response['confidence']} ({response.get('confidence_level', 'Unknown')})")
            
            if response.get("sources_used"):
                print(f"📚 Sources: {', '.join(response['sources_used'])}")
            
            if response.get("response_time"):
                print(f"⏱️ Response time: {response['response_time']}s")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
