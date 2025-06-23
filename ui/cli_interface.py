#!/usr/bin/env python3
"""
Command Line Interface for the Cybersecurity Chatbot
"""

import sys
import os
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text
from datetime import datetime
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot import CybersecurityChatbot
from utils.config import Config

console = Console()

class CybersecurityCLI:
    """Command line interface for the cybersecurity chatbot"""
    
    def __init__(self):
        self.chatbot = None
        self.console = Console()
        self.running = True
    
    def initialize_chatbot(self):
        """Initialize the chatbot with progress indicator"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing cybersecurity chatbot...", total=None)
            
            try:
                self.chatbot = CybersecurityChatbot()
                progress.update(task, description="✅ Chatbot initialized successfully!")
                return True
            except Exception as e:
                progress.update(task, description=f"❌ Failed to initialize: {e}")
                return False
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# 🛡️ Cybersecurity Chatbot

Welcome to your AI-powered cybersecurity knowledge assistant!

## What I can help with:
- **CVE Vulnerabilities**: Get detailed information about specific CVEs
- **OWASP Guidelines**: Learn about web application security best practices  
- **MITRE ATT&CK**: Understand adversarial tactics and techniques
- **Exploit Methods**: Learn about attack vectors and countermeasures

## Available Commands:
- `/help` - Show detailed help information
- `/stats` - Display session statistics
- `/sources` - List available data sources
- `/clear` - Clear conversation history
- `/examples` - Show example queries
- `/export` - Export conversation history
- `/quit` or `Ctrl+C` - Exit the chatbot

## Tips:
- Be specific in your questions for better results
- Use technical cybersecurity terms
- Ask about specific CVE IDs, MITRE techniques, or OWASP categories
        """
        
        panel = Panel(
            Markdown(welcome_text),
            title="🛡️ Cybersecurity Chatbot",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def display_response(self, response):
        """Display chatbot response with formatting"""
        # Main answer
        answer_panel = Panel(
            Markdown(response["answer"]),
            title="🤖 Cybersecurity Assistant",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(answer_panel)
        
        # Metadata table
        if any([response.get("confidence"), response.get("response_time"), response.get("sources_used")]):
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            if response.get("confidence"):
                confidence = response["confidence"]
                confidence_level = response.get("confidence_level", "Unknown")
                confidence_color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.6 else "red"
                table.add_row(
                    "📊 Confidence", 
                    f"[{confidence_color}]{confidence} ({confidence_level})[/{confidence_color}]"
                )
            
            if response.get("response_time"):
                table.add_row("⏱️ Response Time", f"{response['response_time']}s")
            
            if response.get("sources_used"):
                sources = ", ".join(response["sources_used"])
                table.add_row("📚 Sources", sources)
            
            self.console.print(table)
        
        # Source documents (if available and user wants to see them)
        if response.get("source_documents"):
            if click.confirm("\n📄 Would you like to see the source documents?", default=False):
                self.display_source_documents(response["source_documents"])
    
    def display_source_documents(self, documents):
        """Display source documents"""
        for i, doc in enumerate(documents, 1):
            # Document header
            source = doc.metadata.get("source", "Unknown")
            doc_id = (
                doc.metadata.get("cve_id") or 
                doc.metadata.get("technique_id") or 
                doc.metadata.get("exploit_name") or 
                f"Doc {i}"
            )
            
            title = f"📄 Source {i}: {source} - {doc_id}"
            
            # Metadata
            metadata_text = ""
            if doc.metadata.get("severity"):
                metadata_text += f"⚠️ Severity: {doc.metadata['severity']} | "
            if doc.metadata.get("cvss_score"):
                metadata_text += f"📊 CVSS: {doc.metadata['cvss_score']} | "
            if doc.metadata.get("doc_type"):
                metadata_text += f"📝 Type: {doc.metadata['doc_type']}"
            
            # Content preview
            content = doc.page_content
            if len(content) > 800:
                content = content[:800] + "\n...[Content truncated]"
            
            document_content = f"{metadata_text}\n\n{content}" if metadata_text else content
            
            doc_panel = Panel(
                document_content,
                title=title,
                border_style="dim blue",
                padding=(1, 2)
            )
            self.console.print(doc_panel)
    
    def display_help(self):
        """Display detailed help"""
        help_text = """
# 🛡️ Cybersecurity Chatbot Help

## Available Commands:

### System Commands:
- `/help` - Show this help message
- `/stats` - Display session statistics and performance metrics
- `/sources` - List all available data sources and their status
- `/clear` - Clear conversation history
- `/examples` - Show example queries you can try
- `/export [format]` - Export conversation (json/txt)
- `/quit` - Exit the application

### Query Modes:
- **Normal Mode**: Ask any cybersecurity question
- **Filter Mode**: Use `/filter <source>` to search specific sources
  - Available sources: CVE, OWASP, MITRE, Exploits

## Example Queries:

### CVE Vulnerabilities:
- "Tell me about CVE-2023-1234"
- "What are the most critical SQL injection CVEs?"
- "Show me recent Apache vulnerabilities"

### OWASP Security:
- "What is cross-site scripting and how to prevent it?"
- "Explain the OWASP Top 10"
- "How to implement secure authentication?"

### MITRE ATT&CK:
- "What is privilege escalation?"
- "How do attackers perform lateral movement?"
- "Explain persistence techniques"

### Exploit Techniques:
- "How do buffer overflow attacks work?"
- "What are common web application exploits?"
- "How to defend against social engineering?"

### General Security:
- "How to secure a web application?"
- "What is a zero-day vulnerability?"
- "Best practices for network security"

## Tips for Better Results:
1. **Be Specific**: Use exact CVE IDs, MITRE technique IDs, or OWASP categories
2. **Use Technical Terms**: Include relevant cybersecurity terminology
3. **Ask Follow-ups**: Build on previous answers for deeper understanding
4. **Request Details**: Ask for mitigation strategies, detection methods, etc.
        """
        
        help_panel = Panel(
            Markdown(help_text),
            title="📖 Help & Documentation",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(help_panel)
    
    def display_stats(self):
        """Display session statistics"""
        if not self.chatbot:
            self.console.print("❌ Chatbot not initialized", style="red")
            return
        
        stats = self.chatbot.session_stats
        uptime = datetime.now() - stats["session_start"]
        
        # Main stats table
        table = Table(title="📊 Session Statistics", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=30)
        
        table.add_row("Queries Processed", str(stats["queries_processed"]))
        table.add_row("Session Duration", str(uptime).split('.')[0])
        table.add_row("Average Response Time", f"{stats['average_response_time']:.2f}s")
        table.add_row("Sources Accessed", str(len(stats["sources_used"])))
        table.add_row("Conversation Exchanges", str(len(self.chatbot.conversation_history)))
        
        self.console.print(table)
        
        # Sources used
        if stats["sources_used"]:
            sources_text = ", ".join(sorted(stats["sources_used"]))
            sources_panel = Panel(
                sources_text,
                title="📚 Data Sources Used This Session",
                border_style="green"
            )
            self.console.print(sources_panel)
    
    def display_sources(self):
        """Display available data sources"""
        if not self.chatbot:
            self.console.print("❌ Chatbot not initialized", style="red")
            return
        
        try:
            vector_stats = self.chatbot.rag_chain.vector_store.get_statistics()
            sources = vector_stats.get("sources", {})
            
            table = Table(title="📚 Available Data Sources", show_header=True, header_style="bold blue")
            table.add_column("Source", style="cyan", width=15)
            table.add_column("Documents", style="white", width=12)
            table.add_column("Description", style="white", width=50)
            
            source_descriptions = {
                "CVE": "Common Vulnerabilities and Exposures database",
                "OWASP": "Web application security guidelines and best practices",
                "MITRE": "Adversarial tactics, techniques, and procedures (ATT&CK)",
                "Exploits": "Exploit techniques and countermeasures"
            }
            
            for source, count in sources.items():
                description = source_descriptions.get(source, "Security knowledge base")
                table.add_row(source, str(count), description)
            
            total_docs = vector_stats.get("total_documents", 0)
            table.add_row("TOTAL", str(total_docs), "Combined knowledge base", style="bold green")
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"❌ Error retrieving source information: {e}", style="red")
    
    def display_examples(self):
        """Display example queries"""
        examples_text = """
# 💡 Example Queries to Try

## CVE Vulnerabilities:
- `Tell me about CVE-2023-1234`
- `What are the most critical CVEs for Apache?`
- `Show me SQL injection vulnerabilities`
- `What's the CVSS score for CVE-2022-1234?`

## OWASP Security Guidelines:
- `What is cross-site scripting (XSS)?`
- `How to prevent injection attacks?`
- `Explain the OWASP Top 10`
- `What is security misconfiguration?`

## MITRE ATT&CK Framework:
- `What is privilege escalation?`
- `How do attackers perform lateral movement?`
- `Explain persistence techniques`
- `What is T1055 process injection?`

## Exploit Techniques:
- `How do buffer overflow attacks work?`
- `What are common web application exploits?`
- `How to defend against social engineering?`
- `What is a reverse shell?`

## General Security Questions:
- `How to secure a web application?`
- `What is a zero-day vulnerability?`
- `Best practices for network security`
- `How to implement defense in depth?`

## Advanced Queries:
- `Compare SQL injection and XSS attacks`
- `What's the relationship between CVE-2023-1234 and MITRE techniques?`
- `How do OWASP guidelines help prevent common exploits?`
        """
        
        examples_panel = Panel(
            Markdown(examples_text),
            title="💡 Example Queries",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(examples_panel)
    
    def export_conversation(self, format_type="json"):
        """Export conversation history"""
        if not self.chatbot or not self.chatbot.conversation_history:
            self.console.print("📝 No conversation history to export", style="yellow")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cybersec_chat_{timestamp}.{format_type}"
        
        try:
            export_data = self.chatbot.export_conversation(format_type)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(export_data)
            
            self.console.print(f"✅ Conversation exported to: {filename}", style="green")
            
        except Exception as e:
            self.console.print(f"❌ Export failed: {e}", style="red")
    
    def handle_command(self, command):
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == "/help":
            self.display_help()
        elif command == "/stats":
            self.display_stats()
        elif command == "/sources":
            self.display_sources()
        elif command == "/clear":
            if self.chatbot:
                self.chatbot.conversation_history.clear()
                self.console.print("✅ Conversation history cleared", style="green")
        elif command == "/examples":
            self.display_examples()
        elif command.startswith("/export"):
            parts = command.split()
            format_type = parts[1] if len(parts) > 1 and parts[1] in ["json", "txt"] else "json"
            self.export_conversation(format_type)
        elif command in ["/quit", "/exit"]:
            self.running = False
            self.console.print("👋 Thank you for using the Cybersecurity Chatbot!", style="blue")
        else:
            self.console.print(f"❌ Unknown command: {command}. Type '/help' for available commands.", style="red")
    
    def run(self):
        """Main CLI loop"""
        self.display_welcome()
        
        if not self.initialize_chatbot():
            self.console.print("❌ Failed to initialize chatbot. Exiting.", style="red")
            return
        
        self.console.print("\n🚀 Ready! Type your cybersecurity questions or '/help' for commands.\n")
        
        try:
            while self.running:
                # Get user input
                user_input = click.prompt(
                    Text("🔍", style="blue bold"),
                    type=str,
                    prompt_suffix=" ",
                    show_default=False
                ).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue
                
                # Process regular query
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("🤔 Processing your question...", total=None)
                    
                    try:
                        response = self.chatbot.chat(user_input)
                        progress.update(task, description="✅ Response ready!")
                        
                    except Exception as e:
                        progress.update(task, description=f"❌ Error: {e}")
                        self.console.print(f"❌ Error processing query: {e}", style="red")
                        continue
                
                # Display response
                self.console.print()  # Add spacing
                self.display_response(response)
                self.console.print()  # Add spacing
                
        except KeyboardInterrupt:
            self.console.print("\n👋 Goodbye! Thanks for using the Cybersecurity Chatbot!", style="blue")
        except Exception as e:
            self.console.print(f"\n❌ Unexpected error: {e}", style="red")

@click.command()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', help='Path to configuration file')
def main(debug, config):
    """
    🛡️ Cybersecurity Chatbot CLI
    
    An AI-powered assistant for cybersecurity knowledge and guidance.
    """
    if debug:
        Config.DEBUG = True
        Config.LOG_LEVEL = "DEBUG"
    
    if config:
        # Load custom config if provided
        pass
    
    # Setup logging
    Config.setup_logging()
    
    # Run CLI
    cli = CybersecurityCLI()
    cli.run()

if __name__ == "__main__":
    main()