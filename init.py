#!/usr/bin/env python3
"""
Cybersecurity Chatbot Initialization Script

This script helps initialize the cybersecurity chatbot system by:
- Validating the environment
- Setting up required directories
- Initializing the vector store
- Running basic tests
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.config import Config
from utils.helpers import ConfigValidator, SecurityUtils, format_file_size
from src.data_loader import CybersecurityDataLoader
from src.vector_store import CybersecurityVectorStore

console = Console()

class CybersecInitializer:
    """Initialize the cybersecurity chatbot system"""
    
    def __init__(self):
        self.console = Console()
        self.issues = []
        
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# 🛡️ Cybersecurity Chatbot Initialization

Welcome to the setup process for your AI-powered cybersecurity assistant!

This script will:
- ✅ Validate your environment
- 📁 Create necessary directories  
- 🔧 Check configuration
- 📊 Initialize the vector store
- 🧪 Run basic functionality tests

Let's get started!
        """
        
        panel = Panel(
            Markdown(welcome_text),
            title="🚀 System Initialization",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def validate_environment(self) -> bool:
        """Validate the environment setup"""
        self.console.print("\n🔍 [bold blue]Validating Environment...[/bold blue]")
        
        validations = ConfigValidator.validate_environment()
        
        # Create validation table
        table = Table(title="Environment Validation", show_header=True, header_style="bold blue")
        table.add_column("Component", style="cyan", width=25)
        table.add_column("Status", width=10)
        table.add_column("Details", style="white", width=40)
        
        for component, is_valid in validations.items():
            status = "✅ Pass" if is_valid else "❌ Fail"
            status_style = "green" if is_valid else "red"
            
            details = self._get_validation_details(component, is_valid)
            
            table.add_row(
                component.replace('_', ' ').title(),
                f"[{status_style}]{status}[/{status_style}]",
                details
            )
            
            if not is_valid:
                self.issues.append(f"{component}: {details}")
        
        self.console.print(table)
        
        all_valid = all(validations.values())
        if not all_valid:
            self.console.print(f"\n⚠️ [yellow]Found {len(self.issues)} issues that need attention[/yellow]")
        
        return all_valid
    
    def _get_validation_details(self, component: str, is_valid: bool) -> str:
        """Get detailed validation information"""
        if component == "python_version":
            import sys
            version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            return f"Python {version}" if is_valid else f"Python {version} (requires 3.8+)"
        
        elif component == "required_packages":
            return "All packages installed" if is_valid else "Missing required packages"
        
        elif component == "data_directory":
            return f"Directory exists: {Config.DATA_DIR}" if is_valid else f"Missing: {Config.DATA_DIR}"
        
        elif component == "vector_store_path":
            path = Path(Config.VECTOR_STORE_PATH).parent
            return f"Directory accessible: {path}" if is_valid else f"Cannot access: {path}"
        
        elif component == "log_directory":
            path = Path(Config.LOG_FILE).parent
            return f"Directory accessible: {path}" if is_valid else f"Cannot access: {path}"
        
        return "Check passed" if is_valid else "Check failed"
    
    def check_configuration(self) -> bool:
        """Check configuration validity"""
        self.console.print("\n⚙️ [bold blue]Checking Configuration...[/bold blue]")
        
        config_issues = []
        
        # Check API keys
        if not Config.OPENAI_API_KEY and not Config.ANTHROPIC_API_KEY:
            config_issues.append("No LLM API key configured (OpenAI or Anthropic required)")
        
        if Config.OPENAI_API_KEY and not ConfigValidator.validate_api_key(Config.OPENAI_API_KEY, "openai"):
            config_issues.append("Invalid OpenAI API key format")
        
        if Config.ANTHROPIC_API_KEY and not ConfigValidator.validate_api_key(Config.ANTHROPIC_API_KEY, "anthropic"):
            config_issues.append("Invalid Anthropic API key format")
        
        # Check model configuration
        if not ConfigValidator.validate_model_name(Config.LLM_MODEL):
            config_issues.append(f"Invalid LLM model name: {Config.LLM_MODEL}")
        
        # Display results
        if config_issues:
            for issue in config_issues:
                self.console.print(f"❌ [red]{issue}[/red]")
            self.issues.extend(config_issues)
            return False
        else:
            self.console.print("✅ [green]Configuration is valid[/green]")
            return True
    
    def setup_directories(self) -> bool:
        """Create necessary directories"""
        self.console.print("\n📁 [bold blue]Setting up Directories...[/bold blue]")
        
        directories = [
            Config.DATA_DIR,
            Path(Config.VECTOR_STORE_PATH).parent,
            Path(Config.LOG_FILE).parent,
            "logs",
            "exports"
        ]
        
        created_dirs = []
        failed_dirs = []
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(directory))
            except Exception as e:
                failed_dirs.append(f"{directory}: {e}")
        
        # Report results
        for directory in created_dirs:
            self.console.print(f"✅ [green]Created/verified: {directory}[/green]")
        
        for error in failed_dirs:
            self.console.print(f"❌ [red]Failed: {error}[/red]")
            self.issues.append(f"Directory creation failed: {error}")
        
        return len(failed_dirs) == 0
    
    def check_data_sources(self) -> Dict[str, bool]:
        """Check availability of data sources"""
        self.console.print("\n📊 [bold blue]Checking Data Sources...[/bold blue]")
        
        data_sources = Config.get_data_sources()
        source_status = {}
        
        table = Table(title="Data Sources Status", show_header=True, header_style="bold blue")
        table.add_column("Source", style="cyan", width=15)
        table.add_column("File", style="white", width=30)
        table.add_column("Status", width=10)
        table.add_column("Size", width=10)
        
        for source_name, filepath in data_sources.items():
            path = Path(filepath)
            if path.exists():
                status = "✅ Found"
                status_style = "green"
                size = format_file_size(path.stat().st_size)
                source_status[source_name] = True
            else:
                status = "❌ Missing"
                status_style = "red"
                size = "N/A"
                source_status[source_name] = False
            
            table.add_row(
                source_name.upper(),
                str(path.name),
                f"[{status_style}]{status}[/{status_style}]",
                size
            )
        
        self.console.print(table)
        
        missing_sources = [name for name, exists in source_status.items() if not exists]
        if missing_sources:
            self.console.print(f"\n⚠️ [yellow]Missing data files: {', '.join(missing_sources)}[/yellow]")
            self.console.print("💡 [blue]You can create sample data files or add your own data[/blue]")
        
        return source_status
    
    def initialize_vector_store(self, force_rebuild: bool = False) -> bool:
        """Initialize the vector store"""
        self.console.print("\n🗄️ [bold blue]Initializing Vector Store...[/bold blue]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                
                if force_rebuild:
                    task = progress.add_task("Rebuilding vector store...", total=None)
                    vector_store = CybersecurityVectorStore(force_rebuild=True)
                else:
                    task = progress.add_task("Loading/creating vector store...", total=None)
                    vector_store = CybersecurityVectorStore()
                
                progress.update(task, description="✅ Vector store ready!")
            
            # Get statistics
            stats = vector_store.get_statistics()
            
            # Display results
            if stats.get("total_documents", 0) > 0:
                self.console.print(f"✅ [green]Vector store initialized with {stats['total_documents']} documents[/green]")
                
                # Show source breakdown
                sources = stats.get("sources", {})
                if sources:
                    for source, count in sources.items():
                        self.console.print(f"   📄 {source}: {count} documents")
                
                return True
            else:
                self.console.print("⚠️ [yellow]Vector store created but no documents loaded[/yellow]")
                self.issues.append("No documents in vector store - add data files")
                return False
                
        except Exception as e:
            self.console.print(f"❌ [red]Vector store initialization failed: {e}[/red]")
            self.issues.append(f"Vector store error: {e}")
            return False
    
    def run_basic_tests(self) -> bool:
        """Run basic functionality tests"""
        self.console.print("\n🧪 [bold blue]Running Basic Tests...[/bold blue]")
        
        tests_passed = 0
        total_tests = 3
        
        try:
            # Test 1: Import core modules
            from src.chatbot import CybersecurityChatbot
            self.console.print("✅ [green]Core modules import successfully[/green]")
            tests_passed += 1
            
            # Test 2: Initialize chatbot
            chatbot = CybersecurityChatbot()
            self.console.print("✅ [green]Chatbot initialization successful[/green]")
            tests_passed += 1
            
            # Test 3: Test basic query
            test_query = "What is cybersecurity?"
            response = chatbot.chat(test_query)
            if response and response.get("answer"):
                self.console.print("✅ [green]Basic query test successful[/green]")
                tests_passed += 1
            else:
                self.console.print("❌ [red]Basic query test failed[/red]")
                self.issues.append("Basic query test failed")
            
        except Exception as e:
            self.console.print(f"❌ [red]Test failed: {e}[/red]")
            self.issues.append(f"Basic test error: {e}")
        
        self.console.print(f"\n📊 Tests passed: {tests_passed}/{total_tests}")
        return tests_passed == total_tests
    
    def display_summary(self):
        """Display initialization summary"""
        self.console.print(f"\n📋 [bold blue]Initialization Summary[/bold blue]")
        
        if not self.issues:
            success_panel = Panel(
                "🎉 [green]All checks passed! Your cybersecurity chatbot is ready to use.[/green]\n\n"
                "Next steps:\n"
                "• Run the CLI: [cyan]python ui/cli_interface.py[/cyan]\n"
                "• Start web UI: [cyan]streamlit run ui/streamlit_app.py[/cyan]\n"
                "• Add more data: Place JSON files in the [cyan]data/[/cyan] directory",
                title="✅ Success",
                border_style="green"
            )
            self.console.print(success_panel)
        else:
            issue_text = "\n".join([f"• {issue}" for issue in self.issues])
            
            issues_panel = Panel(
                f"⚠️ [yellow]Found {len(self.issues)} issues:[/yellow]\n\n{issue_text}\n\n"
                "Please resolve these issues before using the chatbot.\n"
                "Check the documentation for troubleshooting guidance.",
                title="⚠️ Issues Found",
                border_style="yellow"
            )
            self.console.print(issues_panel)
    
    def run_initialization(self, force_rebuild: bool = False, skip_tests: bool = False):
        """Run the complete initialization process"""
        self.display_welcome()
        
        # Step 1: Validate environment
        env_valid = self.validate_environment()
        
        # Step 2: Check configuration
        config_valid = self.check_configuration()
        
        # Step 3: Setup directories
        dirs_created = self.setup_directories()
        
        # Step 4: Check data sources
        data_status = self.check_data_sources()
        
        # Step 5: Initialize vector store
        vector_store_ready = self.initialize_vector_store(force_rebuild)
        
        # Step 6: Run tests (if not skipped)
        if not skip_tests and vector_store_ready:
            tests_passed = self.run_basic_tests()
        else:
            tests_passed = True
        
        # Step 7: Display summary
        self.display_summary()
        
        return len(self.issues) == 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Initialize the Cybersecurity Chatbot system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of the vector store"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip basic functionality tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        Config.DEBUG = True
        Config.LOG_LEVEL = "DEBUG"
    
    Config.setup_logging()
    
    # Run initialization
    initializer = CybersecInitializer()
    success = initializer.run_initialization(
        force_rebuild=args.force_rebuild,
        skip_tests=args.skip_tests
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
