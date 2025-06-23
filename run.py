#!/usr/bin/env python3
"""
Cybersecurity Chatbot Application Launcher

This script provides an easy way to start the cybersecurity chatbot in different modes:
- Web UI (Streamlit)
- Command Line Interface
- Initialize system
- Development mode
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

class CybersecLauncher:
    """Application launcher for the cybersecurity chatbot"""
    
    def __init__(self):
        self.console = Console()
        self.project_root = Path(__file__).parent
        
    def display_banner(self):
        """Display application banner"""
        banner = """
# 🛡️ Cybersecurity Chatbot

Your AI-powered cybersecurity knowledge assistant
        """
        
        panel = Panel(
            Markdown(banner),
            title="🚀 Cybersecurity Assistant",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def check_requirements(self) -> bool:
        """Check if requirements are installed"""
        try:
            import langchain
            import streamlit
            import chromadb
            return True
        except ImportError as e:
            self.console.print(f"❌ [red]Missing required packages: {e}[/red]")
            self.console.print("💡 [blue]Run: pip install -r requirements.txt[/blue]")
            return False
    
    def run_initialization(self, force_rebuild: bool = False):
        """Run system initialization"""
        self.console.print("🔧 [blue]Initializing system...[/blue]")
        
        try:
            cmd = [sys.executable, "init.py"]
            if force_rebuild:
                cmd.append("--force-rebuild")
            
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print("✅ [green]Initialization completed successfully![/green]")
                return True
            else:
                self.console.print(f"❌ [red]Initialization failed:[/red]")
                self.console.print(result.stderr)
                return False
                
        except Exception as e:
            self.console.print(f"❌ [red]Error during initialization: {e}[/red]")
            return False
    
    def start_web_ui(self, host: str = "localhost", port: int = 8501, auto_open: bool = True):
        """Start the Streamlit web interface"""
        self.console.print(f"🌐 [blue]Starting web interface on http://{host}:{port}[/blue]")
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "ui/streamlit_app.py",
                "--server.address", host,
                "--server.port", str(port),
                "--server.headless", "true" if not auto_open else "false",
                "--browser.gatherUsageStats", "false"
            ]
            
            # Start Streamlit
            process = subprocess.Popen(cmd, cwd=self.project_root)
            
            if auto_open:
                # Wait a moment for server to start
                time.sleep(3)
                webbrowser.open(f"http://{host}:{port}")
            
            self.console.print(f"✅ [green]Web interface started! Access at: http://{host}:{port}[/green]")
            self.console.print("Press [bold red]Ctrl+C[/bold red] to stop the server")
            
            # Wait for process
            process.wait()
            
        except KeyboardInterrupt:
            self.console.print("\n👋 [blue]Shutting down web interface...[/blue]")
            process.terminate()
        except Exception as e:
            self.console.print(f"❌ [red]Error starting web interface: {e}[/red]")
    
    def start_cli(self):
        """Start the command line interface"""
        self.console.print("💻 [blue]Starting command line interface...[/blue]")
        
        try:
            cmd = [sys.executable, "ui/cli_interface.py"]
            subprocess.run(cmd, cwd=self.project_root)
        except Exception as e:
            self.console.print(f"❌ [red]Error starting CLI: {e}[/red]")
    
    def show_status(self):
        """Show system status"""
        self.console.print("📊 [blue]System Status[/blue]")
        
        # Check components
        status_table = Table(title="Component Status", show_header=True, header_style="bold blue")
        status_table.add_column("Component", style="cyan", width=20)
        status_table.add_column("Status", width=15)
        status_table.add_column("Details", style="white", width=40)
        
        # Check Python environment
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_status = "✅ OK" if sys.version_info >= (3, 8) else "❌ Old"
        status_table.add_row("Python", python_status, f"Version {python_version}")
        
        # Check packages
        try:
            import langchain
            import streamlit
            import chromadb
            packages_status = "✅ OK"
            packages_detail = "All required packages installed"
        except ImportError:
            packages_status = "❌ Missing"
            packages_detail = "Some packages missing"
        
        status_table.add_row("Packages", packages_status, packages_detail)
        
        # Check configuration
        env_file = Path(self.project_root) / ".env"
        if env_file.exists():
            config_status = "✅ Found"
            config_detail = "Environment file exists"
        else:
            config_status = "⚠️ Missing"
            config_detail = "No .env file found"
        
        status_table.add_row("Configuration", config_status, config_detail)
        
        # Check data directory
        data_dir = Path(self.project_root) / "data"
        if data_dir.exists() and any(data_dir.glob("*.json")):
            data_status = "✅ OK"
            data_detail = f"Found {len(list(data_dir.glob('*.json')))} data files"
        else:
            data_status = "⚠️ Empty"
            data_detail = "No data files found"
        
        status_table.add_row("Data Files", data_status, data_detail)
        
        # Check vector store
        vector_dir = Path(self.project_root) / "vector_store"
        if vector_dir.exists() and any(vector_dir.iterdir()):
            vector_status = "✅ Ready"
            vector_detail = "Vector store initialized"
        else:
            vector_status = "⚠️ Missing"
            vector_detail = "Vector store not initialized"
        
        status_table.add_row("Vector Store", vector_status, vector_detail)
        
        self.console.print(status_table)
    
    def show_help(self):
        """Show help information"""
        help_text = """
## Available Commands:

- `python run.py web` - Start web interface (Streamlit)
- `python run.py cli` - Start command line interface
- `python run.py init` - Initialize the system
- `python run.py status` - Show system status
- `python run.py help` - Show this help

## Options:

- `--host` - Web interface host (default: localhost)
- `--port` - Web interface port (default: 8501)
- `--no-browser` - Don't auto-open browser
- `--force-rebuild` - Force rebuild vector store

## Examples:

```bash
# Start web interface on custom port
python run.py web --port 8080

# Initialize with force rebuild
python run.py init --force-rebuild

# Start web on all interfaces
python run.py web --host 0.0.0.0
```
        """
        
        help_panel = Panel(
            Markdown(help_text),
            title="📖 Help & Usage",
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(help_panel)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Cybersecurity Chatbot Application Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["web", "cli", "init", "status", "help"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for web interface (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for web interface (default: 8501)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open browser for web interface"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of vector store during initialization"
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = CybersecLauncher()
    launcher.display_banner()
    
    # Check requirements for most commands
    if args.command not in ["help", "status"] and not launcher.check_requirements():
        sys.exit(1)
    
    # Execute command
    if args.command == "web":
        launcher.start_web_ui(
            host=args.host,
            port=args.port,
            auto_open=not args.no_browser
        )
    
    elif args.command == "cli":
        launcher.start_cli()
    
    elif args.command == "init":
        success = launcher.run_initialization(force_rebuild=args.force_rebuild)
        sys.exit(0 if success else 1)
    
    elif args.command == "status":
        launcher.show_status()
    
    elif args.command == "help":
        launcher.show_help()

if __name__ == "__main__":
    main()