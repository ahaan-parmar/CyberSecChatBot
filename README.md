# Cybersecurity Chatbot

This project provides an AI powered assistant for cybersecurity topics. It includes a command line interface and a Streamlit web UI. The chatbot uses a retrieval‑augmented generation pipeline over several knowledge bases such as CVE, OWASP and MITRE ATT&CK.

## Features

- **Command line interface** with rich formatting
- **Streamlit web UI** with conversation history, statistics and knowledge base information
- **RAG pipeline** powered by LangChain and a Chroma vector store
- **Pluggable data sources** stored under `data/`

## Setup

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the initialization script to build the vector store:

```bash
python init.py
```

## Usage

### Command Line

Start the CLI interface:

```bash
python run.py cli
```

### Web UI

Launch the Streamlit application:

```bash
python run.py web
```

The app will open in your browser (default: http://localhost:8501).

## Development

The repository contains optional development tools such as `black`, `flake8` and `pytest`. No tests are included yet, but running `pytest` ensures the environment is correctly configured.

