#!/usr/bin/env python3
"""
Setup script for Cybersecurity Chatbot
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file if it exists
readme_path = Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="cybersecurity-chatbot",
    version="1.0.0",
    author="Cybersecurity Team",
    author_email="security@example.com",
    description="AI-powered cybersecurity knowledge assistant using LangChain and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cybersecurity-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "transformers[torch]>=4.30.0",
        ],
        "cloud": [
            "boto3>=1.26.0",
            "azure-storage-blob>=12.14.0",
            "google-cloud-storage>=2.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cybersec-chat=ui.cli_interface:main",
            "cybersec-web=ui.streamlit_app:main",
            "cybersec-init=src.data_loader:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
        "data": ["*.json"],
        "utils": ["*.py"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cybersecurity-chatbot/issues",
        "Source": "https://github.com/yourusername/cybersecurity-chatbot",
        "Documentation": "https://cybersecurity-chatbot.readthedocs.io/",
    },
    keywords="cybersecurity, AI, chatbot, security, langchain, RAG, OWASP, CVE, MITRE",
)