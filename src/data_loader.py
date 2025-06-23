import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import Config, DATA_SCHEMAS

logger = logging.getLogger(__name__)

class CybersecurityDataLoader:
    """Loads and processes cybersecurity data from various sources"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_schemas = DATA_SCHEMAS
    
    def validate_data_structure(self, data: List[Dict], schema_name: str) -> bool:
        """Validate data structure against schema"""
        if not data:
            logger.warning(f"No data provided for {schema_name}")
            return False
        
        schema = self.data_schemas.get(schema_name)
        if not schema:
            logger.warning(f"No schema found for {schema_name}")
            return True  # Skip validation if no schema
        
        required_fields = schema.get("required_fields", [])
        
        for item in data[:5]:  # Check first 5 items
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                logger.warning(f"Missing required fields in {schema_name}: {missing_fields}")
                return False
        
        return True
    
    def load_cve_data(self, filepath: str) -> List[Document]:
        """Load and process CVE data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cve_data = json.load(f)
            
            if not self.validate_data_structure(cve_data, "cve"):
                logger.error(f"Invalid CVE data structure in {filepath}")
                return []
            
            documents = []
            for cve in cve_data:
                content = self._format_cve_content(cve)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "CVE",
                        "source_file": filepath,
                        "cve_id": cve.get('id', 'Unknown'),
                        "severity": cve.get('severity', 'Unknown'),
                        "cvss_score": cve.get('cvss_score', 0),
                        "doc_type": "vulnerability"
                    }
                )
                documents.append(doc)
            
            # Split documents if they're too large
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(split_docs)} CVE document chunks from {len(documents)} CVEs")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading CVE data from {filepath}: {e}")
            return []
    
    def _format_cve_content(self, cve: Dict) -> str:
        """Format CVE data into readable content"""
        content_parts = [
            f"CVE ID: {cve.get('id', 'N/A')}",
            f"Description: {cve.get('description', 'N/A')}",
            f"CVSS Score: {cve.get('cvss_score', 'N/A')}",
            f"Severity: {cve.get('severity', 'N/A')}"
        ]
        
        if cve.get('published_date'):
            content_parts.append(f"Published: {cve['published_date']}")
        
        if cve.get('affected_products'):
            products = ', '.join(cve['affected_products'][:5])  # Limit to first 5
            if len(cve['affected_products']) > 5:
                products += f" and {len(cve['affected_products']) - 5} more"
            content_parts.append(f"Affected Products: {products}")
        
        if cve.get('references'):
            refs = ', '.join(cve['references'][:3])  # Limit to first 3
            content_parts.append(f"References: {refs}")
        
        return '\n'.join(content_parts)
    
    def load_owasp_data(self, filepath: str) -> List[Document]:
        """Load and process OWASP Top 10 data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                owasp_data = json.load(f)
            
            if not self.validate_data_structure(owasp_data, "owasp"):
                logger.error(f"Invalid OWASP data structure in {filepath}")
                return []
            
            documents = []
            for item in owasp_data:
                content = self._format_owasp_content(item)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "OWASP",
                        "source_file": filepath,
                        "rank": item.get('rank', 0),
                        "category": item.get('category', 'Unknown'),
                        "doc_type": "security_guidance"
                    }
                )
                documents.append(doc)
            
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(split_docs)} OWASP document chunks from {len(documents)} categories")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading OWASP data from {filepath}: {e}")
            return []
    
    def _format_owasp_content(self, item: Dict) -> str:
        """Format OWASP data into readable content"""
        content_parts = [
            f"OWASP Rank: {item.get('rank', 'N/A')}",
            f"Category: {item.get('category', 'N/A')}",
            f"Description: {item.get('description', 'N/A')}"
        ]
        
        if item.get('impact'):
            content_parts.append(f"Impact: {item['impact']}")
        
        if item.get('prevention'):
            content_parts.append(f"Prevention: {item['prevention']}")
        
        if item.get('examples'):
            examples = ', '.join(item['examples'][:3])  # Limit to first 3
            content_parts.append(f"Examples: {examples}")
        
        return '\n'.join(content_parts)
    
    def load_mitre_data(self, filepath: str) -> List[Document]:
        """Load and process MITRE ATT&CK data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                mitre_data = json.load(f)
            
            if not self.validate_data_structure(mitre_data, "mitre"):
                logger.error(f"Invalid MITRE data structure in {filepath}")
                return []
            
            documents = []
            for technique in mitre_data:
                content = self._format_mitre_content(technique)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "MITRE",
                        "source_file": filepath,
                        "technique_id": technique.get('id', 'Unknown'),
                        "tactic": technique.get('tactic', 'Unknown'),
                        "doc_type": "attack_technique"
                    }
                )
                documents.append(doc)
            
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(split_docs)} MITRE document chunks from {len(documents)} techniques")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading MITRE data from {filepath}: {e}")
            return []
    
    def _format_mitre_content(self, technique: Dict) -> str:
        """Format MITRE data into readable content"""
        content_parts = [
            f"MITRE ATT&CK ID: {technique.get('id', 'N/A')}",
            f"Name: {technique.get('name', 'N/A')}",
            f"Tactic: {technique.get('tactic', 'N/A')}",
            f"Description: {technique.get('description', 'N/A')}"
        ]
        
        if technique.get('detection'):
            content_parts.append(f"Detection: {technique['detection']}")
        
        if technique.get('mitigation'):
            content_parts.append(f"Mitigation: {technique['mitigation']}")
        
        if technique.get('platforms'):
            platforms = ', '.join(technique['platforms'])
            content_parts.append(f"Platforms: {platforms}")
        
        return '\n'.join(content_parts)
    
    def load_exploit_data(self, filepath: str) -> List[Document]:
        """Load and process exploit payload data"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                exploit_data = json.load(f)
            
            if not self.validate_data_structure(exploit_data, "exploits"):
                logger.error(f"Invalid exploit data structure in {filepath}")
                return []
            
            documents = []
            for exploit in exploit_data:
                content = self._format_exploit_content(exploit)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "Exploits",
                        "source_file": filepath,
                        "exploit_name": exploit.get('name', 'Unknown'),
                        "exploit_type": exploit.get('type', 'Unknown'),
                        "risk_level": exploit.get('risk_level', 'Unknown'),
                        "doc_type": "exploit_technique"
                    }
                )
                documents.append(doc)
            
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(split_docs)} exploit document chunks from {len(documents)} exploits")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading exploit data from {filepath}: {e}")
            return []
    
    def _format_exploit_content(self, exploit: Dict) -> str:
        """Format exploit data into readable content"""
        content_parts = [
            f"Exploit Name: {exploit.get('name', 'N/A')}",
            f"Type: {exploit.get('type', 'N/A')}",
            f"Description: {exploit.get('description', 'N/A')}"
        ]
        
        if exploit.get('target'):
            content_parts.append(f"Target: {exploit['target']}")
        
        if exploit.get('risk_level'):
            content_parts.append(f"Risk Level: {exploit['risk_level']}")
        
        # Note: We include payload info but sanitize it for educational purposes
        if exploit.get('payload'):
            payload_info = exploit['payload'][:200] + "..." if len(exploit['payload']) > 200 else exploit['payload']
            content_parts.append(f"Payload Information: {payload_info}")
        
        if exploit.get('countermeasures'):
            content_parts.append(f"Countermeasures: {exploit['countermeasures']}")
        
        return '\n'.join(content_parts)
    
    def load_all_data(self) -> List[Document]:
        """Load all cybersecurity data sources"""
        all_documents = []
        data_sources = Config.get_data_sources()
        
        for source_name, filepath in data_sources.items():
            if not Path(filepath).exists():
                logger.warning(f"Data file not found: {filepath}")
                continue
            
            try:
                if source_name == "cve":
                    docs = self.load_cve_data(filepath)
                elif source_name == "owasp":
                    docs = self.load_owasp_data(filepath)
                elif source_name == "mitre":
                    docs = self.load_mitre_data(filepath)
                elif source_name == "exploits":
                    docs = self.load_exploit_data(filepath)
                else:
                    logger.warning(f"Unknown data source: {source_name}")
                    continue
                
                all_documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} documents from {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {source_name} data: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        stats = {}
        data_sources = Config.get_data_sources()
        
        for source_name, filepath in data_sources.items():
            if not Path(filepath).exists():
                stats[source_name] = {"status": "file_not_found", "count": 0}
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats[source_name] = {
                    "status": "loaded",
                    "count": len(data),
                    "file_size": Path(filepath).stat().st_size,
                    "schema_valid": self.validate_data_structure(data, source_name)
                }
            except Exception as e:
                stats[source_name] = {"status": f"error: {e}", "count": 0}
        
        return stats

def main():
    """Main function to load and test data loading"""
    Config.setup_logging()
    loader = CybersecurityDataLoader()
    
    # Get data statistics
    stats = loader.get_data_statistics()
    print("Data Source Statistics:")
    for source, info in stats.items():
        print(f"  {source}: {info}")
    
    # Load all documents
    documents = loader.load_all_data()
    print(f"\nTotal documents loaded: {len(documents)}")
    
    if documents:
        print("\nSample document:")
        print(f"Source: {documents[0].metadata.get('source')}")
        print(f"Content preview: {documents[0].page_content[:200]}...")

if __name__ == "__main__":
    main()
