"""
Configuration management for Medical RAG System
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Central configuration for the Medical RAG System"""
    
    # Neo4j Configuration
    neo4j_uri: str = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user: str = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password: str = os.environ.get('NEO4J_PASSWORD', '')
    
    # API Keys
    openai_api_key: str = os.environ.get('OPENAI_API_KEY', '')
    umls_api_key: str = os.environ.get('UMLS_API_KEY', '')
    
    # Paths
    quickumls_path: str = os.environ.get('QUICKUMLS_PATH', '')
    pubmed_data_path: str = os.environ.get('PUBMED_DATA_PATH', 'data/pubmed/pubmed_corpus.json')
    
    # Processing Configuration
    max_text_length: int = 4096
    min_text_length: int = 100
    
    # Search Configuration
    default_edge_limit: int = 10
    default_node_limit: int = 5
    
    # LLM Configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1000
    
    # Logging
    log_path: str = "logs/medical_rag.log"
    stats_log_path: str = "logs/neo4j_stats.log"
    
    def validate(self) -> bool:
        """Validate required configuration"""
        errors = []
        
        if not self.neo4j_password:
            errors.append("NEO4J_PASSWORD is required")
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
            
        if not self.umls_api_key:
            errors.append("UMLS_API_KEY is required")
            
        if not os.path.exists(self.quickumls_path):
            errors.append(f"QuickUMLS path does not exist: {self.quickumls_path}")
            
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True


# Global config instance
config = Config()