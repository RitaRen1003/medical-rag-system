# Medical Knowledge Graph RAG System

A medical knowledge retrieval and generation system that combines Neo4j graph database, UMLS medical terminology, and LLM for intelligent medical question answering.

## Features

- **Graph-based Knowledge Storage**: Store medical literature in Neo4j graph database
- **UMLS Integration**: Extract and enrich medical terms using QuickUMLS and UMLS API
- **RAG Pipeline**: Retrieve relevant facts from the graph and generate answers using LLM
- **Modular Architecture**: Clean, professional code structure with separate modules

## Prerequisites

- Python 3.8+
- Neo4j Database (local or remote)
- UMLS API Key (get from [UMLS Terminology Services](https://uts.nlm.nih.gov/uts/))
- OpenAI API Key (for LLM generation)
- QuickUMLS installation (follow [QuickUMLS setup guide](https://github.com/Georgetown-IR-Lab/QuickUMLS))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RitaRen1003/medical-rag-system.git
cd medical-rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your credentials
```

## Configuration

Update the `.env` file with your credentials:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Keys
OPENAI_API_KEY=your_openai_api_key
UMLS_API_KEY=your_umls_api_key

# Paths
QUICKUMLS_PATH=/path/to/your/QuickUMLS
PUBMED_DATA_PATH=/path/to/pubmed_corpus.json
```

## Usage

### 1. Import Medical Literature to Neo4j

```bash
python -m medical_rag.data_import
```

This will:
- Clear the existing database
- Import PubMed articles from the JSON corpus
- Build necessary indices and constraints

### 2. Query the System

```python
from medical_rag import MedicalRAGSystem

# Initialize the system
rag_system = MedicalRAGSystem()

# Ask a medical question
question = "What are the advantages of PEG-PR-26 compared to natural AMPs?"
answer = await rag_system.answer_question(question)

print(answer)
```

### 3. Enrich Graph with UMLS Terms

```python
from medical_rag import enrich_graph_with_umls

# Enrich existing nodes with UMLS medical concepts
await enrich_graph_with_umls()
```

### 4. Get Database Statistics

```bash
python -m medical_rag.graph_stats
```

#### Example Usage
```bash
# Import PubMed data
python -m medical_rag.data_import

# Enrich graph with UMLS concepts
python -m medical_rag.graph_enrichment

# Interactive Q&A mode
python main.py interactive

# Single query
python main.py query --query "What are the mechanisms of MRSA resistance?"

# Generate database statistics
python main.py stats

# Run demos
python main.py demo
```

## Project Structure

```
medical_rag/
├── __init__.py
├── config.py           # Configuration management
├── data_import.py      # Import PubMed data to Neo4j
├── graph_manager.py    # Neo4j database operations
├── umls_processor.py   # UMLS term extraction and processing
├── rag_pipeline.py     # RAG question-answering pipeline
├── graph_stats.py      # Database statistics
└── main.py            # Main entry point with examples

requirements.txt
.env.example
README.md
```

## Module Descriptions

- **config.py**: Centralized configuration management
- **graph_manager.py**: Handles all Neo4j operations (CRUD, search, enrichment)
- **umls_processor.py**: UMLS API integration and medical term processing
- **rag_pipeline.py**: Combines graph search, UMLS enrichment, and LLM generation
- **data_import.py**: Batch import of medical literature
- **graph_stats.py**: Generate comprehensive database statistics

## Example Queries

```python
# Simple medical question
"What are the side effects of aspirin?"

# Complex research question
"How do gelatinase-responsive peptides compare to traditional antibiotics for MRSA?"

# Mechanism inquiry
"What is the mechanism of action of ACE inhibitors?"
```

## API Reference

### MedicalRAGSystem

```python
class MedicalRAGSystem:
    async def answer_question(
        self, 
        query: str, 
        include_umls: bool = True,
        max_facts: int = 10
    ) -> str:
        """
        Answer a medical question using graph + UMLS + LLM
        
        Args:
            query: The medical question
            include_umls: Whether to include UMLS term enrichment
            max_facts: Maximum number of facts to retrieve
            
        Returns:
            Generated answer with citations
        """
```

### GraphManager

```python
class GraphManager:
    async def search_edges(self, query: str, limit: int = 10) -> List[Edge]
    async def search_nodes(self, query: str, limit: int = 5) -> List[Node]
    async def add_umls_concept(self, cui: str, details: dict) -> None
    async def link_node_to_umls(self, node_id: str, cui: str) -> None
```

