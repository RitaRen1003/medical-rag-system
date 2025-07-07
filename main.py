"""
Main entry point for Medical RAG System
Provides example usage and command-line interface
"""
import asyncio
import logging
import argparse
from typing import Optional

from medical_rag.config import config
from medical_rag.rag_pipeline import MedicalRAGSystem
from medical_rag.data_import import import_pubmed_corpus
from medical_rag.graph_enrichment import enrich_graph_with_umls
from medical_rag.graph_stats import generate_statistics_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_query():
    """Demonstrate basic question answering"""
    print("\n" + "="*60)
    print("DEMO: Basic Medical Question Answering")
    print("="*60)
    
    async with MedicalRAGSystem() as rag_system:
        query = (
            "What are the potential advantages of the gelatinase-responsive "
            "self-assembled antimicrobial peptide (PEG-PR-26) compared to "
            "natural AMPs in combating methicillin-resistant Staphylococcus "
            "aureus (MRSA) infections?"
        )
        
        print(f"\nQuestion: {query}")
        print("\nGenerating answer...")
        
        # Get answer
        answer = await rag_system.answer_question(query)
        
        print(f"\nAnswer:\n{answer}")
        
        # Get full response with metadata
        print("\n" + "-"*40)
        print("Getting full response with metadata...")
        full_response = await rag_system.get_full_response(query)
        
        print(f"\nMetadata:")
        print(f"- Facts used: {full_response['metadata']['num_facts']}")
        print(f"- Nodes used: {full_response['metadata']['num_nodes']}")
        print(f"- UMLS terms: {full_response['metadata']['num_umls_terms']}")
        
        if full_response['context']['umls_terms']:
            print(f"\nSample UMLS terms identified:")
            for term in full_response['context']['umls_terms']:
                print(f"  - {term}")


async def demo_multiple_queries():
    """Demonstrate multiple query handling"""
    print("\n" + "="*60)
    print("DEMO: Multiple Medical Queries")
    print("="*60)
    
    queries = [
        "What are the mechanisms of antibiotic resistance in MRSA?",
        "How do antimicrobial peptides work against bacteria?",
        "What are the latest treatments for drug-resistant infections?"
    ]
    
    async with MedicalRAGSystem() as rag_system:
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*40}")
            print(f"Query {i}: {query}")
            print("-"*40)
            
            answer = await rag_system.answer_question(
                query,
                include_umls=True,
                max_facts=5
            )
            
            print(f"Answer: {answer[:300]}...")  # Show first 300 chars


async def demo_umls_enrichment():
    """Demonstrate UMLS term extraction and enrichment"""
    print("\n" + "="*60)
    print("DEMO: UMLS Medical Term Extraction")
    print("="*60)
    
    from medical_rag.umls_processor import UMLSProcessor
    
    umls_processor = UMLSProcessor()
    
    test_texts = [
        "Patient presents with acute myocardial infarction and hypertension.",
        "Treatment with ACE inhibitors showed significant improvement in cardiac function.",
        "MRSA infection resistant to methicillin and other beta-lactam antibiotics."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        print("Extracted terms:")
        
        terms = umls_processor.process_text(text, include_details=True)
        
        for term in terms:
            print(f"\n  - Term: {term['term']}")
            print(f"    CUI: {term['cui']}")
            if term.get('semantic_types'):
                print(f"    Types: {', '.join(term['semantic_types'][:2])}")
            if term.get('definitions'):
                print(f"    Definition: {term['definitions'][0][:100]}...")


async def interactive_mode():
    """Run interactive question-answering mode"""
    print("\n" + "="*60)
    print("Medical RAG Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    
    async with MedicalRAGSystem() as rag_system:
        while True:
            try:
                query = input("\nEnter your medical question: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("Exiting interactive mode...")
                    break
                    
                if not query:
                    print("Please enter a valid question.")
                    continue
                    
                print("\nProcessing your question...")
                answer = await rag_system.answer_question(query)
                
                print(f"\nAnswer:\n{answer}")
                
                # Optionally show metadata
                show_meta = input("\nShow metadata? (y/n): ").strip().lower()
                if show_meta == 'y':
                    full_response = await rag_system.get_full_response(query)
                    print(f"\nMetadata: {full_response['metadata']}")
                    
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"An error occurred: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Medical RAG System - Knowledge Graph-based Medical Q&A"
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'import', 'enrich', 'stats', 'interactive', 'query'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Medical question to answer (for query command)'
    )
    
    parser.add_argument(
        '--pubmed-path',
        type=str,
        help='Path to PubMed JSON corpus (for import command)'
    )
    
    parser.add_argument(
        '--no-umls',
        action='store_true',
        help='Disable UMLS enrichment in queries'
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    if not config.validate():
        print("Please fix configuration errors before running.")
        return
    
    # Execute command
    if args.command == 'demo':
        asyncio.run(demo_basic_query())
        asyncio.run(demo_multiple_queries())
        asyncio.run(demo_umls_enrichment())
        
    elif args.command == 'import':
        path = args.pubmed_path or config.pubmed_data_path
        asyncio.run(import_pubmed_corpus(path))
        
    elif args.command == 'enrich':
        asyncio.run(enrich_graph_with_umls())
        
    elif args.command == 'stats':
        asyncio.run(generate_statistics_report())
        
    elif args.command == 'interactive':
        asyncio.run(interactive_mode())
        
    elif args.command == 'query':
        if not args.query:
            print("Please provide a query with --query parameter")
            return
            
        async def run_query():
            async with MedicalRAGSystem() as rag_system:
                answer = await rag_system.answer_question(
                    args.query,
                    include_umls=not args.no_umls
                )
                print(f"\nQuestion: {args.query}")
                print(f"\nAnswer: {answer}")
                
        asyncio.run(run_query())


if __name__ == "__main__":
    main()