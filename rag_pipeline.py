"""
RAG Pipeline for Medical Question Answering
Combines graph search, UMLS enrichment, and LLM generation
"""
import logging
import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import config
from .graph_manager import GraphManager, Edge, Node
from .umls_processor import UMLSProcessor

logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = config.openai_api_key


@dataclass
class RAGContext:
    """Container for RAG context components"""
    query: str
    graph_facts: List[str]
    graph_nodes: List[str]
    umls_terms: List[Dict[str, Any]]
    formatted_context: str


class MedicalRAGPipeline:
    """Main RAG pipeline for medical question answering"""
    
    def __init__(self, graph_manager: GraphManager, umls_processor: UMLSProcessor):
        self.graph_manager = graph_manager
        self.umls_processor = umls_processor
        
    async def answer_question(
        self, 
        query: str,
        include_umls: bool = True,
        max_facts: int = 10,
        max_nodes: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a medical question using graph + UMLS + LLM
        
        Args:
            query: The medical question
            include_umls: Whether to include UMLS term enrichment
            max_facts: Maximum number of facts to retrieve
            max_nodes: Maximum number of nodes to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing question: {query[:100]}...")
        
        # 1. Build RAG context
        context = await self._build_rag_context(
            query, include_umls, max_facts, max_nodes
        )
        
        # 2. Generate answer using LLM
        answer = await self._generate_answer(context)
        
        # 3. Prepare response
        response = {
            'answer': answer,
            'query': query,
            'metadata': {
                'num_facts': len(context.graph_facts),
                'num_nodes': len(context.graph_nodes),
                'num_umls_terms': len(context.umls_terms),
                'model': config.llm_model
            },
            'context': {
                'facts': context.graph_facts[:3],  # Sample for response
                'umls_terms': [t['term'] for t in context.umls_terms[:3]]
            }
        }
        
        logger.info("Successfully generated answer")
        return response
        
    async def _build_rag_context(
        self,
        query: str,
        include_umls: bool,
        max_facts: int,
        max_nodes: int
    ) -> RAGContext:
        """Build comprehensive RAG context from multiple sources"""
        
        # 1. Search graph for relevant facts (edges)
        edges = await self.graph_manager.search_edges(query, max_facts)
        graph_facts = self._format_edges_as_facts(edges)
        
        # 2. Search graph for relevant entities (nodes)
        nodes = await self.graph_manager.search_nodes(query, max_nodes)
        graph_nodes = self._format_nodes_as_context(nodes)
        
        # 3. Extract and enrich UMLS terms if requested
        umls_terms = []
        umls_context = ""
        if include_umls:
            umls_terms = self.umls_processor.process_text(query, include_details=True)
            umls_context = self.umls_processor.format_terms_for_context(umls_terms)
            
        # 4. Format complete context
        formatted_context = self._format_complete_context(
            graph_facts, graph_nodes, umls_context
        )
        
        return RAGContext(
            query=query,
            graph_facts=graph_facts,
            graph_nodes=graph_nodes,
            umls_terms=umls_terms,
            formatted_context=formatted_context
        )
        
    def _format_edges_as_facts(self, edges: List[Edge]) -> List[str]:
        """Format edges as fact statements"""
        facts = []
        
        for edge in edges:
            source = edge.source_node_name or f"Entity_{edge.source_node_uuid[:8]}"
            target = edge.target_node_name or f"Entity_{edge.target_node_uuid[:8]}"
            fact_str = f"{edge.fact} (Source: {source}; Target: {target})"
            facts.append(fact_str)
            
        return facts
        
    def _format_nodes_as_context(self, nodes: List[Node]) -> List[str]:
        """Format nodes as context statements"""
        node_contexts = []
        
        for node in nodes:
            summary = node.summary
            if len(summary) > 200:
                summary = summary[:200] + "..."
                
            context_str = f"{node.name}: {summary}"
            node_contexts.append(context_str)
            
        return node_contexts
        
    def _format_complete_context(
        self,
        facts: List[str],
        nodes: List[str],
        umls_context: str
    ) -> str:
        """Format all context components into a single string"""
        context_parts = []
        
        # Add facts
        if facts:
            context_parts.append("Relevant Facts from Knowledge Graph:")
            for i, fact in enumerate(facts, 1):
                context_parts.append(f"{i}. {fact}")
            context_parts.append("")
            
        # Add node summaries
        if nodes:
            context_parts.append("Relevant Entity Summaries:")
            for i, node in enumerate(nodes, 1):
                context_parts.append(f"{i}. {node}")
            context_parts.append("")
            
        # Add UMLS context
        if umls_context:
            context_parts.append(umls_context)
            context_parts.append("")
            
        return "\n".join(context_parts)
        
    async def _generate_answer(self, context: RAGContext) -> str:
        """Generate answer using LLM with the provided context"""
        
        # Build prompt
        prompt = self._build_prompt(context)
        
        logger.info("Generating answer with LLM...")
        
        try:
            response = openai.ChatCompletion.create(
                model=config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful biomedical expert assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
            
    def _build_prompt(self, context: RAGContext) -> str:
        """Build the prompt for LLM"""
        prompt = f"""You are answering a medical question based on a knowledge graph of PubMed literature and UMLS medical terminology.

Use the provided facts, entity summaries, and medical term definitions as the primary evidence for your answer. 
Cite specific facts when possible and ensure medical accuracy.

User Question:
{context.query}

{context.formatted_context}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Uses the provided facts as supporting evidence
3. Incorporates relevant medical terminology accurately
4. Is clear and accessible to medical professionals

Answer:"""
        
        return prompt
        
    async def enrich_node_with_umls(self, node_uuid: str) -> bool:
        """
        Enrich a specific node with UMLS concepts
        
        Args:
            node_uuid: UUID of the node to enrich
            
        Returns:
            Success status
        """
        logger.info(f"Enriching node {node_uuid} with UMLS concepts...")
        
        try:
            # Get node content
            node = await self.graph_manager.get_node_by_uuid(node_uuid)
            if not node:
                logger.warning(f"Node {node_uuid} not found")
                return False
                
            # Extract text content
            text = node.get('summary', '') or node.get('content', '')
            if not text:
                logger.warning(f"No text content in node {node_uuid}")
                return False
                
            # Extract UMLS terms
            terms = self.umls_processor.process_text(text, include_details=True)
            
            # Add UMLS concepts to graph and link to node
            for term in terms:
                cui = term['cui']
                
                # Add UMLS concept if not exists
                await self.graph_manager.add_umls_concept(cui, term)
                
                # Link node to UMLS concept
                await self.graph_manager.link_node_to_umls(node_uuid, cui)
                
            logger.info(f"Successfully enriched node with {len(terms)} UMLS concepts")
            return True
            
        except Exception as e:
            logger.error(f"Error enriching node with UMLS: {e}")
            return False


class MedicalRAGSystem:
    """High-level interface for the Medical RAG System"""
    
    def __init__(self):
        self.graph_manager = None
        self.umls_processor = None
        self.pipeline = None
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def initialize(self):
        """Initialize all components"""
        self.graph_manager = GraphManager()
        await self.graph_manager.initialize()
        
        self.umls_processor = UMLSProcessor()
        
        self.pipeline = MedicalRAGPipeline(
            self.graph_manager,
            self.umls_processor
        )
        
        logger.info("Medical RAG System initialized")
        
    async def close(self):
        """Close all connections"""
        if self.graph_manager:
            await self.graph_manager.close()
            
    async def answer_question(self, query: str, **kwargs) -> str:
        """
        Answer a medical question
        
        Args:
            query: The medical question
            **kwargs: Additional parameters for the pipeline
            
        Returns:
            The generated answer
        """
        result = await self.pipeline.answer_question(query, **kwargs)
        return result['answer']
        
    async def get_full_response(self, query: str, **kwargs) -> Dict[str, Any]:
        """Get full response with metadata"""
        return await self.pipeline.answer_question(query, **kwargs)