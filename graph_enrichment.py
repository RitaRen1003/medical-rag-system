"""
Graph Enrichment Module
Enriches Neo4j graph with UMLS medical concepts
"""
import logging
from tqdm import tqdm
from typing import List, Dict, Any

from .config import config
from .graph_manager import GraphManager
from .umls_processor import UMLSProcessor

logger = logging.getLogger(__name__)


class GraphEnricher:
    """Enrich graph nodes with UMLS medical concepts"""
    
    def __init__(self, graph_manager: GraphManager, umls_processor: UMLSProcessor):
        self.graph_manager = graph_manager
        self.umls_processor = umls_processor
        
    async def enrich_all_nodes(self, node_limit: int = None):
        """
        Enrich all nodes in the graph with UMLS concepts
        
        Args:
            node_limit: Maximum number of nodes to process (None for all)
        """
        logger.info("Starting graph enrichment with UMLS concepts...")
        
        # Get all nodes
        nodes = await self._get_all_nodes(limit=node_limit)
        logger.info(f"Found {len(nodes)} nodes to enrich")
        
        success_count = 0
        error_count = 0
        
        for node in tqdm(nodes, desc="Enriching nodes"):
            try:
                enriched = await self._enrich_node(node)
                if enriched:
                    success_count += 1
            except Exception as e:
                logger.error(f"Error enriching node {node.get('uuid', '')}: {e}")
                error_count += 1
                
        logger.info(f"Enrichment complete: {success_count} successful, {error_count} errors")
        
    async def enrich_nodes_by_label(self, label: str, limit: int = None):
        """
        Enrich nodes with a specific label
        
        Args:
            label: Node label to filter by
            limit: Maximum number of nodes to process
        """
        logger.info(f"Enriching nodes with label '{label}'...")
        
        nodes = await self._get_nodes_by_label(label, limit)
        logger.info(f"Found {len(nodes)} nodes with label '{label}'")
        
        for node in tqdm(nodes, desc=f"Enriching {label} nodes"):
            await self._enrich_node(node)
            
    async def _get_all_nodes(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get all nodes from the graph"""
        query = "MATCH (n) RETURN n"
        if limit:
            query += f" LIMIT {limit}"
            
        result = await self.graph_manager.graphiti.driver.execute_query(query)
        return [record['n'] for record in result[0]] if result else []
        
    async def _get_nodes_by_label(self, label: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get nodes by label"""
        query = f"MATCH (n:{label}) RETURN n"
        if limit:
            query += f" LIMIT {limit}"
            
        result = await self.graph_manager.graphiti.driver.execute_query(query)
        return [record['n'] for record in result[0]] if result else []
        
    async def _enrich_node(self, node: Dict[str, Any]) -> bool:
        """Enrich a single node with UMLS concepts"""
        node_uuid = node.get('uuid', '')
        if not node_uuid:
            return False
            
        # Extract text content
        text = self._extract_node_text(node)
        if not text:
            return False
            
        # Extract UMLS terms
        terms = self.umls_processor.process_text(text, include_details=True)
        if not terms:
            return False
            
        # Add UMLS concepts and create relationships
        for term in terms:
            try:
                # Add UMLS concept node
                umls_uuid = await self.graph_manager.add_umls_concept(
                    term['cui'],
                    term
                )
                
                # Create relationship
                await self._create_umls_relationship(
                    node_uuid,
                    term['cui'],
                    term['similarity']
                )
                
            except Exception as e:
                logger.error(f"Error adding UMLS concept {term['cui']}: {e}")
                
        return True
        
    def _extract_node_text(self, node: Dict[str, Any]) -> str:
        """Extract text content from a node"""
        # Try different possible text fields
        text_fields = ['summary', 'content', 'episode_body', 'description', 'text']
        
        for field in text_fields:
            if field in node and node[field]:
                return str(node[field])
                
        # Fallback: concatenate name and any string values
        texts = []
        if 'name' in node:
            texts.append(str(node['name']))
            
        for key, value in node.items():
            if isinstance(value, str) and len(value) > 50:
                texts.append(value)
                
        return ' '.join(texts)
        
    async def _create_umls_relationship(
        self,
        node_uuid: str,
        cui: str,
        similarity: float
    ):
        """Create relationship between node and UMLS concept"""
        query = """
        MATCH (n {uuid: $node_uuid})
        MATCH (u {name: $umls_name})
        MERGE (n)-[r:HAS_UMLS_CONCEPT {similarity: $similarity}]->(u)
        RETURN r
        """
        
        await self.graph_manager.graphiti.driver.execute_query(
            query,
            node_uuid=node_uuid,
            umls_name=f"UMLS_{cui}",
            similarity=similarity
        )
        
    async def add_umls_hierarchy(self, cui: str):
        """
        Add UMLS concept hierarchy (parent/child relationships)
        
        Args:
            cui: Concept Unique Identifier
        """
        logger.info(f"Adding UMLS hierarchy for {cui}...")
        
        # Get relations from UMLS
        relations = self.umls_processor.get_cui_relations(cui)
        
        for relation in relations:
            if relation['relationLabel'] in ['RB', 'RN']:  # Broader/Narrower
                related_cui = relation['relatedId']
                
                # Add related concept
                related_details = self.umls_processor.get_cui_details(related_cui)
                if related_details:
                    await self.graph_manager.add_umls_concept(
                        related_cui,
                        related_details
                    )
                    
                    # Create hierarchical relationship
                    rel_type = 'BROADER_THAN' if relation['relationLabel'] == 'RB' else 'NARROWER_THAN'
                    
                    query = f"""
                    MATCH (c1 {{name: $cui1}})
                    MATCH (c2 {{name: $cui2}})
                    MERGE (c1)-[:{rel_type}]->(c2)
                    """
                    
                    await self.graph_manager.graphiti.driver.execute_query(
                        query,
                        cui1=f"UMLS_{cui}",
                        cui2=f"UMLS_{related_cui}"
                    )


async def enrich_graph_with_umls(
    node_limit: int = None,
    label_filter: str = None
):
    """
    Main function to enrich graph with UMLS concepts
    
    Args:
        node_limit: Maximum number of nodes to process
        label_filter: Only process nodes with this label
    """
    async with GraphManager() as graph_manager:
        umls_processor = UMLSProcessor()
        enricher = GraphEnricher(graph_manager, umls_processor)
        
        if label_filter:
            await enricher.enrich_nodes_by_label(label_filter, node_limit)
        else:
            await enricher.enrich_all_nodes(node_limit)


if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run enrichment
    asyncio.run(enrich_graph_with_umls())