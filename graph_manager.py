"""
Neo4j Graph Manager for Medical RAG System
Handles all graph database operations including search, import, and enrichment
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from dataclasses import dataclass

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """Edge representation from graph search"""
    uuid: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None


@dataclass
class Node:
    """Node representation from graph search"""
    uuid: str
    name: str
    summary: str
    labels: List[str]
    created_at: datetime
    attributes: Dict[str, Any]


class GraphManager:
    """Manages all Neo4j graph operations"""
    
    def __init__(self):
        self.graphiti = Graphiti(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password
        )
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def initialize(self):
        """Initialize graph connection and indices"""
        logger.info("Initializing graph connection and indices...")
        await self.graphiti.build_indices_and_constraints()
        
    async def close(self):
        """Close graph connection"""
        await self.graphiti.close()
        logger.info("Graph connection closed")
        
    async def clear_database(self):
        """Clear all nodes and relationships from the database"""
        logger.info("Clearing Neo4j database...")
        try:
            await self.graphiti.driver.execute_query("MATCH (n) DETACH DELETE n")
            logger.info("Successfully cleared Neo4j database")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
            
    async def search_edges(self, query: str, limit: int = None) -> List[Edge]:
        """
        Search for edges (facts) related to the query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of Edge objects
        """
        limit = limit or config.default_edge_limit
        logger.info(f"Searching edges for query: {query[:50]}...")
        
        try:
            raw_edges = await self.graphiti.search(query)
            edges = []
            
            for raw_edge in raw_edges[:limit]:
                edge = Edge(
                    uuid=getattr(raw_edge, 'uuid', ''),
                    fact=getattr(raw_edge, 'fact', ''),
                    source_node_uuid=getattr(raw_edge, 'source_node_uuid', ''),
                    target_node_uuid=getattr(raw_edge, 'target_node_uuid', ''),
                    source_node_name=getattr(raw_edge, 'source_node_name', None),
                    target_node_name=getattr(raw_edge, 'target_node_name', None),
                    valid_at=getattr(raw_edge, 'valid_at', None),
                    invalid_at=getattr(raw_edge, 'invalid_at', None)
                )
                edges.append(edge)
                
            logger.info(f"Found {len(edges)} edges")
            return edges
            
        except Exception as e:
            logger.error(f"Error searching edges: {e}")
            return []
            
    async def search_nodes(self, query: str, limit: int = None) -> List[Node]:
        """
        Search for nodes (entities) related to the query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of Node objects
        """
        limit = limit or config.default_node_limit
        logger.info(f"Searching nodes for query: {query[:50]}...")
        
        try:
            # Configure node search
            node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            node_search_config.limit = limit
            
            search_result = await self.graphiti._search(
                query=query,
                config=node_search_config
            )
            
            nodes = []
            for raw_node in getattr(search_result, "nodes", []):
                node = Node(
                    uuid=getattr(raw_node, 'uuid', ''),
                    name=getattr(raw_node, 'name', '') or getattr(raw_node, 'title', ''),
                    summary=getattr(raw_node, 'summary', ''),
                    labels=getattr(raw_node, 'labels', []),
                    created_at=getattr(raw_node, 'created_at', datetime.now()),
                    attributes=getattr(raw_node, 'attributes', {})
                )
                nodes.append(node)
                
            logger.info(f"Found {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Error searching nodes: {e}")
            return []
            
    async def add_episode(self, name: str, content: str, source: str, 
                         reference_time: datetime) -> str:
        """
        Add a new episode (document) to the graph
        
        Args:
            name: Episode name (e.g., paper title)
            content: Episode content
            source: Source description
            reference_time: Reference timestamp
            
        Returns:
            Episode UUID
        """
        episode = await self.graphiti.add_episode(
            name=name,
            episode_body=content,
            source_description=source,
            reference_time=reference_time,
        )
        return episode.uuid
        
    async def add_umls_concept(self, cui: str, details: Dict[str, Any]) -> str:
        """
        Add a UMLS concept as a node in the graph
        
        Args:
            cui: Concept Unique Identifier
            details: Dictionary containing concept details
            
        Returns:
            Node UUID
        """
        logger.info(f"Adding UMLS concept {cui} to graph...")
        
        try:
            # Create concept description
            semantic_types = ", ".join(details.get('semantic_types', []))
            definitions = "\n".join(details.get('definitions', []))
            
            content = f"""UMLS Concept: {details.get('name', cui)}
CUI: {cui}
Semantic Types: {semantic_types}
Definitions: {definitions}"""
            
            # Add as episode/node
            episode = await self.graphiti.add_episode(
                name=f"UMLS_{cui}",
                episode_body=content,
                source_description="UMLS Metathesaurus",
                reference_time=datetime.now(timezone.utc),
            )
            
            logger.info(f"Successfully added UMLS concept {cui}")
            return episode.uuid
            
        except Exception as e:
            logger.error(f"Error adding UMLS concept: {e}")
            raise
            
    async def link_node_to_umls(self, node_uuid: str, cui: str, 
                               relationship_type: str = "HAS_UMLS_CONCEPT") -> bool:
        """
        Create a relationship between a node and a UMLS concept
        
        Args:
            node_uuid: UUID of the source node
            cui: UMLS Concept Unique Identifier
            relationship_type: Type of relationship
            
        Returns:
            Success status
        """
        logger.info(f"Linking node {node_uuid} to UMLS concept {cui}...")
        
        try:
            query = """
            MATCH (n {uuid: $node_uuid})
            MATCH (u {name: $umls_name})
            MERGE (n)-[r:%s]->(u)
            RETURN r
            """ % relationship_type
            
            await self.graphiti.driver.execute_query(
                query,
                node_uuid=node_uuid,
                umls_name=f"UMLS_{cui}"
            )
            
            logger.info(f"Successfully linked node to UMLS concept")
            return True
            
        except Exception as e:
            logger.error(f"Error linking node to UMLS: {e}")
            return False
            
    async def get_node_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get a node by its UUID"""
        try:
            query = """
            MATCH (n {uuid: $uuid})
            RETURN n
            """
            result = await self.graphiti.driver.execute_query(query, uuid=uuid)
            if result and result[0]:
                return result[0][0]
            return None
        except Exception as e:
            logger.error(f"Error getting node: {e}")
            return None
            
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {}
        
        try:
            # Node and relationship counts
            result = await self.graphiti.driver.execute_query(
                "MATCH (n) RETURN count(n) as node_count"
            )
            stats['total_nodes'] = result[0][0]['node_count'] if result else 0
            
            result = await self.graphiti.driver.execute_query(
                "MATCH ()-[r]->() RETURN count(r) as rel_count"
            )
            stats['total_relationships'] = result[0][0]['rel_count'] if result else 0
            
            # Get label distribution
            result = await self.graphiti.driver.execute_query(
                """
                CALL db.labels() YIELD label
                RETURN label, size([(n) WHERE label IN labels(n) | n]) as count
                ORDER BY count DESC
                """
            )
            stats['label_distribution'] = {
                row['label']: row['count'] for row in result[0]
            } if result else {}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return stats