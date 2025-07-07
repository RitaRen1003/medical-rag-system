"""
Graph Statistics Module
Generates comprehensive statistics about the Neo4j database
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

from .config import config
from .graph_manager import GraphManager

logger = logging.getLogger(__name__)

# Configure stats logger
stats_logger = logging.getLogger("neo4j_stats")
stats_handler = logging.FileHandler(config.stats_log_path)
stats_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
stats_logger.addHandler(stats_handler)
stats_logger.setLevel(logging.INFO)


class GraphStatistics:
    """Generate and log Neo4j database statistics"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.driver = graph_manager.graphiti.driver
        
    async def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive database statistics"""
        logger.info("Generating database statistics...")
        
        stats = {}
        
        # Basic counts
        stats.update(await self._get_basic_counts())
        
        # Label and relationship statistics
        stats.update(await self._get_label_stats())
        stats.update(await self._get_relationship_stats())
        
        # Property statistics
        stats.update(await self._get_property_stats())
        
        # Graph structure statistics
        stats.update(await self._get_structure_stats())
        
        # UMLS enrichment statistics
        stats.update(await self._get_umls_stats())
        
        # Performance statistics
        stats.update(await self._get_performance_stats())
        
        # Log all statistics
        self._log_statistics(stats)
        
        return stats
        
    async def _get_basic_counts(self) -> Dict[str, Any]:
        """Get basic node and relationship counts"""
        stats = {}
        
        # Total nodes
        result = await self.driver.execute_query(
            "MATCH (n) RETURN count(n) AS count"
        )
        stats['total_nodes'] = result[0][0]['count'] if result else 0
        
        # Total relationships
        result = await self.driver.execute_query(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        )
        stats['total_relationships'] = result[0][0]['count'] if result else 0
        
        # Unique labels
        result = await self.driver.execute_query(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        stats['unique_labels'] = result[0][0]['labels'] if result else []
        stats['num_unique_labels'] = len(stats['unique_labels'])
        
        # Unique relationship types
        result = await self.driver.execute_query(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS types"
        )
        stats['unique_relationship_types'] = result[0][0]['types'] if result else []
        stats['num_unique_relationship_types'] = len(stats['unique_relationship_types'])
        
        return stats
        
    async def _get_label_stats(self) -> Dict[str, Any]:
        """Get statistics about node labels"""
        stats = {}
        
        # Node count by label
        result = await self.driver.execute_query(
            """
            CALL db.labels() YIELD label
            RETURN label, size([(n) WHERE label IN labels(n) | n]) AS count
            ORDER BY count DESC
            """
        )
        
        label_counts = [(row['label'], row['count']) for row in result[0]] if result else []
        stats['label_distribution'] = dict(label_counts)
        stats['top_3_labels'] = label_counts[:3]
        
        return stats
        
    async def _get_relationship_stats(self) -> Dict[str, Any]:
        """Get statistics about relationships"""
        stats = {}
        
        # Relationship count by type
        result = await self.driver.execute_query(
            """
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN relationshipType, 
                   size(()-[r]->() WHERE type(r) = relationshipType) AS count
            ORDER BY count DESC
            """
        )
        
        rel_counts = [(row['relationshipType'], row['count']) for row in result[0]] if result else []
        stats['relationship_type_distribution'] = dict(rel_counts)
        stats['top_3_relationship_types'] = rel_counts[:3]
        
        return stats
        
    async def _get_property_stats(self) -> Dict[str, Any]:
        """Get statistics about node and relationship properties"""
        stats = {}
        
        # Average properties per node
        result = await self.driver.execute_query(
            "MATCH (n) RETURN avg(size(keys(n))) AS avg_props"
        )
        stats['avg_properties_per_node'] = result[0][0]['avg_props'] if result else 0
        
        # Average properties per relationship
        result = await self.driver.execute_query(
            "MATCH ()-[r]->() RETURN avg(size(keys(r))) AS avg_props"
        )
        stats['avg_properties_per_relationship'] = result[0][0]['avg_props'] if result else 0
        
        return stats
        
    async def _get_structure_stats(self) -> Dict[str, Any]:
        """Get graph structure statistics"""
        stats = {}
        
        # Average node degree
        result = await self.driver.execute_query(
            "MATCH (n) RETURN avg(size((n)--()))AS avg_degree"
        )
        stats['avg_node_degree'] = result[0][0]['avg_degree'] if result else 0
        
        # Isolated nodes
        result = await self.driver.execute_query(
            "MATCH (n) WHERE size((n)--()) = 0 RETURN count(n) AS count"
        )
        stats['isolated_nodes'] = result[0][0]['count'] if result else 0
        
        # Most connected nodes
        result = await self.driver.execute_query(
            """
            MATCH (n)
            WITH n, size((n)--()) AS degree
            ORDER BY degree DESC
            LIMIT 5
            RETURN n.name AS name, n.uuid AS uuid, degree
            """
        )
        stats['most_connected_nodes'] = [
            {'name': row['name'], 'uuid': row['uuid'], 'degree': row['degree']}
            for row in result[0]
        ] if result else []
        
        return stats
        
    async def _get_umls_stats(self) -> Dict[str, Any]:
        """Get UMLS enrichment statistics"""
        stats = {}
        
        # Count UMLS concept nodes
        result = await self.driver.execute_query(
            "MATCH (n) WHERE n.name STARTS WITH 'UMLS_' RETURN count(n) AS count"
        )
        stats['umls_concept_nodes'] = result[0][0]['count'] if result else 0
        
        # Count nodes with UMLS relationships
        result = await self.driver.execute_query(
            """
            MATCH (n)-[:HAS_UMLS_CONCEPT]->()
            RETURN count(DISTINCT n) AS count
            """
        )
        stats['nodes_with_umls'] = result[0][0]['count'] if result else 0
        
        # UMLS coverage percentage
        if stats.get('total_nodes', 0) > 0:
            non_umls_nodes = stats['total_nodes'] - stats['umls_concept_nodes']
            if non_umls_nodes > 0:
                stats['umls_coverage_percentage'] = (
                    stats['nodes_with_umls'] / non_umls_nodes * 100
                )
            else:
                stats['umls_coverage_percentage'] = 0
        else:
            stats['umls_coverage_percentage'] = 0
            
        # Top semantic types
        result = await self.driver.execute_query(
            """
            MATCH (n) WHERE n.name STARTS WITH 'UMLS_'
            UNWIND n.semantic_types AS semantic_type
            RETURN semantic_type, count(*) AS count
            ORDER BY count DESC
            LIMIT 10
            """
        )
        stats['top_semantic_types'] = [
            {'type': row['semantic_type'], 'count': row['count']}
            for row in result[0]
        ] if result else []
        
        return stats
        
    async def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance-related statistics"""
        stats = {}
        
        # Test query performance
        test_queries = [
            ("Simple node match", "MATCH (n) RETURN n LIMIT 10"),
            ("Pattern match", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10"),
            ("Text search", "MATCH (n) WHERE n.name CONTAINS 'disease' RETURN n LIMIT 10")
        ]
        
        query_times = []
        for query_name, query in test_queries:
            start_time = time.time()
            await self.driver.execute_query(query)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            query_times.append({
                'query': query_name,
                'time_ms': round(elapsed_time, 2)
            })
            
        stats['query_performance'] = query_times
        
        # Index information
        result = await self.driver.execute_query("CALL db.indexes()")
        stats['num_indexes'] = len(result[0]) if result else 0
        
        # Constraint information
        result = await self.driver.execute_query("CALL db.constraints()")
        stats['num_constraints'] = len(result[0]) if result else 0
        
        return stats
        
    def _log_statistics(self, stats: Dict[str, Any]):
        """Log statistics to file"""
        stats_logger.info("="*60)
        stats_logger.info("Neo4j Database Statistics Report")
        stats_logger.info(f"Generated at: {datetime.now()}")
        stats_logger.info("="*60)
        
        # Basic counts
        stats_logger.info(f"Total nodes: {stats.get('total_nodes', 0)}")
        stats_logger.info(f"Total relationships: {stats.get('total_relationships', 0)}")
        stats_logger.info(f"Unique node labels: {stats.get('num_unique_labels', 0)}")
        stats_logger.info(f"Unique relationship types: {stats.get('num_unique_relationship_types', 0)}")
        
        # Top labels and relationships
        stats_logger.info("\nTop 3 node labels:")
        for label, count in stats.get('top_3_labels', []):
            stats_logger.info(f"  - {label}: {count}")
            
        stats_logger.info("\nTop 3 relationship types:")
        for rel_type, count in stats.get('top_3_relationship_types', []):
            stats_logger.info(f"  - {rel_type}: {count}")
            
        # Property statistics
        stats_logger.info(f"\nAverage properties per node: {stats.get('avg_properties_per_node', 0):.2f}")
        stats_logger.info(f"Average properties per relationship: {stats.get('avg_properties_per_relationship', 0):.2f}")
        
        # Structure statistics
        stats_logger.info(f"\nAverage node degree: {stats.get('avg_node_degree', 0):.2f}")
        stats_logger.info(f"Isolated nodes: {stats.get('isolated_nodes', 0)}")
        
        # UMLS statistics
        stats_logger.info(f"\nUMLS concept nodes: {stats.get('umls_concept_nodes', 0)}")
        stats_logger.info(f"Nodes with UMLS enrichment: {stats.get('nodes_with_umls', 0)}")
        stats_logger.info(f"UMLS coverage: {stats.get('umls_coverage_percentage', 0):.1f}%")
        
        # Performance
        stats_logger.info(f"\nNumber of indexes: {stats.get('num_indexes', 0)}")
        stats_logger.info(f"Number of constraints: {stats.get('num_constraints', 0)}")
        
        stats_logger.info("\nQuery performance:")
        for perf in stats.get('query_performance', []):
            stats_logger.info(f"  - {perf['query']}: {perf['time_ms']} ms")
            
        stats_logger.info("="*60)
        
        print(f"\nStatistics report written to: {config.stats_log_path}")


async def generate_statistics_report():
    """Main function to generate statistics report"""
    async with GraphManager() as graph_manager:
        stats_generator = GraphStatistics(graph_manager)
        await stats_generator.generate_full_report()


if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate report
    asyncio.run(generate_statistics_report())