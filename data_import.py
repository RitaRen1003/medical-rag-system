"""
Data Import Module for Medical RAG System
Handles importing PubMed articles into Neo4j graph database
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from .config import config
from .graph_manager import GraphManager

logger = logging.getLogger(__name__)


class PubMedImporter:
    """Import PubMed articles into Neo4j graph"""
    
    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager
        self.max_text_length = config.max_text_length
        self.min_text_length = config.min_text_length
        
    async def import_from_json(self, file_path: str, clear_db: bool = True):
        """
        Import PubMed articles from JSON file
        
        Args:
            file_path: Path to PubMed JSON corpus
            clear_db: Whether to clear database before import
        """
        logger.info(f"Starting PubMed import from {file_path}")
        
        # Validate file
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PubMed corpus not found: {file_path}")
            
        # Clear database if requested
        if clear_db:
            await self.graph_manager.clear_database()
            
        # Load data
        logger.info("Loading PubMed data...")
        with open(file_path, "r", encoding="utf-8") as f:
            pubmed_dict = json.load(f)
            
        logger.info(f"Found {len(pubmed_dict)} papers to process")
        
        # Process papers
        success_count = 0
        error_count = 0
        
        for paper_id, paper in tqdm(pubmed_dict.items(), desc="Importing papers"):
            try:
                await self._import_paper(paper_id, paper)
                success_count += 1
            except Exception as e:
                logger.error(f"Error importing paper {paper_id}: {e}")
                error_count += 1
                
        logger.info(f"Import complete: {success_count} successful, {error_count} errors")
        
    async def _import_paper(self, paper_id: str, paper: Dict):
        """Import a single paper"""
        
        # Build content from metadata
        base_content = self._build_base_content(paper)
        
        # Process full text
        full_text = paper.get('paper_full_text', '')
        processed_full_text = self._process_full_text(full_text)
        
        # Combine content
        episode_content = base_content + processed_full_text
        
        # Set reference time
        reference_time = self._get_reference_time(paper)
        
        # Add to graph
        await self.graph_manager.add_episode(
            name=paper.get('paper_title', paper_id),
            content=episode_content,
            source=f"{paper.get('paper_journal', 'Unknown Journal')}, {paper.get('paper_year', 'Unknown Year')}",
            reference_time=reference_time
        )
        
    def _build_base_content(self, paper: Dict) -> str:
        """Build base content from paper metadata"""
        return (
            f"Title: {paper.get('paper_title', '')}\n"
            f"Authors: {paper.get('paper_authors', '')}\n"
            f"Journal: {paper.get('paper_journal', '')}\n"
            f"Year: {paper.get('paper_year', '')}\n"
            f"Abstract: {paper.get('paper_abstract', '')}\n\n"
        )
        
    def _process_full_text(self, full_text: str) -> str:
        """Process full text according to length constraints"""
        if len(full_text) < self.min_text_length:
            return ""
        return full_text[:self.max_text_length]
        
    def _get_reference_time(self, paper: Dict) -> datetime:
        """Extract reference time from paper year"""
        try:
            year = int(paper.get("paper_year", ""))
            return datetime(year, 1, 1, tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)


async def import_pubmed_corpus(
    file_path: str = None,
    clear_db: bool = True
):
    """
    Main function to import PubMed corpus
    
    Args:
        file_path: Path to corpus file (uses config default if not provided)
        clear_db: Whether to clear database before import
    """
    file_path = file_path or config.pubmed_data_path
    
    async with GraphManager() as graph_manager:
        importer = PubMedImporter(graph_manager)
        await importer.import_from_json(file_path, clear_db)


if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run import
    asyncio.run(import_pubmed_corpus())