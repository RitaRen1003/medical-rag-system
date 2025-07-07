"""
UMLS Medical Term Processor
Handles extraction and enrichment of medical terms using QuickUMLS and UMLS API
"""
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from quickumls import QuickUMLS

from .config import config

logger = logging.getLogger(__name__)


class UMLSProcessor:
    """Process medical terms using UMLS"""
    
    def __init__(self):
        self.api_key = config.umls_api_key
        self.matcher = None
        self._initialize_quickumls()
        
    def _initialize_quickumls(self):
        """Initialize QuickUMLS matcher"""
        try:
            if config.quickumls_path:
                self.matcher = QuickUMLS(config.quickumls_path)
                logger.info("QuickUMLS initialized successfully")
            else:
                logger.warning("QuickUMLS path not configured")
        except Exception as e:
            logger.error(f"Error initializing QuickUMLS: {e}")
            self.matcher = None
            
    def extract_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical terms from text using QuickUMLS
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted medical terms with CUIs
        """
        if not self.matcher:
            logger.warning("QuickUMLS not available, skipping term extraction")
            return []
            
        logger.info(f"Extracting medical terms from text: {text[:100]}...")
        
        try:
            # Match medical terms
            matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
            
            terms = []
            for match_group in matches:
                if match_group:
                    match = match_group[0]  # Best match
                    term = {
                        'cui': match['cui'],
                        'term': match['ngram'],
                        'similarity': match.get('similarity', 1.0),
                        'start': match.get('start', 0),
                        'end': match.get('end', len(match['ngram']))
                    }
                    terms.append(term)
                    
            logger.info(f"Extracted {len(terms)} medical terms")
            return terms
            
        except Exception as e:
            logger.error(f"Error extracting medical terms: {e}")
            return []
            
    def get_cui_details(self, cui: str) -> Dict[str, Any]:
        """
        Get detailed information about a CUI from UMLS API
        
        Args:
            cui: Concept Unique Identifier
            
        Returns:
            Dictionary with CUI details including semantic types
        """
        if not self.api_key:
            logger.warning("UMLS API key not configured")
            return {}
            
        base_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            result = data.get('result', {})
            
            # Extract relevant information
            details = {
                'cui': cui,
                'name': result.get('name', ''),
                'semantic_types': [st['name'] for st in result.get('semanticTypes', [])],
                'definition': result.get('definitions', []),
                'atoms': result.get('atomCount', 0),
                'relations': result.get('relationCount', 0)
            }
            
            logger.info(f"Retrieved details for CUI {cui}")
            return details
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"Authentication failed for CUI {cui}. Check UMLS API key.")
            else:
                logger.error(f"HTTP error for CUI {cui}: {e}")
        except Exception as e:
            logger.error(f"Error getting CUI details: {e}")
            
        return {}
        
    def get_cui_definitions(self, cui: str) -> List[str]:
        """
        Get all definitions for a CUI from UMLS API
        
        Args:
            cui: Concept Unique Identifier
            
        Returns:
            List of definitions
        """
        if not self.api_key:
            return []
            
        base_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/definitions"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            definitions = [
                item['value'] 
                for item in data.get('result', []) 
                if 'value' in item
            ]
            
            logger.info(f"Retrieved {len(definitions)} definitions for CUI {cui}")
            return definitions
            
        except Exception as e:
            logger.debug(f"No definitions found for CUI {cui}: {e}")
            return []
            
    def get_cui_relations(self, cui: str) -> List[Dict[str, Any]]:
        """
        Get relations for a CUI from UMLS API
        
        Args:
            cui: Concept Unique Identifier
            
        Returns:
            List of relations
        """
        if not self.api_key:
            return []
            
        base_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/relations"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            relations = []
            
            for item in data.get('result', []):
                relation = {
                    'relationLabel': item.get('relationLabel', ''),
                    'relatedId': item.get('relatedId', ''),
                    'relatedIdName': item.get('relatedIdName', ''),
                    'rootSource': item.get('rootSource', ''),
                    'groupId': item.get('groupId', '')
                }
                relations.append(relation)
                
            logger.info(f"Retrieved {len(relations)} relations for CUI {cui}")
            return relations
            
        except Exception as e:
            logger.debug(f"Error getting relations for CUI {cui}: {e}")
            return []
            
    def process_text(self, text: str, include_details: bool = True) -> List[Dict[str, Any]]:
        """
        Process text to extract medical terms and optionally get their details
        
        Args:
            text: Input text to process
            include_details: Whether to fetch detailed information from UMLS API
            
        Returns:
            List of medical terms with full information
        """
        # Extract medical terms
        terms = self.extract_medical_terms(text)
        
        if not include_details:
            return terms
            
        # Enrich with UMLS details
        enriched_terms = []
        for term in terms:
            cui = term['cui']
            
            # Get basic details
            details = self.get_cui_details(cui)
            
            # Get definitions
            definitions = self.get_cui_definitions(cui)
            
            # Combine information
            enriched_term = {
                **term,
                **details,
                'definitions': definitions
            }
            
            enriched_terms.append(enriched_term)
            
        return enriched_terms
        
    def format_terms_for_context(self, terms: List[Dict[str, Any]]) -> str:
        """
        Format UMLS terms for inclusion in LLM context
        
        Args:
            terms: List of UMLS terms with details
            
        Returns:
            Formatted string for LLM context
        """
        if not terms:
            return ""
            
        context_parts = ["Medical Terms and Concepts:"]
        
        for term in terms:
            term_info = [f"\n- Term: {term.get('term', '')} (CUI: {term.get('cui', '')})"]
            
            if term.get('semantic_types'):
                term_info.append(f"  Types: {', '.join(term['semantic_types'])}")
                
            if term.get('definitions'):
                term_info.append(f"  Definition: {term['definitions'][0][:200]}...")
                
            context_parts.extend(term_info)
            
        return "\n".join(context_parts)