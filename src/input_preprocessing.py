"""
Component 1: Input Preprocessing for FPL Graph-RAG
Handles intent classification, entity extraction, and input embedding
"""

import re
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from .config import (
    INTENT_TYPES, 
    ENTITY_TYPES, 
    POSITION_MAPPINGS, 
    STAT_KEYWORDS,
    FPL_TEAMS,
    get_embedding_model
)

# Try to import spaCy for better entity extraction
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        USE_SPACY = True
    except OSError:
        # If model not found, use basic tokenization
        USE_SPACY = False
        print("⚠️ spaCy model 'en_core_web_sm' not found. Using basic tokenization.")
        print("   Install with: python -m spacy download en_core_web_sm")
except ImportError:
    USE_SPACY = False
    print("⚠️ spaCy not installed. Using basic tokenization.")
    print("   Install with: pip install spacy && python -m spacy download en_core_web_sm")

class FPLInputPreprocessor:
    """
    Handles preprocessing of user queries for FPL assistant.
    Uses centralized configuration to ensure consistency across the system.
    """
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize preprocessor with embedding model.
        
        Args:
            embedding_model_name: Name of sentence transformer model (defaults to config value)
        """
        model_name = embedding_model_name or get_embedding_model()
        self.embedding_model = SentenceTransformer(model_name)
        
        # Use centralized intent patterns from config
        self.intent_patterns = INTENT_TYPES
        
        # Use centralized entity lists from config
        self.positions = list(POSITION_MAPPINGS.keys())
        self.stat_types = list(STAT_KEYWORDS.keys())
        
    def classify_intent(self, query: str) -> str:
        """
        Classify user intent using rule-based approach with priority handling
        
        Args:
            query: User query string
            
        Returns:
            Detected intent
        """
        query_lower = query.lower()
        
        # Priority check for comparison queries (higher priority)
        if any(keyword in query_lower for keyword in ["compare", "vs", "versus", "better", "difference between"]):
            return "comparison"
        
        # Check each intent pattern
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return "general_question"
        
        # Return intent with highest score
        return max(intent_scores.items(), key=lambda x: x[1])[0]
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract FPL-specific entities from query using centralized entity definitions.
        Uses spaCy NER if available, otherwise falls back to pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary of extracted entities (players, teams, positions, seasons, gameweeks, statistics)
        """
        entities = {entity_type: [] for entity_type in ENTITY_TYPES}
        
        query_lower = query.lower()
        
        # Extract positions using centralized mappings
        for position_variant, normalized in POSITION_MAPPINGS.items():
            if position_variant.lower() in query_lower:
                if normalized not in entities["positions"]:
                    entities["positions"].append(normalized)
        
        # Extract statistics using centralized keywords
        for stat_name, stat_variations in STAT_KEYWORDS.items():
            if any(variant in query_lower for variant in stat_variations):
                if stat_name not in entities["statistics"]:
                    entities["statistics"].append(stat_name)
        
        # Extract seasons (e.g., "2023", "2023/24", "2022-23", "season 2023")
        season_pattern = r'20\d{2}(?:[/-]\d{2})?'
        seasons = re.findall(season_pattern, query)
        # Normalize season format to match database (e.g., "2022" -> "2022-23")
        normalized_seasons = []
        for s in seasons:
            if '-' not in s and '/' not in s:
                # Single year like "2022" -> "2022-23"
                normalized_seasons.append(f"{s}-{str(int(s[-2:]) + 1).zfill(2)}")
            elif '/' in s:
                # Convert "2022/23" to "2022-23"
                normalized_seasons.append(s.replace('/', '-'))
            else:
                normalized_seasons.append(s)
        entities["seasons"] = normalized_seasons if normalized_seasons else ["2022-23"]  # Default to 2022-23
        
        # Extract gameweeks (e.g., "gameweek 10", "GW10", "week 5")
        gameweek_pattern = r'(?:gameweek|gw|week)\s*(\d+)'
        gameweeks = re.findall(gameweek_pattern, query_lower)
        entities["gameweeks"] = gameweeks
        
        # Extract player names using spaCy if available, otherwise use pattern matching
        if USE_SPACY:
            entities["players"] = self._extract_players_with_spacy(query)
        else:
            entities["players"] = self._extract_players_basic(query)
        
        # Extract team names using centralized team list
        for team in FPL_TEAMS:
            if team.lower() in query_lower:
                entities["teams"].append(team)
        
        return entities
    
    def _extract_players_with_spacy(self, query: str) -> List[str]:
        """
        Extract player names using spaCy NER.
        
        Args:
            query: User query string
            
        Returns:
            List of player names
        """
        doc = nlp(query)
        players = []
        
        # Extract PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                players.append(ent.text)
        
        # If no PERSON entities found, fall back to basic extraction
        if not players:
            players = self._extract_players_basic(query)
        
        return players
    
    def _extract_players_basic(self, query: str) -> List[str]:
        """
        Extract player names using basic pattern matching (fallback method).
        
        Args:
            query: User query string
            
        Returns:
            List of player names
        """
        # Remove common query words and intent keywords
        excluded_words = ["what", "which", "show", "find", "get", "the", "who", "are", "is", 
                         "top", "best", "how", "when", "where", "why", "can", "could", "would",
                         "compare", "vs", "versus", "between", "and", "or", "with", "against",
                         "me", "my", "their", "his", "her", "stats", "performance", "players"]
        
        # Tokenize and filter
        words = query.split()
        potential_names = []
        
        for word in words:
            # Check if word starts with uppercase and is not excluded
            if len(word) > 2 and word[0].isupper() and word.lower() not in excluded_words:
                # Also check it's not a team name
                if word not in FPL_TEAMS:
                    potential_names.append(word)
        
        # If comparison query, treat each capitalized word as separate player
        query_lower = query.lower()
        if len(potential_names) > 1 and any(kw in query_lower for kw in ['compare', 'vs', 'versus']):
            return potential_names
        elif potential_names:
            # Otherwise combine as single full name
            return [" ".join(potential_names)]
        
        return []
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert query to embedding vector with query expansion for better matching.
        
        Args:
            query: User query string
            
        Returns:
            Embedding vector
        """
        # Expand query with context to improve semantic similarity
        expanded_query = self._expand_query(query)
        
        embedding = self.embedding_model.encode(expanded_query)
        return embedding.tolist()
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with FPL-specific context to improve embedding similarity.
        This helps match user intent with detailed player descriptions.
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query with additional context
        """
        query_lower = query.lower()
        expanded = query
        
        # Add position context
        if any(word in query_lower for word in ['forward', 'striker', 'fwd']):
            expanded += " Forward striker attacking player who scores goals"
        elif any(word in query_lower for word in ['midfielder', 'mid']):
            expanded += " Midfielder midfield player who creates plays assists"
        elif any(word in query_lower for word in ['defender', 'def', 'defence', 'defense']):
            expanded += " Defender defensive player clean sheets"
        elif any(word in query_lower for word in ['goalkeeper', 'keeper', 'gkp']):
            expanded += " Goalkeeper keeper saves clean sheets"
        
        # Add performance context
        if any(word in query_lower for word in ['best', 'top', 'elite']):
            expanded += " high points excellent performance elite top-performing"
        
        if any(word in query_lower for word in ['scorer', 'goals', 'goal']):
            expanded += " goals scored striker forward attacking"
        
        if any(word in query_lower for word in ['assist', 'creator', 'playmaker']):
            expanded += " assists creative playmaker midfielder"
        
        if any(word in query_lower for word in ['value', 'budget', 'cheap', 'affordable']):
            expanded += " price cost value budget-friendly affordable"
        
        # Add FPL context
        expanded += " Fantasy Premier League FPL player performance statistics"
        
        return expanded
    
    def preprocess(self, query: str) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent, entities, and embedding
        """
        intent = self.classify_intent(query)
        entities = self.extract_entities(query)
        embedding = self.embed_query(query)
        
        return {
            "query": query,
            "intent": intent,
            "entities": entities,
            "embedding": embedding
        }
    
    def analyze_query(self, query: str) -> Dict:
        """
        Perform detailed analysis of a query for error analysis and improvement.
        
        Args:
            query: User query string
            
        Returns:
            Detailed analysis with confidence scores and suggestions
        """
        result = self.preprocess(query)
        
        # Calculate confidence scores
        query_lower = query.lower()
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Analyze entity extraction quality
        entity_confidence = {
            "players": len(result["entities"]["players"]) > 0,
            "teams": len(result["entities"]["teams"]) > 0,
            "positions": len(result["entities"]["positions"]) > 0,
            "seasons": len(result["entities"]["seasons"]) > 0,
            "gameweeks": len(result["entities"]["gameweeks"]) > 0,
            "statistics": len(result["entities"]["statistics"]) > 0
        }
        
        # Generate suggestions for improvement
        suggestions = []
        if not any(entity_confidence.values()):
            suggestions.append("Query lacks specific FPL entities. Try mentioning player names, positions, or teams.")
        
        if len(intent_scores) == 0:
            suggestions.append("Intent unclear. Try using keywords like 'top', 'compare', 'stats', or 'recommend'.")
        elif len(intent_scores) > 2:
            suggestions.append("Multiple intents detected. Consider breaking into separate queries.")
        
        return {
            **result,
            "analysis": {
                "intent_scores": intent_scores,
                "entity_confidence": entity_confidence,
                "entities_found": sum(1 for v in entity_confidence.values() if v),
                "suggestions": suggestions,
                "query_complexity": "simple" if len(query.split()) < 5 else "complex"
            }
        }


# Example usage and testing
if __name__ == "__main__":
    preprocessor = FPLInputPreprocessor()
    
    # Test queries
    test_queries = [
        "Who are the top forwards in 2023?",
        "Show me Erling Haaland's stats",
        "Compare Salah and Son performance",
        "Best midfielders under 8 million",
        "Arsenal players with most clean sheets"
    ]
    
    print("Testing FPL Input Preprocessor\n")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = preprocessor.preprocess(query)
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print(f"Embedding shape: {len(result['embedding'])}")
        print("-" * 60)
