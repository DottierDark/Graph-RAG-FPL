"""
Component 1: Input Preprocessing for FPL Graph-RAG
Handles intent classification, entity extraction, and input embedding
"""

import re
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class FPLInputPreprocessor:
    """
    Handles preprocessing of user queries for FPL assistant
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize preprocessor with embedding model
        
        Args:
            embedding_model_name: Name of sentence transformer model
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Define intent keywords for rule-based classification
        self.intent_patterns = {
            "player_search": ["player", "find player", "show me", "who is", "about"],
            "player_performance": ["stats", "points", "performance", "how many", "scored", "statistics"],
            "team_analysis": ["team", "squad", "lineup", "formation"],
            "recommendation": ["recommend", "suggest", "best", "top", "should i pick", "transfer"],
            "comparison": ["compare", "vs", "versus", "better", "difference between"],
            "fixture": ["fixture", "next game", "upcoming", "schedule", "playing against"],
            "price": ["price", "cost", "value", "budget", "cheap", "expensive"]
        }
        
        # FPL-specific entities
        self.positions = ["GKP", "DEF", "MID", "FWD", "goalkeeper", "defender", "midfielder", "forward"]
        self.stat_types = ["goals", "assists", "points", "clean sheets", "bonus", "saves", "minutes"]
        
    def classify_intent(self, query: str) -> str:
        """
        Classify user intent using rule-based approach
        
        Args:
            query: User query string
            
        Returns:
            Detected intent
        """
        query_lower = query.lower()
        
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
        Extract FPL-specific entities from query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            "players": [],
            "teams": [],
            "positions": [],
            "seasons": [],
            "gameweeks": [],
            "statistics": []
        }
        
        query_lower = query.lower()
        
        # Extract positions
        for position in self.positions:
            if position.lower() in query_lower:
                entities["positions"].append(position.upper()[:3])  # Normalize to 3-letter code
        
        # Extract statistics
        for stat in self.stat_types:
            if stat in query_lower:
                entities["statistics"].append(stat)
        
        # Extract seasons (e.g., "2023", "2023/24", "season 2023")
        season_pattern = r'20\d{2}(?:/\d{2})?'
        seasons = re.findall(season_pattern, query)
        entities["seasons"] = seasons if seasons else ["2023"]  # Default to 2023
        
        # Extract gameweeks (e.g., "gameweek 10", "GW10", "week 5")
        gameweek_pattern = r'(?:gameweek|gw|week)\s*(\d+)'
        gameweeks = re.findall(gameweek_pattern, query_lower)
        entities["gameweeks"] = gameweeks
        
        # Extract player names (capitalized words not matching other categories)
        # This is simplified - in production you'd use NER or match against known players
        words = query.split()
        potential_names = []
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Check if it's not a common word
                if word.lower() not in ["what", "which", "show", "find", "get", "the"]:
                    potential_names.append(word)
        
        # Combine consecutive capitalized words as full names
        if potential_names:
            entities["players"] = [" ".join(potential_names)]
        
        # Extract team names (simplified - you'd match against actual FPL teams)
        fpl_teams = [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
            "Luton", "Man City", "Man Utd", "Newcastle", "Nottm Forest",
            "Sheffield Utd", "Tottenham", "West Ham", "Wolves", "Burnley"
        ]
        
        for team in fpl_teams:
            if team.lower() in query_lower:
                entities["teams"].append(team)
        
        return entities
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert query to embedding vector
        
        Args:
            query: User query string
            
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(query)
        return embedding.tolist()
    
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
