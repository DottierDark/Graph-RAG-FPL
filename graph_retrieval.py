"""
Component 2: Graph Retrieval Layer for FPL Graph-RAG
Handles both baseline Cypher queries and embedding-based retrieval
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from config import (
    get_neo4j_config,
    get_embedding_model,
    RETRIEVAL_LIMITS,
    CYPHER_QUERY_TEMPLATES,
    ERROR_MESSAGES
)


class FPLGraphRetriever:
    """
    Handles retrieval from Neo4j Knowledge Graph using Cypher queries and embeddings.
    Uses centralized configuration for queries, limits, and connection settings.
    """
    
    def __init__(self, uri: str = None, username: str = None, password: str = None, embedding_model_name: str = None):
        """
        Initialize graph retriever with optional override parameters.
        Defaults to centralized config values.
        
        Args:
            uri: Neo4j connection URI (defaults to config)
            username: Neo4j username (defaults to config)
            password: Neo4j password (defaults to config)
            embedding_model_name: Name of embedding model (defaults to config)
        """
        # Use centralized configuration with optional overrides
        neo4j_config = get_neo4j_config()
        self.uri = uri or neo4j_config['uri']
        self.username = username or neo4j_config['username']
        self.password = password or neo4j_config['password']
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Initialize embedding model using config
        model_name = embedding_model_name or get_embedding_model()
        self.embedding_model = SentenceTransformer(model_name)
        
        # Use centralized query templates from config
        self.query_templates = CYPHER_QUERY_TEMPLATES
        
        # FAISS vector store for embeddings (separate from Neo4j)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.player_metadata = []
        self.cache_dir = Path("vector_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Try to load cached embeddings
        self._load_vector_cache()
    
    def close(self):
        """Close Neo4j connection gracefully"""
        if self.driver:
            self.driver.close()
    
    def _execute_query_safely(self, query_type: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Execute a Cypher query with error handling and centralized limits.
        Helper method to reduce code duplication across retrieval methods.
        
        Args:
            query_type: Type of query from templates
            params: Parameters for the query
            
        Returns:
            List of result dictionaries (empty list on error)
        """
        if query_type not in self.query_templates:
            print(f"Warning: {ERROR_MESSAGES['query_not_found']}: {query_type}")
            return []
        
        try:
            query = self.query_templates[query_type]
            
            with self.driver.session() as session:
                result = session.run(query, params)
                records = [dict(record) for record in result]
                
                # Apply centralized retrieval limits
                limit = RETRIEVAL_LIMITS.get(query_type, RETRIEVAL_LIMITS['default'])
                return records[:limit]
                
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def execute_query(self, query_type: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Execute a Cypher query (public interface).
        
        Args:
            query_type: Type of query from templates
            params: Parameters for the query
            
        Returns:
            List of result dictionaries
        """
        return self._execute_query_safely(query_type, params)
    
    def baseline_retrieval(self, intent: str, entities: Dict) -> Dict[str, Any]:
        """
        Perform baseline retrieval using Cypher queries.
        Routes queries based on intent and entities using centralized query templates.
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            
        Returns:
            Retrieved information with method, intent, query_type, and data
        """
        results = {
            "method": "baseline",
            "intent": intent,
            "data": []
        }
        
        # Default season - ensure it's in correct format (YYYY-YY)
        seasons = entities.get("seasons", ["2022-23"])
        season = seasons[0] if seasons else "2022-23"
        
        # Route to appropriate query based on intent using helper method
        if intent == "player_search" and entities.get("players"):
            params = {"player_name": entities["players"][0]}
            results["data"] = self._execute_query_safely("player_search", params)
            results["query_type"] = "player_search"
        
        elif intent == "player_performance" and entities.get("players"):
            params = {
                "player_name": entities["players"][0],
                "season": season
            }
            results["data"] = self._execute_query_safely("player_stats", params)
            results["query_type"] = "player_stats"
        
        elif intent == "recommendation" and entities.get("positions"):
            position = entities["positions"][0]
            params = {
                "position": position,
                "season": season
            }
            results["data"] = self._execute_query_safely("top_players_by_position", params)
            results["query_type"] = "top_players_by_position"
        
        elif intent == "comparison" and entities.get("players") and len(entities["players"]) >= 2:
            params = {
                "player_names": entities["players"][:2],
                "season": season
            }
            results["data"] = self._execute_query_safely("player_comparison", params)
            results["query_type"] = "player_comparison"
        
        elif intent == "team_analysis" and entities.get("teams"):
            params = {
                "team_name": entities["teams"][0],
                "season": season
            }
            results["data"] = self.execute_query("team_players", params)
            results["query_type"] = "team_players"
        
        elif intent == "fixture" and entities.get("gameweeks"):
            gameweek = int(entities["gameweeks"][0])
            params = {
                "gameweek": gameweek,
                "season": season
            }
            results["data"] = self.execute_query("best_performers_gameweek", params)
            results["query_type"] = "best_performers_gameweek"
        
        else:
            # Default: show top scorers for forwards
            params = {
                "position": "FWD",
                "season": season
            }
            results["data"] = self.execute_query("top_scorers", params)
            results["query_type"] = "top_scorers"
        
        return results
    
    def create_node_embeddings(self):
        """
        Create embeddings for all Player nodes using FAISS vector store.
        Loads data from Neo4j (read-only), creates embeddings, caches locally.
        """
        # Load player data from Neo4j
        query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE f.season = '2022-23'
            WITH p, 
                 sum(r.total_points) AS total_points,
                 sum(r.goals_scored) AS goals,
                 sum(r.assists) AS assists,
                 sum(r.minutes) AS minutes
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WITH p, total_points, goals, assists, minutes,
                 collect(DISTINCT pos.name) AS positions
            RETURN p.player_name AS name, 
                   positions,
                   p.season AS season,
                   total_points, goals, assists, minutes
            LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            players = [dict(record) for record in result]
        
        if not players:
            print("No players found in Neo4j")
            return 0
        
        # Create text representations and embeddings
        texts = []
        self.player_metadata = []
        
        for player in players:
            positions_str = ', '.join(player.get('positions', ['Unknown']))
            text = f"Player: {player['name']}, Position: {positions_str}, " \
                   f"Season: {player.get('season', '2022-23')}, " \
                   f"Points: {player.get('total_points', 0)}, " \
                   f"Goals: {player.get('goals', 0)}, Assists: {player.get('assists', 0)}"
            texts.append(text)
            self.player_metadata.append(player)
        
        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)
        
        # Save to cache
        self._save_vector_cache()
        
        print(f"Created embeddings for {len(players)} players and cached to {self.cache_dir}")
        return len(players)
    
    def embedding_retrieval(self, query_embedding: List[float], top_k: int = 10) -> Dict[str, Any]:
        """
        Perform embedding-based retrieval using FAISS vector similarity.
        
        Args:
            query_embedding: Query embedding vector or query string
            top_k: Number of results to return
            
        Returns:
            Retrieved information with similarity scores
        """
        if self.index is None or len(self.player_metadata) == 0:
            return {
                "method": "embedding",
                "data": [],
                "message": "No embeddings available. Run create_node_embeddings() first."
            }
        
        # Convert query to embedding if it's a string
        if isinstance(query_embedding, str):
            query_emb = self.embedding_model.encode([query_embedding]).astype('float32')
        else:
            query_emb = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_emb, min(top_k, len(self.player_metadata)))
        
        # Prepare results with similarity scores
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.player_metadata):
                player = self.player_metadata[idx].copy()
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + distance)
                player['similarity'] = float(similarity)
                results.append(player)
        
        return {
            "method": "embedding",
            "data": results
        }
    
    def _load_vector_cache(self) -> bool:
        """
        Load cached FAISS index and player metadata from disk.
        
        Returns:
            True if cache loaded successfully, False otherwise
        """
        index_path = self.cache_dir / "player_embeddings.faiss"
        metadata_path = self.cache_dir / "player_metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                self.player_metadata = pickle.load(f)
            print(f"Loaded {len(self.player_metadata)} player embeddings from cache")
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def _save_vector_cache(self):
        """
        Save FAISS index and player metadata to disk for fast future loading.
        """
        if self.index is None:
            return
        
        index_path = self.cache_dir / "player_embeddings.faiss"
        metadata_path = self.cache_dir / "player_metadata.pkl"
        
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.player_metadata, f)
        
        print(f"Saved embeddings cache to {self.cache_dir}")
    
    def is_embeddings_ready(self) -> bool:
        """
        Check if embeddings are loaded and ready for queries.
        
        Returns:
            True if embeddings are available, False otherwise
        """
        return self.index is not None and len(self.player_metadata) > 0
    
    def hybrid_retrieval(self, intent: str, entities: Dict, query_embedding: List[float]) -> Dict[str, Any]:
        """
        Combine baseline and embedding retrieval
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            query_embedding: Query embedding vector
            
        Returns:
            Combined retrieval results
        """
        baseline_results = self.baseline_retrieval(intent, entities)
        embedding_results = self.embedding_retrieval(query_embedding, top_k=5)
        
        return {
            "baseline": baseline_results,
            "embedding": embedding_results
        }


# Example usage
if __name__ == "__main__":
    # Test with dummy connection (update with your credentials)
    retriever = FPLGraphRetriever(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    print("FPL Graph Retriever initialized")
    print("Available query types:", list(retriever.query_templates.keys()))
