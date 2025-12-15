"""
Component 2: Graph Retrieval Layer for FPL Graph-RAG
Handles both baseline Cypher queries and embedding-based retrieval
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
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
        
        # Default season
        season = entities.get("seasons", ["2022-23"])[0] if entities.get("seasons") else "2022-23"
        
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
        Create embeddings for all Player nodes
        This should be run once during setup
        """
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
        
        # Create text representation for each player
        embeddings_created = 0
        for player in players:
            positions_str = ', '.join(player.get('positions', ['Unknown']))
            text = f"Player: {player['name']}, Position: {positions_str}, " \
                   f"Season: {player.get('season', '2022-23')}, " \
                   f"Points: {player.get('total_points', 0)}, " \
                   f"Goals: {player.get('goals', 0)}, Assists: {player.get('assists', 0)}"
            
            embedding = self.embedding_model.encode(text).tolist()
            
            # Store embedding in Neo4j
            update_query = """
                MATCH (p:Player {player_name: $name, season: $season})
                SET p.embedding = $embedding
            """
            
            with self.driver.session() as session:
                session.run(update_query, {
                    "name": player['name'], 
                    "season": player.get('season', '2022-23'),
                    "embedding": embedding
                })
                embeddings_created += 1
        
        print(f"Created embeddings for {embeddings_created} players")
        return embeddings_created
    
    def embedding_retrieval(self, query_embedding: List[float], top_k: int = 10) -> Dict[str, Any]:
        """
        Perform embedding-based retrieval using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Retrieved information
        """
        # Note: This requires vector index in Neo4j
        # For now, we'll do a simple similarity calculation
        
        query = """
            MATCH (p:Player)
            WHERE p.embedding IS NOT NULL AND p.season = '2022-23'
            RETURN p.player_name AS name, 
                   p.season AS season,
                   p.embedding AS embedding
            LIMIT 200
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            players = [dict(record) for record in result]
        
        if not players:
            return {
                "method": "embedding",
                "data": [],
                "message": "No player embeddings found. Run create_node_embeddings() first."
            }
        
        # Calculate cosine similarity
        similarities = []
        for player in players:
            player_emb = np.array(player['embedding'])
            query_emb = np.array(query_embedding)
            
            similarity = np.dot(query_emb, player_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(player_emb)
            )
            
            # Get additional player info
            info_query = """
                MATCH (p:Player {player_name: $name, season: $season})
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, sum(r.total_points) AS total_points,
                     sum(r.goals_scored) AS goals,
                     sum(r.assists) AS assists
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WITH p, total_points, goals, assists,
                     collect(DISTINCT pos.name) AS positions
                RETURN p.player_name AS name,
                       positions,
                       total_points, goals, assists
            """
            
            with self.driver.session() as session:
                info_result = session.run(info_query, {
                    "name": player['name'],
                    "season": player.get('season', '2022-23')
                })
                info = info_result.single()
                
                if info:
                    similarities.append({
                        "name": info['name'],
                        "positions": info['positions'],
                        "total_points": info['total_points'],
                        "goals": info['goals'],
                        "assists": info['assists'],
                        "similarity": float(similarity)
                    })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "method": "embedding",
            "data": similarities[:top_k]
        }
    
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
