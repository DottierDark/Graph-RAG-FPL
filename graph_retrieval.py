"""
Component 2: Graph Retrieval Layer for FPL Graph-RAG
Handles both baseline Cypher queries and embedding-based retrieval
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class FPLGraphRetriever:
    """
    Handles retrieval from Neo4j Knowledge Graph using Cypher queries and embeddings
    """
    
    def __init__(self, uri: str, username: str, password: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize graph retriever
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            embedding_model_name: Name of embedding model
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Define Cypher query templates
        self.query_templates = {
            "player_search": """
                MATCH (p:Player)
                WHERE toLower(p.name) CONTAINS toLower($player_name)
                RETURN p.name AS name, p.team AS team, p.position AS position, 
                       p.price AS price, p.total_points AS points
                LIMIT 10
            """,
            
            "top_players_by_position": """
                MATCH (p:Player)
                WHERE p.position = $position
                RETURN p.name AS name, p.team AS team, p.total_points AS points,
                       p.price AS price, p.goals AS goals, p.assists AS assists
                ORDER BY p.total_points DESC
                LIMIT 10
            """,
            
            "player_stats": """
                MATCH (p:Player {name: $player_name})
                OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
                RETURN p.name AS name, p.position AS position, p.price AS price,
                       p.total_points AS points, p.goals AS goals, p.assists AS assists,
                       p.clean_sheets AS clean_sheets, p.minutes AS minutes,
                       t.name AS team
            """,
            
            "team_players": """
                MATCH (p:Player)-[:PLAYS_FOR]->(t:Team)
                WHERE toLower(t.name) CONTAINS toLower($team_name)
                RETURN p.name AS name, p.position AS position, p.total_points AS points,
                       p.price AS price
                ORDER BY p.total_points DESC
                LIMIT 15
            """,
            
            "players_by_price": """
                MATCH (p:Player)
                WHERE p.position = $position AND p.price <= $max_price
                RETURN p.name AS name, p.team AS team, p.price AS price,
                       p.total_points AS points, p.goals AS goals, p.assists AS assists
                ORDER BY p.total_points DESC
                LIMIT 10
            """,
            
            "player_comparison": """
                MATCH (p:Player)
                WHERE p.name IN $player_names
                RETURN p.name AS name, p.position AS position, p.price AS price,
                       p.total_points AS points, p.goals AS goals, p.assists AS assists,
                       p.clean_sheets AS clean_sheets, p.bonus AS bonus
            """,
            
            "gameweek_performance": """
                MATCH (p:Player)-[r:PLAYED_IN]->(g:Gameweek)
                WHERE g.gameweek_number = $gameweek AND p.name = $player_name
                RETURN p.name AS name, g.gameweek_number AS gameweek,
                       r.points AS points, r.goals AS goals, r.assists AS assists
            """,
            
            "top_scorers": """
                MATCH (p:Player)
                WHERE p.position = $position
                RETURN p.name AS name, p.team AS team, p.goals AS goals,
                       p.total_points AS points, p.price AS price
                ORDER BY p.goals DESC
                LIMIT 10
            """,
            
            "clean_sheet_leaders": """
                MATCH (p:Player)
                WHERE p.position IN ['GKP', 'DEF']
                RETURN p.name AS name, p.team AS team, p.position AS position,
                       p.clean_sheets AS clean_sheets, p.total_points AS points
                ORDER BY p.clean_sheets DESC
                LIMIT 10
            """,
            
            "fixtures": """
                MATCH (t:Team)-[f:PLAYS_AGAINST]->(opp:Team)
                WHERE toLower(t.name) CONTAINS toLower($team_name)
                  AND f.gameweek = $gameweek
                RETURN t.name AS team, opp.name AS opponent, 
                       f.gameweek AS gameweek, f.difficulty AS difficulty
            """,
            
            "value_picks": """
                MATCH (p:Player)
                WHERE p.position = $position
                WITH p, (toFloat(p.total_points) / toFloat(p.price)) AS value
                RETURN p.name AS name, p.team AS team, p.price AS price,
                       p.total_points AS points, value
                ORDER BY value DESC
                LIMIT 10
            """,
            
            "form_players": """
                MATCH (p:Player)
                WHERE p.position = $position
                RETURN p.name AS name, p.team AS team, p.form AS form,
                       p.total_points AS points, p.price AS price
                ORDER BY toFloat(p.form) DESC
                LIMIT 10
            """
        }
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def execute_query(self, query_type: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Execute a Cypher query
        
        Args:
            query_type: Type of query from templates
            params: Parameters for the query
            
        Returns:
            List of result dictionaries
        """
        if query_type not in self.query_templates:
            return []
        
        query = self.query_templates[query_type]
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def baseline_retrieval(self, intent: str, entities: Dict) -> Dict[str, Any]:
        """
        Perform baseline retrieval using Cypher queries
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            
        Returns:
            Retrieved information
        """
        results = {
            "method": "baseline",
            "intent": intent,
            "data": []
        }
        
        # Route to appropriate query based on intent
        if intent == "player_search" and entities.get("players"):
            params = {"player_name": entities["players"][0]}
            results["data"] = self.execute_query("player_search", params)
            results["query_type"] = "player_search"
        
        elif intent == "player_performance" and entities.get("players"):
            params = {"player_name": entities["players"][0]}
            results["data"] = self.execute_query("player_stats", params)
            results["query_type"] = "player_stats"
        
        elif intent == "recommendation" and entities.get("positions"):
            position = entities["positions"][0]
            if "price" in entities or "value" in entities or "cheap" in intent:
                params = {"position": position, "max_price": 8.0}
                results["data"] = self.execute_query("players_by_price", params)
                results["query_type"] = "players_by_price"
            else:
                params = {"position": position}
                results["data"] = self.execute_query("top_players_by_position", params)
                results["query_type"] = "top_players_by_position"
        
        elif intent == "comparison" and len(entities.get("players", [])) >= 2:
            params = {"player_names": entities["players"][:2]}
            results["data"] = self.execute_query("player_comparison", params)
            results["query_type"] = "player_comparison"
        
        elif intent == "team_analysis" and entities.get("teams"):
            params = {"team_name": entities["teams"][0]}
            results["data"] = self.execute_query("team_players", params)
            results["query_type"] = "team_players"
        
        else:
            # Default: show top players
            params = {"position": "FWD"}
            results["data"] = self.execute_query("top_players_by_position", params)
            results["query_type"] = "top_players_by_position"
        
        return results
    
    def create_node_embeddings(self):
        """
        Create embeddings for all Player nodes
        This should be run once during setup
        """
        query = """
            MATCH (p:Player)
            RETURN p.name AS name, p.position AS position, p.team AS team,
                   p.total_points AS points, p.price AS price,
                   p.goals AS goals, p.assists AS assists
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            players = [dict(record) for record in result]
        
        # Create text representation for each player
        for player in players:
            text = f"Player: {player['name']}, Position: {player['position']}, " \
                   f"Team: {player['team']}, Points: {player['points']}, " \
                   f"Goals: {player['goals']}, Assists: {player['assists']}"
            
            embedding = self.embedding_model.encode(text).tolist()
            
            # Store embedding in Neo4j
            update_query = """
                MATCH (p:Player {name: $name})
                SET p.embedding = $embedding
            """
            
            with self.driver.session() as session:
                session.run(update_query, {"name": player['name'], "embedding": embedding})
        
        print(f"Created embeddings for {len(players)} players")
    
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
            WHERE p.embedding IS NOT NULL
            RETURN p.name AS name, p.position AS position, p.team AS team,
                   p.total_points AS points, p.price AS price,
                   p.embedding AS embedding
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            players = [dict(record) for record in result]
        
        # Calculate cosine similarity
        similarities = []
        for player in players:
            player_emb = np.array(player['embedding'])
            query_emb = np.array(query_embedding)
            
            similarity = np.dot(query_emb, player_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(player_emb)
            )
            
            similarities.append({
                "name": player['name'],
                "position": player['position'],
                "team": player['team'],
                "points": player['points'],
                "price": player['price'],
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
