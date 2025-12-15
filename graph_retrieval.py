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
        
        # Define Cypher query templates (adapted for Milestone 2 schema)
        self.query_templates = {
            "player_search": """
                MATCH (p:Player)
                WHERE toLower(p.player_name) CONTAINS toLower($player_name)
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WITH p, collect(DISTINCT pos.name) AS positions
                RETURN p.player_name AS name, 
                       positions AS positions,
                       p.season AS season
                LIMIT 10
            """,
            
            "top_players_by_position": """
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, pos, sum(r.total_points) AS total_points, 
                     sum(r.goals_scored) AS goals, sum(r.assists) AS assists
                RETURN p.player_name AS name, 
                       pos.name AS position,
                       total_points, goals, assists
                ORDER BY total_points DESC
                LIMIT 10
            """,
            
            "player_stats": """
                MATCH (p:Player {player_name: $player_name})
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, collect(DISTINCT pos.name) AS positions,
                     sum(r.total_points) AS total_points,
                     sum(r.goals_scored) AS goals,
                     sum(r.assists) AS assists,
                     sum(r.minutes) AS minutes,
                     sum(r.clean_sheets) AS clean_sheets
                RETURN p.player_name AS name, 
                       positions,
                       p.season AS season,
                       total_points, goals, assists, minutes, clean_sheets
            """,
            
            "team_players": """
                MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $team_name})
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                WHERE f.season = $season
                WITH p, sum(r.total_points) AS total_points
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WITH p, total_points, collect(DISTINCT pos.name) AS positions
                RETURN p.player_name AS name, 
                       positions,
                       total_points
                ORDER BY total_points DESC
                LIMIT 15
            """,
            
            "players_by_performance": """
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, sum(r.total_points) AS total_points,
                     sum(r.goals_scored) AS goals,
                     sum(r.assists) AS assists
                WHERE total_points >= $min_points
                RETURN p.player_name AS name, 
                       total_points, goals, assists
                ORDER BY total_points DESC
                LIMIT 10
            """,
            
            "player_comparison": """
                MATCH (p:Player)
                WHERE p.player_name IN $player_names
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, sum(r.total_points) AS total_points,
                     sum(r.goals_scored) AS goals,
                     sum(r.assists) AS assists,
                     sum(r.clean_sheets) AS clean_sheets,
                     sum(r.bonus) AS bonus
                OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
                WITH p, total_points, goals, assists, clean_sheets, bonus,
                     collect(DISTINCT pos.name) AS positions
                RETURN p.player_name AS name, 
                       positions,
                       total_points, goals, assists, clean_sheets, bonus
            """,
            
            "gameweek_performance": """
                MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)-[:IN_GAMEWEEK]->(g:Gameweek {gameweek_number: $gameweek})
                WHERE f.season = $season
                RETURN p.player_name AS name, 
                       g.gameweek_number AS gameweek,
                       r.total_points AS points, 
                       r.goals_scored AS goals, 
                       r.assists AS assists,
                       r.minutes AS minutes
            """,
            
            "top_scorers": """
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, sum(r.goals_scored) AS goals,
                     sum(r.total_points) AS total_points
                RETURN p.player_name AS name, 
                       goals, total_points
                ORDER BY goals DESC
                LIMIT 10
            """,
            
            "clean_sheet_leaders": """
                MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
                WHERE pos.name IN ['GKP', 'DEF']
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                WHERE f.season = $season
                WITH p, pos, sum(r.clean_sheets) AS clean_sheets,
                     sum(r.total_points) AS total_points
                RETURN p.player_name AS name, 
                       pos.name AS position,
                       clean_sheets, total_points
                ORDER BY clean_sheets DESC
                LIMIT 10
            """,
            
            "season_overview": """
                MATCH (f:Fixture {season: $season})
                WITH count(DISTINCT f) AS total_fixtures
                MATCH (g:Gameweek)
                MATCH (gw_fixture:Fixture {season: $season})-[:IN_GAMEWEEK]->(g)
                WITH total_fixtures, count(DISTINCT g) AS total_gameweeks
                MATCH (p:Player {season: $season})
                RETURN total_fixtures, total_gameweeks, count(p) AS total_players
            """,
            
            "best_performers_gameweek": """
                MATCH (f:Fixture)-[:IN_GAMEWEEK]->(g:Gameweek {gameweek_number: $gameweek})
                MATCH (p:Player)-[r:PLAYED_IN]->(f)
                WHERE f.season = $season
                RETURN p.player_name AS name,
                       r.total_points AS points,
                       r.goals_scored AS goals,
                       r.assists AS assists
                ORDER BY r.total_points DESC
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
        
        # Default season
        season = entities.get("seasons", ["2022-23"])[0] if entities.get("seasons") else "2022-23"
        
        # Route to appropriate query based on intent
        if intent == "player_search" and entities.get("players"):
            params = {"player_name": entities["players"][0]}
            results["data"] = self.execute_query("player_search", params)
            results["query_type"] = "player_search"
        
        elif intent == "player_performance" and entities.get("players"):
            params = {
                "player_name": entities["players"][0],
                "season": season
            }
            results["data"] = self.execute_query("player_stats", params)
            results["query_type"] = "player_stats"
        
        elif intent == "recommendation" and entities.get("positions"):
            position = entities["positions"][0]
            params = {
                "position": position,
                "season": season
            }
            results["data"] = self.execute_query("top_players_by_position", params)
            results["query_type"] = "top_players_by_position"
        
        elif intent == "comparison" and entities.get("players") and len(entities["players"]) >= 2:
            params = {
                "player_names": entities["players"][:2],
                "season": season
            }
            results["data"] = self.execute_query("player_comparison", params)
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
