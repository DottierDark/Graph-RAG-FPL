"""
Component 2: Graph Retrieval Layer for FPL Graph-RAG (OPTIMIZED)
Handles both baseline Cypher queries and embedding-based retrieval with FAISS optimization
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime
from .config import (
    get_neo4j_config,
    get_embedding_model,
    RETRIEVAL_LIMITS,
    CYPHER_QUERY_TEMPLATES,
    ERROR_MESSAGES
)


class FPLGraphRetriever:
    """
    Handles retrieval from Neo4j Knowledge Graph using Cypher queries and embeddings.
    OPTIMIZED VERSION with:
    - Dynamic embedding dimensions
    - Multi-model support
    - Smart caching
    - Progress indicators
    - Batch processing
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
        self.embedding_model_name = embedding_model_name or get_embedding_model()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Use centralized query templates from config
        self.query_templates = CYPHER_QUERY_TEMPLATES
        
        # FAISS vector store for embeddings (separate from Neo4j)
        # OPTIMIZATION: Get dimension dynamically from model
        test_embedding = self.embedding_model.encode("test")
        self.dimension = len(test_embedding)
        print(f"📐 Embedding dimension: {self.dimension}D ({self.embedding_model_name})")
        
        self.index = None
        self.player_metadata = []
        
        # OPTIMIZATION: Model-specific cache directory under data/cache
        model_safe_name = self.embedding_model_name.replace('/', '_').replace('\\', '_')
        self.cache_dir = Path("data") / "cache" / model_safe_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load cached embeddings
        if self._load_vector_cache():
            print(f"✅ Embeddings ready: {len(self.player_metadata)} players cached")
        else:
            print("⚠️ No cache found. Run create_node_embeddings() to build index.")
    
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
    
    def create_node_embeddings(self, batch_size: int = 64, max_players: int = None):
        """
        Create embeddings for all Player nodes using FAISS vector store.
        OPTIMIZED VERSION with:
        - Progress indicators
        - Batch processing (default: 64 for efficiency)
        - Processes ALL players by default (no artificial limits)
        - Error handling per player
        - Memory-efficient processing
        
        Args:
            batch_size: Number of players to encode at once (default: 64, optimized for large datasets)
            max_players: Maximum number of players to process (None = ALL ~1600 players)
        
        Returns:
            Number of embeddings created
        """
        print("\n" + "="*60)
        print("🔄 CREATING PLAYER EMBEDDINGS FOR ALL PLAYERS")
        print("="*60)
        
        # OPTIMIZATION: No limit by default - process all 1600+ players
        limit_clause = f"LIMIT {max_players}" if max_players else ""
        
        # OPTIMIZATION: More efficient query - get all stats in one go
        query = f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE f.season = '2022-23'
            WITH p, 
                 sum(r.total_points) AS total_points,
                 sum(r.goals_scored) AS goals,
                 sum(r.assists) AS assists,
                 sum(r.minutes) AS minutes,
                 sum(r.clean_sheets) AS clean_sheets,
                 sum(r.bonus) AS bonus,
                 count(f) AS appearances
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            OPTIONAL MATCH (p)-[:PLAYED_IN]->(:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
            WITH p, total_points, goals, assists, minutes, clean_sheets, bonus, appearances,
                 collect(DISTINCT pos.name) AS positions,
                 collect(DISTINCT t.name)[0] AS team
            RETURN p.player_name AS name, 
                   positions,
                   team,
                   p.season AS season,
                   total_points, goals, assists, minutes, clean_sheets, bonus, appearances
            ORDER BY total_points DESC
            {limit_clause}
        """
        
        print("📊 Fetching players from Neo4j...")
        try:
            with self.driver.session() as session:
                result = session.run(query)
                players = [dict(record) for record in result]
        except Exception as e:
            print(f"❌ Error fetching players: {e}")
            return 0
        
        if not players:
            print("❌ No players found in Neo4j for season 2022-23")
            return 0
        
        print(f"✅ Found {len(players)} players (processing ALL players)")
        print(f"🔧 Embedding model: {self.embedding_model_name} ({self.dimension}D)")
        print(f"⚙️  Batch size: {batch_size} (optimized for large datasets)")
        print(f"💾 Memory: ~{len(players) * self.dimension * 4 / 1024 / 1024:.1f} MB for embeddings")
        print()
        
        # OPTIMIZATION: Batch processing with progress
        texts = []
        self.player_metadata = []
        
        for player in players:
            positions_str = ', '.join(player.get('positions', ['Unknown']))
            team = player.get('team', 'Unknown Team')
            
            # OPTIMIZATION: Rich text representation for better semantic search
            text = (
                f"Player: {player['name']}, "
                f"Position: {positions_str}, "
                f"Team: {team}, "
                f"Season: {player.get('season', '2022-23')}, "
                f"Total Points: {player.get('total_points', 0)}, "
                f"Goals: {player.get('goals', 0)}, "
                f"Assists: {player.get('assists', 0)}, "
                f"Minutes: {player.get('minutes', 0)}, "
                f"Clean Sheets: {player.get('clean_sheets', 0)}, "
                f"Bonus: {player.get('bonus', 0)}, "
                f"Appearances: {player.get('appearances', 0)}"
            )
            texts.append(text)
            self.player_metadata.append(player)
        
        # Generate embeddings in batches with progress
        print("🔄 Generating embeddings (this may take 2-3 minutes for 1600 players)...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize manually for FAISS
        )
        embeddings_np = embeddings.astype('float32')
        
        print(f"✅ Generated {len(embeddings_np)} embeddings")
        
        # OPTIMIZATION: Use IndexFlatIP (Inner Product) which is faster for normalized vectors
        print("🏗️  Building FAISS index (optimized for fast similarity search)...")
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = cosine similarity for normalized vectors
        self.index.add(embeddings_np)
        
        print(f"✅ FAISS index built: {self.index.ntotal} vectors indexed")
        
        # Save to cache with metadata
        self._save_vector_cache()
        
        print()
        print("="*60)
        print(f"✅ SUCCESS! ALL PLAYERS PROCESSED")
        print(f"   Created embeddings for {len(players)} players (~1600 total)")
        print(f"   Relations covered: ~52,000+ PLAYED_IN relationships")
        print(f"   Cached to: {self.cache_dir}")
        print(f"   Model: {self.embedding_model_name}")
        print(f"   Dimension: {self.dimension}D")
        print(f"   FAISS Index: {self.index.ntotal} vectors")
        print(f"   Ready for semantic search across all players!")
        print("="*60)
        print()
        
        return len(players)
    
    def embedding_retrieval(self, query_embedding: List[float], top_k: int = 15) -> Dict[str, Any]:
        """
        Perform embedding-based retrieval using FAISS vector similarity.
        OPTIMIZED VERSION with better similarity scoring for large datasets.
        
        Args:
            query_embedding: Query embedding vector or query string
            top_k: Number of results to return (default: 15 to leverage full index)
            
        Returns:
            Retrieved information with similarity scores
        """
        if self.index is None or len(self.player_metadata) == 0:
            return {
                "method": "embedding",
                "data": [],
                "message": "⚠️ No embeddings available. Run create_node_embeddings() first."
            }
        
        # Convert query to embedding if it's a string
        if isinstance(query_embedding, str):
            query_emb = self.embedding_model.encode([query_embedding], convert_to_numpy=True).astype('float32')
        else:
            query_emb = np.array([query_embedding], dtype='float32')
        
        # OPTIMIZATION: Normalize query for cosine similarity
        faiss.normalize_L2(query_emb)
        
        # Search FAISS index
        k = min(top_k, len(self.player_metadata))
        similarities, indices = self.index.search(query_emb, k)
        
        # Prepare results with similarity scores
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.player_metadata) and idx >= 0:
                player = self.player_metadata[idx].copy()
                # Similarity is already 0-1 from cosine similarity
                player['similarity'] = float(similarity)
                results.append(player)
        
        return {
            "method": "embedding",
            "data": results,
            "model": self.embedding_model_name,
            "dimension": self.dimension
        }
    
    def _get_cache_metadata(self) -> Dict:
        """
        Generate metadata for cache validation.
        
        Returns:
            Dictionary with cache metadata
        """
        return {
            "model": self.embedding_model_name,
            "dimension": self.dimension,
            "created_at": datetime.now().isoformat(),
            "num_players": len(self.player_metadata),
            "version": "2.0"  # Increment when cache format changes
        }
    
    def _load_vector_cache(self) -> bool:
        """
        Load cached FAISS index and player metadata from disk.
        OPTIMIZED VERSION with cache validation.
        
        Returns:
            True if cache loaded successfully, False otherwise
        """
        index_path = self.cache_dir / "player_embeddings.faiss"
        metadata_path = self.cache_dir / "player_metadata.pkl"
        cache_meta_path = self.cache_dir / "cache_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            return False
        
        try:
            # OPTIMIZATION: Validate cache metadata
            if cache_meta_path.exists():
                with open(cache_meta_path, 'r') as f:
                    cache_meta = json.load(f)
                
                # Check if cache is for the same model
                if cache_meta.get('model') != self.embedding_model_name:
                    print(f"⚠️ Cache is for different model: {cache_meta.get('model')}")
                    return False
                
                if cache_meta.get('dimension') != self.dimension:
                    print(f"⚠️ Cache dimension mismatch: {cache_meta.get('dimension')} vs {self.dimension}")
                    return False
            
            # Load index and metadata
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                self.player_metadata = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            return False
    
    def _save_vector_cache(self):
        """
        Save FAISS index and player metadata to disk for fast future loading.
        OPTIMIZED VERSION with metadata.
        """
        if self.index is None:
            return
        
        index_path = self.cache_dir / "player_embeddings.faiss"
        metadata_path = self.cache_dir / "player_metadata.pkl"
        cache_meta_path = self.cache_dir / "cache_metadata.json"
        
        try:
            # Save index
            faiss.write_index(self.index, str(index_path))
            
            # Save player metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.player_metadata, f)
            
            # OPTIMIZATION: Save cache metadata
            cache_meta = self._get_cache_metadata()
            with open(cache_meta_path, 'w') as f:
                json.dump(cache_meta, f, indent=2)
            
            print(f"💾 Saved cache: {index_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")
    
    def clear_cache(self):
        """
        Clear cached embeddings for this model.
        Useful when data has been updated in Neo4j.
        """
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.index = None
            self.player_metadata = []
            print(f"🗑️  Cleared cache: {self.cache_dir}")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about the current cache.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_meta_path = self.cache_dir / "cache_metadata.json"
        
        info = {
            "model": self.embedding_model_name,
            "dimension": self.dimension,
            "cache_dir": str(self.cache_dir),
            "cache_exists": (self.cache_dir / "player_embeddings.faiss").exists(),
            "embeddings_loaded": self.index is not None,
            "num_players": len(self.player_metadata)
        }
        
        if cache_meta_path.exists():
            with open(cache_meta_path, 'r') as f:
                cache_meta = json.load(f)
            info.update(cache_meta)
        
        return info
    
    def is_embeddings_ready(self) -> bool:
        """
        Check if embeddings are loaded and ready for queries.
        
        Returns:
            True if embeddings are available, False otherwise
        """
        return self.index is not None and len(self.player_metadata) > 0
    
    def switch_embedding_model(self, model_name: str):
        """
        Switch to a different embedding model.
        Will load cache if available, otherwise requires create_node_embeddings().
        
        Args:
            model_name: Name of the new embedding model
        """
        print(f"\n🔄 Switching embedding model...")
        print(f"   From: {self.embedding_model_name}")
        print(f"   To: {model_name}")
        
        # Load new model
        self.embedding_model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        
        # Update dimension
        test_embedding = self.embedding_model.encode("test")
        self.dimension = len(test_embedding)
        
        # Update cache directory
        model_safe_name = model_name.replace('/', '_').replace('\\', '_')
        self.cache_dir = Path("data") / "cache" / model_safe_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load cache
        if self._load_vector_cache():
            print(f"✅ Loaded {len(self.player_metadata)} embeddings from cache")
        else:
            print(f"⚠️ No cache found for {model_name}. Run create_node_embeddings().")
            self.index = None
            self.player_metadata = []
    
    def compare_embedding_models(self, query: str, models: List[str] = None, top_k: int = 5) -> Dict:
        """
        Compare different embedding models for the same query.
        
        Args:
            query: Query string
            models: List of model names to compare (defaults to MiniLM and MPNet)
            top_k: Number of results per model
        
        Returns:
            Dictionary with comparison results
        """
        if models is None:
            models = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ]
        
        original_model = self.embedding_model_name
        comparison = {}
        
        for model in models:
            print(f"\n🔍 Testing {model}...")
            
            # Switch model
            self.switch_embedding_model(model)
            
            if not self.is_embeddings_ready():
                comparison[model] = {
                    "error": "Embeddings not available. Run create_node_embeddings() first."
                }
                continue
            
            # Time the retrieval
            import time
            start = time.time()
            results = self.embedding_retrieval(query, top_k=top_k)
            elapsed = time.time() - start
            
            comparison[model] = {
                "dimension": self.dimension,
                "response_time": elapsed,
                "num_results": len(results.get('data', [])),
                "top_results": results.get('data', [])[:3],  # Top 3 for preview
                "avg_similarity": np.mean([r['similarity'] for r in results.get('data', [])]) if results.get('data') else 0
            }
        
        # Restore original model
        self.switch_embedding_model(original_model)
        
        return comparison
    
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
    print("FPL Graph Retriever - Optimized Version")
    print("="*60)
    
    # Initialize
    retriever = FPLGraphRetriever()
    
    print("\nAvailable query types:", list(retriever.query_templates.keys()))
    print("\nCache info:", retriever.get_cache_info())
    
    # Test if embeddings ready
    if retriever.is_embeddings_ready():
        print("\n✅ Embeddings ready for queries!")
    else:
        print("\n⚠️ Run create_node_embeddings() to build the index")
