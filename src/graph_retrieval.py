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
    ERROR_MESSAGES,
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

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        embedding_model_name: str = None,
    ):
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
        self.uri = uri or neo4j_config["uri"]
        self.username = username or neo4j_config["username"]
        self.password = password or neo4j_config["password"]

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

        # Initialize embedding model using config
        self.embedding_model_name = embedding_model_name or get_embedding_model()
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Use centralized query templates from config
        self.query_templates = CYPHER_QUERY_TEMPLATES

        # FAISS vector store for embeddings (separate from Neo4j)
        # OPTIMIZATION: Get dimension dynamically from model
        test_embedding = self.embedding_model.encode("test")
        self.dimension = len(test_embedding)
        print(
            f"📐 Embedding dimension: {self.dimension}D ({self.embedding_model_name})"
        )

        self.index = None
        self.player_metadata = []

        # OPTIMIZATION: Model-specific cache directory under data/cache
        model_safe_name = self.embedding_model_name.replace("/", "_").replace("\\", "_")
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

    def _execute_query_safely(
        self, query_type: str, params: Dict[str, Any]
    ) -> List[Dict]:
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
                limit = RETRIEVAL_LIMITS.get(query_type, RETRIEVAL_LIMITS["default"])
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
        results = {"method": "baseline", "intent": intent, "data": []}

        # Default season - ensure it's in correct format (YYYY-YY)
        seasons = entities.get("seasons", ["2022-23"])
        season = seasons[0] if seasons else "2022-23"

        # Route to appropriate query based on intent using helper method
        if intent == "player_search" and entities.get("players"):
            params = {"player_name": entities["players"][0]}
            results["data"] = self._execute_query_safely("player_search", params)
            results["query_type"] = "player_search"

        elif intent == "player_performance" and entities.get("players"):
            params = {"player_name": entities["players"][0], "season": season}
            results["data"] = self._execute_query_safely("player_stats", params)
            results["query_type"] = "player_stats"

        elif intent == "recommendation" and entities.get("positions"):
            position = entities["positions"][0]
            params = {"position": position, "season": season}
            results["data"] = self._execute_query_safely(
                "top_players_by_position", params
            )
            results["query_type"] = "top_players_by_position"

        elif (
            intent == "comparison"
            and entities.get("players")
            and len(entities["players"]) >= 2
        ):
            params = {"player_names": entities["players"][:2], "season": season}
            results["data"] = self._execute_query_safely("player_comparison", params)
            results["query_type"] = "player_comparison"

        elif intent == "team_analysis" and entities.get("teams"):
            params = {"team_name": entities["teams"][0], "season": season}
            results["data"] = self.execute_query("team_players", params)
            results["query_type"] = "team_players"

        elif intent == "fixture" and entities.get("gameweeks"):
            gameweek = int(entities["gameweeks"][0])
            params = {"gameweek": gameweek, "season": season}
            results["data"] = self.execute_query("best_performers_gameweek", params)
            results["query_type"] = "best_performers_gameweek"

        else:
            # Default: show top scorers for forwards
            params = {"position": "FWD", "season": season}
            results["data"] = self.execute_query("top_scorers", params)
            results["query_type"] = "top_scorers"

        return results

    def create_node_embeddings(self, batch_size: int = 64, max_players: int = None):
        """
        Create position-aware embeddings for all Player nodes.

        KEY ENHANCEMENT: Position is mentioned MULTIPLE TIMES in text
        to ensure embeddings strongly associate players with their positions.
        """
        print("\n" + "=" * 60)
        print("🔄 CREATING POSITION-AWARE PLAYER EMBEDDINGS")
        print("=" * 60)

        limit_clause = f"LIMIT {max_players}" if max_players else ""

        # Query to get comprehensive player data
        query = f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            
            WITH p,
                collect(DISTINCT pos.name) AS all_positions,
                collect(DISTINCT f.season) AS seasons,
                sum(COALESCE(r.total_points, 0)) AS total_points,
                sum(COALESCE(r.goals_scored, 0)) AS total_goals,
                sum(COALESCE(r.assists, 0)) AS total_assists,
                sum(COALESCE(r.minutes, 0)) AS total_minutes,
                sum(COALESCE(r.clean_sheets, 0)) AS total_clean_sheets,
                sum(COALESCE(r.bonus, 0)) AS total_bonus,
                count(DISTINCT f) AS total_appearances
            
            OPTIONAL MATCH (p)-[:PLAYED_IN]->(f2:Fixture)
            OPTIONAL MATCH (f2)-[:HAS_HOME_TEAM]->(ht:Team)
            OPTIONAL MATCH (f2)-[:HAS_AWAY_TEAM]->(at:Team)
            
            WITH p, all_positions, seasons, total_points, total_goals, total_assists,
                total_minutes, total_clean_sheets, total_bonus, total_appearances,
                collect(DISTINCT COALESCE(ht.name, at.name)) AS teams
            
            RETURN p.player_name AS name,
                all_positions AS positions,
                teams,
                seasons,
                total_points,
                total_goals,
                total_assists,
                total_minutes,
                total_clean_sheets,
                total_bonus,
                total_appearances,
                COALESCE(p.now_cost, 0) AS price
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
            print("❌ No players found")
            return 0

        print(f"✅ Found {len(players)} players")
        print(f"🔧 Creating POSITION-FOCUSED text representations...")
        print()

        # Create POSITION-FOCUSED text representations
        texts = []
        self.player_metadata = []

        for i, player in enumerate(players):
            # Get position with fallback
            positions_list = player.get("positions", [])
            if not positions_list or positions_list == [None]:
                primary_position = "Unknown"
                position_full = "Unknown Position"
            else:
                valid_positions = [p for p in positions_list if p]
                primary_position = valid_positions[0] if valid_positions else "Unknown"

                # Map to full names
                position_map = {
                    "FWD": "Forward",
                    "MID": "Midfielder",
                    "DEF": "Defender",
                    "GKP": "Goalkeeper",
                }
                position_full = position_map.get(primary_position, primary_position)

            # Get team
            teams_list = player.get("teams", [])
            valid_teams = [t for t in teams_list if t] if teams_list else []
            primary_team = valid_teams[0] if valid_teams else "Unknown Team"

            # Get stats
            total_points = player.get("total_points", 0) or 0
            total_goals = player.get("total_goals", 0) or 0
            total_assists = player.get("total_assists", 0) or 0
            clean_sheets = player.get("total_clean_sheets", 0) or 0
            appearances = player.get("total_appearances", 0) or 0

            # CRITICAL: Create text with HEAVY position emphasis
            # Position is mentioned 3-4 times to ensure strong embedding association

            text = f"{player['name']} plays as a {position_full} ({primary_position}). "
            text += f"He is a {primary_team} {position_full}. "

            # Position-specific role description
            if primary_position == "FWD":
                text += f"As a forward and striker, {player['name']} is an attacking player. "
                text += f"Forward {player['name']} scored {total_goals} goals "
            elif primary_position == "MID":
                text += f"As a midfielder, {player['name']} plays in midfield. "
                text += f"Midfielder {player['name']} contributed {total_goals} goals "
            elif primary_position == "DEF":
                text += f"As a defender, {player['name']} plays in defense. "
                text += f"Defender {player['name']} kept {clean_sheets} clean sheets and scored {total_goals} goals "
            elif primary_position == "GKP":
                text += (
                    f"As a goalkeeper, {player['name']} is the {primary_team} keeper. "
                )
                text += f"Goalkeeper {player['name']} kept {clean_sheets} clean sheets "
            else:
                text += f"{player['name']} scored {total_goals} goals "

            text += (
                f"and provided {total_assists} assists in {appearances} appearances. "
            )
            text += f"Total FPL points: {total_points}. "

            # Add performance level with position
            if total_points > 200:
                text += f"Elite {position_full}. "
            elif total_points > 150:
                text += f"Top-performing {position_full}. "
            elif total_points > 100:
                text += f"Solid {position_full}. "
            else:
                text += f"Rotation {position_full}. "

            # Final position emphasis
            text += f"Position: {primary_position} {position_full}."

            texts.append(text)

            # Enhanced metadata
            player_meta = player.copy()
            player_meta["primary_position"] = primary_position
            player_meta["position_full"] = position_full
            player_meta["primary_team"] = primary_team
            player_meta["text_representation"] = text
            self.player_metadata.append(player_meta)

            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(players)} players...")

        print(f"✅ Created {len(texts)} position-aware representations")

        # Example of what the text looks like
        if texts:
            print(f"\n📝 Example representation:")
            print(f"   {texts[0][:200]}...")
        print()

        # Generate embeddings
        print("🔄 Generating embeddings...")
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            embeddings_np = embeddings.astype("float32")
            print(f"✅ Generated {len(embeddings_np)} embeddings")
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return 0

        # Build FAISS index
        print("🏗️  Building FAISS index...")
        try:
            import faiss

            faiss.normalize_L2(embeddings_np)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings_np)
            print(f"✅ FAISS index built: {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error building index: {e}")
            return 0

        # Save cache
        self._save_vector_cache()

        print()
        print("=" * 60)
        print("✅ POSITION-AWARE EMBEDDINGS READY")
        print(f"   Players: {len(self.player_metadata)}")
        print(f"   Model: {self.embedding_model_name}")
        print("=" * 60)
        print()

        return len(players)

    def embedding_retrieval(
        self,
        query_embedding,
        entities: Dict = None,
        top_k: int = 15,
        min_similarity: float = 0.3,
    ) -> dict:
        """
        Retrieve players using embedding similarity.
        Filter by entities extracted from input preprocessing.
        """
        if self.index is None or len(self.player_metadata) == 0:
            return {
                "method": "embedding",
                "data": [],
                "error": True,
                "message": "⚠️ No embeddings available",
            }

        try:
            import faiss
            import numpy as np

            # Get the embedding vector
            if isinstance(query_embedding, list):
                query_emb = np.array([query_embedding], dtype="float32")
            else:
                query_emb = query_embedding

            # Normalize
            faiss.normalize_L2(query_emb)

            # Extract filters from entities (from input preprocessing)
            position_filter = None
            season_filter = None

            if entities:
                # Get position filter
                positions = entities.get("positions", [])
                if positions:
                    position_filter = positions[0]
                    print(f"   📍 Filtering by position: {position_filter}")

                # Get season filter
                seasons = entities.get("seasons", [])
                if seasons:
                    season_filter = seasons[0]
                    print(f"   📅 Filtering by season: {season_filter}")

            # Search for similar embeddings
            search_k = min(top_k * 5, len(self.player_metadata))
            similarities, indices = self.index.search(query_emb, search_k)

            # Filter results based on entities
            results = []
            seen_players = set()

            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity < min_similarity:
                    continue

                if idx < 0 or idx >= len(self.player_metadata):
                    continue

                player = self.player_metadata[idx].copy()

                # Skip duplicates
                player_name = player.get("name", "")
                if player_name in seen_players:
                    continue
                seen_players.add(player_name)

                # Apply position filter (from entities)
                if position_filter:
                    player_pos = player.get("primary_position", "")
                    if player_pos != position_filter:
                        continue

                # Apply season filter (from entities)
                if season_filter:
                    player_season = player.get("season", "")
                    if player_season and player_season != season_filter:
                        continue

                player["similarity"] = float(similarity)
                results.append(player)

                if len(results) >= top_k:
                    break

            # Sort by similarity
            if results:
                results.sort(key=lambda x: x["similarity"], reverse=True)

            print(f"📊 Found {len(results)} players after filtering")
            if results:
                print(
                    f"   Top: {results[0].get('name')} - {results[0].get('total_points')} pts"
                )

            return {
                "method": "embedding",
                "data": results,
                "model": self.embedding_model_name,
                "filters_applied": {
                    "position": position_filter,
                    "season": season_filter,
                },
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            return {
                "method": "embedding",
                "data": [],
                "error": True,
                "message": f"Error: {str(e)}",
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
            "version": "2.0",  # Increment when cache format changes
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
                with open(cache_meta_path, "r") as f:
                    cache_meta = json.load(f)

                # Check if cache is for the same model
                if cache_meta.get("model") != self.embedding_model_name:
                    print(f"⚠️ Cache is for different model: {cache_meta.get('model')}")
                    return False

                if cache_meta.get("dimension") != self.dimension:
                    print(
                        f"⚠️ Cache dimension mismatch: {cache_meta.get('dimension')} vs {self.dimension}"
                    )
                    return False

            # Load index and metadata
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, "rb") as f:
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
            with open(metadata_path, "wb") as f:
                pickle.dump(self.player_metadata, f)

            # OPTIMIZATION: Save cache metadata
            cache_meta = self._get_cache_metadata()
            with open(cache_meta_path, "w") as f:
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
            "num_players": len(self.player_metadata),
        }

        if cache_meta_path.exists():
            with open(cache_meta_path, "r") as f:
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
        model_safe_name = model_name.replace("/", "_").replace("\\", "_")
        self.cache_dir = Path("data") / "cache" / model_safe_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Try to load cache
        if self._load_vector_cache():
            print(f"✅ Loaded {len(self.player_metadata)} embeddings from cache")
        else:
            print(f"⚠️ No cache found for {model_name}. Run create_node_embeddings().")
            self.index = None
            self.player_metadata = []

    def compare_embedding_models(
        self, query: str, models: List[str] = None, top_k: int = 5
    ) -> Dict:
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
                "sentence-transformers/all-mpnet-base-v2",
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
                "num_results": len(results.get("data", [])),
                "top_results": results.get("data", [])[:3],  # Top 3 for preview
                "avg_similarity": (
                    np.mean([r["similarity"] for r in results.get("data", [])])
                    if results.get("data")
                    else 0
                ),
            }

        # Restore original model
        self.switch_embedding_model(original_model)

        return comparison

    def hybrid_retrieval(
        self, intent: str, entities: Dict, query_embedding: List[float]
    ) -> Dict[str, Any]:
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

        return {"baseline": baseline_results, "embedding": embedding_results}


# Example usage
if __name__ == "__main__":
    print("FPL Graph Retriever - Optimized Version")
    print("=" * 60)

    # Initialize
    retriever = FPLGraphRetriever()

    print("\nAvailable query types:", list(retriever.query_templates.keys()))
    print("\nCache info:", retriever.get_cache_info())

    # Test if embeddings ready
    if retriever.is_embeddings_ready():
        print("\n✅ Embeddings ready for queries!")
    else:
        print("\n⚠️ Run create_node_embeddings() to build the index")
