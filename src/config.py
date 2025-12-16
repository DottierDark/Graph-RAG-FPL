"""
Central Configuration File for FPL Graph-RAG System
===================================================
This file contains all constants, default parameters, and configuration settings
used across the entire system. Centralizing these values follows DRY principles
and makes the system easier to maintain and configure.

Design Decision: All configurable parameters are in one place rather than
scattered across multiple files, making it easier to tune the system.
"""

# ==================== DATABASE CONFIGURATION ====================
# Neo4j connection settings - default values for local development
NEO4J_CONFIG = {
    "uri": "neo4j+s://81e3f773.databases.neo4j.io",  # Neo4j Bolt connection
    "username": "neo4j",              # Default username
    "password": "password",           # Change in .env file
    "database": "neo4j"               # Default database name
}


# ==================== MODEL CONFIGURATION ====================
# Embedding models for semantic similarity
# Design Decision: Using smaller models for faster inference
EMBEDDING_MODELS = {
    "small": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims, fast
    "large": "sentence-transformers/all-mpnet-base-v2",  # 768 dims, better quality
    "default": "sentence-transformers/all-MiniLM-L6-v2"
}

# LLM model configurations
# Design Decision: Support multiple providers for flexibility
LLM_MODELS = {
    "openai": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.3,      # Low temperature for factual responses
        "max_tokens": 500,       # Limit response length
        "provider": "openai"
    },
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "temperature": 0.3,
        "max_tokens": 500,
        "provider": "huggingface"
    },
    "gemma": {
        "model_name": "google/gemma-2-2b-it",
        "temperature": 0.3,
        "max_tokens": 500,
        "provider": "huggingface"
    },
    "llama": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "temperature": 0.3,
        "max_tokens": 500,
        "provider": "huggingface",
        "requires_auth": True  # Requires accepting license on HuggingFace
    },
    "phi": {
        "model_name": "microsoft/phi-2",
        "temperature": 0.3,
        "max_tokens": 500,
        "provider": "huggingface"
    }
}

# Default LLM to use when no API keys provided
DEFAULT_LLM = "rule-based-fallback"


# ==================== RETRIEVAL CONFIGURATION ====================
# Number of results to return for different query types
# Design Decision: Increased limits to better utilize the full dataset (1600 players, 52k+ relations)
RETRIEVAL_LIMITS = {
    "player_search": 20,         # Player name searches (show more matches)
    "top_players": 20,            # Top performers lists (more comprehensive)
    "team_players": 30,           # All players in a team (teams have many players)
    "comparison": 10,             # Player comparisons (show more context)
    "fixtures": 15,               # Upcoming fixtures
    "embedding_search": 15,       # Semantic similarity results (leverage full index)
    "default": 20,                # Default increased for large dataset
    "top_players_by_position": 20,
    "top_scorers": 20,
    "best_performers_gameweek": 20,
    # New queries
    "total_gameweeks": 1,
    "total_players": 1,
    "teams_only_in_2021_22": 5,
    "max_points_2022_23": 1,
    "elneny_matches_participated": 1,
    "rule_query": 50,
}

# Similarity thresholds for embedding-based retrieval
# Design Decision: Adjusted thresholds after query expansion improvements
SIMILARITY_THRESHOLDS = {
    "high_confidence": 0.75,   # Very similar (increased from 0.8)
    "medium_confidence": 0.55,  # Somewhat similar (decreased from 0.6)
    "low_confidence": 0.40,     # Minimum acceptable (decreased from 0.5)
    "default": 0.50            # Balanced threshold (decreased from 0.6)
}

# Cypher Query Templates (adapted for Milestone 2 schema)
# Design Decision: Centralized queries to maintain consistency
CYPHER_QUERY_TEMPLATES = {
    "player_search": """
        MATCH (p:Player)
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, collect(DISTINCT pos.name) AS positions
        RETURN p.player_name AS name, 
            positions AS positions
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
    """,
    "player_stats": """
        MATCH (p:Player)
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        WITH p
        ORDER BY size(p.player_name) ASC
        LIMIT 1
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
    """,
    "player_comparison": """
        UNWIND $player_names AS search_name
        MATCH (p:Player)
        WHERE toLower(p.player_name) CONTAINS toLower(search_name)
        WITH search_name, p
        ORDER BY size(p.player_name) ASC
        WITH search_name, collect(p)[0] AS matched_player
        WITH collect(matched_player) AS players
        UNWIND players AS p
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
    """,
    "players_by_price": """
        MATCH (p:Player)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        WHERE f.season = $season AND p.now_cost <= $max_price
        WITH p, sum(r.total_points) AS total_points
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, total_points, collect(DISTINCT pos.name) AS positions
        WHERE p.now_cost > 0
        RETURN p.player_name AS name, 
               positions,
               p.now_cost AS price,
               total_points,
               (toFloat(total_points) / p.now_cost) AS value
        ORDER BY total_points DESC
    """,
    "form_players": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE f.season = $season
        WITH p, f, r
        ORDER BY f.event DESC
        WITH p, avg(r.total_points) AS recent_form
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, recent_form, collect(DISTINCT pos.name) AS positions
        RETURN p.player_name AS name,
               positions,
               recent_form
        ORDER BY recent_form DESC
    """,
    "fixtures": """
        MATCH (f:Fixture)
        WHERE f.season = $season AND f.event = $gameweek
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        RETURN f.event AS gameweek,
               home.name AS home_team,
               away.name AS away_team,
               f.team_h_score AS home_score,
               f.team_a_score AS away_score,
               f.kickoff_time AS kickoff
        ORDER BY f.kickoff_time
    """,
    "best_performers_gameweek": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        MATCH (f)-[:IN_GAMEWEEK]->(g:Gameweek {gameweek_number: $gameweek})
        WHERE f.season = $season
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, r, collect(DISTINCT pos.name) AS positions
        RETURN p.player_name AS name,
               positions,
               r.total_points AS points,
               r.goals_scored AS goals,
               r.assists AS assists
        ORDER BY r.total_points DESC
    """,
    # New queries from M2 assignments
    "total_gameweeks": """
        MATCH (g:Gameweek)
        RETURN count(g) AS total_gameweeks
    """,
    "total_players": """
        MATCH (p:Player)
        RETURN count(p) AS total_players
    """,
    "teams_only_in_2021_22": """
        MATCH (f1:Fixture {season: '2021-22'})-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
        WITH collect(DISTINCT t.name) AS teams_2021_22
        MATCH (f2:Fixture {season: '2022-23'})-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t2:Team)
        WITH teams_2021_22, collect(DISTINCT t2.name) AS teams_2022_23
        UNWIND teams_2021_22 AS team
        WITH team, teams_2022_23
        WHERE NOT team IN teams_2022_23
        WITH team
        ORDER BY team
        RETURN collect(team) AS teams_only_in_2021_22
    """,
    "max_points_2022_23": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: '2022-23'})
        WITH p, r.total_points AS points
        ORDER BY points DESC
        RETURN points AS max_points, p.player_name AS player
        LIMIT 1
    """,
    "elneny_matches_participated": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        WHERE p.player_name CONTAINS 'Elneny' OR p.player_name CONTAINS 'ElNeny' OR p.player_name = 'Mohamed Elneny'
        WITH p, r, f
        WHERE r.minutes > 0
        RETURN count(DISTINCT f) AS matches_participated
    """,
    "rule_query": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WITH p, collect(pos.name) AS positions
        WHERE size(positions) = 2 AND 'DEF' IN positions
        UNWIND positions AS position
        WITH p, positions, position
        WHERE position <> 'DEF'
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, positions, position, f, r,
          CASE WHEN r.minutes > 0 AND r.minutes < 60 THEN 1 ELSE 0 END +
          CASE WHEN r.minutes >= 60 THEN 2 ELSE 0 END +
          (r.assists * 3) +
          (r.penalties_missed * -2) +
          (r.own_goals * -2) +
          (r.yellow_cards * -1) +
          (r.red_cards * -3) +
          (r.goals_scored * 6) +
          CASE WHEN r.minutes >= 60 AND r.clean_sheets > 0 THEN 4 ELSE 0 END +
          CASE WHEN r.goals_conceded >= 2 THEN toInteger(r.goals_conceded / 2) * -1 ELSE 0 END
          AS def_score,
          r.total_points AS actual_points
        WITH p, positions, position, f, r, def_score, actual_points,
          CASE WHEN def_score = actual_points THEN true ELSE false END AS matches_def
        WHERE matches_def = false
        WITH p, positions, position, 
          count(f) AS not_def_fixtures,
          collect({
            def_score: def_score,
            season: f.season,
            fixture_number: f.fixture_number,
            matches_def: matches_def,
            total_points: actual_points
          }) AS fixture_details
        MATCH (p)-[:PLAYED_IN]->(all_fixtures:Fixture)
        WITH p, positions, position, not_def_fixtures, fixture_details, count(DISTINCT all_fixtures) AS total_fixtures
        RETURN 
          p.player_name AS player,
          not_def_fixtures,
          total_fixtures,
          fixture_details
        ORDER BY p.player_name
    """,
}


# ==================== INPUT PROCESSING CONFIGURATION ====================
# Intent types supported by the system with keyword patterns
# Design Decision: 7 intents cover most FPL query scenarios
INTENT_TYPES = {
    "player_search": ["player", "find player", "show me", "who is", "about"],
    "player_performance": ["stats", "points", "performance", "how many", "scored", "statistics"],
    "team_analysis": ["team", "squad", "lineup", "formation"],
    "recommendation": ["recommend", "suggest", "best", "top", "should i pick", "transfer"],
    "comparison": ["compare", "vs", "versus", "better", "difference between"],
    "fixture": ["fixture", "next game", "upcoming", "schedule", "playing against"],
    "price": ["price", "cost", "value", "budget", "cheap", "expensive"],
    "rule_query": ["two positions", "dual position", "multiple positions", "plays as defender"],
    "database_stats": ["how many", "total", "count", "which teams", "what teams", "teams were", "teams only"]
}

# Entity types to extract from queries
ENTITY_TYPES = [
    "players",  # Player names
    "teams",  # Team names
    "positions",  # FWD, MID, DEF, GKP
    "seasons",  # 2021-22, 2022-23, etc.
    "gameweeks",  # GW1, GW38, etc.
    "statistics",  # goals, assists, clean_sheets, etc.
]

# Position mappings - handle various forms
# Design Decision: Normalize user input to standard codes
POSITION_MAPPINGS = {
    "forward": "FWD",
    "forwards": "FWD",
    "striker": "FWD",
    "strikers": "FWD",
    "fwd": "FWD",
    "midfielder": "MID",
    "midfielders": "MID",
    "mid": "MID",
    "defender": "DEF",
    "defenders": "DEF",
    "def": "DEF",
    "goalkeeper": "GKP",
    "goalkeepers": "GKP",
    "gkp": "GKP",
    "keeper": "GKP",
    "keepers": "GKP"
}

# Statistics keywords mapping
# Design Decision: Map user terms to database field names
STAT_KEYWORDS = {
    "goals": ["goals", "goals_scored", "goal"],
    "assists": ["assists", "assist"],
    "points": ["points", "total_points", "score"],
    "price": ["price", "cost", "now_cost", "value"],
    "clean sheets": ["clean sheets", "clean_sheets", "cleansheets", "cs"],
    "saves": ["saves", "save"],
    "bonus": ["bonus", "bonus_points"],
    "form": ["form"],
    "minutes": ["minutes", "mins", "played"]
}

# FPL Teams (2023-24 season)
# Design Decision: Centralized team list for entity extraction
FPL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
    "Luton", "Man City", "Man Utd", "Newcastle", "Nottm Forest",
    "Sheffield Utd", "Tottenham", "West Ham", "Wolves", "Burnley"
]


# ==================== UI CONFIGURATION ====================
# Streamlit page settings
STREAMLIT_CONFIG = {
    "page_title": "FPL Graph-RAG Assistant",
    "page_icon": "⚽",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "description": """
This assistant uses a Graph-RAG pipeline to answer Fantasy Premier League questions:
1. **Input Preprocessing**: Intent classification & entity extraction
2. **Graph Retrieval**: Cypher queries + embedding-based search
3. **LLM Generation**: Grounded response using multiple LLMs
"""
}

# Example queries for UI
EXAMPLE_QUERIES = [
    "Who are the top forwards in 2022-23?",
    "Show me Erling Haaland's stats",
    "Best midfielders under 8 million",
    # "Compare Salah and Son performance",
    # "Arsenal players with most clean sheets",
    "Top scorers in gameweek 10",
    "Which defenders have the best value?",
    # "Liverpool's upcoming fixtures",
    "How many gameweeks were played during the seasons?",
    "How many player nodes were created?",
    "What teams were in 2021-22 but not 2022-23?",
    "What was the biggest total_point value in 2022-23?",
    "How many matches did Mohamed Elneny participate in?",
    "Find players with two positions where one is DEF"
]


# ==================== PROMPT ENGINEERING ====================
# LLM persona for FPL queries
# Design Decision: Strict persona prevents hallucination
DEFAULT_PERSONA = """You are an expert Fantasy Premier League (FPL) assistant. 
You provide accurate, data-driven advice based on player statistics and performance.
You only use information from the provided context and never make up statistics.
If you don't have enough information, you clearly state that."""

# Prompt template structure
# Design Decision: Structured prompts improve response quality
PROMPT_TEMPLATE = """### PERSONA
{persona}

### CONTEXT (FPL Knowledge Graph Data)
{context}

### TASK
Answer the following user question using ONLY the information provided in the context above.

CRITICAL RULES:
3. NEVER make up statistics, player names, or any data not present in the context.
4. Only cite statistics that appear in the context above.

Be concise and specific, citing relevant statistics from the context.

### USER QUESTION
{query}

### ANSWER
"""


# ==================== ERROR HANDLING ====================
# Retry settings for API calls
# Design Decision: Implement exponential backoff for resilience
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,      # seconds
    "max_delay": 10.0,       # seconds
    "exponential_base": 2
}

# Timeout settings (seconds)
TIMEOUT_CONFIG = {
    "neo4j_query": 30,      # Cypher query timeout
    "llm_api": 60,          # LLM API timeout
    "embedding": 10,         # Embedding generation timeout
    "ui_response": 30       # UI response timeout
}

# Error messages
ERROR_MESSAGES = {
    "neo4j_connection_failed": "Failed to connect to Neo4j. Ensure the database is running and credentials are correct.",
    "no_results": "No results found in the knowledge graph for this query.",
    "query_not_found": "Query type not found in templates",
    "openai_key_missing": "OpenAI API key not configured. Using fallback mode.",
    "hf_key_missing": "HuggingFace client not initialized. Please provide HF_TOKEN.",
    "hf_init_failed": "Could not initialize HuggingFace client",
    "llm_call_failed": "Error calling LLM API",
    "embedding_error": "Error generating embeddings. Using baseline retrieval only.",
    "invalid_query": "Could not understand the query. Please rephrase."
}


# ==================== PERFORMANCE SETTINGS ====================
# Caching settings
# Design Decision: Cache embeddings to improve performance
CACHE_CONFIG = {
    "enable_embedding_cache": True,
    "cache_size_mb": 100,
    "cache_ttl_seconds": 3600  # 1 hour
}

# Batch processing settings
BATCH_CONFIG = {
    "embedding_batch_size": 32,
    "query_batch_size": 10
}


# ==================== VALIDATION RULES ====================
# Input validation rules
# Design Decision: Validate inputs early to prevent errors
VALIDATION_RULES = {
    "max_query_length": 500,        # characters
    "min_query_length": 3,          # characters
    "max_player_names": 5,          # in comparison queries
    "valid_seasons": ["2021-22", "2022-23", "2023-24"],
    "valid_gameweeks": range(1, 39)  # GW1 to GW38
}


# ==================== LOGGING CONFIGURATION ====================
# Logging settings
# Design Decision: Detailed logs for debugging, info logs for production
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "fpl_graph_rag.log",
    "enable_file_logging": False  # Set to True for production
}


# ==================== FEATURE FLAGS ====================
# Enable/disable features
# Design Decision: Feature flags for gradual rollout and testing
FEATURE_FLAGS = {
    "enable_hybrid_retrieval": True,
    "enable_embedding_search": True,
    "enable_llm_caching": False,  # Not implemented yet
    "enable_query_suggestions": True,
    "enable_model_comparison": True,
    "enable_visualization": True
}


# ==================== EXPORT ====================
# Export commonly used config groups
def get_neo4j_config():
    """Get Neo4j configuration with environment variable overrides"""
    import os
    return {
        "uri": os.getenv("NEO4J_URI", NEO4J_CONFIG["uri"]),
        "username": os.getenv("NEO4J_USERNAME", NEO4J_CONFIG["username"]),
        "password": os.getenv("NEO4J_PASSWORD", NEO4J_CONFIG["password"]),
        "database": os.getenv("NEO4J_DATABASE", NEO4J_CONFIG["database"])
    }


def get_llm_config(model_name):
    """
    Get LLM configuration for a specific model
    
    Args:
        model_name: Name of the LLM model
        
    Returns:
        dict: Model configuration
    """
    return LLM_MODELS.get(model_name, LLM_MODELS["openai"])


def get_embedding_model(size="default"):
    """
    Get embedding model name
    
    Args:
        size: "small", "large", or "default"
        
    Returns:
        str: Model name
    """
    return EMBEDDING_MODELS.get(size, EMBEDDING_MODELS["default"])


def get_available_queries():
    """
    Get list of available Cypher queries with descriptions.
    
    Returns:
        List of dicts with query info
    """
    query_descriptions = {
        "player_search": "Search for players by name",
        "top_players_by_position": "Get top performers by position",
        "player_stats": "Get detailed player statistics",
        "team_players": "Get all players from a team",
        "player_comparison": "Compare multiple players",
        "gameweek_performance": "Get player performance in specific gameweek",
        "top_scorers": "Get top goal scorers by position",
        "clean_sheet_leaders": "Get defenders/goalkeepers with most clean sheets",
        "players_by_price": "Find players within price range",
        "form_players": "Get players in best recent form",
        "fixtures": "Get fixtures for a gameweek",
        "best_performers_gameweek": "Get best performers in a gameweek"
    }
    
    return [
        {"name": key, "description": query_descriptions.get(key, "Query description")}
        for key in CYPHER_QUERY_TEMPLATES.keys()
    ]


def get_query_count():
    """
    Get total number of available Cypher queries.
    
    Returns:
        int: Number of queries
    """
    return len(CYPHER_QUERY_TEMPLATES)


def get_available_queries():
    """
    Get list of all available Cypher query templates
    
    Returns:
        list: List of query template names with descriptions
    """
    query_descriptions = {
        "player_search": "Search for players by name",
        "top_players_by_position": "Get top players filtered by position",
        "player_stats": "Get detailed statistics for a specific player",
        "team_players": "Get all players from a specific team",
        "player_comparison": "Compare statistics between multiple players",
        "gameweek_performance": "Get player performance for a specific gameweek",
        "top_scorers": "Get top goal scorers by position",
        "clean_sheet_leaders": "Get defenders/goalkeepers with most clean sheets",
        "players_by_price": "Find players within a price range",
        "form_players": "Get players with best recent form",
        "fixtures": "Get fixtures for a specific gameweek",
        "best_performers_gameweek": "Get best performing players in a gameweek"
    }
    
    return [
        {"name": name, "description": query_descriptions.get(name, "No description")}
        for name in CYPHER_QUERY_TEMPLATES.keys()
    ]


def get_query_count():
    """
    Get total number of available Cypher queries
    
    Returns:
        int: Number of query templates
    """
    return len(CYPHER_QUERY_TEMPLATES)


# Configuration summary for debugging
def print_config_summary():
    """Print configuration summary - useful for debugging"""
    print("=" * 60)
    print("FPL GRAPH-RAG CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Default Embedding Model: {EMBEDDING_MODELS['default']}")
    print(f"Default LLM: {DEFAULT_LLM}")
    print(f"Supported Intents: {len(INTENT_TYPES)}")
    print(f"Supported Entities: {len(ENTITY_TYPES)}")
    print(f"Cypher Templates: {get_query_count()}")
    print(f"Feature Flags: {sum(FEATURE_FLAGS.values())}/{len(FEATURE_FLAGS)} enabled")
    print("=" * 60)
    print("\nAvailable Cypher Queries:")
    for query in get_available_queries():
        print(f"  - {query['name']}: {query['description']}")
    print("=" * 60)


if __name__ == "__main__":
    # Test configuration loading
    print_config_summary()
    print("\nNeo4j Config:", get_neo4j_config())
    print("\nGPT-3.5 Config:", get_llm_config("openai"))
    print("\nDefault Embedding Model:", get_embedding_model())
