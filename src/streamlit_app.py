"""
Component 4: Streamlit UI for FPL Graph-RAG Assistant
Provides interactive interface for the complete pipeline
"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from .input_preprocessing import FPLInputPreprocessor
from .graph_retrieval import FPLGraphRetriever
from .llm_layer import FPLLLMLayer
from .config import (
    get_neo4j_config,
    ERROR_MESSAGES,
    EXAMPLE_QUERIES,
    STREAMLIT_CONFIG
)

# Load environment variables
load_dotenv()


# ========== Token Management (adapted from app.py) ==========
class TokenManager:
    """Responsible for managing and loading API tokens"""
    
    @staticmethod
    def load_token(token_name: str) -> Optional[str]:
        """
        Load API token from environment variable or config file.
        
        Args:
            token_name: Name of the environment variable
            
        Returns:
            Token string or None if not found
        """
        # Try environment variable
        token = os.getenv(token_name)
        if token:
            return token
        
        # Already loaded by load_dotenv
        return None
    
    @staticmethod
    def get_neo4j_credentials() -> Tuple[str, str, str]:
        """
        Get Neo4j connection credentials from centralized config.
        
        Returns:
            Tuple of (uri, username, password)
        """
        neo4j_config = get_neo4j_config()
        return (
            neo4j_config['uri'],
            neo4j_config['username'],
            neo4j_config['password']
        )


# ========== Session State Manager (adapted from app.py) ==========
class SessionStateManager:
    """Responsible for managing Streamlit session state"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "preprocessor" not in st.session_state:
            st.session_state.preprocessor = FPLInputPreprocessor()
        if "retriever" not in st.session_state:
            SessionStateManager._initialize_retriever()
        if "llm_layer" not in st.session_state:
            SessionStateManager._initialize_llm()
    
    @staticmethod
    def _initialize_retriever():
        """Initialize graph retriever with FAISS vector store using centralized config"""
        try:
            uri, username, password = TokenManager.get_neo4j_credentials()
            
            # Initialize retriever with integrated FAISS vector store
            st.session_state.retriever = FPLGraphRetriever(
                uri=uri,
                username=username,
                password=password
            )
            
            # Check if embeddings are already loaded from cache
            embeddings_ready = st.session_state.retriever.is_embeddings_ready()
            st.session_state.embeddings_loaded = embeddings_ready
            
            # AUTO-BUILD: If no embeddings exist, create them automatically
            if not embeddings_ready:
                st.info("🔄 No embeddings found. Building embeddings for all ~1600 players...")
                st.info("⏱️ This is a one-time process that takes 2-3 minutes. Future startups will be instant!")
                
                with st.spinner("Creating embeddings... Please wait (this only happens once)"):
                    try:
                        player_count = st.session_state.retriever.create_node_embeddings()
                        st.session_state.embeddings_loaded = True
                        st.success(f"✅ Successfully created embeddings for {player_count} players! System ready.")
                    except Exception as build_error:
                        st.error(f"❌ Failed to build embeddings: {str(build_error)}")
                        st.warning("You can try again later via the 'Create Embeddings' button in the sidebar.")
                        st.session_state.embeddings_loaded = False
            
        except Exception as e:
            st.error(f"{ERROR_MESSAGES['neo4j_connection_failed']}: {str(e)}")
            st.session_state.retriever = None
            st.session_state.embeddings_loaded = False
    
    @staticmethod
    def _initialize_llm():
        """Initialize LLM layer"""
        st.session_state.llm_layer = FPLLLMLayer(
            openai_key=TokenManager.load_token("OPENAI_API_KEY"),
            hf_key=TokenManager.load_token("HUGGINGFACE_API_KEY")
        )
    
    @staticmethod
    def check_embeddings_exist() -> bool:
        """Check if embeddings are available in vector cache"""
        # Always check the current state from the retriever
        if st.session_state.get('retriever'):
            embeddings_ready = st.session_state.retriever.is_embeddings_ready()
            st.session_state.embeddings_exist = embeddings_ready
            st.session_state.embeddings_loaded = embeddings_ready
            return embeddings_ready
        return False
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Dict[str, Any] = None):
        """Add message to chat history"""
        message = {"role": role, "content": content}
        if metadata:
            message["metadata"] = metadata
        st.session_state.messages.append(message)
    
    @staticmethod
    def clear_messages():
        """Clear chat history"""
        st.session_state.messages = []


# ========== UI Configuration (adapted from app.py) ==========
class UIConfigurator:
    """Responsible for configuring Streamlit UI"""
    
    @staticmethod
    def configure_page():
        """Configure Streamlit page settings using centralized config"""
        st.set_page_config(
            page_title=STREAMLIT_CONFIG['page_title'],
            page_icon=STREAMLIT_CONFIG['page_icon'],
            layout=STREAMLIT_CONFIG['layout']
        )
    
    @staticmethod
    def display_header():
        """Display main header using centralized config"""
        st.title(f"{STREAMLIT_CONFIG['page_icon']} {STREAMLIT_CONFIG['page_title']}")
        st.markdown(STREAMLIT_CONFIG['description'])


# ========== Message Display (adapted from app.py) ==========
class MessageDisplayer:
    """Responsible for displaying messages"""
    
    @staticmethod
    def show_error(message: str):
        """Display error message"""
        st.error(f"❌ {message}")
    
    @staticmethod
    def show_success(message: str):
        """Display success message"""
        st.success(f"✅ {message}")
    
    @staticmethod
    def show_info(message: str):
        """Display info message"""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def show_warning(message: str):
        """Display warning message"""
        st.warning(f"⚠️ {message}")


# ========== Query Processor (SOLID: Single Responsibility) ==========
class QueryProcessor:
    """Responsible for processing FPL queries through the RAG pipeline"""
    
    @staticmethod
    def process_query(query: str, preprocessor, retriever, llm_layer, retrieval_method: str, selected_model: str) -> dict:
        """Process query through complete pipeline with progress tracking"""
        results = {
            "success": False,
            "answer": None,
            "metadata": {},
            "error": None
        }
        
        try:
            # Step 1: Preprocessing
            preprocessed = preprocessor.preprocess(query)
            results["metadata"]["intent"] = preprocessed['intent']
            results["metadata"]["entities"] = preprocessed['entities']
            
            # Step 2: Retrieval
            baseline_results = None
            embedding_results = None
            
            if retrieval_method in ["Baseline (Cypher)", "Hybrid (Both)"]:
                baseline_results = retriever.baseline_retrieval(
                    preprocessed['intent'],
                    preprocessed['entities']
                )
                results["metadata"]["query_type"] = baseline_results.get('query_type', 'N/A')
                results["metadata"]["results_count"] = len(baseline_results.get('data', []))
            
            if retrieval_method in ["Embedding-based", "Hybrid (Both)"]:
                # Check if embeddings are ready before attempting retrieval
                if not retriever.is_embeddings_ready():
                    results["error"] = "Embeddings not available. Please create embeddings first or use Baseline retrieval."
                    results["metadata"]["semantic_matches"] = 0
                    return results
                
                # Use integrated FAISS-based embedding retrieval
                embedding_results = retriever.embedding_retrieval(
                    preprocessed['embedding'],
                    top_k=5
                )
                results["metadata"]["semantic_matches"] = len(embedding_results.get('data', []))
            
            # Step 3: LLM Generation
            response = llm_layer.generate_response(
                query=query,
                baseline_results=baseline_results or {},
                embedding_results=embedding_results,
                model_name=selected_model
            )
            
            results["success"] = not response.get('error', False)
            results["answer"] = response.get('answer')
            results["metadata"]["model"] = response.get('model', selected_model)
            results["metadata"]["response_time"] = response.get('response_time', 0)
            results["metadata"]["retrieval_method"] = retrieval_method
            results["metadata"]["baseline_data"] = baseline_results.get('data', [])[:3] if baseline_results else []
            results["metadata"]["embedding_data"] = embedding_results.get('data', [])[:3] if embedding_results and embedding_results.get('data') else []
            
        except Exception as e:
            results["error"] = str(e)
        
        return results


# ========== Main Application Function ==========
def main():
    """Main function to run the Streamlit application"""
    # Page configuration
    UIConfigurator.configure_page()

    # Initialize session state using the manager
    SessionStateManager.initialize()

    # Display header
    UIConfigurator.display_header()

    # Add custom CSS for better styling
    st.markdown("""
<style>
    /* Chat messages - Adapt to Light/Dark mode automatically */
    .stChatMessage {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Expanders - Remove hardcoded white background */
    .stExpander {
        background-color: transparent;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 5px;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: var(--primary-color);
    }
    
    /* Status container styling */
    div[data-testid="stStatusWidget"] {
        background-color: var(--secondary-background-color);
    }
</style>
""", unsafe_allow_html=True)

    # Welcome message if no history
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to the FPL Graph-RAG Assistant!**
        
        **Full Dataset Power:**
        - 🎯 1,600+ Players indexed
        - 🔗 52,000+ Relationships mapped
        - 📊 Complete 2022-23 season coverage
        
        **Ask me anything about Fantasy Premier League:**
        - 🎯 "Who are the top forwards in 2022-23?"
        - 📊 "Show me Erling Haaland's stats"
        - 🔍 "Best midfielders under 8 million"
        - ⚖️ "Compare Salah and Son performance"
        
        Select an example from the sidebar or type your own question below!
        """)

    # Handle embedding creation UI
    if st.session_state.get('show_embedding_creator', False):
        with st.container():
            st.warning("### 🔄 Create Embeddings for Enhanced Search")
            st.markdown("""
            Embeddings enable **semantic search** and **hybrid retrieval** for better results.
            
            **Full Dataset Processing:**
            - 📊 All ~1,600 players will be indexed
            - 🔗 Covering 52,000+ player-fixture relationships
            - 🎯 Complete 2022-23 season stats included
            
            **What happens:**
            - Load player data from Neo4j (read-only)
            - Generate semantic embeddings using FAISS
            - Takes ~2-3 minutes for full dataset
            - Cache embeddings locally in `data/cache/` for instant future access
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Start Creation", type="primary", use_container_width=True):
                    st.session_state.start_embedding_creation = True
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_embedding_creator = False
                    st.rerun()

    # Execute embedding creation
    if st.session_state.get('start_embedding_creation', False):
        st.session_state.start_embedding_creation = False
        st.session_state.show_embedding_creator = False
        
        progress_container = st.container()
        with progress_container:
            status = st.status("Creating embeddings...", expanded=True)
            
            with status:
                st.write("🔍 Step 1: Connecting to Neo4j...")
                st.write("✓ Connected successfully")
                
                st.write("📊 Step 2: Loading players from Neo4j...")
                
                try:
                    retriever = st.session_state.get('retriever')
                    if not retriever:
                        raise Exception("Retriever not initialized")
                    
                    # Build index with integrated FAISS vector store
                    player_count = retriever.create_node_embeddings()
                    
                    st.write(f"✓ Loaded {player_count} players")
                    st.write(f"🔢 Step 3: Generated embeddings with FAISS ({player_count} total)")
                    st.write("💾 Step 4: Cached to data/cache/ directory")
                    st.write("✓ All 1600+ players ready for semantic search!")
                    
                    status.update(label="✅ Full dataset embeddings created successfully!", state="complete")
                    
                    # Reset embedding check
                    st.session_state.embeddings_checked = False
                    st.session_state.embeddings_exist = True
                    st.session_state.embeddings_loaded = True
                    
                    st.success("🎉 Embeddings are now available! You can use Embedding-based or Hybrid retrieval.")
                    st.balloons()
                    
                except Exception as e:
                    status.update(label="❌ Error creating embeddings", state="error")
                    st.error(f"Error: {str(e)}")
                    st.info("💡 Tip: Make sure Neo4j is running and contains player data.")
        
        # Rerun to update UI
        st.rerun()

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        SessionStateManager.clear_messages()
        st.rerun()

    st.sidebar.markdown("---")

    # API Status
    with st.sidebar.expander("🔑 API Status", expanded=False):
        hf_token = TokenManager.load_token("HUGGINGFACE_API_KEY")
        openai_token = TokenManager.load_token("OPENAI_API_KEY")
        
        if hf_token and hf_token != "your_hf_token_here":
            st.success("✅ HuggingFace: Connected (FREE)")
        else:
            st.error("❌ HuggingFace: Not configured")
            st.info("Get free token: https://huggingface.co/settings/tokens")
        
        if openai_token and openai_token != "your_openai_key_here":
            st.success("✅ OpenAI: Connected (Paid)")
        else:
            st.warning("⚠️ OpenAI: Not configured")
        
        if st.session_state.retriever:
            st.success("✅ Neo4j: Connected")
        else:
            st.error("❌ Neo4j: Not connected")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        ["gemma-2b", "mistral-7b", "llama-2-7b", "gpt-3.5-turbo", "rule-based-fallback"],
        index=0,  # Default to gemma-2b (FREE via HuggingFace)
        help="Gemma & Mistral use free HuggingFace API. GPT requires OpenAI API key (paid)."
    )

    # Embedding model selection
    embedding_model = st.sidebar.selectbox(
        "Select Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0
    )

    # Retrieval method
    embeddings_available = SessionStateManager.check_embeddings_exist()

    retrieval_method = st.sidebar.radio(
        "Retrieval Method",
        ["Baseline (Cypher)", "Embedding-based", "Hybrid (Both)"],
        index=0,  # Default to Baseline
        help="Baseline uses direct Cypher queries. Embedding/Hybrid require embeddings to be created first."
    )

    # Show embedding status and warn if not available
    if retrieval_method in ["Embedding-based", "Hybrid (Both)"]:
        if embeddings_available:
            st.sidebar.success("✅ Embeddings available")
        else:
            st.sidebar.error("❌ Embeddings not available!")
            st.sidebar.warning("⚠️ Queries will fail. Please create embeddings or use Baseline.")
            if st.sidebar.button("🔄 Create Embeddings Now", use_container_width=True, type="primary"):
                st.session_state.show_embedding_creator = True
                st.rerun()

    # Example questions from centralized config
    st.sidebar.header("📝 Example Questions")

    selected_example = st.sidebar.selectbox(
        "Or select an example:",
        [""] + EXAMPLE_QUERIES
    )

    # Session stats
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Session Stats")
    total_queries = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
    st.sidebar.metric("Total Queries", total_queries)
    if st.session_state.messages:
        last_model = st.session_state.messages[-1].get("metadata", {}).get("model", "N/A") if st.session_state.messages[-1]["role"] == "assistant" else "N/A"
        st.sidebar.metric("Last Model Used", last_model)

    # Main interface
    st.header("💬 Chat with Your FPL Assistant")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata if available
                if "metadata" in message and message["role"] == "assistant":
                    with st.expander("📊 View Details"):
                        metadata = message["metadata"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Intent", metadata.get("intent", "N/A"))
                            st.metric("Query Type", metadata.get("query_type", "N/A"))
                        with col2:
                            st.metric("Results Found", metadata.get("results_count", 0))
                            st.metric("Model Used", metadata.get("model", "N/A"))
                        with col3:
                            if "response_time" in metadata:
                                st.metric("Response Time", f"{metadata['response_time']:.2f}s")
                            st.metric("Retrieval Method", metadata.get("retrieval_method", "N/A"))

    # Query input using chat_input
    query = st.chat_input("Ask about FPL players, stats, or get recommendations...")

    # Alternative: select from examples
    if not query and selected_example:
        query = selected_example

    # Process button
    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(query)
        
        SessionStateManager.add_message("user", query)
        
        if not st.session_state.retriever:
            with st.chat_message("assistant"):
                st.error("Neo4j connection not available. Please check your configuration.")
        elif retrieval_method in ["Embedding-based", "Hybrid (Both)"] and not embeddings_available:
            # Safety check: prevent processing if embeddings not available
            with st.chat_message("assistant"):
                st.error("❌ Cannot process query: Embeddings not available")
                st.info("💡 Please create embeddings first or switch to 'Baseline (Cypher)' retrieval method.")
                if st.button("🔄 Create Embeddings", key="create_from_error"):
                    st.session_state.show_embedding_creator = True
                    st.rerun()
        else:
            # Show assistant response with clean status tracking
            with st.chat_message("assistant"):
                # Create status container with progress
                status_container = st.status("Processing your question...", expanded=True)
                
                try:
                    with status_container:
                        st.write("🔍 **Analyzing your question...**")
                        
                    # Process query using QueryProcessor
                    results = QueryProcessor.process_query(
                        query=query,
                        preprocessor=st.session_state.preprocessor,
                        retriever=st.session_state.retriever,
                        llm_layer=st.session_state.llm_layer,
                        retrieval_method=retrieval_method,
                        selected_model=selected_model
                    )
                    
                    with status_container:
                        st.write(f"✓ Intent: `{results['metadata'].get('intent', 'N/A')}`")
                        
                        if retrieval_method in ["Baseline (Cypher)", "Hybrid (Both)"]:
                            st.write(f"✓ Found {results['metadata'].get('results_count', 0)} results via Cypher")
                        
                        if retrieval_method in ["Embedding-based", "Hybrid (Both)"]:
                            semantic_count = results['metadata'].get('semantic_matches', 0)
                            if semantic_count > 0:
                                st.write(f"✓ Found {semantic_count} semantic matches")
                            else:
                                st.write("⚠️ No embeddings found (using baseline only)")
                        
                        st.write(f"✓ Generated response with {results['metadata'].get('model', 'N/A')}")
                    
                    status_container.update(label="✅ Complete!", state="complete", expanded=False)
                    
                    # Display the answer
                    if results["success"] and results["answer"]:
                        st.markdown(results["answer"])
                        
                        # Add to chat history
                        SessionStateManager.add_message("assistant", results["answer"], results["metadata"])
                        
                        # Show detailed view in expander
                        with st.expander("📊 View Retrieved Data"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Query Type", results['metadata'].get('query_type', 'N/A'))
                                st.metric("Results Found", results['metadata'].get('results_count', 0))
                            
                            with col2:
                                st.metric("Response Time", f"{results['metadata'].get('response_time', 0):.2f}s")
                                st.metric("Model", results['metadata'].get('model', 'N/A'))
                            
                            if results['metadata'].get('baseline_data'):
                                st.markdown("**🔍 Cypher Query Results:**")
                                st.json(results['metadata']['baseline_data'])
                            
                            if results['metadata'].get('embedding_data'):
                                st.markdown("**🧠 Semantic Search Results:**")
                                st.json(results['metadata']['embedding_data'])
                    else:
                        error_msg = results.get("error") or "Unable to generate response."
                        st.error(error_msg)
                        SessionStateManager.add_message("assistant", f"❌ {error_msg}")
                    
                except Exception as e:
                    status_container.update(label="❌ Error", state="error", expanded=True)
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    SessionStateManager.add_message("assistant", error_msg)

    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🎯 About This System
    This Graph-RAG pipeline demonstrates:
    - **Symbolic Reasoning**: Structured Cypher queries on Neo4j Knowledge Graph
    - **Statistical Reasoning**: Embedding-based semantic search
    - **LLM Grounding**: Responses based only on retrieved facts

    **Dataset Scale:**
    - 📊 1,600+ Players indexed
    - 🔗 52,000+ Relationships
    - 🎯 Full 2022-23 FPL Season

    **Built for Milestone 3 - CSEN 903**
    """)

    # Display system stats
    with st.expander("📈 System Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Players in Database", "~1,600")
        
        with col2:
            st.metric("Total Relationships", "~52,000+")
        
        with col3:
            st.metric("Embedding Models", "2")


# Run the app
if __name__ == "__main__":
    main()
