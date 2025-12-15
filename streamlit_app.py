"""
Component 4: Streamlit UI for FPL Graph-RAG Assistant
Provides interactive interface for the complete pipeline
"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from input_preprocessing import FPLInputPreprocessor
from graph_retrieval import FPLGraphRetriever
from llm_layer import FPLLLMLayer

# Load environment variables
load_dotenv()


# ========== Token Management (adapted from app.py) ==========
class TokenManager:
    """Responsible for managing and loading API tokens"""
    
    @staticmethod
    def load_token(token_name: str) -> Optional[str]:
        """Load API token from environment variable or config file"""
        # Try environment variable
        token = os.getenv(token_name)
        if token:
            return token
        
        # Try .env file (already loaded by load_dotenv)
        return None
    
    @staticmethod
    def get_neo4j_credentials() -> Tuple[str, str, str]:
        """Get Neo4j connection credentials"""
        return (
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password")
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
        """Initialize graph retriever with error handling"""
        try:
            uri, username, password = TokenManager.get_neo4j_credentials()
            st.session_state.retriever = FPLGraphRetriever(
                uri=uri,
                username=username,
                password=password
            )
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")
            st.session_state.retriever = None
    
    @staticmethod
    def _initialize_llm():
        """Initialize LLM layer"""
        st.session_state.llm_layer = FPLLLMLayer(
            openai_key=TokenManager.load_token("OPENAI_API_KEY"),
            hf_key=TokenManager.load_token("HUGGINGFACE_API_KEY")
        )
    
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
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="FPL Graph-RAG Assistant",
            page_icon="⚽",
            layout="wide"
        )
    
    @staticmethod
    def display_header():
        """Display main header"""
        st.title("⚽ FPL Graph-RAG Assistant")
        st.markdown("""
        This assistant uses a Graph-RAG pipeline to answer Fantasy Premier League questions:
        1. **Input Preprocessing**: Intent classification & entity extraction
        2. **Graph Retrieval**: Cypher queries + embedding-based search
        3. **LLM Generation**: Grounded response using multiple LLMs
        """)


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


# Page configuration
UIConfigurator.configure_page()

# Initialize session state using the manager
SessionStateManager.initialize()

# Display header
UIConfigurator.display_header()

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

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
retrieval_method = st.sidebar.radio(
    "Retrieval Method",
    ["Baseline (Cypher)", "Embedding-based", "Hybrid (Both)"],
    index=2
)

# Example questions
st.sidebar.header("📝 Example Questions")
example_questions = [
    "Who are the top forwards in 2023?",
    "Show me Erling Haaland's stats",
    "Best midfielders under 8 million",
    "Compare Salah and Son performance",
    "Arsenal players with most clean sheets",
    "Top scorers in the league",
    "Best value defenders",
    "Players in good form",
    "Which forwards should I pick?",
    "Show me Man City players"
]

selected_example = st.sidebar.selectbox(
    "Or select an example:",
    [""] + example_questions
)

# Main interface
st.header("💬 Ask Your FPL Question")

# Query input
query = st.text_input(
    "Enter your question:",
    value=selected_example if selected_example else "",
    placeholder="e.g., Who are the top forwards in 2023?"
)

# Process button
if st.button("🔍 Get Answer", type="primary"):
    if not query:
        MessageDisplayer.show_warning("Please enter a question!")
    elif not st.session_state.retriever:
        MessageDisplayer.show_error("Neo4j connection not available. Please check your configuration.")
    else:
        with st.spinner("Processing your question..."):
            try:
                # Create columns for the pipeline visualization
                col1, col2, col3 = st.columns(3)
                
                # Step 1: Input Preprocessing
                with col1:
                    st.subheader("1️⃣ Input Preprocessing")
                    preprocessed = st.session_state.preprocessor.preprocess(query)
                    
                    st.write("**Intent:**", preprocessed['intent'])
                    st.write("**Entities:**")
                    for entity_type, values in preprocessed['entities'].items():
                        if values:
                            st.write(f"- {entity_type}: {', '.join(map(str, values))}")
                    st.write(f"**Embedding Dimension:** {len(preprocessed['embedding'])}")
                
                # Step 2: Graph Retrieval
                with col2:
                    st.subheader("2️⃣ Graph Retrieval")
                    
                    baseline_results = None
                    embedding_results = None
                    
                    if retrieval_method in ["Baseline (Cypher)", "Hybrid (Both)"]:
                        baseline_results = st.session_state.retriever.baseline_retrieval(
                            preprocessed['intent'],
                            preprocessed['entities']
                        )
                        st.write(f"**Cypher Query Type:** {baseline_results.get('query_type', 'N/A')}")
                        st.write(f"**Results Found:** {len(baseline_results.get('data', []))}")
                    
                    if retrieval_method in ["Embedding-based", "Hybrid (Both)"]:
                        embedding_results = st.session_state.retriever.embedding_retrieval(
                            preprocessed['embedding'],
                            top_k=5
                        )
                        st.write(f"**Semantic Matches:** {len(embedding_results.get('data', []))}")
                
                # Step 3: LLM Generation
                with col3:
                    st.subheader("3️⃣ LLM Generation")
                    
                    response = st.session_state.llm_layer.generate_response(
                        query=query,
                        baseline_results=baseline_results or {},
                        embedding_results=embedding_results,
                        model_name=selected_model
                    )
                    
                    st.write(f"**Model:** {response['model']}")
                    if 'response_time' in response:
                        st.write(f"**Response Time:** {response['response_time']:.2f}s")
                    if response.get('error'):
                        MessageDisplayer.show_error("Error generating response")
                
                # Display results in expandable sections
                st.markdown("---")
                st.header("📊 Results")
                
                # Final Answer
                st.subheader("✅ Final Answer")
                if response.get('answer'):
                    st.success(response['answer'])
                else:
                    MessageDisplayer.show_warning("No answer generated")
                
                # Knowledge Graph Context
                with st.expander("🔍 View Knowledge Graph Context"):
                    st.markdown("### Retrieved Information from Neo4j")
                    
                    if baseline_results and baseline_results.get('data'):
                        st.markdown("#### Cypher Query Results")
                        st.json(baseline_results['data'][:5])
                    
                    if embedding_results and embedding_results.get('data'):
                        st.markdown("#### Embedding-Based Results")
                        st.json(embedding_results['data'][:5])
                
                # Full Prompt
                with st.expander("📝 View Full Prompt"):
                    st.code(response.get('prompt', 'N/A'), language="text")
                
                # Model Comparison (optional)
                if st.checkbox("🔬 Compare Multiple Models"):
                    with st.spinner("Comparing models..."):
                        comparisons = st.session_state.llm_layer.compare_models(
                            query,
                            baseline_results or {},
                            embedding_results,
                            models=["gpt-3.5-turbo", "mistral-7b", "rule-based-fallback"]
                        )
                        
                        for model_name, model_response in comparisons.items():
                            st.markdown(f"### {model_name}")
                            st.write(model_response.get('answer', 'N/A'))
                            if 'response_time' in model_response:
                                st.caption(f"Response time: {model_response['response_time']:.2f}s")
                            st.markdown("---")
            
            except Exception as e:
                MessageDisplayer.show_error(f"Error processing query: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
### 🎯 About This System
This Graph-RAG pipeline demonstrates:
- **Symbolic Reasoning**: Structured Cypher queries on Neo4j Knowledge Graph
- **Statistical Reasoning**: Embedding-based semantic search
- **LLM Grounding**: Responses based only on retrieved facts

**Built for Milestone 3 - CSEN 903**
""")

# Display system stats
with st.expander("📈 System Statistics"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cypher Query Templates", "12")
    
    with col2:
        st.metric("Embedding Models", "2")
    
    with col3:
        st.metric("LLM Models", "3+")
