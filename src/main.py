"""
Main Integration Script for FPL Graph-RAG System
Demonstrates the complete pipeline end-to-end
"""

import os
from dotenv import load_dotenv
from input_preprocessing import FPLInputPreprocessor
from graph_retrieval import FPLGraphRetriever
from llm_layer import FPLLLMLayer
import json

# Load environment variables
load_dotenv()

class FPLGraphRAG:
    """
    Complete Graph-RAG pipeline for FPL
    """
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing FPL Graph-RAG System...")
        
        # Component 1: Input Preprocessing
        self.preprocessor = FPLInputPreprocessor()
        print("✓ Input Preprocessor loaded")
        
        # Component 2: Graph Retrieval
        self.retriever = FPLGraphRetriever(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        print("✓ Graph Retriever connected")
        
        # Component 3: LLM Layer
        self.llm_layer = FPLLLMLayer(
            openai_key=os.getenv("OPENAI_API_KEY"),
            hf_key=os.getenv("HUGGINGFACE_API_KEY")
        )
        print("✓ LLM Layer initialized")
        
        print("\n" + "="*60)
        print("FPL Graph-RAG System Ready!")
        print("="*60 + "\n")
    
    def process_query(self, query: str, retrieval_method: str = "hybrid", model: str = "rule-based-fallback"):
        """
        Process a complete query through the pipeline
        
        Args:
            query: User question
            retrieval_method: 'baseline', 'embedding', or 'hybrid'
            model: LLM model to use
            
        Returns:
            Complete results dictionary
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Preprocessing
        print("STEP 1: Input Preprocessing")
        print("-" * 60)
        preprocessed = self.preprocessor.preprocess(query)
        print(f"Intent: {preprocessed['intent']}")
        print(f"Entities: {json.dumps(preprocessed['entities'], indent=2)}")
        print(f"Embedding created: {len(preprocessed['embedding'])} dimensions\n")
        
        # Step 2: Retrieval
        print("STEP 2: Graph Retrieval")
        print("-" * 60)
        
        baseline_results = None
        embedding_results = None
        
        if retrieval_method in ["baseline", "hybrid"]:
            print("Running Cypher queries...")
            baseline_results = self.retriever.baseline_retrieval(
                preprocessed['intent'],
                preprocessed['entities']
            )
            print(f"Cypher query type: {baseline_results.get('query_type', 'N/A')}")
            print(f"Results found: {len(baseline_results.get('data', []))}")
            if baseline_results.get('data'):
                print(f"Sample result: {baseline_results['data'][0]}")
        
        if retrieval_method in ["embedding", "hybrid"]:
            print("\nRunning embedding-based retrieval...")
            embedding_results = self.retriever.embedding_retrieval(
                preprocessed['embedding'],
                top_k=5
            )
            print(f"Semantic matches found: {len(embedding_results.get('data', []))}")
            if embedding_results.get('data'):
                print(f"Top match: {embedding_results['data'][0]}")
        
        print()
        
        # Step 3: LLM Generation
        print("STEP 3: LLM Generation")
        print("-" * 60)
        print(f"Using model: {model}")
        
        response = self.llm_layer.generate_response(
            query=query,
            baseline_results=baseline_results or {},
            embedding_results=embedding_results,
            model_name=model
        )
        
        print(f"Response generated in {response.get('response_time', 0):.2f}s")
        print()
        
        # Display final answer
        print("FINAL ANSWER:")
        print("=" * 60)
        print(response['answer'])
        print("=" * 60)
        
        return {
            "query": query,
            "preprocessing": preprocessed,
            "baseline_results": baseline_results,
            "embedding_results": embedding_results,
            "llm_response": response
        }
    
    def run_demo(self):
        """
        Run a demo with example questions
        """
        demo_questions = [
            "Who are the top forwards in 2023?",
            "Show me Erling Haaland's stats",
            "Best midfielders under 8 million",
            "Compare Salah and Son performance",
            "Arsenal players with most clean sheets"
        ]
        
        print("\n" + "="*60)
        print("RUNNING DEMO WITH EXAMPLE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n\nDEMO QUESTION {i}/{len(demo_questions)}")
            self.process_query(question, retrieval_method="hybrid", model="rule-based-fallback")
            
            if i < len(demo_questions):
                input("\nPress Enter to continue to next question...")
    
    def run_evaluation(self):
        """
        Run evaluation comparing different models
        """
        test_query = "Who are the top forwards in 2023?"
        
        print("\n" + "="*60)
        print("RUNNING MODEL COMPARISON")
        print("="*60)
        
        models = ["gpt-3.5-turbo", "mistral-7b", "rule-based-fallback"]
        
        # Preprocess once
        preprocessed = self.preprocessor.preprocess(test_query)
        baseline_results = self.retriever.baseline_retrieval(
            preprocessed['intent'],
            preprocessed['entities']
        )
        
        # Compare models
        print(f"\nQuery: {test_query}\n")
        
        comparisons = self.llm_layer.compare_models(
            test_query,
            baseline_results,
            models=models
        )
        
        # Display comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        for model, response in comparisons.items():
            print(f"\n--- {model.upper()} ---")
            print(f"Response time: {response.get('response_time', 'N/A')}")
            print(f"Answer: {response['answer'][:200]}...")
            print()
        
        return comparisons
    
    def close(self):
        """Clean up resources"""
        if self.retriever:
            self.retriever.close()
        print("System shut down successfully")


def main():
    """
    Main execution function
    """
    import sys
    
    # Initialize system
    try:
        system = FPLGraphRAG()
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        print("\nMake sure:")
        print("1. Neo4j is running")
        print("2. .env file is configured")
        print("3. All dependencies are installed")
        return
    
    # Interactive mode
    print("\nFPL Graph-RAG System")
    print("Commands:")
    print("  - Type a question to get an answer")
    print("  - Type 'demo' to run demo questions")
    print("  - Type 'eval' to run model comparison")
    print("  - Type 'quit' to exit")
    print()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                break
            
            elif user_input.lower() == 'demo':
                system.run_demo()
            
            elif user_input.lower() == 'eval':
                system.run_evaluation()
            
            else:
                system.process_query(user_input)
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        
        except Exception as e:
            print(f"\nError: {str(e)}")
    
    # Cleanup
    system.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
