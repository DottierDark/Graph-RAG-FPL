#!/usr/bin/env python3
"""
Embedding Management Script
===========================
Comprehensive tool for managing embeddings in FPL Graph-RAG system.

Features:
- Create embeddings with progress tracking
- Compare different embedding models
- Check cache status
- Clear and rebuild cache
- Benchmark performance

Usage:
    python manage_embeddings.py create          # Create embeddings
    python manage_embeddings.py status          # Check cache status
    python manage_embeddings.py compare         # Compare models
    python manage_embeddings.py clear           # Clear cache
    python manage_embeddings.py benchmark       # Run benchmarks
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import optimized retriever
try:
    from .graph_retrieval_optimized import FPLGraphRetriever
except ImportError:
    print("⚠️ Using standard graph_retrieval.py")
    from .graph_retrieval import FPLGraphRetriever


class EmbeddingManager:
    """Manages embedding operations for FPL Graph-RAG"""
    
    def __init__(self):
        print("🔧 Initializing Embedding Manager...")
        self.retriever = FPLGraphRetriever(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        print("✅ Connected to Neo4j")
    
    def create_embeddings(self, batch_size=64, max_players=None):
        """Create embeddings for all players (default: ALL ~1600 players)"""
        print("\n" + "="*60)
        print("CREATE EMBEDDINGS FOR FULL DATASET")
        print("="*60)
        print(f"Processing: {'ALL ~1600 players' if not max_players else f'{max_players} players'}")
        print(f"Batch size: {batch_size} (optimized for large dataset)")
        print()
        
        count = self.retriever.create_node_embeddings(
            batch_size=batch_size,
            max_players=max_players
        )
        
        if count > 0:
            print(f"\n✅ Successfully created {count} embeddings")
        else:
            print("\n❌ No embeddings created. Check Neo4j data.")
        
        return count
    
    def check_status(self):
        """Check embedding cache status"""
        print("\n" + "="*60)
        print("EMBEDDING STATUS")
        print("="*60)
        
        if hasattr(self.retriever, 'get_cache_info'):
            info = self.retriever.get_cache_info()
            
            print(f"\n📊 Cache Information:")
            print(f"   Model: {info.get('model', 'N/A')}")
            print(f"   Dimension: {info.get('dimension', 'N/A')}D")
            print(f"   Cache Directory: {info.get('cache_dir', 'N/A')}")
            print(f"   Cache Exists: {'✅' if info.get('cache_exists') else '❌'}")
            print(f"   Embeddings Loaded: {'✅' if info.get('embeddings_loaded') else '❌'}")
            print(f"   Number of Players: {info.get('num_players', 0)}")
            
            if 'created_at' in info:
                print(f"   Created At: {info['created_at']}")
        else:
            print("\n⚠️ Using basic retriever without cache info")
            print(f"   Embeddings Ready: {'✅' if self.retriever.is_embeddings_ready() else '❌'}")
        
        print()
    
    def compare_models(self, query="top forwards", top_k=5):
        """Compare different embedding models"""
        print("\n" + "="*60)
        print("COMPARE EMBEDDING MODELS")
        print("="*60)
        print(f"Query: {query}")
        print()
        
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # 384D, fast
            "sentence-transformers/all-mpnet-base-v2",  # 768D, better quality
        ]
        
        if hasattr(self.retriever, 'compare_embedding_models'):
            comparison = self.retriever.compare_embedding_models(query, models, top_k)
            
            print("\n📊 Results:")
            print("-" * 60)
            
            for model, results in comparison.items():
                model_name = model.split('/')[-1]
                print(f"\n{model_name}:")
                
                if 'error' in results:
                    print(f"  ❌ {results['error']}")
                    continue
                
                print(f"  Dimension: {results['dimension']}D")
                print(f"  Response Time: {results['response_time']:.3f}s")
                print(f"  Results Found: {results['num_results']}")
                print(f"  Avg Similarity: {results['avg_similarity']:.3f}")
                
                if results['top_results']:
                    print(f"  Top Result: {results['top_results'][0]['name']} ({results['top_results'][0]['similarity']:.3f})")
            
            print()
        else:
            print("⚠️ Comparison not available with basic retriever")
    
    def clear_cache(self):
        """Clear embedding cache"""
        print("\n" + "="*60)
        print("CLEAR CACHE")
        print("="*60)
        
        response = input("\n⚠️ This will delete all cached embeddings. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled")
            return
        
        if hasattr(self.retriever, 'clear_cache'):
            self.retriever.clear_cache()
            print("✅ Cache cleared successfully")
        else:
            # Manual cleanup for basic retriever
            cache_dir = Path("data/cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"✅ Removed {cache_dir}")
            else:
                print("⚠️ No cache directory found")
        
        print()
    
    def benchmark(self, num_queries=10):
        """Benchmark embedding retrieval performance"""
        print("\n" + "="*60)
        print("BENCHMARK EMBEDDINGS")
        print("="*60)
        
        if not self.retriever.is_embeddings_ready():
            print("❌ Embeddings not ready. Run 'create' first.")
            return
        
        test_queries = [
            "top forwards",
            "best midfielders",
            "goalkeepers with clean sheets",
            "Man City players",
            "Arsenal attackers",
            "Liverpool midfielders",
            "highest scoring defenders",
            "value for money players",
            "consistent performers",
            "form players"
        ][:num_queries]
        
        print(f"\n🏃 Running {len(test_queries)} test queries...")
        print()
        
        times = []
        for i, query in enumerate(test_queries, 1):
            start = time.time()
            results = self.retriever.embedding_retrieval(query, top_k=5)
            elapsed = time.time() - start
            times.append(elapsed)
            
            num_results = len(results.get('data', []))
            print(f"  {i}. '{query}': {elapsed:.3f}s ({num_results} results)")
        
        print()
        print("📊 Statistics:")
        print(f"   Total Time: {sum(times):.3f}s")
        print(f"   Average: {sum(times)/len(times):.3f}s")
        print(f"   Min: {min(times):.3f}s")
        print(f"   Max: {max(times):.3f}s")
        print()


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Embedding Management Script")
        print("="*60)
        print("\nUsage:")
        print("  python manage_embeddings.py <command>")
        print("\nCommands:")
        print("  create    - Create embeddings for all players")
        print("  status    - Check cache status")
        print("  compare   - Compare different embedding models")
        print("  clear     - Clear embedding cache")
        print("  benchmark - Run performance benchmarks")
        print("\nExamples:")
        print("  python manage_embeddings.py create")
        print("  python manage_embeddings.py status")
        print("  python manage_embeddings.py compare")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    try:
        manager = EmbeddingManager()
        
        if command == "create":
            manager.create_embeddings()
        
        elif command == "status":
            manager.check_status()
        
        elif command == "compare":
            manager.compare_models()
        
        elif command == "clear":
            manager.clear_cache()
        
        elif command == "benchmark":
            manager.benchmark()
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Run without arguments to see available commands")
            sys.exit(1)
        
        manager.retriever.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
