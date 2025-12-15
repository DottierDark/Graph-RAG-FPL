"""
Quick Test Script - Run this to verify everything works
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)
    
    packages = [
        ('neo4j', 'Neo4j driver'),
        ('streamlit', 'Streamlit'),
        ('sentence_transformers', 'SentenceTransformers'),
        ('dotenv', 'python-dotenv'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} - OK")
        except ImportError:
            print(f"✗ {name} - MISSING")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed!\n")
    return True

def test_env_file():
    """Test if .env file exists"""
    print("=" * 60)
    print("Testing Environment Configuration...")
    print("=" * 60)
    
    if not os.path.exists('.env'):
        print("✗ .env file NOT FOUND")
        print("Create it by copying: cp .env.example .env")
        print("Then edit with your Neo4j credentials")
        return False
    
    print("✓ .env file exists")
    
    # Load and check
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask password
            if 'PASSWORD' in var:
                display_value = '***'
            else:
                display_value = value
            print(f"✓ {var} = {display_value}")
        else:
            print(f"✗ {var} = NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n⚠️  Missing variables: {', '.join(missing)}")
        return False
    
    print("\n✓ Environment configured!\n")
    return True

def test_neo4j_connection():
    """Test Neo4j connection"""
    print("=" * 60)
    print("Testing Neo4j Connection...")
    print("=" * 60)
    
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        load_dotenv()
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'password')
        
        print(f"Connecting to: {uri}")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test query
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record['test'] == 1:
                print("✓ Connection successful!")
                
                # Check for Player nodes
                result = session.run("MATCH (p:Player) RETURN count(p) as count")
                count = result.single()['count']
                print(f"✓ Found {count} Player nodes")
                
                if count == 0:
                    print("\n⚠️  WARNING: No Player nodes found!")
                    print("You need to add FPL data to Neo4j")
                    print("See ACTION_PLAN_URGENT.md - Hour 3")
                    driver.close()
                    return False
                
                # Sample a player
                result = session.run("MATCH (p:Player) RETURN p LIMIT 1")
                player = result.single()
                if player:
                    print(f"✓ Sample player: {dict(player['p'])}")
        
        driver.close()
        print("\n✓ Neo4j ready!\n")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Is Neo4j running? Check http://localhost:7474")
        print("2. Are credentials correct in .env?")
        print("3. Is the URI correct? (bolt://localhost:7687)")
        return False

def test_components():
    """Test if component files exist"""
    print("=" * 60)
    print("Testing Component Files...")
    print("=" * 60)
    
    files = [
        'input_preprocessing.py',
        'graph_retrieval.py',
        'llm_layer.py',
        'streamlit_app.py',
        'main.py'
    ]
    
    missing = []
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print("Make sure all files are in the current directory")
        return False
    
    print("\n✓ All component files present!\n")
    return True

def test_component_imports():
    """Test if components can be imported"""
    print("=" * 60)
    print("Testing Component Imports...")
    print("=" * 60)
    
    try:
        from input_preprocessing import FPLInputPreprocessor
        print("✓ Input Preprocessing - OK")
    except Exception as e:
        print(f"✗ Input Preprocessing - ERROR: {e}")
        return False
    
    try:
        from graph_retrieval import FPLGraphRetriever
        print("✓ Graph Retrieval - OK")
    except Exception as e:
        print(f"✗ Graph Retrieval - ERROR: {e}")
        return False
    
    try:
        from llm_layer import FPLLLMLayer
        print("✓ LLM Layer - OK")
    except Exception as e:
        print(f"✗ LLM Layer - ERROR: {e}")
        return False
    
    print("\n✓ All components can be imported!\n")
    return True

def test_simple_query():
    """Test a simple end-to-end query"""
    print("=" * 60)
    print("Testing Simple Query...")
    print("=" * 60)
    
    try:
        from input_preprocessing import FPLInputPreprocessor
        
        preprocessor = FPLInputPreprocessor()
        query = "Who are the top forwards?"
        
        print(f"Query: {query}")
        result = preprocessor.preprocess(query)
        
        print(f"✓ Intent: {result['intent']}")
        print(f"✓ Entities: {result['entities']}")
        print(f"✓ Embedding: {len(result['embedding'])} dimensions")
        
        print("\n✓ Preprocessing works!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("FPL GRAPH-RAG SYSTEM - QUICK TEST")
    print("=" * 60 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment Config", test_env_file),
        ("Component Files", test_components),
        ("Component Imports", test_component_imports),
        ("Neo4j Connection", test_neo4j_connection),
        ("Simple Query", test_simple_query),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} - EXCEPTION: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is ready to use!")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Test with example questions")
        print("3. Prepare your demo")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Fix the issues above before proceeding")
        print("See ACTION_PLAN_URGENT.md for help")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
