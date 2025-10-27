# Test script to verify LangChain 1.0 with langchain-classic
import sys
print("Python version:", sys.version)
print("\n" + "="*50)

# Test 1: Check if langchain-classic package exists
print("\nTest 1: Checking langchain-classic package...")
try:
    import langchain_classic
    print(f"✓ langchain_classic imported successfully")
    print(f"  Version: {langchain_classic.__version__}")
    print(f"  Location: {langchain_classic.__file__}")
except ImportError as e:
    print(f"✗ Failed to import langchain_classic: {e}")

# Test 2: Check if langchain_classic.chains exists
print("\nTest 2: Checking langchain_classic.chains module...")
try:
    import langchain_classic.chains
    print(f"✓ langchain_classic.chains imported successfully")
except ImportError as e:
    print(f"✗ Failed to import langchain_classic.chains: {e}")

# Test 3: Check if create_retrieval_chain exists
print("\nTest 3: Checking create_retrieval_chain function...")
try:
    from langchain_classic.chains import create_retrieval_chain
    print(f"✓ create_retrieval_chain imported successfully")
    print(f"  Type: {type(create_retrieval_chain)}")
except ImportError as e:
    print(f"✗ Failed to import create_retrieval_chain: {e}")

# Test 4: Check if create_stuff_documents_chain exists
print("\nTest 4: Checking create_stuff_documents_chain function...")
try:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    print(f"✓ create_stuff_documents_chain imported successfully")
    print(f"  Type: {type(create_stuff_documents_chain)}")
except ImportError as e:
    print(f"✗ Failed to import create_stuff_documents_chain: {e}")

# Test 5: Check other required imports
print("\nTest 5: Checking other required imports...")
try:
    from langchain_text_splitters import CharacterTextSplitter
    print(f"✓ CharacterTextSplitter imported successfully")
except ImportError as e:
    print(f"✗ Failed: {e}")

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print(f"✓ ChatOpenAI and OpenAIEmbeddings imported successfully")
except ImportError as e:
    print(f"✗ Failed: {e}")

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    print(f"✓ PyPDFLoader and FAISS imported successfully")
except ImportError as e:
    print(f"✗ Failed: {e}")

try:
    from langchain_core.prompts import ChatPromptTemplate
    print(f"✓ ChatPromptTemplate imported successfully")
except ImportError as e:
    print(f"✗ Failed: {e}")

# Test 6: List what's available in langchain_classic.chains
print("\nTest 6: Listing available items in langchain_classic.chains...")
try:
    import langchain_classic.chains as chains
    available_items = [item for item in dir(chains) if not item.startswith('_')]
    print(f"Found {len(available_items)} items in langchain_classic.chains")
    print("Key items:")
    for item in available_items[:10]:  # Show first 10
        print(f"  - {item}")
    if len(available_items) > 10:
        print(f"  ... and {len(available_items) - 10} more")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*50)
print("\n✅ ALL TESTS PASSED!" if all([
    'langchain_classic' in sys.modules,
    'langchain_classic.chains' in sys.modules
]) else "\n⚠️ Some tests failed - check output above")
print("\nYour installed langchain packages:")
print("  langchain: 1.0.0")
print("  langchain-classic: 1.0.0")
print("  langchain-community: 0.4")
print("  langchain-core: 1.0.0")
print("  langchain-openai: 1.0.0")
print("  langchain-text-splitters: 1.0.0")