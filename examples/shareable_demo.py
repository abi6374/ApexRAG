import asyncio
import subprocess
import sys

# --- 1. Auto-Install Dependencies ---
try:
    import apex_rag
    print(f"✅ Found apex_rag version: {apex_rag.__version__}")
except ImportError:
    print("📦 apex_rag is not installed. Installing it from PyPI now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "apex-rag"])
    import apex_rag
    print("✅ Installation complete!")

# Now we can safely import it
from apex_rag import ApexIndex

# --- 2. The Document to Test ---
SAMPLE_TEXT = """
# Physics 101

## Classical Mechanics
Newton's second law states that the force acting on an object is equal to the mass of that object times its acceleration (F = ma).

## Thermodynamics
The first law of thermodynamics is a version of the law of conservation of energy. It states that energy can neither be created nor destroyed.

# Chemistry 101

## Organic Chemistry
Benzene is an organic chemical compound with the molecular formula C6H6. It is a cyclic hydrocarbon with a continuous pi bond.
"""

# --- 3. The Test Script ---
async def main():
    print("\n🚀 Starting ApexRAG Local-First Engine...")

    try:
        # Initialize the index (using an in-memory SQLite database)
        async with await ApexIndex.create(
            db_url="sqlite+aiosqlite:///demo_index.db",
            model="llama3.1:8b",
            trace_enabled=True,
            verify_leaves=True,
        ) as index:

            print("\n📥 Step 1: Ingesting sample structural document...")
            doc_id = await index.ingest_text(
                text=SAMPLE_TEXT,
                doc_id="science_textbook_demo",
                synthesize_summaries=True  # Calls Ollama to generate summaries
            )
            print(f"✅ Document successfully ingested (ID: {doc_id})\n")

            print("🔍 Step 2: Running an agentic query...")
            question = "What is the molecular formula of Benzene?"
            print(f"User Query: '{question}'\n")

            result = await index.query(question, doc_id)

            print("\n" + "="*40)
            print("RESULTS")
            print("="*40)

            if result and result.verified:
                print(f"🎯 Exact Answer Found in Path: {result.path}")
                print(f"Confidence: {result.confidence * 100:.1f}%\n")
                print("Extracted Content:")
                print(result.content)
            else:
                print("❌ No verified answer found in the document.")

    except ConnectionError:
        print("\n⚠️  OLLAMA CONNECTION ERROR  ⚠️")
        print("ApexRAG requires a local LLM to run. Please make sure you have Ollama installed and running!")
        print("Download at: https://ollama.com")
        print("Run: `ollama serve` and `ollama pull llama3.1` in your terminal.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
