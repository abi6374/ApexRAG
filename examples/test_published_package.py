import asyncio
from apex_rag import ApexIndex

# A sample "document" to ingest
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

async def main():
    print("🚀 Testing ApexRAG library as an end-user...\n")
    
    # 1. Initialize the index (using an in-memory or temporary database for testing)
    async with await ApexIndex.create(
        db_url="sqlite+aiosqlite:///test_user.db", 
        model="llama3.1:8b",
        trace_enabled=True,
        verify_leaves=True,
    ) as index:
        
        print("📥 1. Ingesting sample text...")
        # We can ingest raw text directly instead of a file
        doc_id = await index.ingest_text(
            text=SAMPLE_TEXT,
            doc_id="science_textbook_001",
            synthesize_summaries=True  # Will call Ollama to generate Semantic Maps
        )
        print(f"✅ Document successfully ingested with ID: {doc_id}\n")
        
        print("🔍 2. Running an agentic query...")
        question = "What is the molecular formula of Benzene?"
        print(f"User Query: '{question}'\n")
        
        result = await index.query(question, doc_id)
        
        print("\n--- RESULTS ---")
        if result and result.verified:
            print(f"🎯 Exact Answer Found in Path: {result.path}")
            print(f"Confidence: {result.confidence * 100:.1f}%\n")
            print("Content:")
            print(result.content)
        else:
            print("❌ No verified answer found in the document.")

if __name__ == "__main__":
    asyncio.run(main())
