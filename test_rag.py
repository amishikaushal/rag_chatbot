import os
from src.rag import get_rag_chain

def test_pipeline():
    print("--- Initializing RAG Chain ---")
    try:
        # 1. Load the chain
        chain = get_rag_chain()
        print("✅ Chain loaded successfully.")

        # 2. Define a test query
        # Hint: Ask something specific to the documents you ingested into FAISS
        query = "What is the main topic of the uploaded documents?"
        
        print(f"\n--- Sending Query: {query} ---")
        
        # 3. Run the chain
        # RetrievalQA returns a dict with 'query', 'result', and 'source_documents'
        response = chain.invoke({"query": query})

        # 4. Print Results
        print("\n--- LLM Response ---")
        print(response["result"])

        print("\n--- Retrieved Sources ---")
        for i, doc in enumerate(response["source_documents"]):
            print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
            # print(f"Content snippet: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        # This helps debug if it's an API error or a code error
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()