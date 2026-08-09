from src.vectorstore import get_vectorstore


vectorstore = get_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

question = input("Ask a question: ")

results = retriever.invoke(question)

print(f"\nRetrieved {len(results)} chunks:\n")

for i, document in enumerate(results, 1):
    print("=" * 60)
    print(f"RESULT {i}")
    print("=" * 60)
    for key, value in document.metadata.items():
        print(f"{key}: {value}")
    print("-" * 60)
    print(document.page_content[:1000])
