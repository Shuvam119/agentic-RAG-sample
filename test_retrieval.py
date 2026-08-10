import sys

from src.vectorstore import retrieve


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


question = input("Ask a question: ")

results = retrieve(question, k=5)

print(f"\nRetrieved {len(results)} chunks:\n")

for i, document in enumerate(results, 1):
    print("=" * 60)
    print(f"RESULT {i}")
    print("=" * 60)
    for key, value in document["metadata"].items():
        print(f"{key}: {value}")
    print("-" * 60)
    print(document["text"][:1000])
