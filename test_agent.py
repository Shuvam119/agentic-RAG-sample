from src.agent import create_agent


agent = create_agent()

question = input("Ask CloudDesk: ")

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": question}
        ]
    }
)

answer = result["messages"][-1].content

print("\nANSWER")
print("=" * 60)
print(answer)
