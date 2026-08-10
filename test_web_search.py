from src.web_search_tool import search_web


query = input("Web search: ")

result = search_web.invoke(query)

print("\n")
print(result)
