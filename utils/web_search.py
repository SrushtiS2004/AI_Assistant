from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 3) -> str:
    """
    Searches the web using DuckDuckGo Search API and returns a summarized result.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        
        if not results:
            return "No relevant web search results found."
            
        summarized_results = []
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Content')
            href = res.get('href', '#')
            summarized_results.append(f"Result {i+1}: {title}\n{body}\nSource: {href}")
            
        return "\n\n".join(summarized_results)
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"

if __name__ == "__main__":
    print(search_web("What is Streamlit?"))
