from duckduckgo_search import DDGS

def search_duckduckgo(query, max_results=10):
    """
    Search DuckDuckGo using DDGS and filter results for trusted sources.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                keywords=query,
                region="wt-wt",  # World-wide
                safesearch="moderate",
                timelimit="y",  # Last 1year
                backend="html",  # Use HTML mode
                max_results=max_results,
            )

            filtered_results = []
            for result in results:
                title = result.get("title", "")
                snippet = result.get("body", "")
                link = result.get("href", "")
                
                # Filtering: Select only reliable sources such as Wikipedia, News, and academic materials
                if "wikipedia.org" in link or "news" in link or "scholar.google.com" in link:
                    filtered_results.append({"title": title, "snippet": snippet, "link": link})
            
            return filtered_results
    except Exception as e:
        print(f"❌ Error during DuckDuckGo search: {e}")
        return []
    
