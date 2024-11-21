from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from dotenv import load_dotenv
import os
import wikipediaapi
from functools import lru_cache
from langchain.schema import Document

load_dotenv()
upstage_api_key = os.environ['UPSTAGE_API_KEY']
user_agent = os.environ['USER_AGENT']
MAX_LENGTH = 5000  # Limit for text length to be processed by the model

def truncate_text(text, max_length=MAX_LENGTH):
    """Truncate text to a specified maximum length."""
    return text[:max_length] if len(text) > max_length else text

@lru_cache(maxsize=100)
def fetch_wikipedia_snippet(query, lang='en'):
    """
    Fetch a snippet from Wikipedia based on the query and summarize it using ChatUpstage.
    
    Parameters:
        query (str): The query to search in Wikipedia.
        lang (str): The language code for Wikipedia (default: 'en').
    
    Returns:
        str: A summarized text of the Wikipedia content if the page exists, otherwise None.
    """
    # Initialize Wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, lang)
    page = wiki_wiki.page(query)

    # Check if the Wikipedia page exists
    if page.exists():
        full_text = page.text
        print(f"✅ Wikipedia page fetched for '{query}'")

        # Truncate text to avoid exceeding model limits
        shortened_text = truncate_text(full_text)

        # Initialize ChatUpstage for summarization
        model = ChatUpstage(api_key=os.environ['UPSTAGE_API_KEY'])

        # Prepare the summarization prompt
        prompt = f"""
        Summarize the following article in a concise and clear way:

        {shortened_text}
        """
        
        try:
            # Generate summary using ChatUpstage
            response = model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            return None
    else:
        print(f"❌ Wikipedia page not found for '{query}'")
        return None
    

#result = fetch_wikipedia_snippet("math")
#print(result)
