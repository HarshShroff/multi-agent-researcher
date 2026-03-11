import json
import time
import arxiv
import wikipedia
from ddgs import DDGS

def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search ArXiv for academic papers related to the query.

    Queries the ArXiv preprint repository for scholarly papers matching the given
    topic or keywords. Returns structured metadata including titles, abstracts,
    publication dates, author lists, and direct URLs. Use this tool to find
    peer-reviewed or preprint academic research on scientific and technical topics.

    Args:
        query: A search string describing the topic or keywords to look up on ArXiv.
        max_results: The maximum number of papers to return (default 5, max 20).

    Returns:
        A JSON string containing a list of up to 5 papers, each with fields:
        title, summary (first 500 chars), url, published (YYYY-MM-DD),
        authors (first 3), and type='arxiv'. Returns an error JSON on failure.
    """
    try:
        search = arxiv.Search(query=query, max_results=max_results)
        papers = []
        for result in search.results():
            papers.append({
                "title": result.title,
                "summary": result.summary[:500],
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
                "authors": [a.name for a in result.authors[:3]],
                "type": "arxiv"
            })
        return json.dumps(papers)
    except Exception as e:
        return json.dumps({"error": f"ArXiv search failed: {str(e)}", "query": query})


def search_wikipedia(topic: str) -> str:
    """Retrieve a Wikipedia summary for a given topic.

    Fetches a concise encyclopedic overview of the topic from Wikipedia, providing
    general background knowledge, definitions, and context. Use this tool to get
    foundational information before diving into specialized academic literature.

    Args:
        topic: The subject or concept to look up on Wikipedia.

    Returns:
        A plain-text Wikipedia summary of approximately 7 sentences. Returns an
        error message string if the topic is not found or disambiguation is needed.
    """
    try:
        summary = wikipedia.summary(topic, sentences=7, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=7, auto_suggest=True)
            return summary
        except Exception as inner_e:
            return json.dumps({"error": f"Wikipedia disambiguation failed: {str(inner_e)}", "topic": topic})
    except Exception as e:
        return json.dumps({"error": f"Wikipedia search failed: {str(e)}", "topic": topic})


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for current information about the query using DuckDuckGo.

    Performs a real-time web search to find recent news, blog posts, product pages,
    and other online content related to the topic. Use this tool to discover the
    latest developments, market information, or practical applications that may not
    yet appear in academic databases.

    Args:
        query: A search string with keywords describing the topic or question.
        max_results: The maximum number of web results to return (default 5, max 20).

    Returns:
        A JSON string containing a list of up to 5 web results, each with fields:
        title, body (snippet), and href (URL). Returns an error JSON on failure.
    """
    try:
        time.sleep(1)  
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)}", "query": query})


RESEARCH_TOOLS = [search_arxiv, search_wikipedia, search_web]
