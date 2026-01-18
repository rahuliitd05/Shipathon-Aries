"""
Multi-Source Evidence Scraper for Fact-Checking
Supports: Wikipedia, DuckDuckGo, Google Fact Check API, News Sources
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
import spacy
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import subprocess
import sys

# Try to import duckduckgo_search, fallback gracefully
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️ duckduckgo-search not installed. DuckDuckGo search disabled.")

# Load spaCy model - auto-download if not available
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("📥 Downloading spaCy model (en_core_web_sm)...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model downloaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not download spaCy model: {e}")
        try:
            import en_core_web_sm
            nlp = en_core_web_sm.load()
        except:
        nlp = None
        print("⚠️ spaCy model not found. Entity extraction disabled.")


@dataclass
class EvidenceSource:
    name: str
    reliability_score: float  # 0-1 scale
    source_type: str  # "encyclopedia", "news", "factcheck", "general"


# Source reliability ratings
SOURCES = {
    "wikipedia": EvidenceSource("Wikipedia", 0.85, "encyclopedia"),
    "google_factcheck": EvidenceSource("Google Fact Check", 0.95, "factcheck"),
    "reuters": EvidenceSource("Reuters", 0.90, "news"),
    "ap_news": EvidenceSource("AP News", 0.90, "news"),
    "snopes": EvidenceSource("Snopes", 0.88, "factcheck"),
    "politifact": EvidenceSource("PolitiFact", 0.88, "factcheck"),
    "duckduckgo": EvidenceSource("DuckDuckGo", 0.60, "general"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_entities(text: str) -> List[Dict]:
    """
    Extract named entities from text using spaCy
    """
    if nlp is None:
        # Fallback: extract quoted terms and capitalized words
        entities = []
        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        for q in quoted:
            entities.append({"text": q, "label": "GENERIC"})
        # Find capitalized words (potential proper nouns)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2 and word.lower() not in ['the', 'is', 'was', 'are', 'were', 'has', 'have', 'had']:
                entities.append({"text": word.strip('.,!?'), "label": "GENERIC"})
        return entities[:5] if entities else [{"text": text, "label": "GENERIC"}]
    
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    return entities


def scrape_wikipedia(query: str, max_paragraphs: int = 10) -> List[Dict]:
    """
    Scrape Wikipedia for evidence
    """
    evidence = []
    
    # Try direct page first
    encoded = quote_plus(query.replace(" ", "_"))
    url = f"https://en.wikipedia.org/wiki/{encoded}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        
        if response.status_code == 404:
            # Try Wikipedia search API
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}&format=json&srlimit=3"
            search_resp = requests.get(search_url, headers=HEADERS, timeout=5)
            if search_resp.status_code == 200:
                data = search_resp.json()
                if data.get('query', {}).get('search'):
                    first_result = data['query']['search'][0]
                    title = first_result['title'].replace(" ", "_")
                    url = f"https://en.wikipedia.org/wiki/{quote_plus(title)}"
                    response = requests.get(url, headers=HEADERS, timeout=8)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get article title
            title_tag = soup.find('h1', {'id': 'firstHeading'})
            title = title_tag.text if title_tag else query
            
            # Get main content paragraphs
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                for p in content_div.find_all('p', recursive=True)[:max_paragraphs]:
                    text = p.get_text().strip()
                    # Filter out short or reference-only paragraphs
                    if len(text) > 80 and not text.startswith('['):
                        # Clean up citation markers
                        text = re.sub(r'\[\d+\]', '', text)
                        evidence.append({
                            "source": f"Wikipedia: {title}",
                            "url": url,
                            "text": text,
                            "reliability": SOURCES["wikipedia"].reliability_score,
                            "source_type": "encyclopedia"
                        })
    except Exception as e:
        print(f"Wikipedia scraping error: {e}")
    
    return evidence


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search DuckDuckGo for relevant pages and scrape them
    """
    if not DDGS_AVAILABLE:
        return []
    
    evidence = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            for result in results:
                url = result.get('href', '')
                title = result.get('title', '')
                snippet = result.get('body', '')
                
                # Determine source reliability
                reliability = 0.5
                source_type = "general"
                
                if 'wikipedia.org' in url:
                    reliability = 0.85
                    source_type = "encyclopedia"
                elif 'reuters.com' in url:
                    reliability = 0.90
                    source_type = "news"
                elif 'apnews.com' in url:
                    reliability = 0.90
                    source_type = "news"
                elif 'snopes.com' in url:
                    reliability = 0.88
                    source_type = "factcheck"
                elif 'politifact.com' in url:
                    reliability = 0.88
                    source_type = "factcheck"
                elif any(x in url for x in ['.gov', '.edu']):
                    reliability = 0.80
                    source_type = "official"
                
                if snippet and len(snippet) > 50:
                    evidence.append({
                        "source": title,
                        "url": url,
                        "text": snippet,
                        "reliability": reliability,
                        "source_type": source_type
                    })
                
                # Try to scrape full content from high-reliability sources
                if reliability >= 0.80 and 'wikipedia.org' not in url:
                    try:
                        full_content = scrape_webpage(url)
                        for content in full_content[:3]:
                            content['reliability'] = reliability
                            content['source_type'] = source_type
                            evidence.append(content)
                    except:
                        pass
                        
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
    
    return evidence


def scrape_webpage(url: str, max_paragraphs: int = 5) -> List[Dict]:
    """
    Generic webpage scraper
    """
    evidence = []
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else url
            
            # Try to find main content
            main_content = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find('div', class_=re.compile(r'content|article|post|entry')) or
                soup.body
            )
            
            if main_content:
                for p in main_content.find_all('p')[:max_paragraphs]:
                    text = p.get_text().strip()
                    if len(text) > 80:
                        evidence.append({
                            "source": title[:100],
                            "url": url,
                            "text": text[:1000],
                            "reliability": 0.5,
                            "source_type": "general"
                        })
    except Exception as e:
        print(f"Webpage scraping error for {url}: {e}")
    
    return evidence


def search_google_factcheck_api(query: str) -> List[Dict]:
    """
    Search Google Fact Check Tools API (free, no key needed for basic search)
    """
    evidence = []
    
    try:
        # Google Fact Check Explorer (public endpoint)
        url = f"https://toolbox.google.com/factcheck/api/v1/claimSearch?query={quote_plus(query)}&languageCode=en"
        response = requests.get(url, headers=HEADERS, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            claims = data.get('claims', [])
            
            for claim in claims[:5]:
                claim_text = claim.get('text', '')
                claimant = claim.get('claimant', 'Unknown')
                
                reviews = claim.get('claimReview', [])
                for review in reviews:
                    publisher = review.get('publisher', {}).get('name', 'Fact Checker')
                    rating = review.get('textualRating', '')
                    review_url = review.get('url', '')
                    
                    evidence.append({
                        "source": f"{publisher} (Fact Check)",
                        "url": review_url,
                        "text": f"Claim by {claimant}: \"{claim_text}\" - Rated: {rating}",
                        "reliability": 0.90,
                        "source_type": "factcheck",
                        "rating": rating
                    })
    except Exception as e:
        print(f"Google Fact Check API error: {e}")
    
    return evidence


def fetch_evidence(user_claim: str, use_all_sources: bool = True) -> List[Dict]:
    """
    Main function - Orchestrates multi-source evidence gathering
    """
    all_evidence = []
    
    # 1. Extract entities from claim
    entities = get_entities(user_claim)
    
    # Fallback if no entities
    if not entities:
        entities = [{"text": user_claim, "label": "GENERIC"}]
    
    # 2. Build search queries
    search_queries = []
    
    # Primary: full claim
    search_queries.append(user_claim)
    
    # Secondary: entity-based queries
    for ent in entities[:3]:
        if ent['text'].lower() != user_claim.lower():
            search_queries.append(ent['text'])
    
    # 3. Gather evidence from multiple sources in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for query in search_queries[:3]:
            # Wikipedia
            futures.append(executor.submit(scrape_wikipedia, query))
            
            # DuckDuckGo (if available)
            if use_all_sources and DDGS_AVAILABLE:
                futures.append(executor.submit(search_duckduckgo, query, 3))
        
        # Google Fact Check API
        if use_all_sources:
            futures.append(executor.submit(search_google_factcheck_api, user_claim))
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_evidence.extend(result)
            except Exception as e:
                print(f"Error fetching evidence: {e}")
    
    # 4. Deduplicate and rank evidence
    seen_texts = set()
    unique_evidence = []
    
    for item in all_evidence:
        # Create a hash of the text for deduplication
        text_hash = item['text'][:100].lower()
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_evidence.append(item)
    
    # Sort by reliability score
    unique_evidence.sort(key=lambda x: x.get('reliability', 0.5), reverse=True)
    
    return unique_evidence


# Legacy compatibility
def construct_url(entity_text: str, entity_label: str) -> str:
    """Build URL based on entity type (legacy compatibility)"""
    encoded_text = quote_plus(entity_text)
    
    if entity_label in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
        return f"https://en.wikipedia.org/wiki/{encoded_text}"
    else:
        return f"https://www.google.com/search?q={encoded_text}"


def scrape_content(url: str) -> Tuple[Optional[str], List[str]]:
    """Scrape content from URL (legacy compatibility)"""
    try:
        page = requests.get(url, headers=HEADERS, timeout=5)
        if page.status_code != 200:
            return None, []

        soup = BeautifulSoup(page.content, 'html.parser')
        
        title_tag = soup.find('h1')
        title = title_tag.text if title_tag else "Unknown Source"
        
        paragraphs = []
        for p in soup.find_all('p')[:10]:
            text = p.get_text().strip()
            if len(text) > 50:
                paragraphs.append(text)
                
        return title, paragraphs
    except Exception as e:
        print(f"Scraping Error: {e}")
        return None, []