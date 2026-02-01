"""
Work Package 2 – News intelligence / event discovery.

Pipeline:
  1) Use Gemini to understand the user prompt and extract structured query info.
  2) Generate search queries.
  3) Fetch articles from NewsAPI.
  4) Filter relevant articles.
  5) Use Gemini again to cluster and extract high-level EVENTS.
  6) Optionally store events in MongoDB for later analysis.

This module is imported by mcp_server.py and used in the `search_events` tool.
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optional MongoDB setup
try:
    from pymongo import MongoClient

    mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
    mongo_db = mongo_client["news_intelligence"] if mongo_client else None
    mongo_collection = mongo_db["events"] if mongo_db else None
except Exception:
    mongo_client = None
    mongo_collection = None

# Optional Gemini setup
try:
    import google.generativeai as genai

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        genai = None
except Exception:
    genai = None


def _fix_json(raw_text: str) -> Any:
    """Clean Gemini output and try to extract valid JSON."""
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        json_text = match.group(0)
    else:
        json_text = cleaned

    json_text = json_text.replace("\n", " ").replace("\t", " ")
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)

    return json.loads(json_text)


def _gemini_generate(prompt: str) -> str:
    """Call Gemini and return the raw text content."""
    if genai is None or not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini (google-generativeai) is not configured. Set GEMINI_API_KEY in .env."
        )

    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def extract_query_info(prompt: str) -> Dict[str, Any]:
    """Use Gemini to convert a free-text request into structured query info."""
    full_prompt = f"""
You are an intelligent query-understanding engine.

Extract structured information from ANY user request, across ALL domains:
- Politics, war, technology, economy, sports, science, natural disasters, companies, crypto, products, etc.

Return JSON ONLY in this format:

{{
  "main_topic": "",
  "category": "",
  "subtopics": [],
  "entities": [],
  "countries": [],
  "time_period": {{
     "start_year": null,
     "end_year": null
  }},
  "keywords": []
}}

Definitions:
- "category" = politics, war, tech, sports, economy, business, science, crypto, etc.
- "entities" = people, organizations, companies, products, technologies.
- "countries" = any country mentioned or implied.
- "subtopics" = more specific topics (e.g., inflation, EV batteries, Gaza).
- "keywords" = useful broad keywords.

If no specific year is given, leave time_period as null.

Do not add extra text outside the JSON.

USER REQUEST:
{prompt}
"""
    raw = _gemini_generate(full_prompt)

    try:
        return json.loads(raw)
    except Exception:
        return _fix_json(raw)


def generate_search_queries(info: Dict[str, Any]) -> List[str]:
    """Generate a small set of NewsAPI search queries from structured query info."""
    topic = info.get("main_topic", "") or ""
    keywords = info.get("keywords", []) or []
    entities = info.get("entities", []) or []
    countries = info.get("countries", []) or []
    subtopics = info.get("subtopics", []) or []

    queries = set()

    if topic:
        queries.add(topic)
        queries.add(f"{topic} news")
        queries.add(f"{topic} latest")

    for c in countries:
        c = c.strip()
        if not c:
            continue
        if topic:
            queries.add(f"{c} {topic}")
            queries.add(f"{c} {topic} news")
            queries.add(f"{c} latest {topic}")

    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        queries.add(kw)
        for c in countries:
            queries.add(f"{c} {kw}")

    for e in entities:
        e = e.strip()
        if not e:
            continue
        queries.add(e)
        if topic:
            queries.add(f"{e} {topic}")

    for st in subtopics:
        st = st.strip()
        if not st:
            continue
        if topic:
            queries.add(f"{topic} {st}")
        for c in countries:
            queries.add(f"{c} {st}")

    queries = {q for q in queries if len(q.split()) >= 2}
    return sorted(queries)


def fetch_articles_from_newsapi(queries: List[str]) -> List[Dict[str, Any]]:
    """Fetch articles from NewsAPI for a list of queries."""
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY is not set in .env (required for WP2).")

    all_articles: List[Dict[str, Any]] = []

    for q in queries:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 30,
            "apiKey": NEWS_API_KEY,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            for a in data.get("articles", []):
                a["search_query"] = q
                all_articles.append(a)
        except Exception as e:
            print(f"[WP2] Error fetching query '{q}': {e}")

    return all_articles


def filter_relevant_articles(info: Dict[str, Any], articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply a simple filter on NewsAPI articles based on topic, keywords, countries and time period."""
    topic = (info.get("main_topic") or "").lower()
    keywords = [k.lower() for k in (info.get("keywords") or [])]
    countries = [c.lower() for c in (info.get("countries") or [])]

    start_year = info.get("time_period", {}).get("start_year")
    end_year = info.get("time_period", {}).get("end_year")

    filtered: List[Dict[str, Any]] = []

    for a in articles:
        title = (a.get("title") or "").lower()
        desc = (a.get("description") or "").lower()
        content = (a.get("content") or "").lower()
        text = " ".join([title, desc, content])

        if countries and not any(c in text for c in countries):
            continue

        if topic and topic not in text and not any(kw in text for kw in keywords):
            continue

        if start_year and end_year:
            try:
                year = int((a.get("publishedAt") or "")[:4])
                if not (start_year <= year <= end_year):
                    continue
            except Exception:
                pass

        filtered.append(a)

    return filtered


def extract_events_from_articles(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Use Gemini to cluster filtered articles into structured events."""
    if not articles:
        return {"events": [], "overall_summary": "No relevant articles found."}

    articles_text = ""
    for a in articles:
        articles_text += f"""
TITLE: {a.get('title')}
DATE: {a.get('publishedAt')}
DESCRIPTION: {a.get('description')}
CONTENT: {a.get('content')}
---
"""

    prompt = f"""
You are an advanced news intelligence extraction system.

From the following news articles, extract a list of aggregated EVENTS.
Cluster similar stories into one event when appropriate.

Return JSON ONLY in the format:

{{
  "events": [
    {{
      "event_title": "",
      "date": "",
      "summary": "",
      "impact": "",
      "source_titles": []
    }}
  ],
  "overall_summary": ""
}}

Impact must be one of: "low", "medium", "high".

ARTICLES:
{articles_text}
"""
    raw = _gemini_generate(prompt)

    try:
        return json.loads(raw)
    except Exception:
        return _fix_json(raw)


def save_events_to_mongo(user_prompt: str, events_result: Dict[str, Any]) -> int:
    """
    Save events to MongoDB, if configured.

    Returns the number of documents inserted.
    """
    if mongo_collection is None:
        return 0

    docs = []
    for ev in events_result.get("events", []):
        docs.append(
            {
                "user_prompt": user_prompt,
                "event_title": ev.get("event_title"),
                "date": ev.get("date"),
                "summary": ev.get("summary"),
                "impact": ev.get("impact"),
                "source_titles": ev.get("source_titles"),
                "overall_summary": events_result.get("overall_summary"),
                "timestamp_saved": datetime.utcnow(),
            }
        )

    if docs:
        mongo_collection.insert_many(docs)

    return len(docs)


def run_wp2_pipeline(user_prompt: str) -> Dict[str, Any]:
    """
    Orchestrator for WP2 used by the MCP server.

    Returns a dict with:
      - status, message
      - query_info
      - n_queries, n_articles, n_filtered, n_saved
      - events (as returned by Gemini)
    """
    try:
        info = extract_query_info(user_prompt)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to extract query info from Gemini: {e}",
            "query_info": None,
        }

    queries = generate_search_queries(info)
    if not queries:
        return {
            "status": "error",
            "message": "Could not generate any search queries from the user request.",
            "query_info": info,
        }

    try:
        articles = fetch_articles_from_newsapi(queries)
    except Exception as e:
        return {
            "status": "error",
            "message": f"NewsAPI fetch failed: {e}",
            "query_info": info,
        }

    filtered = filter_relevant_articles(info, articles)

    try:
        events_json = extract_events_from_articles(filtered)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Gemini event extraction failed: {e}",
            "query_info": info,
        }

    saved_count = 0
    try:
        saved_count = save_events_to_mongo(user_prompt, events_json)
    except Exception as e:
        print(f"[WP2] Mongo save failed: {e}")

    return {
        "status": "success",
        "message": "WP2 pipeline completed.",
        "query_info": info,
        "n_queries": len(queries),
        "n_articles": len(articles),
        "n_filtered": len(filtered),
        "n_saved": saved_count,
        "events": events_json,
    }

