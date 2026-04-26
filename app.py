"""
TruthLens API v7.0 — Production Ready with Grok-style Reasoning
National Open University of Nigeria (NOUN)
Developer: Jabir Muhammad Kabir
Supervisor: Dr. Ojeniyi Adebayo

Pipeline:
  1. DistilBERT     — instant linguistic pattern analysis
  2. Tavily Search  — AI-native search with date awareness
  3. Google News    — additional Nigerian + global coverage  
  4. Jina Reader    — reads full article text bypassing blocks
  5. Phi-3.5-mini   — reads articles, reasons like Grok
  6. Smart Fusion   — combines all signals into final verdict
"""

import os, re, time, json, hashlib, logging, random
from datetime import datetime, date
from collections import defaultdict

import torch
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthlens")

# ── Environment ────────────────────────────────────────────────────
MODEL_ID    = os.environ.get("MODEL_ID", "Jabirkabir/truthlens-fakenews-detector")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
TAVILY_KEY  = os.environ.get("TAVILY_KEY", "")

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="TruthLens API", version="7.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Rate limiting ──────────────────────────────────────────────────
_rate = defaultdict(list)

@app.middleware("http")
async def security(request: FastAPIRequest, call_next):
    if request.url.path == "/analyse":
        ip = getattr(request.client, "host", "unknown")
        now = time.time()
        _rate[ip] = [t for t in _rate[ip] if now - t < 60]
        if len(_rate[ip]) >= 20:
            return JSONResponse(status_code=429, content={"error": "Too many requests. Wait 60 seconds."})
        _rate[ip].append(now)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp

# ── Load DistilBERT ────────────────────────────────────────────────
logger.info(f"Loading DistilBERT: {MODEL_ID}")
cls_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
cls_model.eval()
logger.info("DistilBERT ready!")

# ── Cache ──────────────────────────────────────────────────────────
CACHE: dict = {}
CACHE_TTL = 3600 * 3  # 3 hours (shorter for date-sensitive claims)

def cache_key(text: str) -> str:
    # Include today's date in key so JAMB 2026 results
    # get re-searched each day automatically
    today = str(date.today())
    return hashlib.md5(f"{today}:{text.lower().strip()}".encode()).hexdigest()

def cache_get(key: str):
    e = CACHE.get(key)
    if e and time.time() - e["ts"] < CACHE_TTL:
        return e["v"]
    return None

def cache_set(key: str, value):
    if len(CACHE) > 400:
        oldest = sorted(CACHE.items(), key=lambda x: x[1]["ts"])[:100]
        for k, _ in oldest:
            del CACHE[k]
    CACHE[key] = {"ts": time.time(), "v": value}

# ── Counter ────────────────────────────────────────────────────────
COUNTER = {"date": str(date.today()), "searches": 0, "cache_hits": 0}

def tick(cache_hit=False):
    today = str(date.today())
    if COUNTER["date"] != today:
        COUNTER.update({"date": today, "searches": 0, "cache_hits": 0})
    if cache_hit:
        COUNTER["cache_hits"] += 1
    else:
        COUNTER["searches"] += 1

# ── Credible sources ───────────────────────────────────────────────
SOURCES = {
    "punchng.com":         {"region": "Nigeria",     "name": "Punch Nigeria"},
    "vanguardngr.com":     {"region": "Nigeria",     "name": "Vanguard"},
    "dailytrust.com":      {"region": "Nigeria",     "name": "Daily Trust"},
    "premiumtimesng.com":  {"region": "Nigeria",     "name": "Premium Times"},
    "channelstv.com":      {"region": "Nigeria",     "name": "Channels TV"},
    "thecable.ng":         {"region": "Nigeria",     "name": "The Cable"},
    "guardian.ng":         {"region": "Nigeria",     "name": "Guardian Nigeria"},
    "thisdaylive.com":     {"region": "Nigeria",     "name": "ThisDay"},
    "businessday.ng":      {"region": "Nigeria",     "name": "BusinessDay"},
    "leadership.ng":       {"region": "Nigeria",     "name": "Leadership"},
    "tribuneonlineng.com": {"region": "Nigeria",     "name": "Tribune"},
    "sunnewsonline.com":   {"region": "Nigeria",     "name": "The Sun"},
    "blueprint.ng":        {"region": "Nigeria",     "name": "Blueprint"},
    "saharareporters.com": {"region": "Nigeria",     "name": "Sahara Reporters"},
    "legit.ng":            {"region": "Nigeria",     "name": "Legit.ng"},
    "nannews.ng":          {"region": "Nigeria",     "name": "NAN"},
    "arise.tv":            {"region": "Nigeria",     "name": "Arise TV"},
    "tvcnews.tv":          {"region": "Nigeria",     "name": "TVC News"},
    "prnigeria.com":       {"region": "Nigeria",     "name": "PR Nigeria"},
    "myschool.ng":         {"region": "Nigeria-Edu", "name": "MySchool"},
    "schoolgist.com.ng":   {"region": "Nigeria-Edu", "name": "SchoolGist"},
    "myjoyonline.com":     {"region": "Ghana",       "name": "Joy Online"},
    "graphic.com.gh":      {"region": "Ghana",       "name": "Graphic Ghana"},
    "nation.africa":       {"region": "Kenya",       "name": "Nation Africa"},
    "standardmedia.co.ke": {"region": "Kenya",       "name": "Standard Media"},
    "news24.com":          {"region": "S.Africa",    "name": "News24"},
    "dailymaverick.co.za": {"region": "S.Africa",    "name": "Daily Maverick"},
    "reuters.com":         {"region": "Global",      "name": "Reuters"},
    "apnews.com":          {"region": "Global",      "name": "AP News"},
    "bbc.com":             {"region": "Global",      "name": "BBC"},
    "bbc.co.uk":           {"region": "Global",      "name": "BBC"},
    "aljazeera.com":       {"region": "Global",      "name": "Al Jazeera"},
    "theguardian.com":     {"region": "Global",      "name": "The Guardian"},
    "cnn.com":             {"region": "Global",      "name": "CNN"},
    "bloomberg.com":       {"region": "Global",      "name": "Bloomberg"},
    "nytimes.com":         {"region": "Global",      "name": "NY Times"},
    "washingtonpost.com":  {"region": "Global",      "name": "Washington Post"},
    "rfi.fr":              {"region": "Global",      "name": "RFI"},
    "dw.com":              {"region": "Europe",      "name": "DW"},
    "france24.com":        {"region": "Europe",      "name": "France 24"},
    "thehindu.com":        {"region": "India",       "name": "The Hindu"},
    "arabnews.com":        {"region": "M.East",      "name": "Arab News"},
    "scmp.com":            {"region": "Asia",        "name": "SCMP"},
    "cbc.ca":              {"region": "Canada",      "name": "CBC"},
    "abc.net.au":          {"region": "Australia",   "name": "ABC Australia"},
    "dubawa.org":          {"region": "FactChecker", "name": "Dubawa"},
    "africacheck.org":     {"region": "FactChecker", "name": "Africa Check"},
    "pesacheck.org":       {"region": "FactChecker", "name": "PesaCheck"},
    "snopes.com":          {"region": "FactChecker", "name": "Snopes"},
    "factcheck.org":       {"region": "FactChecker", "name": "FactCheck.org"},
    "politifact.com":      {"region": "FactChecker", "name": "PolitiFact"},
    "fullfact.org":        {"region": "FactChecker", "name": "Full Fact"},
    "boomlive.in":         {"region": "FactChecker", "name": "BoomLive"},
}

NAME_LOOKUP = {}
for src, info in SOURCES.items():
    n = info["name"].lower()
    NAME_LOOKUP[n] = src
    parts = n.split()
    if len(parts) >= 2:
        NAME_LOOKUP[" ".join(parts[:2])] = src

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def extract_years(text: str) -> list:
    return [int(y) for y in re.findall(r'\b(20\d{2})\b', text)]

def run_distilbert(text: str) -> dict:
    inputs = cls_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        logits = cls_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    real_p = float(probs[0])
    fake_p = float(probs[1])
    return {
        "real": real_p, "fake": fake_p,
        "prediction": int(probs.argmax()),
        "confidence": int(max(real_p, fake_p) * 100)
    }

def generate_queries(claim: str) -> list:
    """Generate smart search queries including a current-status query"""
    queries = [claim]
    proper  = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
    years   = re.findall(r'\b20\d{2}\b', claim)
    actions = re.findall(
        r'\b(dead|died|death|arrested|elected|resigned|appointed|'
        r'banned|attacked|killed|won|lost|signed|launched|announced|released|out)\b',
        claim.lower()
    )
    if proper and actions:
        queries.append(" ".join(proper[:3]) + " " + actions[0])
    elif proper:
        queries.append(" ".join(proper[:4]))

    # Always add a "current status" query for time-sensitive claims
    # e.g. "Is Tinubu alive 2026?" or "JAMB 2026 results status"
    current_year = datetime.now().year
    if proper:
        status_q = f"{' '.join(proper[:2])} current status {current_year}"
        queries.append(status_q)
    elif years:
        queries.append(claim + f" {current_year}")

    seen, unique = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    logger.info(f"Queries: {unique}")
    return unique[:3]

# ══════════════════════════════════════════════════════════════════
# LAYER 2a — Tavily Search (AI-native, date-aware)
# ══════════════════════════════════════════════════════════════════
def search_tavily(query: str) -> list:
    """
    Tavily is built for AI agents — returns clean article content
    with publication dates. Works perfectly on Railway.
    Free: 1000 searches/month.
    """
    if not TAVILY_KEY:
        logger.warning("No TAVILY_KEY — skipping Tavily search")
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key":        TAVILY_KEY,
                "query":          query,
                "search_depth":   "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results":    8,
                "include_domains": list(SOURCES.keys()),
            },
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for r in data.get("results", []):
                results.append({
                    "href":        r.get("url", ""),
                    "title":       r.get("title", ""),
                    "body":        r.get("content", "")[:500],
                    "raw_content": r.get("raw_content", "")[:3000],
                    "pubdate":     r.get("published_date", ""),
                    "score":       r.get("score", 0),
                    "source":      "tavily"
                })
            # Also get Tavily's own answer summary
            tavily_answer = data.get("answer", "")
            logger.info(f"Tavily: {len(results)} results for '{query[:40]}'")
            if tavily_answer:
                logger.info(f"Tavily answer: {tavily_answer[:150]}")
            return results, tavily_answer
        else:
            logger.error(f"Tavily error: {resp.status_code} {resp.text[:200]}")
            return [], ""
    except Exception as e:
        logger.error(f"Tavily failed: {e}")
        return [], ""

# ══════════════════════════════════════════════════════════════════
# LAYER 2b — Google News RSS (backup, free, unlimited)
# ══════════════════════════════════════════════════════════════════
def search_google_news(query: str, region: str = "US") -> list:
    try:
        q    = requests.utils.quote(query[:200])
        url  = f"https://news.google.com/rss/search?q={q}&hl=en&gl={region}&ceid={region}:en"
        resp = requests.get(
            url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=12
        )
        results = []
        if resp.status_code == 200:
            for item in re.findall(r'<item>(.*?)</item>', resp.text, re.DOTALL)[:10]:
                t   = (re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', item) or
                       re.findall(r'<title>(.*?)</title>', item))
                l   = re.findall(r'<link>(.*?)</link>', item)
                d   = (re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>', item) or
                       re.findall(r'<description>(.*?)</description>', item))
                pub = re.findall(r'<pubDate>(.*?)</pubDate>', item)
                if t:
                    clean_t = re.sub(r'\s+-\s+\S[\S]*\s*$', '',
                              re.sub(r'<.*?>', '', t[0])).strip()
                    results.append({
                        "href":        l[0].strip() if l else "",
                        "title":       clean_t,
                        "body":        re.sub(r'<.*?>', '', d[0]).strip()[:400] if d else "",
                        "raw_content": "",
                        "pubdate":     pub[0].strip() if pub else "",
                        "source":      "google_news"
                    })
        logger.info(f"Google News ({region}): {len(results)} results")
        return results
    except Exception as e:
        logger.warning(f"Google News ({region}) failed: {e}")
        return []

def search_all(queries: list) -> tuple:
    """Run all searches, combine, deduplicate"""
    all_results, seen_urls = [], set()
    tavily_answers = []

    def add(items):
        for r in items:
            url = r.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    for q in queries:
        # Tavily first — best quality + date-aware
        if TAVILY_KEY:
            tv_results, tv_answer = search_tavily(q)
            add(tv_results)
            if tv_answer:
                tavily_answers.append(tv_answer)
        # Google News Nigerian edition
        add(search_google_news(q, "NG"))
        # Google News US edition
        add(search_google_news(q, "US"))
        time.sleep(0.3)

    logger.info(f"Total unique results: {len(all_results)}")
    combined_answer = " ".join(tavily_answers[:2]) if tavily_answers else ""
    return all_results, combined_answer

# ══════════════════════════════════════════════════════════════════
# LAYER 3 — Jina Reader (reads articles bypassing all blocks)
# ══════════════════════════════════════════════════════════════════
def read_article(url: str) -> str:
    """
    Jina AI Reader bypasses Cloudflare, paywalls, bot detection.
    Works on ALL Nigerian and global news sites.
    Free, no API key.
    """
    if not url or "news.google.com" in url:
        return ""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = requests.get(
            jina_url,
            headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/plain",
                "X-Return-Format": "text",
            },
            timeout=15
        )
        if resp.status_code == 200 and len(resp.text) > 200:
            text  = resp.text
            lines = text.split("\n")
            body  = []
            for line in lines:
                if line.startswith(("Title:", "URL:", "Published", "Source:",
                                    "Description:", "---", "===")):
                    continue
                body.append(line)
            clean = "\n".join(body).strip()
            if len(clean) > 200:
                logger.info(f"Jina read {len(clean)} chars from {url[:50]}")
                return clean[:5000]
        return ""
    except Exception as e:
        logger.debug(f"Jina failed {url[:50]}: {e}")
        return ""

# ══════════════════════════════════════════════════════════════════
# Source matching
# ══════════════════════════════════════════════════════════════════
def match_sources(claim: str, results: list) -> list:
    """
    Match results to credible sources.
    Handles Google News URLs (news.google.com) by matching title suffix.
    Also uses Tavily direct URLs.
    """
    found, seen = [], set()
    claim_years = extract_years(claim)

    for r in results:
        url         = (r.get("href") or "").lower()
        title       = r.get("title") or ""
        snippet     = r.get("body") or ""
        pubdate     = r.get("pubdate") or ""
        raw_content = r.get("raw_content") or ""

        # Date filter — check pubdate against claim years
        if claim_years and pubdate:
            pub_years = extract_years(pubdate)
            if pub_years and not any(
                abs(py - cy) <= 1
                for py in pub_years
                for cy in claim_years
            ):
                logger.info(f"Date filtered: {title[:50]} pub={pubdate}")
                continue

        matched = None

        # Method 1: Direct URL match (Tavily returns real URLs)
        for src in SOURCES:
            if src in url and src not in seen:
                matched = src
                break

        # Method 2: Google News title suffix "Headline - Source Name"
        if not matched and " - " in title:
            suffix = title.split(" - ")[-1].strip().lower()
            if suffix in NAME_LOOKUP and NAME_LOOKUP[suffix] not in seen:
                matched = NAME_LOOKUP[suffix]
            else:
                for name, src in NAME_LOOKUP.items():
                    if name in suffix and src not in seen and len(name) > 4:
                        matched = src
                        break

        # Method 3: Domain keyword in title or snippet
        if not matched:
            combined = (title + " " + snippet).lower()
            for src, info in SOURCES.items():
                if src not in seen:
                    core = re.sub(
                        r'\.(com|ng|org|co\.uk|co|tv|africa|net)$', '', src
                    )
                    if len(core) > 5 and core in combined:
                        matched = src
                        break

        if matched:
            seen.add(matched)
            info = SOURCES[matched]
            raw_url = r.get("href") or ""
            article_url = (
                raw_url
                if (raw_url.startswith("http") and "news.google.com" not in raw_url)
                else f"https://{matched}"
            )
            found.append({
                "source":         matched,
                "name":           info["name"],
                "region":         info["region"],
                "url":            f"https://{matched}",
                "article_url":    article_url,
                "title":          title,
                "snippet":        snippet[:300],
                "raw_content":    raw_content,
                "pubdate":        pubdate,
                "is_nigerian":    info["region"] in ("Nigeria", "Nigeria-Edu"),
                "is_factchecker": info["region"] == "FactChecker",
            })

    found.sort(
        key=lambda x: (x["is_factchecker"] * 2 + x["is_nigerian"]),
        reverse=True
    )
    logger.info(f"Sources matched: {[s['name'] for s in found]}")
    return found

# ══════════════════════════════════════════════════════════════════
# LAYER 4 — Phi-3.5-mini reasoning (Grok-style)
# ══════════════════════════════════════════════════════════════════
def reason_with_phi35(
    claim: str,
    articles: list,      # list of (title, full_text, pubdate, source_name)
    tavily_summary: str  # Tavily's own answer about the topic
) -> dict:
    """
    Phi-3.5-mini reads full articles and reasons like Grok.
    
    Key difference from before:
    - Gets full article TEXT not just titles
    - Gets publication dates so it knows WHEN articles were written
    - Gets Tavily's independent web summary
    - Reasons about MEANING not keywords
    
    Example:
    Claim: "Tinubu is dead"
    Article 1: "TikToker arrested for FALSELY claiming Tinubu died"
    Article 2: "Tinubu mourns death of Kano lawmaker"
    Phi-3.5: "Article 1 says someone was arrested for making this 
              false claim. Article 2 shows Tinubu mourning someone — 
              he is clearly alive. VERDICT: FALSE"
    """
    if not HF_TOKEN:
        return {
            "verdict": "unavailable",
            "reasoning": "HuggingFace token not configured",
            "confidence": 0
        }

    # Build evidence block with dates
    evidence_parts = []
    current_date = datetime.now().strftime("%B %Y")

    for i, (title, text, pubdate, source) in enumerate(articles[:5]):
        part = f"[Source {i+1}: {source}]"
        if pubdate:
            part += f" [Published: {pubdate}]"
        part += f"\nHeadline: {title}"
        if text and len(text) > 100:
            part += f"\nContent: {text[:1500]}"
        evidence_parts.append(part)

    if tavily_summary:
        evidence_parts.insert(
            0,
            f"[Web Summary as of {current_date}]\n{tavily_summary}"
        )

    if not evidence_parts:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reasoning": "No articles could be retrieved for analysis",
            "confidence": 0
        }

    evidence_block = "\n\n".join(evidence_parts)

    prompt = f"""You are TruthLens, a professional AI fact-checker similar to Grok.
Today's date is: {datetime.now().strftime("%d %B %Y")}

CLAIM TO VERIFY: "{claim}"

EVIDENCE FROM NEWS SOURCES:
{evidence_block}

YOUR TASK:
Read each source carefully. Think step by step:

1. What does each article ACTUALLY say? 
   - An article saying "person arrested for claiming X" means X is FALSE
   - An article saying "mourns death of Y" when Y is a different person means the subject is ALIVE
   - Check publication dates — old articles may be outdated

2. Is there a TIMING issue?
   - If claim says "2026 results are out" but articles are from 2025, that does not confirm 2026 results
   - Current date is {datetime.now().strftime("%d %B %Y")} — use this for context

3. What does the OVERALL evidence show?
   - Do multiple credible sources directly confirm or deny the claim?
   - Are sources talking about the actual claim or something similar but different?

RESPOND IN THIS EXACT FORMAT:
REASONING: [Explain what each source actually says and your analysis — be specific]
VERDICT: [SUPPORTS_CLAIM or CONTRADICTS_CLAIM or INSUFFICIENT_EVIDENCE]
CONFIDENCE: [HIGH or MEDIUM or LOW]
EXPLANATION: [One clear sentence for the user explaining the verdict]"""

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "microsoft/Phi-3.5-mini-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are TruthLens, a professional AI fact-checker. Today is {datetime.now().strftime('%d %B %Y')}. Be precise, analytical and consider article dates carefully."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 600,
                "temperature": 0.1,
                "stream": False
            },
            timeout=45
        )

        if response.status_code == 200:
            raw = response.json()
            if isinstance(raw, dict) and "choices" in raw:
                text = raw["choices"][0]["message"]["content"]
            elif isinstance(raw, list) and raw:
                text = raw[0].get("generated_text", "")
            else:
                text = str(raw)

            logger.info(f"Phi-3.5 response: {text[:300]}")

            # Parse structured response
            v_match   = re.search(
                r'VERDICT:\s*(SUPPORTS_CLAIM|CONTRADICTS_CLAIM|INSUFFICIENT_EVIDENCE)',
                text, re.IGNORECASE
            )
            r_match   = re.search(
                r'REASONING:\s*(.*?)(?=VERDICT:|$)', text, re.IGNORECASE | re.DOTALL
            )
            c_match   = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', text, re.IGNORECASE)
            e_match   = re.search(r'EXPLANATION:\s*(.*?)(?=\n\n|$)', text, re.IGNORECASE | re.DOTALL)

            verdict     = v_match.group(1).upper() if v_match else "INSUFFICIENT_EVIDENCE"
            reasoning   = r_match.group(1).strip()[:800] if r_match else text[:500]
            confidence  = c_match.group(1).upper() if c_match else "LOW"
            explanation = e_match.group(1).strip()[:200] if e_match else reasoning[:150]

            conf_score = {"HIGH": 0.9, "MEDIUM": 0.65, "LOW": 0.4}.get(confidence, 0.4)

            return {
                "verdict":     verdict,
                "reasoning":   reasoning,
                "explanation": explanation,
                "confidence":  conf_score,
            }

        elif response.status_code == 503:
            logger.warning("Phi-3.5 loading — retrying in 20s")
            time.sleep(20)
            return reason_with_phi35(claim, articles, tavily_summary)
        else:
            err = response.text[:300]
            logger.error(f"Phi-3.5 error: {response.status_code} — {err}")
            return {
                "verdict": "api_error",
                "reasoning": f"API error {response.status_code}: {err}",
                "confidence": 0
            }

    except Exception as e:
        logger.error(f"Phi-3.5 failed: {e}")
        return {"verdict": "error", "reasoning": str(e), "confidence": 0}

# ══════════════════════════════════════════════════════════════════
# LAYER 5 — Smart verdict fusion
# ══════════════════════════════════════════════════════════════════
def fuse_verdict(
    claim: str,
    distilbert: dict,
    matched: list,
    phi35: dict,
    tavily_summary: str
) -> dict:
    """
    Combine all signals. Priority:
    Phi-3.5 reasoning > Fact-checkers > Source count > DistilBERT
    """
    nigerian   = [s for s in matched if s["is_nigerian"]]
    factcheck  = [s for s in matched if s["is_factchecker"]]
    total      = len(matched)
    phi_v      = phi35.get("verdict", "INSUFFICIENT_EVIDENCE")
    phi_conf   = phi35.get("confidence", 0)
    phi_exp    = phi35.get("explanation", "")
    phi_ok     = phi_v not in (
        "unavailable", "api_error", "error",
        "insufficient_evidence", "INSUFFICIENT_EVIDENCE"
    )

    fc_fake = [s for s in factcheck if any(
        w in s["title"].lower()
        for w in ["false","fake","misinformation","misleading","debunked","hoax","satire"]
    )]
    fc_real = [s for s in factcheck if any(
        w in s["title"].lower()
        for w in ["true","confirmed","accurate","verified","correct"]
    )]

    if fc_fake:
        return {
            "verdict": "FAKE", "confidence": 99,
            "title": "Confirmed Misinformation",
            "subtitle": f"Flagged by {fc_fake[0]['name']}: {fc_fake[0]['title'][:60]}"
        }
    if fc_real and phi_v == "SUPPORTS_CLAIM":
        return {
            "verdict": "REAL", "confidence": 99,
            "title": "Fact-Checker + AI Verified",
            "subtitle": f"Confirmed by {fc_real[0]['name']} — {phi_exp}"
        }
    if phi_ok and phi_v == "CONTRADICTS_CLAIM" and phi_conf >= 0.6:
        return {
            "verdict": "FAKE", "confidence": 96,
            "title": "AI Reading Contradicts Claim",
            "subtitle": f"Phi-3.5 read the full articles: {phi_exp}"
        }
    if phi_ok and phi_v == "SUPPORTS_CLAIM" and phi_conf >= 0.6 and total >= 2:
        return {
            "verdict": "REAL", "confidence": 96,
            "title": "Confirmed — AI Read and Verified",
            "subtitle": f"Phi-3.5 analysed {total} sources: {phi_exp}"
        }
    if phi_ok and phi_v == "SUPPORTS_CLAIM" and phi_conf >= 0.6:
        return {
            "verdict": "REAL", "confidence": 88,
            "title": "AI Reading Confirms Claim",
            "subtitle": phi_exp
        }
    if total >= 3 and phi_v != "CONTRADICTS_CLAIM":
        regions = list(dict.fromkeys(s["region"] for s in matched[:4]))
        return {
            "verdict": "REAL", "confidence": 90,
            "title": "Confirmed by Multiple Outlets",
            "subtitle": f"Found in {total} credible outlets: {', '.join(regions)}"
        }
    if len(nigerian) >= 2:
        return {
            "verdict": "REAL", "confidence": 87,
            "title": "Confirmed by Nigerian Outlets",
            "subtitle": f"{nigerian[0]['name']} and {nigerian[1]['name']} report this"
        }
    if total >= 2:
        return {
            "verdict": "REAL", "confidence": 80,
            "title": "Likely Authentic News",
            "subtitle": f"Found in {matched[0]['name']} and {matched[1]['name']}"
        }
    if total == 1 and phi_v != "CONTRADICTS_CLAIM":
        return {
            "verdict": "REAL", "confidence": 63,
            "title": "Possibly Authentic — Limited Sources",
            "subtitle": f"Found in 1 outlet: {matched[0]['name']}. Verify further."
        }
    if phi_ok and phi_v == "CONTRADICTS_CLAIM":
        return {
            "verdict": "FAKE", "confidence": 87,
            "title": "AI Reading Contradicts Claim",
            "subtitle": phi_exp
        }
    if distilbert["prediction"] == 1 and distilbert["confidence"] >= 80:
        return {
            "verdict": "FAKE", "confidence": 80,
            "title": "Likely Fake — AI + No Sources",
            "subtitle": "Misinformation patterns detected. Zero credible sources found worldwide."
        }
    if distilbert["prediction"] == 0 and distilbert["confidence"] >= 80:
        return {
            "verdict": "SUSPICIOUS", "confidence": 58,
            "title": "Suspicious — Cannot Verify",
            "subtitle": "Writing appears credible but no outlet worldwide confirms this claim."
        }
    return {
        "verdict": "MIXED", "confidence": 50,
        "title": "Uncertain — Verify Manually",
        "subtitle": "Mixed signals. Use the fact-checker links below to verify."
    }

# ══════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════
class AnalyseRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "name": "TruthLens API v7.0",
        "institution": "National Open University of Nigeria (NOUN)",
        "developer": "Jabir Muhammad Kabir",
        "supervisor": "Dr. Ojeniyi Adebayo",
        "status": "running",
        "pipeline": [
            "DistilBERT", "TavilySearch", "GoogleNewsRSS",
            "JinaReader", "Phi-3.5-mini", "SmartFusion"
        ],
        "sources": len(SOURCES),
        "tavily_enabled": bool(TAVILY_KEY),
        "phi35_enabled":  bool(HF_TOKEN),
    }

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "tavily":  bool(TAVILY_KEY),
        "phi35":   bool(HF_TOKEN),
    }

@app.get("/stats")
def stats():
    return COUNTER.copy()

@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    text = req.text.strip()
    if not text or len(text) < 8:
        raise HTTPException(400, "Text too short")
    if len(text) > 8000:
        raise HTTPException(400, "Text too long — max 8000 characters")

    clean = re.sub(r'http\S+|www\S+|<.*?>|\s+', ' ', text).strip()[:600]
    current_year  = datetime.now().year
    claim_years   = extract_years(clean)
    future_years  = [y for y in claim_years if y > current_year]

    if future_years:
        return _future_verdict(future_years[0])

    ck = cache_key(clean)
    cached = cache_get(ck)
    if cached:
        tick(cache_hit=True)
        cached["from_cache"] = True
        return cached

    tick(cache_hit=False)

    # Layer 1 — DistilBERT
    db = run_distilbert(clean)
    logger.info(f"DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%")

    # Layer 2 — Query expansion + search
    queries = generate_queries(clean)
    results, tavily_summary = search_all(queries)

    # Source matching
    matched   = match_sources(clean, results)
    factcheck = [s for s in matched if s["is_factchecker"]]

    # Layer 3 — Read full articles with Jina
    articles_for_phi = []
    for src in matched[:5]:
        url = src["article_url"]
        # Use raw_content from Tavily if available (already extracted)
        if src.get("raw_content") and len(src["raw_content"]) > 200:
            full_text = src["raw_content"]
            logger.info(f"Using Tavily content for {src['name']}: {len(full_text)} chars")
        else:
            # Fall back to Jina reader
            full_text = read_article(url) if url and "news.google.com" not in url else ""
        articles_for_phi.append((
            src["title"],
            full_text,
            src["pubdate"],
            src["name"]
        ))

    # Also add high-quality snippets from unmatched results
    for r in results[:5]:
        if r.get("raw_content") and len(r["raw_content"]) > 300:
            title = r.get("title", "")
            src_name = "Web Source"
            for src_domain in SOURCES:
                if src_domain in (r.get("href") or "").lower():
                    src_name = SOURCES[src_domain]["name"]
                    break
            if not any(t == title for t, _, _, _ in articles_for_phi):
                articles_for_phi.append((
                    title,
                    r["raw_content"],
                    r.get("pubdate", ""),
                    src_name
                ))

    scraped_count = sum(1 for _, t, _, _ in articles_for_phi if len(t) > 200)

    # Layer 4 — Phi-3.5 reasoning
    phi35 = reason_with_phi35(clean, articles_for_phi, tavily_summary)
    logger.info(f"Phi-3.5: {phi35.get('verdict')} conf={phi35.get('confidence')}")

    # Layer 5 — Fuse verdict
    fusion     = fuse_verdict(clean, db, matched, phi35, tavily_summary)
    verdict    = fusion["verdict"]
    confidence = fusion["confidence"]

    # Signals
    upper     = sum(1 for w in clean.split() if w.isupper() and len(w) > 2)
    excl      = clean.count("!")
    has_attr  = any(w in clean.lower() for w in
                ["according to","reported","said","confirmed","announced"])
    clickbait = any(w in clean.lower() for w in
                ["shocking","secret","exposed","leaked","won't believe",
                 "they don't want","viral","breaking","must see"])

    phi_sig = {
        "SUPPORTS_CLAIM":        ("Phi-3.5 read articles — SUPPORTS claim",    "pos"),
        "CONTRADICTS_CLAIM":     ("Phi-3.5 read articles — CONTRADICTS claim", "neg"),
        "INSUFFICIENT_EVIDENCE": ("Phi-3.5 — insufficient article content",    "neu"),
    }.get(phi35.get("verdict", ""), ("Phi-3.5 reasoning unavailable", "neu"))

    signals = [
        {"icon":"🌐","label":"Web Sources",
         "value":f"{len(matched)} credible outlets found",
         "type":"pos" if len(matched)>=2 else "neg" if len(matched)==0 else "neu"},
        {"icon":"🤖","label":"DistilBERT AI",
         "value":f"{'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%",
         "type":"pos" if db['prediction']==0 else "neg"},
        {"icon":"🧠","label":"Phi-3.5 Reasoning",
         "value":phi_sig[0], "type":phi_sig[1]},
        {"icon":"📰","label":"Articles Read",
         "value":f"{scraped_count} full articles analysed",
         "type":"pos" if scraped_count>0 else "neu"},
        {"icon":"😱","label":"Tone Analysis",
         "value":"Sensationalist" if upper>2 or excl>2 else "Measured tone",
         "type":"neg" if upper>2 else "pos"},
        {"icon":"🎣","label":"Clickbait",
         "value":"Clickbait detected" if clickbait else "None detected",
         "type":"neg" if clickbait else "pos"},
    ]

    phi_reasoning = phi35.get("reasoning", "") or phi35.get("explanation", "")

    if verdict == "REAL":
        src_list = ", ".join(s["name"] for s in matched[:3]) if matched else "none"
        analysis = (
            f"This content appears CREDIBLE.\n\n"
            f"LAYER 1 — DistilBERT: Linguistic patterns scored "
            f"{'REAL' if db['prediction']==0 else 'FAKE'} at {db['confidence']}% "
            f"(this is the raw AI score before web verification).\n\n"
            f"LAYER 2 — WEB SEARCH: Found {len(matched)} credible outlet(s) "
            f"including {src_list}. Searched via {len(queries)} targeted queries "
            f"with date-aware Tavily + Google News.\n\n"
            f"LAYER 3 — PHI-3.5 READING: {phi_reasoning[:400] if phi_reasoning else 'Articles analysed.'}\n\n"
            f"RECOMMENDATION: Content appears credible. Click the source links below."
        )
    elif verdict in ("FAKE", "SUSPICIOUS"):
        analysis = (
            f"This content is LIKELY FAKE OR MISLEADING.\n\n"
            f"LAYER 1 — DistilBERT: Scored "
            f"{'REAL' if db['prediction']==0 else 'FAKE'} at {db['confidence']}%.\n\n"
            f"LAYER 2 — WEB SEARCH: "
            f"{'Zero credible sources found across 60+ global outlets.' if len(matched)==0 else f'{len(matched)} source(s) found but content contradicts the claim.'} "
            f"Searched via {len(queries)} targeted queries.\n\n"
            f"LAYER 3 — PHI-3.5 READING: {phi_reasoning[:400] if phi_reasoning else 'No supporting evidence found in articles.'}\n\n"
            f"RECOMMENDATION: Do not share. Verify through AfricaCheck or Dubawa."
        )
    else:
        analysis = (
            f"UNCERTAIN RESULT: Mixed signals detected.\n\n"
            f"This may be very recent breaking news, an opinion piece, "
            f"satire, or content outside our training domain.\n\n"
            f"LAYER 1 — DistilBERT: {db['confidence']}% "
            f"{'REAL' if db['prediction']==0 else 'FAKE'}.\n"
            f"LAYER 2 — Web: {len(matched)} source(s) found.\n"
            f"LAYER 3 — Phi-3.5: {phi_reasoning[:200] if phi_reasoning else 'Inconclusive.'}\n\n"
            f"RECOMMENDATION: Check Google News, AfricaCheck or Dubawa before sharing."
        )

    result = {
        "verdict":          verdict,
        "confidence":       confidence,
        "verdict_title":    fusion["title"],
        "verdict_subtitle": fusion["subtitle"],
        "analysis":         analysis,
        "signals":          signals,
        "real_probability": round(db["real"], 4),
        "fake_probability": round(db["fake"], 4),
        "phi3": {
            "verdict":     phi35.get("verdict", ""),
            "reasoning":   phi_reasoning[:600],
            "explanation": phi35.get("explanation", ""),
            "confidence":  phi35.get("confidence", 0),
        },
        "tavily_summary": tavily_summary[:300] if tavily_summary else "",
        "sources_found": [
            {
                "source":         s["source"],
                "name":           s["name"],
                "region":         s["region"],
                "url":            s["url"],
                "article_url":    s["article_url"],
                "title":          s["title"][:120],
                "snippet":        s["snippet"],
                "pubdate":        s["pubdate"],
                "is_factchecker": s["is_factchecker"],
                "is_nigerian":    s["is_nigerian"],
            }
            for s in matched[:8]
        ],
        "fact_checks": [
            {
                "name":        s["name"],
                "url":         s["url"],
                "article_url": s["article_url"],
                "title":       s["title"][:120],
                "snippet":     s["snippet"],
            }
            for s in factcheck[:3]
        ],
        "queries_used":     queries,
        "articles_scraped": scraped_count,
        "search_timestamp": datetime.now().isoformat(),
        "from_cache":       False,
        "daily_stats":      COUNTER.copy(),
    }

    cache_set(ck, result)
    return result


def _future_verdict(year: int) -> dict:
    return {
        "verdict": "FAKE", "confidence": 99,
        "verdict_title": "Future Event Claimed as Fact",
        "verdict_subtitle": f"References {year} which has not happened yet.",
        "analysis": (
            f"TEMPORAL CHECK: This content references {year}, a future year. "
            f"Events that have not occurred cannot be reported as facts.\n\n"
            f"RECOMMENDATION: Do not share."
        ),
        "signals": [
            {"icon":"📅","label":"Date Check",      "value":f"Future year {year}", "type":"neg"},
            {"icon":"🚫","label":"Verdict",         "value":"Impossible — future", "type":"neg"},
            {"icon":"🤖","label":"DistilBERT",      "value":"Temporal check",      "type":"neg"},
            {"icon":"🧠","label":"Phi-3.5",         "value":"Not needed",          "type":"neg"},
            {"icon":"🌐","label":"Web Search",      "value":"Not needed",          "type":"neg"},
            {"icon":"⚠️","label":"Warning",         "value":"Do not share",        "type":"neg"},
        ],
        "real_probability": 0.01, "fake_probability": 0.99,
        "phi3": {"verdict":"N/A","reasoning":"Future date","confidence":0},
        "tavily_summary": "",
        "sources_found": [], "fact_checks": [],
        "queries_used": [], "articles_scraped": 0,
        "search_timestamp": datetime.now().isoformat(),
        "from_cache": False, "daily_stats": {}
    }
