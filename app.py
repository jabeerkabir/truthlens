"""
TruthLens API v6.0 — Production Ready
National Open University of Nigeria (NOUN)
Developer: Jabir Muhammad Kabir
Supervisor: Dr. Ojeniyi Adebayo

Pipeline:
  1. DistilBERT  — instant linguistic analysis
  2. DuckDuckGo  — real-time web search (works on Railway)
  3. Google News RSS — backup search
  4. BeautifulSoup — full article scraping
  5. Phi-3-mini-128k — reads articles, reasons, derives verdict
  6. Smart fusion — combines all signals into final verdict
"""

import os, re, time, json, hashlib, logging, random
from datetime import datetime, date
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
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
MODEL_ID = os.environ.get("MODEL_ID", "Jabirkabir/truthlens-fakenews-detector")
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # HuggingFace token for Phi-3

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="TruthLens API", version="6.0", docs_url=None, redoc_url=None)
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
CACHE_TTL = 3600 * 6  # 6 hours

def cache_key(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

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
    # NIGERIA
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
    # NIGERIA EDUCATION
    "myschool.ng":         {"region": "Nigeria-Edu", "name": "MySchool"},
    "schoolgist.com.ng":   {"region": "Nigeria-Edu", "name": "SchoolGist"},
    "nairaland.com":       {"region": "Nigeria",     "name": "Nairaland"},
    # AFRICA
    "myjoyonline.com":     {"region": "Ghana",       "name": "Joy Online"},
    "graphic.com.gh":      {"region": "Ghana",       "name": "Graphic Ghana"},
    "nation.africa":       {"region": "Kenya",       "name": "Nation Africa"},
    "standardmedia.co.ke": {"region": "Kenya",       "name": "Standard Media"},
    "monitor.co.ug":       {"region": "Uganda",      "name": "Daily Monitor"},
    "news24.com":          {"region": "S.Africa",    "name": "News24"},
    "dailymaverick.co.za": {"region": "S.Africa",    "name": "Daily Maverick"},
    # GLOBAL
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
    # FACT CHECKERS
    "dubawa.org":          {"region": "FactChecker", "name": "Dubawa"},
    "africacheck.org":     {"region": "FactChecker", "name": "Africa Check"},
    "pesacheck.org":       {"region": "FactChecker", "name": "PesaCheck"},
    "snopes.com":          {"region": "FactChecker", "name": "Snopes"},
    "factcheck.org":       {"region": "FactChecker", "name": "FactCheck.org"},
    "politifact.com":      {"region": "FactChecker", "name": "PolitiFact"},
    "fullfact.org":        {"region": "FactChecker", "name": "Full Fact"},
    "boomlive.in":         {"region": "FactChecker", "name": "BoomLive"},
}

# Name lookup for Google News title matching
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
# LAYER 1 HELPERS
# ══════════════════════════════════════════════════════════════════
def extract_years(text: str) -> list:
    return [int(y) for y in re.findall(r'\b(20\d{2})\b', text)]

def run_distilbert(text: str) -> dict:
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = cls_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    real_p = float(probs[0])
    fake_p = float(probs[1])
    pred = int(probs.argmax())
    return {
        "real": real_p,
        "fake": fake_p,
        "prediction": pred,   # 0=REAL 1=FAKE
        "confidence": int(max(real_p, fake_p) * 100)
    }

# ══════════════════════════════════════════════════════════════════
# LAYER 2 — Query expansion
# ══════════════════════════════════════════════════════════════════
def generate_queries(claim: str) -> list:
    queries = [claim]
    proper = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
    years  = re.findall(r'\b20\d{2}\b', claim)
    actions = re.findall(
        r'\b(dead|died|death|arrested|elected|resigned|appointed|'
        r'banned|attacked|killed|won|lost|signed|launched|announced|released)\b',
        claim.lower()
    )
    if proper and actions:
        queries.append(" ".join(proper[:3]) + " " + actions[0])
    elif proper:
        queries.append(" ".join(proper[:4]))
    if years:
        queries.append(claim + " " + years[0])
    elif any(t in claim.lower() for t in ["nigeria","nigerian","abuja","lagos","jamb","waec","inec"]):
        queries.append(claim + " Nigeria")
    seen, unique = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    logger.info(f"Queries: {unique}")
    return unique[:3]

# ══════════════════════════════════════════════════════════════════
# LAYER 3 — Search (DuckDuckGo primary, Google News backup)
# ══════════════════════════════════════════════════════════════════
def search_duckduckgo(query: str) -> list:
    """DuckDuckGo — works perfectly on Railway (no IP blocking)"""
    for attempt in range(3):
        try:
            from duckduckgo_search import DDGS
            time.sleep(random.uniform(1, 3))
            with DDGS(headers={"User-Agent": random.choice(USER_AGENTS)}) as ddg:
                results = list(ddg.text(query, max_results=10, region="wt-wt", safesearch="off"))
            if results:
                logger.info(f"DuckDuckGo: {len(results)} results for '{query[:40]}'")
                return [{"href": r.get("href",""), "title": r.get("title",""), "body": r.get("body","")} for r in results]
        except Exception as e:
            logger.warning(f"DDG attempt {attempt+1}: {e}")
            time.sleep(random.uniform(2, 4))
    return []

def search_google_news(query: str, region: str = "US") -> list:
    """Google News RSS — free, unlimited, no API key"""
    try:
        q    = requests.utils.quote(query[:200])
        ceid = f"{region}:en"
        url  = f"https://news.google.com/rss/search?q={q}&hl=en&gl={region}&ceid={ceid}"
        resp = requests.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=12)
        results = []
        if resp.status_code == 200:
            for item in re.findall(r'<item>(.*?)</item>', resp.text, re.DOTALL)[:12]:
                t   = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', item) or re.findall(r'<title>(.*?)</title>', item)
                l   = re.findall(r'<link>(.*?)</link>', item)
                d   = re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>', item) or re.findall(r'<description>(.*?)</description>', item)
                pub = re.findall(r'<pubDate>(.*?)</pubDate>', item)
                if t:
                    clean_t = re.sub(r'\s+-\s+\S[\S]*\s*$', '', re.sub(r'<.*?>', '', t[0])).strip()
                    results.append({
                        "href":    l[0].strip() if l else "",
                        "title":   clean_t,
                        "body":    re.sub(r'<.*?>', '', d[0]).strip()[:400] if d else "",
                        "pubdate": pub[0].strip() if pub else ""
                    })
        logger.info(f"Google News ({region}): {len(results)} results")
        return results
    except Exception as e:
        logger.warning(f"Google News ({region}) failed: {e}")
        return []

def search_all(queries: list) -> list:
    """Run all queries, combine results, deduplicate"""
    all_results, seen_urls = [], set()
    for q in queries:
        # Try DuckDuckGo first (works on Railway!)
        for r in search_duckduckgo(q):
            url = r.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
        # Always also search Google News for Nigerian + global coverage
        for region in ["NG", "US"]:
            for r in search_google_news(q, region):
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        time.sleep(0.3)
    logger.info(f"Total unique results: {len(all_results)}")
    return all_results

# ══════════════════════════════════════════════════════════════════
# LAYER 4 — Article scraping
# ══════════════════════════════════════════════════════════════════
def scrape_article(url: str) -> str:
    """
    Fetch full article text using Jina AI Reader.
    Jina bypasses Cloudflare, paywalls, bot detection on ALL sites.
    Free, no API key needed.
    Just prepend r.jina.ai/ to any URL.
    """
    if not url or "news.google.com" in url:
        return ""
    try:
        # Method 1: Jina AI Reader — bypasses ALL bot protection
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
            # Clean up Jina metadata headers from response
            text = resp.text
            # Remove Jina header lines (Title:, URL:, Published Time: etc)
            lines = text.split("\n")
            content_lines = []
            skip_header = True
            for line in lines:
                if skip_header and line.startswith(("Title:", "URL:", "Published", "Source:", "Description:", "---", "===")):
                    continue
                skip_header = False
                content_lines.append(line)
            clean = "\n".join(content_lines).strip()
            if len(clean) > 200:
                logger.info(f"Jina scraped {len(clean)} chars from {url[:50]}")
                return clean[:5000]

        # Method 2: Direct BeautifulSoup fallback
        resp2 = requests.get(
            url,
            headers={"User-Agent": random.choice(USER_AGENTS)},
            timeout=8,
            allow_redirects=True
        )
        if resp2.status_code == 200:
            soup = BeautifulSoup(resp2.text, "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside","form","iframe"]):
                tag.decompose()
            for sel in ["article","[class*='article-body']","[class*='story-body']",
                        "[class*='post-content']","main",".content","#content"]:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text(separator=" ", strip=True)
                    if len(text) > 200:
                        return text[:5000]
            paras = soup.find_all("p")
            return " ".join(p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40)[:5000]

        return ""
    except Exception as e:
        logger.debug(f"Scrape failed {url[:50]}: {e}")
        return ""

# ══════════════════════════════════════════════════════════════════
# Source matching — handles Google News title suffix format
# ══════════════════════════════════════════════════════════════════
def match_sources(claim: str, results: list) -> list:
    """
    Match results to credible sources.
    Google News RSS returns news.google.com URLs so we match
    by title suffix: "Buhari dies - Premium Times Nigeria"
    """
    found, seen = [], set()
    claim_years = extract_years(claim)

    for r in results:
        url     = (r.get("href") or "").lower()
        title   = r.get("title") or ""
        snippet = r.get("body")  or ""

        # Date filter
        if claim_years:
            res_years = extract_years(title + " " + snippet)
            if res_years and not any(abs(ry-cy) <= 1 for ry in res_years for cy in claim_years):
                continue

        matched = None

        # Method 1: Direct URL match (DuckDuckGo results have real URLs)
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

        # Method 3: Domain keyword in title/snippet
        if not matched:
            combined = (title + " " + snippet).lower()
            for src, info in SOURCES.items():
                if src not in seen:
                    core = re.sub(r'\.(com|ng|org|co\.uk|co|tv|africa|net)$', '', src)
                    if len(core) > 5 and core in combined:
                        matched = src
                        break

        if matched:
            seen.add(matched)
            info = SOURCES[matched]
            raw_url = r.get("href") or ""
            article_url = raw_url if (raw_url.startswith("http") and "news.google.com" not in raw_url) else f"https://{matched}"
            found.append({
                "source":         matched,
                "name":           info["name"],
                "region":         info["region"],
                "url":            f"https://{matched}",
                "article_url":    article_url,
                "title":          title,
                "snippet":        snippet[:300],
                "pubdate":        r.get("pubdate", ""),
                "is_nigerian":    info["region"] in ("Nigeria","Nigeria-Edu"),
                "is_factchecker": info["region"] == "FactChecker",
            })

    found.sort(key=lambda x: (x["is_factchecker"]*2 + x["is_nigerian"]), reverse=True)
    logger.info(f"Sources matched: {[s['name'] for s in found]}")
    return found

# ══════════════════════════════════════════════════════════════════
# LAYER 5 — Phi-3-mini-128k reasoning via HF Inference API
# ══════════════════════════════════════════════════════════════════
def reason_with_phi3(claim: str, articles: list, source_titles: list) -> dict:
    """
    Send claim + scraped article text to Phi-3-mini-128k.
    Phi-3 reads and UNDERSTANDS the articles then reasons
    whether they support or contradict the claim.
    
    Example:
    Claim: "Tinubu is dead"
    Article: "TikToker arrested for falsely claiming Tinubu died"
    Phi-3: "The article says someone was ARRESTED for FALSELY
            claiming this. Tinubu is clearly ALIVE. FAKE."
    """
    if not HF_TOKEN:
        logger.warning("No HF_TOKEN set — skipping Phi-3 reasoning")
        return {"verdict": "unavailable", "reasoning": "HF token not configured", "confidence": 0}

    # Build evidence block from scraped articles + titles
    evidence_parts = []
    for i, (title, text) in enumerate(zip(source_titles[:4], articles[:4])):
        part = f"[Article {i+1}] {title}\n"
        if text and len(text) > 100:
            part += text[:1500]
        evidence_parts.append(part)

    if not evidence_parts:
        return {"verdict": "insufficient_evidence", "reasoning": "No articles could be scraped", "confidence": 0}

    evidence_block = "\n\n".join(evidence_parts)

    prompt = f"""You are a professional fact-checker. Read the articles below carefully and determine if the claim is TRUE or FALSE.

CLAIM TO VERIFY: "{claim}"

EVIDENCE FROM NEWS ARTICLES:
{evidence_block}

INSTRUCTIONS:
1. Read each article carefully and understand what it is actually saying
2. Determine if each article SUPPORTS or CONTRADICTS the claim
3. Pay careful attention to context — an article about someone being arrested for making a false claim DISPROVES that claim
4. Consider dates — if an article discusses future events as if they happened, that is suspicious
5. Give your final verdict

Respond in this exact format:
REASONING: [explain what each article actually says and whether it supports or contradicts the claim]
VERDICT: [SUPPORTS_CLAIM / CONTRADICTS_CLAIM / INSUFFICIENT_EVIDENCE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
EXPLANATION: [one sentence summary for the user]"""

    try:
        # Call Phi-3-mini-128k via HuggingFace Inference API
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": "microsoft/Phi-3-mini-128k-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional fact-checker. Be concise and precise."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1,
                "stream": False
            },
            timeout=45
        )

        if response.status_code == 200:
            raw = response.json()
            # Chat completions response format
            if isinstance(raw, dict) and "choices" in raw:
                text = raw["choices"][0]["message"]["content"]
            elif isinstance(raw, list) and raw:
                text = raw[0].get("generated_text", "")
            elif isinstance(raw, dict):
                text = raw.get("generated_text", str(raw))
            else:
                text = str(raw)

            logger.info(f"Phi-3 response: {text[:200]}")

            # Parse structured response
            verdict_match = re.search(r'VERDICT:\s*(SUPPORTS_CLAIM|CONTRADICTS_CLAIM|INSUFFICIENT_EVIDENCE)', text, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=VERDICT:|$)', text, re.IGNORECASE | re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', text, re.IGNORECASE)
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=\n|$)', text, re.IGNORECASE)

            verdict = verdict_match.group(1).upper() if verdict_match else "INSUFFICIENT_EVIDENCE"
            reasoning = reasoning_match.group(1).strip()[:600] if reasoning_match else text[:400]
            confidence = confidence_match.group(1).upper() if confidence_match else "LOW"
            explanation = explanation_match.group(1).strip() if explanation_match else reasoning[:150]

            conf_score = {"HIGH": 0.9, "MEDIUM": 0.65, "LOW": 0.4}.get(confidence, 0.4)

            return {
                "verdict":     verdict,
                "reasoning":   reasoning,
                "explanation": explanation,
                "confidence":  conf_score,
                "raw":         text[:800]
            }
        elif response.status_code == 503:
            logger.warning("Phi-3.5 model loading — retrying in 20s")
            time.sleep(20)
            return reason_with_phi3(claim, articles, source_titles)
        else:
            error_detail = response.text[:300] if response.text else "No details"
            logger.error(f"Phi-3.5 API error: {response.status_code} — {error_detail}")
            return {"verdict": "api_error", "reasoning": f"API error {response.status_code}: {error_detail}", "confidence": 0}

    except Exception as e:
        logger.error(f"Phi-3 reasoning failed: {e}")
        return {"verdict": "error", "reasoning": str(e), "confidence": 0}

# ══════════════════════════════════════════════════════════════════
# LAYER 6 — Final verdict fusion
# ══════════════════════════════════════════════════════════════════
def fuse_verdict(claim: str, distilbert: dict, matched: list, phi3: dict) -> dict:
    """
    Combine all signals into final verdict.
    Priority: Phi-3 reasoning > Fact-checkers > Source count > DistilBERT
    """
    nigerian  = [s for s in matched if s["is_nigerian"]]
    factcheck = [s for s in matched if s["is_factchecker"]]
    total     = len(matched)
    phi_v     = phi3.get("verdict", "INSUFFICIENT_EVIDENCE")
    phi_conf  = phi3.get("confidence", 0)
    phi_exp   = phi3.get("explanation", "")
    phi_ok    = phi_v not in ("unavailable", "api_error", "error", "insufficient_evidence", "INSUFFICIENT_EVIDENCE")

    fc_fake = [s for s in factcheck if any(w in s["title"].lower() for w in
               ["false","fake","misinformation","misleading","debunked","hoax","satire"])]
    fc_real = [s for s in factcheck if any(w in s["title"].lower() for w in
               ["true","confirmed","accurate","verified","correct"])]

    # ── Decision tree ──────────────────────────────────────────────
    if fc_fake:
        verdict, confidence = "FAKE", 99
        title_  = "Confirmed Misinformation"
        sub     = f"Flagged by {fc_fake[0]['name']}: {fc_fake[0]['title'][:60]}"

    elif fc_real and phi_v == "SUPPORTS_CLAIM":
        verdict, confidence = "REAL", 99
        title_  = "Fact-Checker Verified + AI Confirmed"
        sub     = f"Verified by {fc_real[0]['name']} — {phi_exp}"

    elif phi_ok and phi_v == "CONTRADICTS_CLAIM" and phi_conf >= 0.65:
        verdict, confidence = "FAKE", 97
        title_  = "AI Reading Contradicts Claim"
        sub     = f"Phi-3 read the articles and found: {phi_exp}"

    elif phi_ok and phi_v == "SUPPORTS_CLAIM" and phi_conf >= 0.65 and total >= 2:
        verdict, confidence = "REAL", 97
        title_  = "Confirmed — AI Read and Verified"
        sub     = f"Phi-3 read {total} articles: {phi_exp}"

    elif phi_ok and phi_v == "SUPPORTS_CLAIM" and phi_conf >= 0.65 and total >= 1:
        verdict, confidence = "REAL", 92
        title_  = "AI Reading Confirms Claim"
        sub     = f"Phi-3 read the articles and found: {phi_exp}"

    elif phi_ok and phi_v == "SUPPORTS_CLAIM" and total == 0:
        verdict, confidence = "REAL", 72
        title_  = "AI Reading Finds Support — No Major Outlets"
        sub     = f"{phi_exp}. Verify through additional sources."

    elif total >= 3 and phi_v != "CONTRADICTS_CLAIM":
        verdict, confidence = "REAL", 90
        regions = list(dict.fromkeys(s["region"] for s in matched[:4]))
        title_  = "Confirmed by Multiple Outlets"
        sub     = f"Found in {total} credible outlets: {', '.join(regions)}"

    elif len(nigerian) >= 2:
        verdict, confidence = "REAL", 88
        title_  = "Confirmed by Nigerian Outlets"
        sub     = f"{nigerian[0]['name']} and {nigerian[1]['name']} report this"

    elif total >= 2:
        verdict, confidence = "REAL", 80
        title_  = "Likely Authentic News"
        sub     = f"Found in {matched[0]['name']} and {matched[1]['name']}"

    elif total == 1 and phi_v != "CONTRADICTS_CLAIM":
        verdict, confidence = "REAL", 65
        title_  = "Possibly Authentic"
        sub     = f"Found in 1 outlet: {matched[0]['name']}. Verify further."

    elif phi_ok and phi_v == "CONTRADICTS_CLAIM":
        verdict, confidence = "FAKE", 88
        title_  = "AI Reading Contradicts Claim"
        sub     = f"Phi-3 analysis: {phi_exp}"

    elif distilbert["prediction"] == 1 and distilbert["confidence"] >= 80:
        verdict, confidence = "FAKE", 82
        title_  = "Likely Fake — AI + No Sources"
        sub     = "Misinformation patterns detected. Zero credible sources found worldwide."

    elif distilbert["prediction"] == 0 and distilbert["confidence"] >= 80:
        verdict, confidence = "SUSPICIOUS", 60
        title_  = "Suspicious — Cannot Verify"
        sub     = "Writing appears credible but no outlet worldwide confirms this claim."

    else:
        verdict, confidence = "MIXED", 50
        title_  = "Uncertain — Verify Manually"
        sub     = "Mixed signals. Use the fact-checker links below to verify."

    return {"verdict": verdict, "confidence": confidence, "title": title_, "subtitle": sub}

# ══════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════
class AnalyseRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "name": "TruthLens API v6.0",
        "institution": "National Open University of Nigeria (NOUN)",
        "developer": "Jabir Muhammad Kabir",
        "supervisor": "Dr. Ojeniyi Adebayo",
        "status": "running",
        "pipeline": ["DistilBERT","DuckDuckGo","GoogleNewsRSS","BeautifulSoup","Phi3-mini-128k","SmartFusion"],
        "sources": len(SOURCES)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "phi3": bool(HF_TOKEN)}

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
    current_year = datetime.now().year
    claim_years  = extract_years(clean)
    future_years = [y for y in claim_years if y > current_year]

    # Instant reject: future year
    if future_years:
        return _future_verdict(future_years[0])

    # Cache check
    ck = cache_key(clean)
    cached = cache_get(ck)
    if cached:
        tick(cache_hit=True)
        cached["from_cache"] = True
        return cached

    tick(cache_hit=False)

    # ── Layer 1: DistilBERT ────────────────────────────────────────
    db = run_distilbert(clean)
    logger.info(f"DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%")

    # ── Layer 2: Query expansion ───────────────────────────────────
    queries = generate_queries(clean)

    # ── Layer 3: Search ────────────────────────────────────────────
    results = search_all(queries)

    # ── Source matching ────────────────────────────────────────────
    matched = match_sources(clean, results)
    factcheck = [s for s in matched if s["is_factchecker"]]

    # ── Layer 4: Scrape articles ───────────────────────────────────
    scraped_texts  = []
    scraped_titles = []
    for src in matched[:5]:
        if src["article_url"] and "news.google.com" not in src["article_url"]:
            text_content = scrape_article(src["article_url"])
            if text_content and len(text_content) > 100:
                scraped_texts.append(text_content)
                scraped_titles.append(src["title"])
                logger.info(f"Scraped {len(text_content)} chars from {src['name']}")

    # Also use snippets as evidence when scraping fails
    if not scraped_texts:
        for r in results[:8]:
            if r.get("body") and len(r.get("body","")) > 50:
                scraped_texts.append(r["body"])
                scraped_titles.append(r.get("title",""))

    # ── Layer 5: Phi-3-mini reasoning ─────────────────────────────
    phi3 = reason_with_phi3(clean, scraped_texts, scraped_titles)
    logger.info(f"Phi-3: {phi3.get('verdict')} conf={phi3.get('confidence')}")

    # ── Layer 6: Fuse verdict ──────────────────────────────────────
    fusion = fuse_verdict(clean, db, matched, phi3)
    verdict    = fusion["verdict"]
    confidence = fusion["confidence"]

    # ── Signals ────────────────────────────────────────────────────
    upper  = sum(1 for w in clean.split() if w.isupper() and len(w) > 2)
    excl   = clean.count("!")
    has_attr = any(w in clean.lower() for w in ["according to","reported","said","confirmed","announced"])
    clickbait = any(w in clean.lower() for w in ["shocking","secret","exposed","leaked","won't believe","they don't want","viral","breaking"])

    phi_signal = {
        "SUPPORTS_CLAIM":       ("AI read articles — SUPPORTS claim", "pos"),
        "CONTRADICTS_CLAIM":    ("AI read articles — CONTRADICTS claim", "neg"),
        "INSUFFICIENT_EVIDENCE":("AI reading — insufficient evidence", "neu"),
    }.get(phi3.get("verdict",""), ("AI reasoning unavailable", "neu"))

    signals = [
        {"icon":"🌐","label":"Web Sources",    "value":f"{len(matched)} credible outlets found",              "type":"pos" if len(matched)>=2 else "neg" if len(matched)==0 else "neu"},
        {"icon":"🤖","label":"DistilBERT AI",  "value":f"{'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%","type":"pos" if db['prediction']==0 else "neg"},
        {"icon":"🧠","label":"Phi-3 Reasoning","value":phi_signal[0],                                         "type":phi_signal[1]},
        {"icon":"📰","label":"Articles Scraped","value":f"{len(scraped_texts)} full articles read",            "type":"pos" if len(scraped_texts)>0 else "neu"},
        {"icon":"😱","label":"Tone Analysis",  "value":"Sensationalist" if upper>2 or excl>2 else "Measured", "type":"neg" if upper>2 else "pos"},
        {"icon":"🎣","label":"Clickbait",      "value":"Clickbait detected" if clickbait else "None detected","type":"neg" if clickbait else "pos"},
    ]

    # ── Analysis text ──────────────────────────────────────────────
    phi_reasoning = phi3.get("reasoning","") or phi3.get("explanation","")
    if verdict == "REAL":
        src_list = ", ".join(s["name"] for s in matched[:3]) if matched else "none"
        analysis = (
            f"This content appears CREDIBLE.\n\n"
            f"LAYER 1 — DistilBERT AI: Classified as {'REAL' if db['prediction']==0 else 'FAKE'} "
            f"at {db['confidence']}% confidence based on linguistic patterns.\n\n"
            f"LAYER 2 — WEB SEARCH: Found {len(matched)} credible outlet(s) including {src_list}. "
            f"Searched via {len(queries)} targeted queries using DuckDuckGo and Google News.\n\n"
            f"LAYER 3 — PHI-3 READING: {phi_reasoning[:300] if phi_reasoning else 'Article reasoning completed.'}\n\n"
            f"RECOMMENDATION: Content appears credible. Click the source links below to read the full articles."
        )
    elif verdict in ("FAKE", "SUSPICIOUS"):
        analysis = (
            f"This content is LIKELY FAKE OR MISLEADING.\n\n"
            f"LAYER 1 — DistilBERT AI: Classified as {'REAL' if db['prediction']==0 else 'FAKE'} "
            f"at {db['confidence']}% confidence.\n\n"
            f"LAYER 2 — WEB SEARCH: {'Zero credible sources found across 60+ global outlets.' if len(matched)==0 else f'Found {len(matched)} source(s) but content contradicts the claim.'} "
            f"Searched via {len(queries)} targeted queries.\n\n"
            f"LAYER 3 — PHI-3 READING: {phi_reasoning[:300] if phi_reasoning else 'No supporting articles found.'}\n\n"
            f"RECOMMENDATION: Do not share. Verify through AfricaCheck (africacheck.org) or Dubawa (dubawa.org)."
        )
    else:
        analysis = (
            f"UNCERTAIN RESULT: Mixed signals from our pipeline.\n\n"
            f"This may be very recent breaking news, an opinion piece, satire, or outside our training domain.\n\n"
            f"LAYER 1 — DistilBERT: {db['confidence']}% confidence {'REAL' if db['prediction']==0 else 'FAKE'}.\n"
            f"LAYER 2 — Web: {len(matched)} source(s) found.\n"
            f"LAYER 3 — Phi-3: {phi_reasoning[:200] if phi_reasoning else 'Inconclusive.'}\n\n"
            f"RECOMMENDATION: Search Google News and check AfricaCheck or Dubawa before sharing."
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
            "verdict":     phi3.get("verdict",""),
            "reasoning":   phi_reasoning[:500],
            "confidence":  phi3.get("confidence", 0),
        },
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
        "articles_scraped": len(scraped_texts),
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
        "analysis": f"TEMPORAL CHECK: This content references {year}, a future year. Events that have not occurred cannot be reported as facts.\n\nRECOMMENDATION: Do not share.",
        "signals": [
            {"icon":"📅","label":"Date Check",     "value":f"Future year {year} detected","type":"neg"},
            {"icon":"🚫","label":"Verdict",        "value":"Impossible — future event",   "type":"neg"},
            {"icon":"🤖","label":"DistilBERT",     "value":"Temporal impossibility",      "type":"neg"},
            {"icon":"🧠","label":"Phi-3 Reasoning","value":"Not needed",                  "type":"neg"},
            {"icon":"🌐","label":"Web Search",     "value":"Not needed",                  "type":"neg"},
            {"icon":"⚠️","label":"Warning",        "value":"Do not share",                "type":"neg"},
        ],
        "real_probability": 0.01, "fake_probability": 0.99,
        "phi3": {"verdict":"N/A","reasoning":"Future date detected","confidence":0},
        "sources_found": [], "fact_checks": [],
        "queries_used": [], "articles_scraped": 0,
        "search_timestamp": datetime.now().isoformat(),
        "from_cache": False, "daily_stats": {}
    }
