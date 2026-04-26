"""
TruthLens API v8.0 — Production Ready
National Open University of Nigeria (NOUN)
Developer: Jabir Muhammad Kabir
Supervisor: Dr. Ojeniyi Adebayo
"""

import os, re, time, hashlib, logging, random
from datetime import datetime, date
from collections import defaultdict

import torch
import torch.nn.functional as F
import requests
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truthlens")

# ── Environment ────────────────────────────────────────────────────
MODEL_ID   = os.environ.get("MODEL_ID", "Jabirkabir/truthlens-fakenews-detector")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
TAVILY_KEY = os.environ.get("TAVILY_KEY", "")

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="TruthLens API", version="8.0", docs_url=None, redoc_url=None)
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
            return JSONResponse(status_code=429, content={"error": "Too many requests."})
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
CACHE_TTL = 3600 * 3

def cache_key(text: str) -> str:
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
for _src, _info in SOURCES.items():
    _n = _info["name"].lower()
    NAME_LOOKUP[_n] = _src
    _parts = _n.split()
    if len(_parts) >= 2:
        NAME_LOOKUP[" ".join(_parts[:2])] = _src

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
    probs = F.softmax(logits, dim=-1)[0]
    real_p = float(probs[0])
    fake_p = float(probs[1])
    return {
        "real": real_p, "fake": fake_p,
        "prediction": int(probs.argmax()),
        "confidence": int(max(real_p, fake_p) * 100)
    }

def generate_queries(claim: str) -> list:
    queries = [claim]
    proper  = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
    years   = re.findall(r'\b20\d{2}\b', claim)
    actions = re.findall(
        r'\b(dead|died|death|arrested|elected|resigned|appointed|'
        r'banned|killed|won|lost|announced|released|out|offers|offering)\b',
        claim.lower()
    )
    if proper and actions:
        queries.append(" ".join(proper[:3]) + " " + actions[0])
    elif proper:
        queries.append(" ".join(proper[:4]))
    current_year = datetime.now().year
    if proper:
        queries.append(f"{' '.join(proper[:2])} {current_year}")
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
# LAYER 2a — Tavily Search
# ══════════════════════════════════════════════════════════════════
def search_tavily(query: str) -> tuple:
    """
    Returns (list_of_results, answer_string)
    NOT restricting domains so it searches the whole web
    then we filter by credible sources after
    """
    if not TAVILY_KEY:
        logger.warning("TAVILY_KEY not set — skipping")
        return [], ""
    try:
        payload = {
            "api_key":             TAVILY_KEY,
            "query":               query,
            "search_depth":        "advanced",
            "include_answer":      True,
            "include_raw_content": True,
            "max_results":         10,
        }
        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=20
        )
        logger.info(f"Tavily status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for r in data.get("results", []):
                results.append({
                    "href":        r.get("url", ""),
                    "title":       r.get("title", ""),
                    "body":        r.get("content", "")[:500],
                    "raw_content": r.get("raw_content", "")[:4000],
                    "pubdate":     r.get("published_date", ""),
                    "source":      "tavily"
                })
            answer = data.get("answer", "") or ""
            logger.info(f"Tavily: {len(results)} results, answer={bool(answer)}")
            return results, answer
        else:
            logger.error(f"Tavily error {resp.status_code}: {resp.text[:200]}")
            return [], ""
    except Exception as e:
        logger.error(f"Tavily exception: {e}")
        return [], ""

# ══════════════════════════════════════════════════════════════════
# LAYER 2b — Google News RSS
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
                    clean_t = re.sub(
                        r'\s+-\s+\S[\S]*\s*$', '',
                        re.sub(r'<.*?>', '', t[0])
                    ).strip()
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
    all_results, seen_urls = [], set()
    tavily_answers = []

    def add(items):
        for r in items:
            url = r.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    for q in queries:
        if TAVILY_KEY:
            tv_results, tv_answer = search_tavily(q)
            add(tv_results)
            if tv_answer and len(tv_answer) > 20:
                tavily_answers.append(tv_answer)
        add(search_google_news(q, "NG"))
        add(search_google_news(q, "US"))
        time.sleep(0.3)

    logger.info(f"Total unique results: {len(all_results)}")
    return all_results, " ".join(tavily_answers[:2])

# ══════════════════════════════════════════════════════════════════
# LAYER 3 — Jina Reader
# ══════════════════════════════════════════════════════════════════
def read_article(url: str) -> str:
    if not url or "news.google.com" in url:
        return ""
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/plain",
            },
            timeout=15
        )
        if resp.status_code == 200 and len(resp.text) > 200:
            lines = resp.text.split("\n")
            body = [
                l for l in lines
                if not l.startswith(
                    ("Title:", "URL:", "Published", "Source:", "Description:", "---", "===")
                )
            ]
            clean = "\n".join(body).strip()
            if len(clean) > 200:
                logger.info(f"Jina: {len(clean)} chars from {url[:50]}")
                return clean[:5000]
        return ""
    except Exception as e:
        logger.debug(f"Jina failed {url[:50]}: {e}")
        return ""

# ══════════════════════════════════════════════════════════════════
# Source matching
# ══════════════════════════════════════════════════════════════════
def match_sources(claim: str, results: list) -> list:
    found, seen = [], set()
    claim_years = extract_years(claim)

    for r in results:
        url     = (r.get("href") or "").lower()
        title   = r.get("title") or ""
        snippet = r.get("body")  or ""
        pubdate = r.get("pubdate") or ""

        if claim_years and pubdate:
            pub_years = extract_years(pubdate)
            if pub_years and not any(
                abs(py - cy) <= 1
                for py in pub_years for cy in claim_years
            ):
                logger.info(f"Date filtered: {title[:50]}")
                continue

        matched = None

        # Method 1: Direct URL
        for src in SOURCES:
            if src in url and src not in seen:
                matched = src
                break

        # Method 2: Google News title suffix
        if not matched and " - " in title:
            suffix = title.split(" - ")[-1].strip().lower()
            if suffix in NAME_LOOKUP and NAME_LOOKUP[suffix] not in seen:
                matched = NAME_LOOKUP[suffix]
            else:
                for name, src in NAME_LOOKUP.items():
                    if name in suffix and src not in seen and len(name) > 4:
                        matched = src
                        break

        # Method 3: Domain keyword in text
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
            article_url = (
                raw_url
                if raw_url.startswith("http") and "news.google.com" not in raw_url
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
                "raw_content":    r.get("raw_content", ""),
                "pubdate":        pubdate,
                "is_nigerian":    info["region"] in ("Nigeria", "Nigeria-Edu"),
                "is_factchecker": info["region"] == "FactChecker",
            })

    found.sort(key=lambda x: (x["is_factchecker"] * 2 + x["is_nigerian"]), reverse=True)
    logger.info(f"Sources matched: {[s['name'] for s in found]}")
    return found

# ══════════════════════════════════════════════════════════════════
# LAYER 4 — Phi-3.5-mini reasoning
# ══════════════════════════════════════════════════════════════════
def reason_with_phi35(
    claim: str,
    articles: list,
    tavily_summary: str
) -> dict:
    if not HF_TOKEN:
        return {"verdict": "unavailable", "reasoning": "HF_TOKEN not set", "confidence": 0}

    evidence_parts = []
    today = datetime.now().strftime("%d %B %Y")

    if tavily_summary and len(tavily_summary) > 30:
        evidence_parts.append(
            f"[Web Summary — searched today {today}]\n{tavily_summary[:600]}"
        )

    for i, (title, text, pubdate, source) in enumerate(articles[:5]):
        part = f"[Article {i+1} — {source}]"
        if pubdate:
            part += f" [Date: {pubdate}]"
        part += f"\nHeadline: {title}"
        if text and len(text) > 100:
            part += f"\nContent: {text[:2000]}"
        evidence_parts.append(part)

    if not evidence_parts:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reasoning": "No articles retrieved",
            "confidence": 0
        }

    evidence = "\n\n".join(evidence_parts)

    prompt = f"""You are TruthLens, a professional AI fact-checker. Today is {today}.

CLAIM: "{claim}"

EVIDENCE:
{evidence}

Analyse each article carefully:
- An article about someone being ARRESTED for claiming X means X is FALSE
- Check if articles are actually about the claim or just mention related words
- Check publication dates — old articles may not reflect current reality
- "Tinubu mourns someone" does NOT mean Tinubu is dead
- "FUT Minna offers law" — check if this is confirmed by the university itself

RESPOND EXACTLY LIKE THIS:
REASONING: [what each article actually says, step by step]
VERDICT: SUPPORTS_CLAIM or CONTRADICTS_CLAIM or INSUFFICIENT_EVIDENCE
CONFIDENCE: HIGH or MEDIUM or LOW
EXPLANATION: [one sentence for the user]"""

    try:
        resp = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type":  "application/json"
            },
            json={
                "model": "microsoft/Phi-3.5-mini-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are TruthLens, a professional fact-checker. Today is {today}. Be precise and analytical."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 700,
                "temperature": 0.1,
                "stream": False
            },
            timeout=50
        )

        logger.info(f"Phi-3.5 status: {resp.status_code}")

        if resp.status_code == 200:
            raw  = resp.json()
            if isinstance(raw, dict) and "choices" in raw:
                text = raw["choices"][0]["message"]["content"]
            elif isinstance(raw, list) and raw:
                text = raw[0].get("generated_text", "")
            else:
                text = str(raw)

            logger.info(f"Phi-3.5 raw response: {text[:400]}")

            v_m = re.search(
                r'VERDICT[:\s]+(SUPPORTS_CLAIM|CONTRADICTS_CLAIM|INSUFFICIENT_EVIDENCE)',
                text, re.IGNORECASE
            )
            r_m = re.search(r'REASONING[:\s]+(.*?)(?=VERDICT:|$)', text, re.IGNORECASE | re.DOTALL)
            c_m = re.search(r'CONFIDENCE[:\s]+(HIGH|MEDIUM|LOW)', text, re.IGNORECASE)
            e_m = re.search(r'EXPLANATION[:\s]+(.*?)(?=\n\n|$)', text, re.IGNORECASE | re.DOTALL)

            verdict     = v_m.group(1).upper() if v_m else "INSUFFICIENT_EVIDENCE"
            reasoning   = r_m.group(1).strip()[:800] if r_m else text[:600]
            confidence  = c_m.group(1).upper() if c_m else "LOW"
            explanation = e_m.group(1).strip()[:250] if e_m else reasoning[:150]
            conf_score  = {"HIGH": 0.9, "MEDIUM": 0.65, "LOW": 0.4}.get(confidence, 0.4)

            return {
                "verdict":     verdict,
                "reasoning":   reasoning,
                "explanation": explanation,
                "confidence":  conf_score,
            }

        elif resp.status_code == 503:
            logger.warning("Phi-3.5 loading — retry in 20s")
            time.sleep(20)
            return reason_with_phi35(claim, articles, tavily_summary)
        else:
            err = resp.text[:300]
            logger.error(f"Phi-3.5 {resp.status_code}: {err}")
            return {
                "verdict":   "api_error",
                "reasoning": f"Phi-3.5 API error {resp.status_code}: {err}",
                "confidence": 0
            }

    except Exception as e:
        logger.error(f"Phi-3.5 exception: {e}")
        return {"verdict": "error", "reasoning": str(e), "confidence": 0}

# ══════════════════════════════════════════════════════════════════
# LAYER 5 — Verdict fusion
# ══════════════════════════════════════════════════════════════════
def fuse_verdict(distilbert: dict, matched: list, phi35: dict) -> dict:
    nigerian  = [s for s in matched if s["is_nigerian"]]
    factcheck = [s for s in matched if s["is_factchecker"]]
    total     = len(matched)
    phi_v     = phi35.get("verdict", "INSUFFICIENT_EVIDENCE")
    phi_conf  = phi35.get("confidence", 0)
    phi_exp   = phi35.get("explanation", "")
    phi_ok    = phi_v not in (
        "unavailable", "api_error", "error",
        "insufficient_evidence", "INSUFFICIENT_EVIDENCE", ""
    )

    fc_fake = [s for s in factcheck if any(
        w in s["title"].lower()
        for w in ["false","fake","misinformation","misleading","debunked","hoax","satire"]
    )]
    fc_real = [s for s in factcheck if any(
        w in s["title"].lower()
        for w in ["true","confirmed","accurate","verified","correct"]
    )]

    # Priority 1 — Fact checkers
    if fc_fake:
        return {"verdict":"FAKE","confidence":99,
                "title":"Confirmed Misinformation",
                "subtitle":f"Flagged by {fc_fake[0]['name']}: {fc_fake[0]['title'][:60]}"}
    if fc_real:
        return {"verdict":"REAL","confidence":99,
                "title":"Fact-Checker Verified",
                "subtitle":f"Confirmed by {fc_real[0]['name']}"}

    # Priority 2 — Phi-3.5 reasoning (WINS over source count)
    if phi_ok and phi_v == "CONTRADICTS_CLAIM" and phi_conf >= 0.5:
        return {"verdict":"FAKE","confidence":95,
                "title":"AI Reading Contradicts Claim",
                "subtitle":f"Phi-3.5 read the articles: {phi_exp}"}

    if phi_ok and phi_v == "SUPPORTS_CLAIM" and phi_conf >= 0.65 and total >= 2:
        return {"verdict":"REAL","confidence":95,
                "title":"Confirmed — AI Read and Verified",
                "subtitle":f"Phi-3.5 confirmed across {total} sources: {phi_exp}"}

    if phi_ok and phi_v == "SUPPORTS_CLAIM" and phi_conf >= 0.65:
        return {"verdict":"REAL","confidence":85,
                "title":"AI Reading Confirms Claim",
                "subtitle":phi_exp}

    # Priority 3 — Source count (only if Phi did not contradict)
    if total >= 3 and phi_v != "CONTRADICTS_CLAIM":
        regions = list(dict.fromkeys(s["region"] for s in matched[:4]))
        return {"verdict":"REAL","confidence":85,
                "title":"Confirmed by Multiple Outlets",
                "subtitle":f"Found in {total} credible outlets: {', '.join(regions)}"}

    if len(nigerian) >= 2 and phi_v != "CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":82,
                "title":"Confirmed by Nigerian Outlets",
                "subtitle":f"{nigerian[0]['name']} and {nigerian[1]['name']}"}

    if total >= 2 and phi_v != "CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":75,
                "title":"Likely Authentic News",
                "subtitle":f"Found in {matched[0]['name']} and {matched[1]['name']}"}

    if total == 1 and phi_v != "CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":60,
                "title":"Possibly Authentic",
                "subtitle":f"1 outlet: {matched[0]['name']}. Verify further."}

    # Priority 4 — DistilBERT fallback
    if distilbert["prediction"] == 1 and distilbert["confidence"] >= 80:
        return {"verdict":"FAKE","confidence":78,
                "title":"Likely Fake — No Sources Found",
                "subtitle":"Misinformation patterns detected. Zero credible sources found."}

    if distilbert["prediction"] == 0 and distilbert["confidence"] >= 80:
        return {"verdict":"SUSPICIOUS","confidence":55,
                "title":"Suspicious — Cannot Verify",
                "subtitle":"Writing appears credible but no outlet confirms this claim."}

    return {"verdict":"MIXED","confidence":50,
            "title":"Uncertain — Verify Manually",
            "subtitle":"Mixed signals. Use the fact-checker links below."}

# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════
class AnalyseRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "name":    "TruthLens API v8.0",
        "status":  "running",
        "tavily":  bool(TAVILY_KEY),
        "phi35":   bool(HF_TOKEN),
        "sources": len(SOURCES),
    }

@app.get("/health")
def health():
    return {"status":"healthy","tavily":bool(TAVILY_KEY),"phi35":bool(HF_TOKEN)}

@app.get("/stats")
def stats():
    return COUNTER.copy()

@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    text = req.text.strip()
    if not text or len(text) < 8:
        raise HTTPException(400, "Text too short")
    if len(text) > 8000:
        raise HTTPException(400, "Text too long")

    clean        = re.sub(r'http\S+|www\S+|<.*?>|\s+', ' ', text).strip()[:600]
    current_year = datetime.now().year
    claim_years  = extract_years(clean)
    future_years = [y for y in claim_years if y > current_year]

    if future_years:
        return _future_verdict(future_years[0])

    ck = cache_key(clean)
    cached = cache_get(ck)
    if cached:
        tick(cache_hit=True)
        cached["from_cache"] = True
        return cached

    tick()

    # Layer 1
    db = run_distilbert(clean)
    logger.info(f"DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%")

    # Layer 2
    queries = generate_queries(clean)
    results, tavily_summary = search_all(queries)

    # Source matching
    matched   = match_sources(clean, results)
    factcheck = [s for s in matched if s["is_factchecker"]]

    # Layer 3 — Read articles
    articles_for_phi = []
    for src in matched[:5]:
        raw = src.get("raw_content", "")
        if raw and len(raw) > 200:
            full_text = raw
        else:
            url = src["article_url"]
            full_text = read_article(url) if url and "news.google.com" not in url else ""
        articles_for_phi.append((
            src["title"], full_text, src["pubdate"], src["name"]
        ))

    # Add Tavily raw content from unmatched results
    for r in results[:8]:
        rc = r.get("raw_content", "")
        if rc and len(rc) > 300:
            title = r.get("title", "")
            if not any(t == title for t, _, _, _ in articles_for_phi):
                src_name = "Web"
                for sd in SOURCES:
                    if sd in (r.get("href") or "").lower():
                        src_name = SOURCES[sd]["name"]
                        break
                articles_for_phi.append(
                    (title, rc, r.get("pubdate",""), src_name)
                )

    scraped = sum(1 for _, t, _, _ in articles_for_phi if len(t) > 200)

    # Layer 4 — Phi-3.5
    phi35 = reason_with_phi35(clean, articles_for_phi, tavily_summary)
    logger.info(f"Phi-3.5: verdict={phi35.get('verdict')} conf={phi35.get('confidence')}")

    # Layer 5 — Fuse
    fusion     = fuse_verdict(db, matched, phi35)
    verdict    = fusion["verdict"]
    confidence = fusion["confidence"]

    # Signals
    upper     = sum(1 for w in clean.split() if w.isupper() and len(w) > 2)
    excl      = clean.count("!")
    clickbait = any(w in clean.lower() for w in
                ["shocking","secret","exposed","leaked","won't believe","viral","breaking"])

    phi_sig = {
        "SUPPORTS_CLAIM":        ("Read articles — SUPPORTS claim",       "pos"),
        "CONTRADICTS_CLAIM":     ("Read articles — CONTRADICTS claim",    "neg"),
        "INSUFFICIENT_EVIDENCE": ("Insufficient article content",          "neu"),
    }.get(phi35.get("verdict",""), ("Reasoning unavailable — check HF_TOKEN", "neu"))

    signals = [
        {"icon":"🌐","label":"Web Sources",
         "value":f"{len(matched)} credible outlets found",
         "type":"pos" if len(matched)>=2 else "neg" if len(matched)==0 else "neu"},
        {"icon":"🤖","label":"DistilBERT",
         "value":f"{'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%",
         "type":"pos" if db['prediction']==0 else "neg"},
        {"icon":"🧠","label":"Phi-3.5 Reading",
         "value":phi_sig[0],"type":phi_sig[1]},
        {"icon":"📰","label":"Articles Read",
         "value":f"{scraped} full articles analysed",
         "type":"pos" if scraped>0 else "neu"},
        {"icon":"😱","label":"Tone",
         "value":"Sensationalist" if upper>2 or excl>2 else "Measured",
         "type":"neg" if upper>2 else "pos"},
        {"icon":"🎣","label":"Clickbait",
         "value":"Detected" if clickbait else "None detected",
         "type":"neg" if clickbait else "pos"},
    ]

    phi_reasoning = phi35.get("reasoning","") or phi35.get("explanation","") or "No reasoning returned"

    if verdict == "REAL":
        src_list = ", ".join(s["name"] for s in matched[:3]) if matched else "none"
        analysis = (
            f"This content appears CREDIBLE.\n\n"
            f"LAYER 1 — DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} at {db['confidence']}% "
            f"(raw AI score before web verification).\n\n"
            f"LAYER 2 — WEB: Found {len(matched)} outlet(s) including {src_list}. "
            f"Searched via {len(queries)} queries using Tavily + Google News.\n\n"
            f"LAYER 3 — PHI-3.5: {phi_reasoning[:500]}\n\n"
            f"RECOMMENDATION: Content appears credible. Click sources below to verify."
        )
    elif verdict in ("FAKE","SUSPICIOUS"):
        analysis = (
            f"This content is LIKELY FAKE OR MISLEADING.\n\n"
            f"LAYER 1 — DistilBERT: {db['confidence']}% {'REAL' if db['prediction']==0 else 'FAKE'}.\n\n"
            f"LAYER 2 — WEB: {'Zero credible sources found.' if not matched else f'{len(matched)} sources found but content contradicts the claim.'}\n\n"
            f"LAYER 3 — PHI-3.5: {phi_reasoning[:500]}\n\n"
            f"RECOMMENDATION: Do not share. Verify through AfricaCheck or Dubawa."
        )
    else:
        analysis = (
            f"UNCERTAIN: Mixed signals.\n\n"
            f"DistilBERT: {db['confidence']}% {'REAL' if db['prediction']==0 else 'FAKE'}. "
            f"Sources: {len(matched)} found. "
            f"Phi-3.5: {phi_reasoning[:300]}\n\n"
            f"RECOMMENDATION: Check AfricaCheck or Dubawa before sharing."
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
            "verdict":     phi35.get("verdict",""),
            "reasoning":   phi_reasoning,
            "explanation": phi35.get("explanation",""),
            "confidence":  phi35.get("confidence", 0),
        },
        "tavily_summary":   tavily_summary[:300] if tavily_summary else "",
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
        "articles_scraped": scraped,
        "search_timestamp": datetime.now().isoformat(),
        "from_cache":       False,
        "daily_stats":      COUNTER.copy(),
    }

    cache_set(ck, result)
    return result


def _future_verdict(year: int) -> dict:
    return {
        "verdict":"FAKE","confidence":99,
        "verdict_title":"Future Event Claimed as Fact",
        "verdict_subtitle":f"References {year} which has not happened yet.",
        "analysis":f"This content references {year}, a future year. Cannot be reported as fact.\n\nDo not share.",
        "signals":[
            {"icon":"📅","label":"Date","value":f"Future year {year}","type":"neg"},
            {"icon":"🚫","label":"Verdict","value":"Impossible","type":"neg"},
            {"icon":"🤖","label":"DistilBERT","value":"N/A","type":"neg"},
            {"icon":"🧠","label":"Phi-3.5","value":"N/A","type":"neg"},
            {"icon":"🌐","label":"Web","value":"N/A","type":"neg"},
            {"icon":"⚠️","label":"Warning","value":"Do not share","type":"neg"},
        ],
        "real_probability":0.01,"fake_probability":0.99,
        "phi3":{"verdict":"N/A","reasoning":"Future date","confidence":0},
        "tavily_summary":"",
        "sources_found":[],"fact_checks":[],
        "queries_used":[],"articles_scraped":0,
        "search_timestamp":datetime.now().isoformat(),
        "from_cache":False,"daily_stats":{}
    }
