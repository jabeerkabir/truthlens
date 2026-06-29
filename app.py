"""
TruthLens API v10.0 — Hardened Fact Verification
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

MODEL_ID   = os.environ.get("MODEL_ID", "Jabirkabir/truthlens-fakenews-detector")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
TAVILY_KEY = os.environ.get("TAVILY_KEY", "")
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")

LLM_MODEL = "llama-3.1-8b-instant"

app = FastAPI(title="TruthLens API", version="10.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET","POST"], allow_headers=["Content-Type"],
)

_rate = defaultdict(list)

@app.middleware("http")
async def security(request: FastAPIRequest, call_next):
    if request.url.path == "/analyse":
        ip = getattr(request.client, "host", "unknown")
        now = time.time()
        _rate[ip] = [t for t in _rate[ip] if now - t < 60]
        if len(_rate[ip]) >= 20:
            return JSONResponse(status_code=429, content={"error":"Too many requests."})
        _rate[ip].append(now)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp

logger.info(f"Loading DistilBERT: {MODEL_ID}")
cls_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
cls_model.eval()
logger.info("DistilBERT ready!")

CACHE: dict = {}
CACHE_TTL        = 3600 * 3  # 3 hours — standard claims
CACHE_TTL_SPORT  = 300        # 5 min  — live sports
CACHE_TTL_DEATH  = 600        # 10 min — death/alive claims (can change fast)
CACHE_TTL_BREAK  = 900        # 15 min — breaking news

def cache_key(text):
    return hashlib.md5(f"{date.today()}:{text.lower().strip()}".encode()).hexdigest()

def cache_get(key, ttl=CACHE_TTL):
    e = CACHE.get(key)
    if e and time.time() - e["ts"] < ttl:
        return e["v"]
    return None

def cache_set(key, value):
    if len(CACHE) > 400:
        for k, _ in sorted(CACHE.items(), key=lambda x: x[1]["ts"])[:100]:
            del CACHE[k]
    CACHE[key] = {"ts": time.time(), "v": value}

COUNTER = {"date": str(date.today()), "searches": 0, "cache_hits": 0}

def tick(cache_hit=False):
    today = str(date.today())
    if COUNTER["date"] != today:
        COUNTER.update({"date": today, "searches": 0, "cache_hits": 0})
    COUNTER["cache_hits" if cache_hit else "searches"] += 1

# ── Known satire / unreliable domains — verdict always FAKE ──
SATIRE_DOMAINS = {
    "theonion.com", "babylonbee.com", "waterfordwhispersnews.com",
    "thedailymash.co.uk", "newsthump.com", "naijafake.com",
    "realnewsrightnow.com", "empirenews.net", "worldnewsdailyreport.com",
    "huzlers.com", "stuppid.com", "clickhole.com",
}

SOURCES = {
    "punchng.com":         {"region":"Nigeria",     "name":"Punch Nigeria"},
    "vanguardngr.com":     {"region":"Nigeria",     "name":"Vanguard"},
    "dailytrust.com":      {"region":"Nigeria",     "name":"Daily Trust"},
    "premiumtimesng.com":  {"region":"Nigeria",     "name":"Premium Times"},
    "channelstv.com":      {"region":"Nigeria",     "name":"Channels TV"},
    "thecable.ng":         {"region":"Nigeria",     "name":"The Cable"},
    "guardian.ng":         {"region":"Nigeria",     "name":"Guardian Nigeria"},
    "thisdaylive.com":     {"region":"Nigeria",     "name":"ThisDay"},
    "businessday.ng":      {"region":"Nigeria",     "name":"BusinessDay"},
    "leadership.ng":       {"region":"Nigeria",     "name":"Leadership"},
    "tribuneonlineng.com": {"region":"Nigeria",     "name":"Tribune"},
    "sunnewsonline.com":   {"region":"Nigeria",     "name":"The Sun"},
    "blueprint.ng":        {"region":"Nigeria",     "name":"Blueprint"},
    "saharareporters.com": {"region":"Nigeria",     "name":"Sahara Reporters"},
    "legit.ng":            {"region":"Nigeria",     "name":"Legit.ng"},
    "nannews.ng":          {"region":"Nigeria",     "name":"NAN"},
    "arise.tv":            {"region":"Nigeria",     "name":"Arise TV"},
    "tvcnews.tv":          {"region":"Nigeria",     "name":"TVC News"},
    "prnigeria.com":       {"region":"Nigeria",     "name":"PR Nigeria"},
    "myschool.ng":         {"region":"Nigeria-Edu", "name":"MySchool"},
    "schoolgist.com.ng":   {"region":"Nigeria-Edu", "name":"SchoolGist"},
    "nuc.edu.ng":          {"region":"Nigeria-Edu", "name":"NUC Nigeria"},
    "jamb.gov.ng":         {"region":"Nigeria-Edu", "name":"JAMB"},
    "waec.org.ng":         {"region":"Nigeria-Edu", "name":"WAEC Nigeria"},
    "myjoyonline.com":     {"region":"Ghana",       "name":"Joy Online"},
    "graphic.com.gh":      {"region":"Ghana",       "name":"Graphic Ghana"},
    "nation.africa":       {"region":"Kenya",       "name":"Nation Africa"},
    "standardmedia.co.ke": {"region":"Kenya",       "name":"Standard Media"},
    "news24.com":          {"region":"S.Africa",    "name":"News24"},
    "dailymaverick.co.za": {"region":"S.Africa",    "name":"Daily Maverick"},
    "reuters.com":         {"region":"Global",      "name":"Reuters"},
    "apnews.com":          {"region":"Global",      "name":"AP News"},
    "bbc.com":             {"region":"Global",      "name":"BBC"},
    "bbc.co.uk":           {"region":"Global",      "name":"BBC"},
    "aljazeera.com":       {"region":"Global",      "name":"Al Jazeera"},
    "theguardian.com":     {"region":"Global",      "name":"The Guardian"},
    "cnn.com":             {"region":"Global",      "name":"CNN"},
    "bloomberg.com":       {"region":"Global",      "name":"Bloomberg"},
    "nytimes.com":         {"region":"Global",      "name":"NY Times"},
    "washingtonpost.com":  {"region":"Global",      "name":"Washington Post"},
    "rfi.fr":              {"region":"Global",      "name":"RFI"},
    "dw.com":              {"region":"Europe",      "name":"DW"},
    "france24.com":        {"region":"Europe",      "name":"France 24"},
    "who.int":             {"region":"Health",      "name":"WHO"},
    "cdc.gov":             {"region":"Health",      "name":"CDC"},
    "ncdc.gov.ng":         {"region":"Health",      "name":"NCDC Nigeria"},
    "arabnews.com":        {"region":"M.East",      "name":"Arab News"},
    "dubawa.org":          {"region":"FactChecker", "name":"Dubawa"},
    "africacheck.org":     {"region":"FactChecker", "name":"Africa Check"},
    "pesacheck.org":       {"region":"FactChecker", "name":"PesaCheck"},
    "snopes.com":          {"region":"FactChecker", "name":"Snopes"},
    "factcheck.org":       {"region":"FactChecker", "name":"FactCheck.org"},
    "politifact.com":      {"region":"FactChecker", "name":"PolitiFact"},
    "fullfact.org":        {"region":"FactChecker", "name":"Full Fact"},
    "boomlive.in":         {"region":"FactChecker", "name":"BoomLive"},
}

# Tier 1 = most trusted (Reuters, AP, BBC, WHO, CDC)
TIER1_SOURCES = {
    "reuters.com","apnews.com","bbc.com","bbc.co.uk","who.int","cdc.gov",
    "aljazeera.com","theguardian.com","nytimes.com","washingtonpost.com",
    "bloomberg.com","france24.com","dw.com",
}

NAME_LOOKUP = {}
for _s, _i in SOURCES.items():
    _n = _i["name"].lower()
    NAME_LOOKUP[_n] = _s
    _p = _n.split()
    if len(_p) >= 2:
        NAME_LOOKUP[" ".join(_p[:2])] = _s

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

def extract_years(text):
    return [int(y) for y in re.findall(r'\b(20\d{2})\b', text)]

def run_distilbert(text):
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = cls_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    r, f = float(probs[0]), float(probs[1])
    return {"real":r,"fake":f,"prediction":int(probs.argmax()),"confidence":int(max(r,f)*100)}

# ══════════════════════════════════════════════════════════════════
# CLAIM ANALYSIS — type, sensitivity flags, satire check
# ══════════════════════════════════════════════════════════════════
def detect_claim_type(claim: str) -> str:
    c = claim.lower()
    scores = {
        "education": sum(1 for k in [
            "university","polytechnic","college","school","offers","offering",
            "course","programme","program","faculty","department","admission",
            "jamb","waec","neco","cut off","result","llb","mbbs","law","medicine",
            "engineering","accredited","scholarship","degree","postgraduate"
        ] if k in c),
        "political": sum(1 for k in [
            "president","governor","senator","minister","government","politician",
            "arrested","impeached","resigned","elected","coup","protest",
            "policy","bill","passed","tinubu","buhari","obi","atiku","parliament"
        ] if k in c),
        "health": sum(1 for k in [
            "vaccine","virus","disease","outbreak","cure","treatment","hospital",
            "covid","monkeypox","cholera","ebola","cancer","drug","who","cdc","ncdc"
        ] if k in c),
        "business": sum(1 for k in [
            "naira","dollar","economy","inflation","gdp","cbn","bank","stock",
            "price","rate","fuel","petrol","subsidy","budget","billion","million"
        ] if k in c),
        "entertainment": sum(1 for k in [
            "music","song","album","artist","movie","film","concert","award",
            "grammy","oscars","nollywood","afrobeats","singer","actor","actress"
        ] if k in c),
        "sports": sum(1 for k in [
            "football","soccer","basketball","tennis","cricket","super eagles",
            "premier league","world cup","champions league","goal","match","team",
            "beat","won","lost","defeated","score","final","semifinal","knockout",
            "fixture","tournament","league","cup","vs","versus","draw","nil"
        ] if k in c),
        "science": sum(1 for k in [
            "research","study","scientists","nasa","space","climate","environment",
            "discovery","experiment","published","journal","found","shows"
        ] if k in c),
    }
    claim_type = max(scores, key=scores.get)
    if scores[claim_type] < 1:
        claim_type = "general"
    logger.info(f"Claim type: {claim_type} scores={scores}")
    return claim_type

def is_death_claim(claim: str) -> bool:
    """Detect any death/alive claim regardless of category."""
    c = claim.lower()
    death_patterns = [
        r'\b(is\s+)?dead\b', r'\bdied\b', r'\bdeath\b', r'\bkilled\b',
        r'\bpassed\s+away\b', r'\bno\s+more\b', r'\bin\s+a\s+coma\b',
        r'\bcritical\s+condition\b', r'\bshot\s+dead\b', r'\bmurdered\b',
        r'\ris\s+alive\b', r'\bstill\s+alive\b', r'\bnot\s+dead\b',
    ]
    return any(re.search(p, c) for p in death_patterns)

def is_live_sports_claim(claim: str) -> bool:
    """Detect claims about live or very recent sports results."""
    c = claim.lower()
    result_words = [
        "beat","won","lost","defeated","scored","wins","loses","beats",
        "victory","drew","draw","thrashed","hammered","crushed","eliminated",
        r'\d+-\d+', r'\d+\s*-\s*\d+',
    ]
    sport_words = [
        "world cup","premier league","champions league","match","game",
        "football","soccer","final","semifinal","cup","tournament","fixture",
    ]
    has_result = any(re.search(w, c) for w in result_words)
    has_sport  = any(w in c for w in sport_words)
    return has_result and has_sport

def is_breaking_news(claim: str) -> bool:
    """Detect breaking/urgent news that should not be cached long."""
    c = claim.lower()
    return any(w in c for w in [
        "breaking","just in","just now","developing","urgent","confirmed just",
        "minutes ago","hours ago","today","this morning","this evening",
    ])

def detect_satire(text: str, results: list) -> bool:
    """Check if the source is a known satire site."""
    c = text.lower()
    satire_signals = ["satire","parody","fictional","comedy news","not real"]
    if any(s in c for s in satire_signals):
        return True
    for r in results:
        url = (r.get("href") or "").lower()
        if any(sd in url for sd in SATIRE_DOMAINS):
            return True
    return False

# ══════════════════════════════════════════════════════════════════
# QUERY GENERATION
# ══════════════════════════════════════════════════════════════════
def generate_queries(claim: str, claim_type: str, is_death: bool = False) -> list:
    queries = [claim]
    proper  = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
    current_year = datetime.now().year

    # Death claims always get an "is X alive" query regardless of category
    if is_death and proper:
        queries.append(f"is {' '.join(proper[:2])} alive {current_year}")
        queries.append(f"{' '.join(proper[:2])} death confirmed {current_year}")

    elif claim_type == "education":
        unis = re.findall(
            r'\b(FUT\s*Minna|FUTMINNA|UniAbuja|UNILAG|UNN|OAU|ABU|BUK|FUTA|'
            r'FUNAAB|[A-Z]{2,}\s+[A-Z][a-z]+)\b', claim
        )
        subjects = re.findall(
            r'\b(law|medicine|engineering|pharmacy|nursing|accounting|'
            r'economics|computer science|architecture|LLB|MBBS)\b',
            claim, re.IGNORECASE
        )
        if unis:
            uni = unis[0].strip()
            queries.append(f"{uni} approved courses NUC accredited {current_year}")
            if subjects:
                queries.append(f"{uni} {subjects[0]} programme faculty accredited")
        queries.append(f"{claim} NUC JAMB verified {current_year}")

    elif claim_type == "political":
        if proper:
            queries.append(f"{' '.join(proper[:2])} latest news {current_year}")
            # Also search for arrest-related context to catch "arrested for spreading rumour"
            queries.append(f"{' '.join(proper[:2])} statement confirmed {current_year}")

    elif claim_type == "health":
        queries.append(f"{claim} WHO official statement {current_year}")
        queries.append(f"{claim} NCDC Nigeria {current_year}")

    elif claim_type == "business":
        queries.append(f"{claim} CBN official {current_year}")
        queries.append(f"{claim} Nigeria economy verified {current_year}")

    elif claim_type == "entertainment":
        if proper:
            queries.append(f"{' '.join(proper[:2])} {current_year} official confirmed")
        queries.append(f"{claim} entertainment news official {current_year}")

    elif claim_type == "sports":
        queries.append(f"{claim} final score full time result {current_year}")
        queries.append(f"{claim} FT score confirmed")

    elif claim_type == "science":
        queries.append(f"{claim} peer reviewed study {current_year}")
        queries.append(f"{claim} scientific evidence {current_year}")

    else:
        if proper:
            queries.append(f"{' '.join(proper[:3])} Nigeria {current_year}")
        queries.append(f"{claim} fact check {current_year}")

    seen, unique = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    logger.info(f"Queries ({claim_type}, death={is_death}): {unique[:4]}")
    return unique[:4]

# ══════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════
def search_tavily(query: str, claim_type: str) -> tuple:
    if not TAVILY_KEY:
        logger.warning("TAVILY_KEY empty — skipping Tavily")
        return [], ""
    try:
        include_domains = []
        if claim_type == "education":
            include_domains = ["nuc.edu.ng","jamb.gov.ng","myschool.ng","schoolgist.com.ng"]
            uni_map = {
                "futminna":"futminna.edu.ng","unilag":"unilag.edu.ng",
                "unn":"unn.edu.ng","oau":"oauife.edu.ng","abu":"abu.edu.ng",
                "buk":"buk.edu.ng","futa":"futa.edu.ng","funaab":"funaab.edu.ng",
            }
            for uni, domain in uni_map.items():
                if uni in query.lower():
                    include_domains.insert(0, domain)
                    break
        elif claim_type == "health":
            include_domains = ["who.int","cdc.gov","ncdc.gov.ng","reuters.com","apnews.com"]
        elif claim_type == "political":
            include_domains = [
                "premiumtimesng.com","punchng.com","vanguardngr.com",
                "bbc.com","reuters.com","apnews.com","aljazeera.com",
            ]
        elif claim_type == "sports":
            include_domains = [
                "bbc.com","bbc.co.uk","reuters.com","apnews.com",
                "theguardian.com","espn.com","fifa.com","cnn.com",
            ]

        payload = {
            "api_key":             TAVILY_KEY,
            "query":               query,
            "search_depth":        "advanced",
            "include_answer":      True,
            "include_raw_content": True,
            "max_results":         10,
        }
        if include_domains:
            payload["include_domains"] = include_domains

        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        logger.info(f"Tavily [{claim_type}] '{query[:40]}': status={resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            results = [{
                "href":        r.get("url",""),
                "title":       r.get("title",""),
                "body":        r.get("content","")[:500],
                "raw_content": r.get("raw_content","")[:4000],
                "pubdate":     r.get("published_date",""),
                "source":      "tavily"
            } for r in data.get("results",[])]
            answer = data.get("answer","") or ""
            logger.info(f"Tavily: {len(results)} results, answer={len(answer)} chars")
            return results, answer
        else:
            logger.error(f"Tavily error {resp.status_code}: {resp.text[:200]}")
            return [], ""
    except Exception as e:
        logger.error(f"Tavily exception: {e}")
        return [], ""

def search_google_news(query: str, region: str = "US") -> list:
    try:
        q    = requests.utils.quote(query[:200])
        url  = f"https://news.google.com/rss/search?q={q}&hl=en&gl={region}&ceid={region}:en"
        resp = requests.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=12)
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
                    clean_t = re.sub(r'\s+-\s+\S[\S]*\s*$','',re.sub(r'<.*?>','',t[0])).strip()
                    results.append({
                        "href":        l[0].strip() if l else "",
                        "title":       clean_t,
                        "body":        re.sub(r'<.*?>','',d[0]).strip()[:400] if d else "",
                        "raw_content": "",
                        "pubdate":     pub[0].strip() if pub else "",
                        "source":      "google_news"
                    })
        logger.info(f"Google News ({region}): {len(results)} results")
        return results
    except Exception as e:
        logger.warning(f"Google News failed: {e}")
        return []

def search_all(queries: list, claim_type: str) -> tuple:
    all_results, seen_urls = [], set()
    tavily_answers = []

    def add(items):
        for r in items:
            url = r.get("href","")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    for q in queries:
        if TAVILY_KEY:
            tv_r, tv_a = search_tavily(q, claim_type)
            add(tv_r)
            if tv_a and len(tv_a) > 20:
                tavily_answers.append(tv_a)
        add(search_google_news(q, "NG"))
        add(search_google_news(q, "US"))
        time.sleep(0.3)

    logger.info(f"Total unique results: {len(all_results)}")
    return all_results, " ".join(tavily_answers[:3])

def read_article(url: str) -> str:
    if not url or "news.google.com" in url:
        return ""
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"User-Agent": random.choice(USER_AGENTS), "Accept":"text/plain"},
            timeout=15
        )
        if resp.status_code == 200 and len(resp.text) > 200:
            lines = resp.text.split("\n")
            body  = [l for l in lines if not l.startswith(
                ("Title:","URL:","Published","Source:","Description:","---","===")
            )]
            clean = "\n".join(body).strip()
            if len(clean) > 200:
                return clean[:5000]
        return ""
    except:
        return ""

def match_sources(claim: str, results: list) -> list:
    found, seen = [], set()
    claim_years = extract_years(claim)
    for r in results:
        url     = (r.get("href") or "").lower()
        title   = r.get("title") or ""
        snippet = r.get("body")  or ""
        pubdate = r.get("pubdate") or ""

        # Skip satire domains entirely
        if any(sd in url for sd in SATIRE_DOMAINS):
            logger.info(f"Skipping satire domain: {url[:60]}")
            continue

        if claim_years and pubdate:
            pub_years = extract_years(pubdate)
            if pub_years and not any(abs(py-cy)<=1 for py in pub_years for cy in claim_years):
                continue

        matched = None

        # Priority 1: exact domain match in URL
        for src in SOURCES:
            if src in url and src not in seen:
                matched = src
                break

        # Priority 2: source name in article title suffix " - Source Name"
        if not matched and " - " in title:
            suffix = title.split(" - ")[-1].strip().lower()
            if suffix in NAME_LOOKUP and NAME_LOOKUP[suffix] not in seen:
                matched = NAME_LOOKUP[suffix]
            else:
                for name, src in NAME_LOOKUP.items():
                    if name in suffix and src not in seen and len(name) > 5:
                        matched = src
                        break

        # Priority 3: core domain word in combined text — but only if > 6 chars
        # to avoid matching "guardian.ng" when text just says "guardian"
        if not matched:
            combined = (title + " " + snippet).lower()
            for src, info in SOURCES.items():
                if src not in seen:
                    core = re.sub(r'\.(com|ng|org|co\.uk|co|tv|africa|net|gov|in|ke|gh|za)$', '', src)
                    # Require core word to be at least 7 chars to avoid false matches
                    if len(core) >= 7 and re.search(r'\b' + re.escape(core) + r'\b', combined):
                        matched = src
                        break

        if matched:
            seen.add(matched)
            info = SOURCES[matched]
            raw_url = r.get("href") or ""
            article_url = (
                raw_url if raw_url.startswith("http") and "news.google.com" not in raw_url
                else f"https://{matched}"
            )
            found.append({
                "source":      matched,
                "name":        info["name"],
                "region":      info["region"],
                "url":         f"https://{matched}",
                "article_url": article_url,
                "title":       title,
                "snippet":     snippet[:300],
                "raw_content": r.get("raw_content",""),
                "pubdate":     pubdate,
                "is_nigerian":    info["region"] in ("Nigeria","Nigeria-Edu"),
                "is_factchecker": info["region"] == "FactChecker",
                "is_tier1":       matched in TIER1_SOURCES,
            })

    found.sort(key=lambda x: (x["is_factchecker"]*3 + x["is_tier1"]*2 + x["is_nigerian"]), reverse=True)
    logger.info(f"Sources matched: {[s['name'] for s in found]}")
    return found

# ══════════════════════════════════════════════════════════════════
# LLM REASONING — Groq
# ══════════════════════════════════════════════════════════════════
def reason_with_llm(
    claim: str,
    articles: list,
    tavily_summary: str,
    claim_type: str,
    is_death: bool = False,
    is_live_sport: bool = False,
) -> dict:
    if not GROQ_KEY:
        logger.warning("GROQ_API_KEY not set — skipping LLM reasoning")
        return {"verdict":"unavailable","reasoning":"GROQ_API_KEY not configured","confidence":0,"model_used":"none"}

    today = datetime.now().strftime("%d %B %Y")

    evidence_parts = []
    if tavily_summary and len(tavily_summary) > 30:
        evidence_parts.append(f"[Web Summary — {today}]\n{tavily_summary[:600]}")

    for i, (title, text, pubdate, source) in enumerate(articles[:5]):
        part = f"[Source {i+1}: {source}]"
        if pubdate:
            part += f" [Date: {pubdate}]"
        part += f"\nHeadline: {title}"
        if text and len(text) > 100:
            part += f"\nContent: {text[:2000]}"
        evidence_parts.append(part)

    if not evidence_parts:
        return {"verdict":"INSUFFICIENT_EVIDENCE","reasoning":"No articles found to analyse","confidence":0,"model_used":"none"}

    evidence = "\n\n".join(evidence_parts)

    # Build specific guidance per claim type and sensitivity
    if is_death:
        type_guide = (
            "This is a DEATH/ALIVE claim — the highest sensitivity category. Rules:\n"
            "1. Someone 'mourning' X does NOT confirm X is dead.\n"
            "2. Someone 'arrested for claiming X is dead' means X is ALIVE and the claim is FALSE.\n"
            "3. Only return SUPPORTS_CLAIM if a Tier-1 outlet (Reuters, AP, BBC, Al Jazeera) "
            "explicitly confirms the death with direct reporting, not citing social media.\n"
            "4. A single blog, social post, or indirect reference = INSUFFICIENT_EVIDENCE.\n"
            "5. Any source debunking the death = CONTRADICTS_CLAIM immediately.\n"
            "6. When in doubt: INSUFFICIENT_EVIDENCE. Never guess on death claims."
        )
    elif is_live_sport:
        type_guide = (
            "This is a LIVE SPORTS RESULT claim. Rules:\n"
            "1. You MUST find a source that explicitly states the FINAL SCORE with 'FT', "
            "'full time', 'final score', or 'full-time result'.\n"
            "2. A source mentioning a goal scorer (e.g. 'Havertz scored') does NOT confirm "
            "who won — the match may still be in progress or ended differently.\n"
            "3. A source saying 'X leads Y' means the match is still ongoing.\n"
            "4. ONLY return SUPPORTS_CLAIM if a source explicitly states the final result "
            "AND it matches the claim exactly.\n"
            "5. If match is in progress or only partial events are reported = INSUFFICIENT_EVIDENCE.\n"
            "6. If the final result contradicts the claim = CONTRADICTS_CLAIM."
        )
    else:
        type_guide = {
            "education": (
                "For education claims: NUC, JAMB, or the official university website must confirm "
                "the programme exists. News articles reporting a claim are NOT sufficient — "
                "only official academic bodies count. If no official source confirms it, "
                "verdict is CONTRADICTS_CLAIM."
            ),
            "political": (
                "For political claims: check if major credible outlets directly report this. "
                "An arrest related to the claim (e.g. 'man arrested for spreading rumour') "
                "means the original claim is FALSE. Policy claims need official government sources. "
                "Be strict — do not say SUPPORTS_CLAIM from a single blog or opinion piece."
            ),
            "health": (
                "For health claims: WHO, CDC, NCDC are the highest authority. "
                "Anecdotal reports or social media screenshots are never sufficient. "
                "Require official health body confirmation for SUPPORTS_CLAIM."
            ),
            "business": (
                "For business/economic claims: CBN, official government releases, or Tier-1 "
                "financial outlets (Bloomberg, Reuters) are required. Social media figures "
                "and unverified market data are not sufficient."
            ),
            "entertainment": (
                "For entertainment claims (awards, releases, events): require official "
                "announcements from the artist/label or coverage by major entertainment outlets. "
                "Fan pages, blogs, and social media posts alone are NOT sufficient. "
                "For award claims, the official awarding body must have announced it."
            ),
            "sports": (
                "For sports claims: require confirmed final scores or official league/team "
                "announcements. In-progress match reports are not sufficient."
            ),
            "science": (
                "For science claims: require peer-reviewed publications, official research "
                "institution statements, or Tier-1 science journalism. "
                "Preprints and blog summaries are not sufficient."
            ),
            "general": (
                "Check if multiple credible sources directly confirm this specific claim. "
                "One source is not enough for SUPPORTS_CLAIM on an unusual claim."
            ),
        }.get(claim_type, "Check if credible sources directly confirm this specific claim.")

    prompt = f"""You are TruthLens, a professional AI fact-checker built for journalists. Today is {today}.

CLAIM TO VERIFY: "{claim}"
CLAIM CATEGORY: {claim_type.upper()}{"  ⚠️ DEATH/ALIVE CLAIM — EXTREME CAUTION" if is_death else ""}{"  ⚠️ LIVE SPORTS — RESULT UNCONFIRMED UNTIL EXPLICITLY STATED" if is_live_sport else ""}

GUIDANCE FOR THIS CLAIM TYPE:
{type_guide}

EVIDENCE FROM SOURCES:
{evidence}

UNIVERSAL RULES (apply to every claim):
1. Read what each source ACTUALLY says — quote specific phrases, do not paraphrase loosely
2. "Arrested for spreading rumour that X happened" = X did NOT happen, claim is FALSE
3. "Mourning X" or "praying for X" does NOT confirm X died
4. Old articles that predate the claimed event are irrelevant
5. A source discussing the RUMOUR of a claim is not confirming the claim
6. MEDIUM confidence = not enough for SUPPORTS_CLAIM on death, sports results, or political claims
7. If evidence is mixed or indirect, use INSUFFICIENT_EVIDENCE — it protects readers
8. NEVER fabricate or infer details not present in the sources

RESPOND IN EXACTLY THIS FORMAT — nothing before REASONING, nothing after EXPLANATION:
REASONING: [Quote specific phrases from each source. State exactly what each source says.]
VERDICT: SUPPORTS_CLAIM or CONTRADICTS_CLAIM or INSUFFICIENT_EVIDENCE
CONFIDENCE: HIGH or MEDIUM or LOW
EXPLANATION: [One sentence summary for a non-expert reader]"""

    try:
        logger.info(f"Calling Groq API with model: {LLM_MODEL}")
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       LLM_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  800,
                "temperature": 0.1,
            },
            timeout=30,
        )
        logger.info(f"Groq status: {resp.status_code}")

        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"]
            logger.info(f"Groq response ({len(text)} chars): {text[:300]}")

            v_m = re.search(r'VERDICT[:\s]+(SUPPORTS_CLAIM|CONTRADICTS_CLAIM|INSUFFICIENT_EVIDENCE)', text, re.IGNORECASE)
            r_m = re.search(r'REASONING[:\s]+(.*?)(?=VERDICT:|CONFIDENCE:|$)', text, re.IGNORECASE | re.DOTALL)
            c_m = re.search(r'CONFIDENCE[:\s]+(HIGH|MEDIUM|LOW)', text, re.IGNORECASE)
            e_m = re.search(r'EXPLANATION[:\s]+(.*?)(?=\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)

            verdict     = v_m.group(1).upper() if v_m else "INSUFFICIENT_EVIDENCE"
            reasoning   = r_m.group(1).strip()[:800] if r_m else text[:600]
            confidence  = c_m.group(1).upper() if c_m else "MEDIUM"
            explanation = e_m.group(1).strip()[:250] if e_m else reasoning[:150]
            conf_score  = {"HIGH": 0.9, "MEDIUM": 0.65, "LOW": 0.4}.get(confidence, 0.5)

            return {
                "verdict":     verdict,
                "reasoning":   reasoning,
                "explanation": explanation,
                "confidence":  conf_score,
                "model_used":  LLM_MODEL,
            }

        elif resp.status_code == 429:
            logger.error("Groq rate limit hit")
            return {"verdict":"INSUFFICIENT_EVIDENCE","reasoning":"Groq rate limit. Try again shortly.","confidence":0,"model_used":"none"}
        else:
            logger.error(f"Groq error {resp.status_code}: {resp.text[:200]}")
            return {"verdict":"INSUFFICIENT_EVIDENCE","reasoning":f"Groq API error: {resp.status_code}","confidence":0,"model_used":"none"}

    except Exception as e:
        logger.error(f"Groq exception: {e}")
        return {"verdict":"INSUFFICIENT_EVIDENCE","reasoning":f"Groq request failed: {str(e)}","confidence":0,"model_used":"none"}

# ══════════════════════════════════════════════════════════════════
# VERDICT FUSION — 10-layer, sensitivity-aware
# ══════════════════════════════════════════════════════════════════
def fuse_verdict(
    distilbert: dict,
    matched: list,
    llm: dict,
    claim_type: str = "general",
    is_live_sport: bool = False,
    is_death: bool = False,
    is_satire: bool = False,
) -> dict:

    # ── Layer 0: Satire — immediate FAKE ──
    if is_satire:
        return {
            "verdict": "FAKE",
            "confidence": 98,
            "title": "Satire / Parody Content",
            "subtitle": "This content originates from a known satire or parody source. It is not real news.",
        }

    nigerian  = [s for s in matched if s["is_nigerian"]]
    factcheck = [s for s in matched if s["is_factchecker"]]
    tier1     = [s for s in matched if s.get("is_tier1")]
    total     = len(matched)
    llm_v     = llm.get("verdict", "INSUFFICIENT_EVIDENCE")
    llm_conf  = llm.get("confidence", 0)
    llm_exp   = llm.get("explanation", "")
    llm_ok    = llm_v in ("SUPPORTS_CLAIM", "CONTRADICTS_CLAIM")

    fc_fake = [s for s in factcheck if any(w in s["title"].lower() for w in
               ["false","fake","misinformation","misleading","debunked","hoax","not true","fabricated"])]
    fc_real = [s for s in factcheck if any(w in s["title"].lower() for w in
               ["true","confirmed","accurate","verified","correct","legit"])]

    # ── Layer 1: Fact-checker is always highest authority ──
    if fc_fake:
        return {"verdict":"FAKE","confidence":99,
                "title":"Confirmed Misinformation",
                "subtitle":f"Flagged by {fc_fake[0]['name']}: {fc_fake[0]['title'][:70]}"}
    if fc_real:
        return {"verdict":"REAL","confidence":99,
                "title":"Fact-Checker Verified",
                "subtitle":f"Confirmed by {fc_real[0]['name']}"}

    # ── Layer 2: LLM CONTRADICTS overrides everything else ──
    if llm_ok and llm_v == "CONTRADICTS_CLAIM" and llm_conf >= 0.4:
        return {"verdict":"FAKE","confidence":int(llm_conf * 100),
                "title":"AI Reading Contradicts Claim",
                "subtitle": llm_exp or "Sources directly contradict this claim."}

    # ── Layer 3: Death claims — require Tier-1 + HIGH confidence only ──
    if is_death:
        if llm_ok and llm_v == "SUPPORTS_CLAIM" and llm_conf >= 0.9 and len(tier1) >= 1:
            return {"verdict":"REAL","confidence":88,
                    "title":"Death Confirmed by Major Outlet",
                    "subtitle":f"Confirmed by {tier1[0]['name']}: {llm_exp}"}
        if llm_v == "SUPPORTS_CLAIM" and llm_conf < 0.9:
            return {"verdict":"MIXED","confidence":42,
                    "title":"Death Claim — Cannot Confirm",
                    "subtitle":"Sources are ambiguous. Require direct confirmation from Reuters, AP, or BBC before sharing."}
        if llm_v == "INSUFFICIENT_EVIDENCE":
            return {"verdict":"MIXED","confidence":38,
                    "title":"Death Claim — Unverified",
                    "subtitle":"No Tier-1 source has confirmed this. Do not share until officially confirmed."}

    # ── Layer 4: Live sports — require HIGH confidence + explicit final result ──
    if is_live_sport or claim_type == "sports":
        if llm_ok and llm_v == "SUPPORTS_CLAIM" and llm_conf >= 0.9:
            return {"verdict":"REAL","confidence":90,
                    "title":"Sports Result Confirmed",
                    "subtitle":f"Final result confirmed: {llm_exp}"}
        return {"verdict":"MIXED","confidence":45,
                "title":"Sports Result Not Yet Confirmed",
                "subtitle":"Sources report match events but no confirmed full-time result found. Check BBC Sport or ESPN directly."}

    # ── Layer 5: LLM SUPPORTS + HIGH confidence + Tier-1 source ──
    if llm_ok and llm_v == "SUPPORTS_CLAIM" and llm_conf >= 0.9 and len(tier1) >= 1:
        return {"verdict":"REAL","confidence":95,
                "title":"Confirmed by Tier-1 Source + AI Verification",
                "subtitle":f"{tier1[0]['name']} confirmed: {llm_exp}"}

    # ── Layer 6: LLM SUPPORTS + HIGH confidence + multiple sources ──
    if llm_ok and llm_v == "SUPPORTS_CLAIM" and llm_conf >= 0.9 and total >= 2:
        return {"verdict":"REAL","confidence":92,
                "title":"Confirmed — AI Read and Verified",
                "subtitle":f"AI confirmed across {total} sources: {llm_exp}"}

    # ── Layer 7: LLM SUPPORTS + MEDIUM confidence + 3+ sources ──
    # (Not enough for death/sports — already handled above)
    if llm_ok and llm_v == "SUPPORTS_CLAIM" and llm_conf >= 0.65 and total >= 3:
        return {"verdict":"REAL","confidence":80,
                "title":"Likely Confirmed by Multiple Sources",
                "subtitle":f"Supporting evidence found across {total} outlets: {llm_exp}"}

    # ── Layer 8: LLM SUPPORTS + MEDIUM confidence + 1-2 sources ──
    if llm_ok and llm_v == "SUPPORTS_CLAIM" and llm_conf >= 0.65 and total >= 1:
        return {"verdict":"REAL","confidence":68,
                "title":"Partially Confirmed — Verify Further",
                "subtitle":f"{llm_exp} — found in {matched[0]['name']}. Check additional sources."}

    # ── Layer 9: LLM INSUFFICIENT_EVIDENCE ──
    if llm_v == "INSUFFICIENT_EVIDENCE":
        if total >= 2:
            return {"verdict":"MIXED","confidence":45,
                    "title":"Cannot Verify — Insufficient Evidence",
                    "subtitle":f"Found {total} related sources but none explicitly confirm this claim. Verify manually."}
        return {"verdict":"MIXED","confidence":38,
                "title":"Cannot Verify — No Confirming Sources",
                "subtitle":"No sources found that directly confirm this claim. Use fact-checkers below."}

    # ── Layer 10: Source count only (LLM unavailable / low confidence) ──
    if total >= 3 and llm_v != "CONTRADICTS_CLAIM":
        regions = list(dict.fromkeys(s["region"] for s in matched[:4]))
        return {"verdict":"REAL","confidence":72,
                "title":"Found in Multiple Outlets",
                "subtitle":f"Reported in {total} outlets across {', '.join(regions)}. AI reasoning unavailable."}
    if len(nigerian) >= 2 and llm_v != "CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":68,
                "title":"Reported by Nigerian Outlets",
                "subtitle":f"Found in {nigerian[0]['name']} and {nigerian[1]['name']}. Verify further."}
    if total == 1 and llm_v != "CONTRADICTS_CLAIM":
        return {"verdict":"MIXED","confidence":42,
                "title":"Single Source — Verify Further",
                "subtitle":f"Only found in {matched[0]['name']}. Check additional outlets before sharing."}

    # ── Fallback: DistilBERT only ──
    if distilbert["prediction"] == 1 and distilbert["confidence"] >= 80:
        return {"verdict":"FAKE","confidence":72,
                "title":"Likely Fake — No Sources Found",
                "subtitle":"Misinformation patterns detected and zero credible sources found."}
    if distilbert["prediction"] == 0 and distilbert["confidence"] >= 80:
        return {"verdict":"SUSPICIOUS","confidence":50,
                "title":"Suspicious — Cannot Verify",
                "subtitle":"Writing appears credible but no outlet confirms this. Verify before sharing."}

    return {"verdict":"MIXED","confidence":40,
            "title":"Uncertain — Verify Manually",
            "subtitle":"Mixed signals. Use the fact-checker links below to investigate."}

# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════
class AnalyseRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "name":      "TruthLens API v10.0",
        "status":    "running",
        "tavily":    bool(TAVILY_KEY),
        "llm":       bool(GROQ_KEY),
        "llm_model": LLM_MODEL,
        "sources":   len(SOURCES),
    }

@app.get("/health")
def health():
    return {"status":"healthy","tavily":bool(TAVILY_KEY),"llm":bool(GROQ_KEY)}

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

    # Clean and extract core claim
    # For long text (full article), extract first 600 chars as the claim
    # but keep full text available for context
    clean = re.sub(r'http\S+|www\S+|<.*?>|\s+', ' ', text).strip()
    claim = clean[:600]  # what we verify

    current_year = datetime.now().year
    claim_years  = extract_years(claim)
    # Only block years clearly in the future (not current year)
    future_years = [y for y in claim_years if y > current_year + 1]
    if future_years:
        return _future_verdict(future_years[0])

    # Detect all sensitivity flags upfront
    claim_type   = detect_claim_type(claim)
    is_death     = is_death_claim(claim)
    live_sport   = is_live_sports_claim(claim)
    is_breaking  = is_breaking_news(claim)

    # Choose cache TTL based on sensitivity
    if live_sport:
        ttl = CACHE_TTL_SPORT
    elif is_death or is_breaking:
        ttl = CACHE_TTL_DEATH
    else:
        ttl = CACHE_TTL

    ck = cache_key(claim)
    cached = cache_get(ck, ttl=ttl)
    if cached:
        tick(cache_hit=True)
        cached["from_cache"] = True
        return cached
    tick()

    db      = run_distilbert(claim)
    queries = generate_queries(claim, claim_type, is_death=is_death)
    results, tavily_summary = search_all(queries, claim_type)

    # Check satire before going further
    is_satire = detect_satire(claim, results)

    matched   = match_sources(claim, results)
    factcheck = [s for s in matched if s["is_factchecker"]]

    articles_for_llm = []
    for src in matched[:6]:
        raw = src.get("raw_content", "")
        full_text = raw if raw and len(raw) > 200 else read_article(src["article_url"])
        articles_for_llm.append((src["title"], full_text, src["pubdate"], src["name"]))

    for r in results[:8]:
        rc = r.get("raw_content", "")
        if rc and len(rc) > 300:
            title = r.get("title", "")
            if not any(t == title for t, _, _, _ in articles_for_llm):
                sn = next(
                    (SOURCES[sd]["name"] for sd in SOURCES if sd in (r.get("href") or "").lower()),
                    "Web"
                )
                articles_for_llm.append((title, rc, r.get("pubdate", ""), sn))

    scraped = sum(1 for _, t, _, _ in articles_for_llm if len(t) > 200)

    llm = reason_with_llm(
        claim, articles_for_llm, tavily_summary, claim_type,
        is_death=is_death, is_live_sport=live_sport,
    )
    fusion = fuse_verdict(
        db, matched, llm,
        claim_type=claim_type,
        is_live_sport=live_sport,
        is_death=is_death,
        is_satire=is_satire,
    )

    verdict    = fusion["verdict"]
    confidence = fusion["confidence"]
    model_used = llm.get("model_used", "unknown")

    # Sensitivity flags for UI
    clickbait = any(w in claim.lower() for w in [
        "shocking","secret","exposed","leaked","viral","breaking",
        "you won't believe","bombshell","explosive",
    ])

    # Build signals
    sensitivity_flags = []
    if is_death:     sensitivity_flags.append("⚠️ Death Claim")
    if live_sport:   sensitivity_flags.append("⚡ Live Sports")
    if is_breaking:  sensitivity_flags.append("🔴 Breaking News")
    if is_satire:    sensitivity_flags.append("🎭 Satire Detected")

    tier1_found = [s for s in matched if s.get("is_tier1")]

    llm_sig = {
        "SUPPORTS_CLAIM":        ("AI read articles — SUPPORTS claim", "pos"),
        "CONTRADICTS_CLAIM":     ("AI read articles — CONTRADICTS claim", "neg"),
        "INSUFFICIENT_EVIDENCE": ("Insufficient evidence in articles", "neu"),
    }.get(llm.get("verdict", ""), ("LLM unavailable", "neu"))

    signals = [
        {"icon":"🌐","label":"Web Sources",
         "value":f"{len(matched)} credible outlets" + (f" ({len(tier1_found)} Tier-1)" if tier1_found else ""),
         "type":"pos" if len(matched) >= 2 else "neg" if len(matched) == 0 else "neu"},
        {"icon":"🤖","label":"DistilBERT",
         "value":f"{'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%",
         "type":"pos" if db['prediction'] == 0 else "neg"},
        {"icon":"🧠","label":"AI Reasoning",
         "value":llm_sig[0], "type":llm_sig[1]},
        {"icon":"📰","label":"Articles Read",
         "value":f"{scraped} articles analysed",
         "type":"pos" if scraped > 0 else "neu"},
        {"icon":"🎯","label":"Claim Category",
         "value":claim_type.upper() + (f" ({', '.join(sensitivity_flags)})" if sensitivity_flags else ""),
         "type":"neu"},
        {"icon":"🎣","label":"Clickbait",
         "value":"Detected" if clickbait else "None",
         "type":"neg" if clickbait else "pos"},
    ]

    llm_reasoning = llm.get("reasoning", "") or llm.get("explanation", "") or "No reasoning returned."
    src_list = ", ".join(s["name"] for s in matched[:3]) if matched else "none"

    if verdict == "REAL":
        analysis = (
            f"CREDIBLE — {claim_type.upper()} CLAIM\n\n"
            f"LAYER 1 — DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}% (raw score before web check).\n\n"
            f"LAYER 2 — WEB SEARCH [{claim_type}]: Found {len(matched)} outlet(s) including {src_list}. "
            f"Tier-1 sources: {len(tier1_found)}. "
            f"Searched via {len(queries)} targeted queries via Tavily + Google News.\n\n"
            f"LAYER 3 — AI READING ({model_used}): {llm_reasoning[:500]}\n\n"
            f"RECOMMENDATION: Content appears credible. Always verify by clicking sources directly."
        )
    elif verdict in ("FAKE", "SUSPICIOUS"):
        analysis = (
            f"LIKELY FAKE — {claim_type.upper()} CLAIM\n\n"
            f"LAYER 1 — DistilBERT: {db['confidence']}% {'REAL' if db['prediction']==0 else 'FAKE'}.\n\n"
            f"LAYER 2 — WEB SEARCH [{claim_type}]: "
            f"{'Zero credible sources found.' if not matched else f'{len(matched)} source(s) found — content contradicts or does not support claim.'}\n\n"
            f"LAYER 3 — AI READING ({model_used}): {llm_reasoning[:500]}\n\n"
            f"RECOMMENDATION: Do not share. Verify through AfricaCheck, Dubawa, or Snopes."
        )
    else:
        recommendation = (
            "For live sports, check BBC Sport, ESPN, or FIFA.com directly." if live_sport
            else "For death claims, wait for Reuters, AP, or BBC confirmation." if is_death
            else "Check AfricaCheck or Dubawa before sharing."
        )
        analysis = (
            f"UNVERIFIED — {claim_type.upper()} CLAIM\n\n"
            f"LAYER 1 — DistilBERT: {db['confidence']}%.\n\n"
            f"LAYER 2 — WEB SEARCH: Found {len(matched)} related source(s): {src_list}.\n\n"
            f"LAYER 3 — AI READING ({model_used}): {llm_reasoning[:400]}\n\n"
            f"RECOMMENDATION: {recommendation}"
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
            "verdict":     llm.get("verdict", ""),
            "reasoning":   llm_reasoning,
            "explanation": llm.get("explanation", ""),
            "confidence":  llm.get("confidence", 0),
            "model":       model_used,
        },
        "claim_type":       claim_type,
        "sensitivity": {
            "is_death":    is_death,
            "is_sport":    live_sport,
            "is_breaking": is_breaking,
            "is_satire":   is_satire,
        },
        "tavily_summary": tavily_summary[:300] if tavily_summary else "",
        "sources_found": [{
            "source":        s["source"],
            "name":          s["name"],
            "region":        s["region"],
            "url":           s["url"],
            "article_url":   s["article_url"],
            "title":         s["title"][:120],
            "snippet":       s["snippet"],
            "pubdate":       s["pubdate"],
            "is_factchecker":s["is_factchecker"],
            "is_nigerian":   s["is_nigerian"],
            "is_tier1":      s.get("is_tier1", False),
        } for s in matched[:8]],
        "fact_checks": [{
            "name":        s["name"],
            "url":         s["url"],
            "article_url": s["article_url"],
            "title":       s["title"][:120],
            "snippet":     s["snippet"],
        } for s in factcheck[:3]],
        "queries_used":       queries,
        "articles_scraped":   scraped,
        "search_timestamp":   datetime.now().isoformat(),
        "from_cache":         False,
        "daily_stats":        COUNTER.copy(),
    }
    cache_set(ck, result)
    return result


def _future_verdict(year):
    return {
        "verdict":          "FAKE",
        "confidence":       99,
        "verdict_title":    "Future Event Claimed as Fact",
        "verdict_subtitle": f"References {year} which has not occurred yet.",
        "analysis":         f"This content references {year}, a future year. It cannot be reported as fact.\n\nDo not share.",
        "signals": [
            {"icon":"📅","label":"Date",       "value":f"Future year {year}", "type":"neg"},
            {"icon":"🚫","label":"Verdict",    "value":"Impossible",          "type":"neg"},
            {"icon":"🤖","label":"DistilBERT", "value":"N/A",                 "type":"neg"},
            {"icon":"🧠","label":"AI Reading", "value":"N/A",                 "type":"neg"},
            {"icon":"🌐","label":"Web",        "value":"N/A",                 "type":"neg"},
            {"icon":"⚠️","label":"Warning",   "value":"Do not share",        "type":"neg"},
        ],
        "real_probability": 0.01,
        "fake_probability": 0.99,
        "phi3":    {"verdict":"N/A","reasoning":"Future date","confidence":0,"model":"none"},
        "claim_type":       "temporal",
        "sensitivity":      {"is_death":False,"is_sport":False,"is_breaking":False,"is_satire":False},
        "tavily_summary":   "",
        "sources_found":    [],
        "fact_checks":      [],
        "queries_used":     [],
        "articles_scraped": 0,
        "search_timestamp": datetime.now().isoformat(),
        "from_cache":       False,
        "daily_stats":      {},
    }
