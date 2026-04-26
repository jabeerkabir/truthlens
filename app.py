"""
TruthLens API v9.0 — Grok-style Claim-Aware Search
National Open University of Nigeria (NOUN)
Developer: Jabir Muhammad Kabir
Supervisor: Dr. Ojeniyi Adebayo

Key upgrade: Detects claim TYPE then searches the RIGHT sources
  Political claim  → BBC, Reuters, Punch, Vanguard
  Education claim  → NUC, JAMB, MySchool, university site
  Health claim     → WHO, NHS, medical journals
  Business claim   → Bloomberg, BusinessDay, Reuters
  General claim    → broad web search
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

app = FastAPI(title="TruthLens API", version="9.0", docs_url=None, redoc_url=None)
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
CACHE_TTL = 3600 * 3

def cache_key(text):
    return hashlib.md5(f"{date.today()}:{text.lower().strip()}".encode()).hexdigest()

def cache_get(key):
    e = CACHE.get(key)
    if e and time.time() - e["ts"] < CACHE_TTL:
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
    "nabteb.gov.ng":       {"region":"Nigeria-Edu", "name":"NABTEB"},
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
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

# ══════════════════════════════════════════════════════════════════
# CLAIM TYPE DETECTOR — This is the Grok-style intelligence
# ══════════════════════════════════════════════════════════════════
def detect_claim_type(claim: str) -> dict:
    """
    Detect what TYPE of claim this is so we search the RIGHT sources.
    Like Grok does — different claims need different search strategies.
    """
    c = claim.lower()

    # Education/University claims
    edu_keywords = [
        "university","polytechnic","college","school","offers","offering",
        "course","programme","program","faculty","department","admission",
        "jamb","waec","neco","cut off","cutoff","result","result","unilag",
        "unn","oau","uniabuja","futminna","futa","abu","buk","unimaid",
        "funaab","abubakar","tafawa","balewa","federal","state university",
        "llb","mbbs","law","medicine","engineering","accredited"
    ]
    # Nigerian universities pattern
    uni_pattern = r'\b(uni|fut|futa|funaab|abu|buk|aaua|lasu|unilag|unn|oau)\b'

    # Political/government claims
    political_keywords = [
        "president","governor","senator","minister","government","politician",
        "tinubu","buhari","obi","atiku","lula","biden","trump","prime minister",
        "dead","died","death","arrested","impeached","resigned","elected",
        "coup","protest","policy","bill","law passed","federal"
    ]

    # Health claims
    health_keywords = [
        "vaccine","virus","disease","outbreak","cure","treatment","hospital",
        "covid","monkeypox","cholera","ebola","cancer","drug","medicine",
        "who","cdc","ncdc","health","medical","doctor","clinical"
    ]

    # Business/economy claims
    business_keywords = [
        "naira","dollar","economy","inflation","gdp","cbn","bank","stock",
        "price","rate","fuel","petrol","subsidy","budget","revenue","billion"
    ]

    edu_score  = sum(1 for k in edu_keywords if k in c)
    pol_score  = sum(1 for k in political_keywords if k in c)
    health_score = sum(1 for k in health_keywords if k in c)
    biz_score  = sum(1 for k in business_keywords if k in c)
    has_uni    = bool(re.search(uni_pattern, c))

    if edu_score >= 2 or has_uni:
        claim_type = "education"
    elif pol_score >= 2:
        claim_type = "political"
    elif health_score >= 2:
        claim_type = "health"
    elif biz_score >= 2:
        claim_type = "business"
    else:
        claim_type = "general"

    logger.info(f"Claim type: {claim_type} (edu={edu_score} pol={pol_score} health={health_score})")
    return {"type": claim_type}

def generate_queries(claim: str, claim_type: str) -> list:
    """
    Generate smart queries based on claim type.
    Education claims search university sites + NUC + JAMB.
    Political claims search news outlets.
    """
    queries = [claim]
    proper  = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
    current_year = datetime.now().year

    if claim_type == "education":
        # Extract university name
        unis = re.findall(
            r'\b(FUT\s*Minna|FUTMINNA|UniAbuja|UNILAG|UNN|OAU|ABU|BUK|FUTA|FUNAAB|'
            r'[A-Z]{2,}\s+[A-Z][a-z]+|[A-Z][a-z]+\s+University|[A-Z][a-z]+\s+Polytechnic)\b',
            claim
        )
        subjects = re.findall(
            r'\b(law|medicine|engineering|pharmacy|nursing|accounting|'
            r'economics|computer science|architecture|dentistry|LLB|MBBS)\b',
            claim, re.IGNORECASE
        )
        if unis:
            uni = unis[0].strip()
            # Search university website directly
            queries.append(f"{uni} approved courses programmes {current_year}")
            if subjects:
                queries.append(f"{uni} {subjects[0]} faculty accredited NUC")
            else:
                queries.append(f"{uni} NUC accredited programmes list")
        queries.append(f"{claim} NUC JAMB {current_year}")

    elif claim_type == "political":
        if proper:
            # Search for current status
            queries.append(f"{' '.join(proper[:2])} latest news {current_year}")
            queries.append(f"is {' '.join(proper[:2])} alive {current_year}")

    elif claim_type == "health":
        queries.append(f"{claim} WHO official {current_year}")
        queries.append(f"{claim} NCDC Nigeria health {current_year}")

    elif claim_type == "business":
        queries.append(f"{claim} CBN official {current_year}")
        queries.append(f"{claim} Nigeria economy {current_year}")

    else:
        if proper:
            queries.append(f"{' '.join(proper[:3])} Nigeria {current_year}")
        queries.append(f"{claim} fact check {current_year}")

    seen, unique = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    logger.info(f"Queries ({claim_type}): {unique}")
    return unique[:4]

# ══════════════════════════════════════════════════════════════════
# SEARCH FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def search_tavily(query: str, claim_type: str = "general") -> tuple:
    if not TAVILY_KEY:
        logger.warning("TAVILY_KEY not set")
        return [], ""
    try:
        # For education claims search specific sites
        include_domains = []
        if claim_type == "education":
            include_domains = [
                "nuc.edu.ng", "jamb.gov.ng", "myschool.ng",
                "schoolgist.com.ng", "waec.org.ng"
            ]
            # Add university domain if detected
            uni_match = re.search(
                r'(futminna|unilag|unn|oau|abu|buk|futa|funaab)',
                query.lower()
            )
            if uni_match:
                uni = uni_match.group(1)
                domain_map = {
                    "futminna": "futminna.edu.ng",
                    "unilag":   "unilag.edu.ng",
                    "unn":      "unn.edu.ng",
                    "oau":      "oauife.edu.ng",
                    "abu":      "abu.edu.ng",
                    "buk":      "buk.edu.ng",
                    "futa":     "futa.edu.ng",
                    "funaab":   "funaab.edu.ng",
                }
                if uni in domain_map:
                    include_domains.insert(0, domain_map[uni])

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

        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=20
        )
        logger.info(f"Tavily status: {resp.status_code} for '{query[:40]}'")
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
            logger.info(f"Tavily: {len(results)} results, answer={len(answer)} chars")
            return results, answer
        else:
            logger.error(f"Tavily {resp.status_code}: {resp.text[:200]}")
            return [], ""
    except Exception as e:
        logger.error(f"Tavily exception: {e}")
        return [], ""

def search_google_news(query: str, region: str = "US") -> list:
    try:
        q    = requests.utils.quote(query[:200])
        url  = f"https://news.google.com/rss/search?q={q}&hl=en&gl={region}&ceid={region}:en"
        resp = requests.get(
            url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=12
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
                        r'\s+-\s+\S[\S]*\s*$', '', re.sub(r'<.*?>', '', t[0])
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
        # Google News for political/general
        if claim_type in ("political","general","business"):
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
                logger.info(f"Jina: {len(clean)} chars from {url[:50]}")
                return clean[:5000]
        return ""
    except Exception as e:
        logger.debug(f"Jina failed: {e}")
        return ""

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
                abs(py-cy) <= 1 for py in pub_years for cy in claim_years
            ):
                continue

        matched = None
        for src in SOURCES:
            if src in url and src not in seen:
                matched = src; break

        if not matched and " - " in title:
            suffix = title.split(" - ")[-1].strip().lower()
            if suffix in NAME_LOOKUP and NAME_LOOKUP[suffix] not in seen:
                matched = NAME_LOOKUP[suffix]
            else:
                for name, src in NAME_LOOKUP.items():
                    if name in suffix and src not in seen and len(name) > 4:
                        matched = src; break

        if not matched:
            combined = (title + " " + snippet).lower()
            for src, info in SOURCES.items():
                if src not in seen:
                    core = re.sub(r'\.(com|ng|org|co\.uk|co|tv|africa|net|gov)$','',src)
                    if len(core) > 5 and core in combined:
                        matched = src; break

        if matched:
            seen.add(matched)
            info = SOURCES[matched]
            raw_url = r.get("href") or ""
            article_url = (
                raw_url if raw_url.startswith("http") and "news.google.com" not in raw_url
                else f"https://{matched}"
            )
            found.append({
                "source":matched,"name":info["name"],"region":info["region"],
                "url":f"https://{matched}","article_url":article_url,
                "title":title,"snippet":snippet[:300],
                "raw_content":r.get("raw_content",""),
                "pubdate":pubdate,
                "is_nigerian":info["region"] in ("Nigeria","Nigeria-Edu"),
                "is_factchecker":info["region"]=="FactChecker",
            })

    found.sort(key=lambda x:(x["is_factchecker"]*2+x["is_nigerian"]),reverse=True)
    logger.info(f"Sources matched: {[s['name'] for s in found]}")
    return found

def extract_years(text):
    return [int(y) for y in re.findall(r'\b(20\d{2})\b', text)]

def run_distilbert(text):
    inputs = cls_tokenizer(text,return_tensors="pt",truncation=True,max_length=256,padding=True)
    with torch.no_grad():
        logits = cls_model(**inputs).logits
    probs = F.softmax(logits,dim=-1)[0]
    r,f   = float(probs[0]),float(probs[1])
    return {"real":r,"fake":f,"prediction":int(probs.argmax()),"confidence":int(max(r,f)*100)}

# ══════════════════════════════════════════════════════════════════
# PHI-3.5 REASONING — Grok-style with claim type awareness
# ══════════════════════════════════════════════════════════════════
def reason_with_phi35(claim, articles, tavily_summary, claim_type):
    if not HF_TOKEN:
        return {"verdict":"unavailable","reasoning":"HF_TOKEN not set","confidence":0}

    today = datetime.now().strftime("%d %B %Y")
    evidence_parts = []

    if tavily_summary and len(tavily_summary) > 30:
        evidence_parts.append(
            f"[Web Summary — as of {today}]\n{tavily_summary[:800]}"
        )

    for i,(title,text,pubdate,source) in enumerate(articles[:6]):
        part = f"[Source {i+1}: {source}]"
        if pubdate: part += f" [Published: {pubdate}]"
        part += f"\nHeadline: {title}"
        if text and len(text) > 100:
            part += f"\nContent: {text[:2500]}"
        evidence_parts.append(part)

    if not evidence_parts:
        return {"verdict":"INSUFFICIENT_EVIDENCE","reasoning":"No articles retrieved","confidence":0}

    evidence = "\n\n".join(evidence_parts)

    # Claim-type specific instructions for Phi-3.5
    type_instructions = {
        "education": """For education claims:
- Check if the university OFFICIALLY lists this programme on their website
- Check NUC (National Universities Commission) accreditation
- Check JAMB approved course list
- MySchool.ng and SchoolGist are reliable for Nigerian university courses
- If no official source confirms it, the claim is likely FALSE""",

        "political": """For political claims:
- Check if major credible outlets (BBC, Reuters, AP, Punch, Vanguard) directly report this
- An article about someone MOURNING does not mean the subject is dead
- An article about someone being ARRESTED for making a claim means the claim is FALSE
- Check the most RECENT articles — political situations change quickly""",

        "health": """For health claims:
- WHO and CDC are the highest authority sources
- NCDC is the authority for Nigerian health claims
- Check if official health bodies have made statements
- Be very careful — false health claims can cause harm""",

        "general": """For general claims:
- Look for direct confirmation from credible sources
- Check publication dates carefully
- Consider whether the claim matches what articles actually say""",
    }

    specific_guidance = type_instructions.get(claim_type, type_instructions["general"])

    prompt = f"""You are TruthLens, an AI fact-checker. Today is {today}.

CLAIM: "{claim}"
CLAIM TYPE: {claim_type.upper()}

{specific_guidance}

EVIDENCE FOUND:
{evidence}

ANALYSE CAREFULLY:
1. Does any source DIRECTLY confirm or deny this claim?
2. Are sources about the actual claim or just related topics?
3. For education: is there official university/NUC/JAMB confirmation?
4. For political: check if news is about the person or just mentions them
5. Consider dates — old news may not apply today

RESPOND EXACTLY:
REASONING: [detailed step by step analysis of what each source says]
VERDICT: SUPPORTS_CLAIM or CONTRADICTS_CLAIM or INSUFFICIENT_EVIDENCE
CONFIDENCE: HIGH or MEDIUM or LOW
EXPLANATION: [one clear sentence for the user]"""

    try:
        resp = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct/v1/chat/completions",
            headers={"Authorization":f"Bearer {HF_TOKEN}","Content-Type":"application/json"},
            json={
                "model": "microsoft/Phi-3.5-mini-instruct",
                "messages": [
                    {"role":"system","content":f"You are TruthLens, a professional fact-checker. Today is {today}."},
                    {"role":"user","content":prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.1,
                "stream": False
            },
            timeout=50
        )

        logger.info(f"Phi-3.5 status: {resp.status_code}")

        if resp.status_code == 200:
            raw = resp.json()
            if isinstance(raw,dict) and "choices" in raw:
                text = raw["choices"][0]["message"]["content"]
            elif isinstance(raw,list) and raw:
                text = raw[0].get("generated_text","")
            else:
                text = str(raw)

            logger.info(f"Phi-3.5 response: {text[:400]}")

            v_m = re.search(r'VERDICT[:\s]+(SUPPORTS_CLAIM|CONTRADICTS_CLAIM|INSUFFICIENT_EVIDENCE)',text,re.IGNORECASE)
            r_m = re.search(r'REASONING[:\s]+(.*?)(?=VERDICT:|$)',text,re.IGNORECASE|re.DOTALL)
            c_m = re.search(r'CONFIDENCE[:\s]+(HIGH|MEDIUM|LOW)',text,re.IGNORECASE)
            e_m = re.search(r'EXPLANATION[:\s]+(.*?)(?=\n\n|$)',text,re.IGNORECASE|re.DOTALL)

            verdict    = v_m.group(1).upper() if v_m else "INSUFFICIENT_EVIDENCE"
            reasoning  = r_m.group(1).strip()[:1000] if r_m else text[:700]
            confidence = c_m.group(1).upper() if c_m else "LOW"
            explanation= e_m.group(1).strip()[:250] if e_m else reasoning[:150]
            conf_score = {"HIGH":0.9,"MEDIUM":0.65,"LOW":0.4}.get(confidence,0.4)

            return {"verdict":verdict,"reasoning":reasoning,"explanation":explanation,"confidence":conf_score}

        elif resp.status_code == 503:
            logger.warning("Phi-3.5 loading — retry 20s")
            time.sleep(20)
            return reason_with_phi35(claim,articles,tavily_summary,claim_type)
        else:
            err = resp.text[:300]
            logger.error(f"Phi-3.5 {resp.status_code}: {err}")
            return {"verdict":"api_error","reasoning":f"API {resp.status_code}: {err}","confidence":0}

    except Exception as e:
        logger.error(f"Phi-3.5 exception: {e}")
        return {"verdict":"error","reasoning":str(e),"confidence":0}

def fuse_verdict(distilbert, matched, phi35):
    nigerian  = [s for s in matched if s["is_nigerian"]]
    factcheck = [s for s in matched if s["is_factchecker"]]
    total     = len(matched)
    phi_v     = phi35.get("verdict","INSUFFICIENT_EVIDENCE")
    phi_conf  = phi35.get("confidence",0)
    phi_exp   = phi35.get("explanation","")
    phi_ok    = phi_v not in ("unavailable","api_error","error","insufficient_evidence","INSUFFICIENT_EVIDENCE","")

    fc_fake = [s for s in factcheck if any(w in s["title"].lower() for w in ["false","fake","misinformation","misleading","debunked","hoax"])]
    fc_real = [s for s in factcheck if any(w in s["title"].lower() for w in ["true","confirmed","accurate","verified","correct"])]

    if fc_fake:
        return {"verdict":"FAKE","confidence":99,"title":"Confirmed Misinformation","subtitle":f"Flagged by {fc_fake[0]['name']}"}
    if fc_real:
        return {"verdict":"REAL","confidence":99,"title":"Fact-Checker Verified","subtitle":f"Confirmed by {fc_real[0]['name']}"}
    if phi_ok and phi_v=="CONTRADICTS_CLAIM" and phi_conf>=0.5:
        return {"verdict":"FAKE","confidence":95,"title":"AI Reading Contradicts Claim","subtitle":phi_exp}
    if phi_ok and phi_v=="SUPPORTS_CLAIM" and phi_conf>=0.65 and total>=2:
        return {"verdict":"REAL","confidence":95,"title":"Confirmed — AI Read and Verified","subtitle":f"Phi-3.5 confirmed across {total} sources: {phi_exp}"}
    if phi_ok and phi_v=="SUPPORTS_CLAIM" and phi_conf>=0.65:
        return {"verdict":"REAL","confidence":85,"title":"AI Reading Confirms Claim","subtitle":phi_exp}
    if total>=3 and phi_v!="CONTRADICTS_CLAIM":
        regions=list(dict.fromkeys(s["region"] for s in matched[:4]))
        return {"verdict":"REAL","confidence":85,"title":"Confirmed by Multiple Outlets","subtitle":f"Found in {total} outlets: {', '.join(regions)}"}
    if len(nigerian)>=2 and phi_v!="CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":82,"title":"Confirmed by Nigerian Outlets","subtitle":f"{nigerian[0]['name']} and {nigerian[1]['name']}"}
    if total>=2 and phi_v!="CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":75,"title":"Likely Authentic News","subtitle":f"Found in {matched[0]['name']} and {matched[1]['name']}"}
    if total==1 and phi_v!="CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":60,"title":"Possibly Authentic","subtitle":f"1 outlet: {matched[0]['name']}. Verify further."}
    if distilbert["prediction"]==1 and distilbert["confidence"]>=80:
        return {"verdict":"FAKE","confidence":78,"title":"Likely Fake — No Sources","subtitle":"Misinformation patterns + zero credible sources found."}
    if distilbert["prediction"]==0 and distilbert["confidence"]>=80:
        return {"verdict":"SUSPICIOUS","confidence":55,"title":"Suspicious — Cannot Verify","subtitle":"Writing appears credible but no outlet confirms this."}
    return {"verdict":"MIXED","confidence":50,"title":"Uncertain — Verify Manually","subtitle":"Mixed signals. Use fact-checker links below."}

class AnalyseRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"name":"TruthLens API v9.0","status":"running","tavily":bool(TAVILY_KEY),"phi35":bool(HF_TOKEN),"sources":len(SOURCES)}

@app.get("/health")
def health():
    return {"status":"healthy","tavily":bool(TAVILY_KEY),"phi35":bool(HF_TOKEN)}

@app.get("/stats")
def stats():
    return COUNTER.copy()

@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    text = req.text.strip()
    if not text or len(text)<8: raise HTTPException(400,"Text too short")
    if len(text)>8000: raise HTTPException(400,"Text too long")

    clean = re.sub(r'http\S+|www\S+|<.*?>|\s+',' ',text).strip()[:600]
    current_year = datetime.now().year
    claim_years  = extract_years(clean)
    future_years = [y for y in claim_years if y>current_year]

    if future_years:
        return _future_verdict(future_years[0])

    ck = cache_key(clean)
    cached = cache_get(ck)
    if cached:
        tick(cache_hit=True)
        cached["from_cache"] = True
        return cached
    tick()

    db = run_distilbert(clean)
    logger.info(f"DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%")

    claim_info = detect_claim_type(clean)
    claim_type = claim_info["type"]

    queries = generate_queries(clean, claim_type)
    results, tavily_summary = search_all(queries, claim_type)

    matched   = match_sources(clean, results)
    factcheck = [s for s in matched if s["is_factchecker"]]

    articles_for_phi = []
    for src in matched[:6]:
        raw = src.get("raw_content","")
        if raw and len(raw)>200:
            full_text = raw
        else:
            url = src["article_url"]
            full_text = read_article(url) if url and "news.google.com" not in url else ""
        articles_for_phi.append((src["title"],full_text,src["pubdate"],src["name"]))

    for r in results[:8]:
        rc = r.get("raw_content","")
        if rc and len(rc)>300:
            title = r.get("title","")
            if not any(t==title for t,_,_,_ in articles_for_phi):
                sn = "Web"
                for sd in SOURCES:
                    if sd in (r.get("href") or "").lower():
                        sn = SOURCES[sd]["name"]; break
                articles_for_phi.append((title,rc,r.get("pubdate",""),sn))

    scraped = sum(1 for _,t,_,_ in articles_for_phi if len(t)>200)

    phi35 = reason_with_phi35(clean, articles_for_phi, tavily_summary, claim_type)
    logger.info(f"Phi-3.5: verdict={phi35.get('verdict')} conf={phi35.get('confidence')}")

    fusion     = fuse_verdict(db, matched, phi35)
    verdict    = fusion["verdict"]
    confidence = fusion["confidence"]

    upper     = sum(1 for w in clean.split() if w.isupper() and len(w)>2)
    excl      = clean.count("!")
    clickbait = any(w in clean.lower() for w in ["shocking","secret","exposed","leaked","viral","breaking"])

    phi_sig = {
        "SUPPORTS_CLAIM":        ("Read articles — SUPPORTS claim",    "pos"),
        "CONTRADICTS_CLAIM":     ("Read articles — CONTRADICTS claim", "neg"),
        "INSUFFICIENT_EVIDENCE": ("Insufficient article content",       "neu"),
    }.get(phi35.get("verdict",""),("Check HF_TOKEN in Railway Variables","neu"))

    signals = [
        {"icon":"🌐","label":"Web Sources","value":f"{len(matched)} credible outlets","type":"pos" if len(matched)>=2 else "neg" if len(matched)==0 else "neu"},
        {"icon":"🤖","label":"DistilBERT","value":f"{'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%","type":"pos" if db['prediction']==0 else "neg"},
        {"icon":"🧠","label":"Phi-3.5 Reasoning","value":phi_sig[0],"type":phi_sig[1]},
        {"icon":"📰","label":"Articles Read","value":f"{scraped} articles analysed","type":"pos" if scraped>0 else "neu"},
        {"icon":"🎯","label":"Claim Type","value":claim_type.upper(),"type":"neu"},
        {"icon":"🎣","label":"Clickbait","value":"Detected" if clickbait else "None","type":"neg" if clickbait else "pos"},
    ]

    phi_reasoning = phi35.get("reasoning","") or phi35.get("explanation","") or "No reasoning returned from Phi-3.5. Check HF_TOKEN in Railway Variables."

    if verdict=="REAL":
        src_list=", ".join(s["name"] for s in matched[:3]) if matched else "none"
        analysis=(f"This content appears CREDIBLE.\n\nCLAIM TYPE: {claim_type.upper()} — searched the most relevant sources for this type of claim.\n\nLAYER 1 — DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} at {db['confidence']}% (raw score before web verification).\n\nLAYER 2 — WEB: Found {len(matched)} outlet(s) including {src_list}. Used {len(queries)} targeted queries via Tavily + Google News.\n\nLAYER 3 — PHI-3.5: {phi_reasoning[:500]}\n\nRECOMMENDATION: Content appears credible. Click sources below.")
    elif verdict in ("FAKE","SUSPICIOUS"):
        analysis=(f"This content is LIKELY FAKE OR MISLEADING.\n\nCLAIM TYPE: {claim_type.upper()} — searched the most relevant sources.\n\nLAYER 1 — DistilBERT: {db['confidence']}% {'REAL' if db['prediction']==0 else 'FAKE'}.\n\nLAYER 2 — WEB: {'Zero credible sources found.' if not matched else f'{len(matched)} sources found but content contradicts the claim.'}\n\nLAYER 3 — PHI-3.5: {phi_reasoning[:500]}\n\nRECOMMENDATION: Do not share. Verify through AfricaCheck or Dubawa.")
    else:
        analysis=(f"UNCERTAIN: Mixed signals.\n\nCLAIM TYPE: {claim_type.upper()}.\nDistilBERT: {db['confidence']}%. Sources: {len(matched)}.\nPhi-3.5: {phi_reasoning[:300]}\n\nRECOMMENDATION: Check AfricaCheck or Dubawa.")

    result = {
        "verdict":verdict,"confidence":confidence,
        "verdict_title":fusion["title"],"verdict_subtitle":fusion["subtitle"],
        "analysis":analysis,"signals":signals,
        "real_probability":round(db["real"],4),"fake_probability":round(db["fake"],4),
        "phi3":{"verdict":phi35.get("verdict",""),"reasoning":phi_reasoning,"explanation":phi35.get("explanation",""),"confidence":phi35.get("confidence",0)},
        "claim_type":claim_type,
        "tavily_summary":tavily_summary[:300] if tavily_summary else "",
        "sources_found":[{"source":s["source"],"name":s["name"],"region":s["region"],"url":s["url"],"article_url":s["article_url"],"title":s["title"][:120],"snippet":s["snippet"],"pubdate":s["pubdate"],"is_factchecker":s["is_factchecker"],"is_nigerian":s["is_nigerian"]} for s in matched[:8]],
        "fact_checks":[{"name":s["name"],"url":s["url"],"article_url":s["article_url"],"title":s["title"][:120],"snippet":s["snippet"]} for s in factcheck[:3]],
        "queries_used":queries,"articles_scraped":scraped,
        "search_timestamp":datetime.now().isoformat(),
        "from_cache":False,"daily_stats":COUNTER.copy(),
    }
    cache_set(ck, result)
    return result

def _future_verdict(year):
    return {
        "verdict":"FAKE","confidence":99,
        "verdict_title":"Future Event Claimed as Fact",
        "verdict_subtitle":f"References {year} which has not happened yet.",
        "analysis":f"This content references {year}, a future year. Cannot be fact.\n\nDo not share.",
        "signals":[{"icon":"📅","label":"Date","value":f"Future year {year}","type":"neg"},{"icon":"🚫","label":"Verdict","value":"Impossible","type":"neg"},{"icon":"🤖","label":"DistilBERT","value":"N/A","type":"neg"},{"icon":"🧠","label":"Phi-3.5","value":"N/A","type":"neg"},{"icon":"🌐","label":"Web","value":"N/A","type":"neg"},{"icon":"⚠️","label":"Warning","value":"Do not share","type":"neg"}],
        "real_probability":0.01,"fake_probability":0.99,
        "phi3":{"verdict":"N/A","reasoning":"Future date detected","confidence":0},
        "claim_type":"temporal","tavily_summary":"",
        "sources_found":[],"fact_checks":[],"queries_used":[],"articles_scraped":0,
        "search_timestamp":datetime.now().isoformat(),"from_cache":False,"daily_stats":{}
    }
