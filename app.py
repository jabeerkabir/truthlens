"""
TruthLens API v9.0 — Grok-style Claim-Aware Search
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

# LLM for reasoning — Mistral is free on HF Inference API
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

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
# CLAIM TYPE DETECTOR
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
            "dead","died","death","arrested","impeached","resigned","elected",
            "coup","protest","policy","bill","passed","tinubu","buhari","obi","atiku"
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
            "premier league","world cup","champions league","goal","match","team"
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

def generate_queries(claim: str, claim_type: str) -> list:
    queries = [claim]
    proper  = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
    current_year = datetime.now().year

    if claim_type == "education":
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
            queries.append(f"is {' '.join(proper[:2])} alive {current_year}")

    elif claim_type == "health":
        queries.append(f"{claim} WHO official statement {current_year}")
        queries.append(f"{claim} NCDC Nigeria {current_year}")

    elif claim_type == "business":
        queries.append(f"{claim} CBN official {current_year}")
        queries.append(f"{claim} Nigeria economy verified {current_year}")

    elif claim_type == "entertainment":
        if proper:
            queries.append(f"{' '.join(proper[:2])} {current_year} confirmed")
        queries.append(f"{claim} entertainment news {current_year}")

    elif claim_type == "sports":
        queries.append(f"{claim} sports news {current_year}")
        queries.append(f"{claim} official confirmed {current_year}")

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
    logger.info(f"Queries ({claim_type}): {unique[:4]}")
    return unique[:4]

# ══════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════
def search_tavily(query: str, claim_type: str) -> tuple:
    if not TAVILY_KEY:
        logger.warning("TAVILY_KEY empty — skipping Tavily")
        return [], ""
    try:
        # For education — search specific authoritative sites
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
            include_domains = ["who.int","cdc.gov","ncdc.gov.ng"]
        elif claim_type == "political":
            include_domains = [
                "premiumtimesng.com","punchng.com","vanguardngr.com",
                "bbc.com","reuters.com","apnews.com","aljazeera.com"
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

        if claim_years and pubdate:
            pub_years = extract_years(pubdate)
            if pub_years and not any(abs(py-cy)<=1 for py in pub_years for cy in claim_years):
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
                    if name in suffix and src not in seen and len(name)>4:
                        matched = src; break

        if not matched:
            combined = (title+" "+snippet).lower()
            for src, info in SOURCES.items():
                if src not in seen:
                    core = re.sub(r'\.(com|ng|org|co\.uk|co|tv|africa|net|gov)$','',src)
                    if len(core)>5 and core in combined:
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

# ══════════════════════════════════════════════════════════════════
# LLM REASONING — Mistral-7B (actually works on HF Inference API)
# ══════════════════════════════════════════════════════════════════
def reason_with_llm(claim: str, articles: list, tavily_summary: str, claim_type: str) -> dict:
    if not HF_TOKEN:
        return {"verdict":"unavailable","reasoning":"HF_TOKEN not configured","confidence":0}

    today = datetime.now().strftime("%d %B %Y")

    # Build evidence
    evidence_parts = []
    if tavily_summary and len(tavily_summary) > 30:
        evidence_parts.append(f"[Web Summary — {today}]\n{tavily_summary[:600]}")

    for i, (title, text, pubdate, source) in enumerate(articles[:5]):
        part = f"[Source {i+1}: {source}]"
        if pubdate: part += f" [Date: {pubdate}]"
        part += f"\nHeadline: {title}"
        if text and len(text) > 100:
            part += f"\nContent: {text[:2000]}"
        evidence_parts.append(part)

    if not evidence_parts:
        return {"verdict":"INSUFFICIENT_EVIDENCE","reasoning":"No articles found","confidence":0}

    evidence = "\n\n".join(evidence_parts)

    type_guide = {
        "education": "For education claims: check if university website or NUC/JAMB officially lists this programme. If no official source confirms it, verdict is CONTRADICTS_CLAIM.",
        "political": "For political claims: check if major news outlets directly report this event. Someone mourning X does not mean X died. Someone arrested for claiming Y means Y is false.",
        "health":    "For health claims: WHO and CDC are highest authority. Check official statements.",
        "business":  "For business claims: check CBN, official government sources, and major financial outlets.",
        "entertainment": "For entertainment claims: check official announcements, verified artist pages, major entertainment outlets.",
        "sports":    "For sports claims: check official league/team announcements and major sports outlets.",
        "science":   "For science claims: check peer-reviewed sources and official research institutions.",
        "general":   "Check if credible sources directly confirm this specific claim.",
    }.get(claim_type, "Check if credible sources directly confirm this claim.")

    prompt = f"""[INST] You are TruthLens, a professional AI fact-checker. Today is {today}.

CLAIM TO VERIFY: "{claim}"
CLAIM CATEGORY: {claim_type.upper()}

GUIDANCE: {type_guide}

EVIDENCE FROM SOURCES:
{evidence}

IMPORTANT RULES:
1. Read what each source ACTUALLY says — not just keywords
2. "Person arrested for claiming X" = X is FALSE
3. "Person mourns Y" does NOT mean person is dead
4. Old articles may not reflect current reality
5. No official university/NUC confirmation = education claim is likely FALSE
6. Multiple major outlets directly confirming = SUPPORTS_CLAIM

RESPOND IN THIS EXACT FORMAT:
REASONING: [explain what each source actually says step by step]
VERDICT: SUPPORTS_CLAIM or CONTRADICTS_CLAIM or INSUFFICIENT_EVIDENCE
CONFIDENCE: HIGH or MEDIUM or LOW
EXPLANATION: [one clear sentence for the user] [/INST]"""

    # Try Mistral first — it IS available on HF Inference API
    models_to_try = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    for model in models_to_try:
        try:
            logger.info(f"Trying LLM: {model}")
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 600,
                        "temperature": 0.1,
                        "do_sample": False,
                        "return_full_text": False
                    }
                },
                timeout=45
            )
            logger.info(f"LLM {model} status: {resp.status_code}")

            if resp.status_code == 200:
                raw = resp.json()
                if isinstance(raw, list) and raw:
                    text = raw[0].get("generated_text", "")
                elif isinstance(raw, dict):
                    text = raw.get("generated_text", str(raw))
                else:
                    text = str(raw)

                if len(text) < 20:
                    logger.warning(f"LLM returned empty response")
                    continue

                logger.info(f"LLM response ({len(text)} chars): {text[:300]}")

                v_m = re.search(r'VERDICT[:\s]+(SUPPORTS_CLAIM|CONTRADICTS_CLAIM|INSUFFICIENT_EVIDENCE)', text, re.IGNORECASE)
                r_m = re.search(r'REASONING[:\s]+(.*?)(?=VERDICT:|CONFIDENCE:|$)', text, re.IGNORECASE|re.DOTALL)
                c_m = re.search(r'CONFIDENCE[:\s]+(HIGH|MEDIUM|LOW)', text, re.IGNORECASE)
                e_m = re.search(r'EXPLANATION[:\s]+(.*?)(?=\n\n|\[|$)', text, re.IGNORECASE|re.DOTALL)

                verdict    = v_m.group(1).upper() if v_m else "INSUFFICIENT_EVIDENCE"
                reasoning  = r_m.group(1).strip()[:800] if r_m else text[:600]
                confidence = c_m.group(1).upper() if c_m else "MEDIUM"
                explanation= e_m.group(1).strip()[:250] if e_m else reasoning[:150]
                conf_score = {"HIGH":0.9,"MEDIUM":0.65,"LOW":0.4}.get(confidence,0.5)

                return {
                    "verdict":     verdict,
                    "reasoning":   reasoning,
                    "explanation": explanation,
                    "confidence":  conf_score,
                    "model_used":  model.split("/")[-1]
                }

            elif resp.status_code == 503:
                logger.warning(f"{model} loading — trying next")
                continue
            else:
                logger.error(f"{model} returned {resp.status_code}: {resp.text[:150]}")
                continue

        except Exception as e:
            logger.error(f"LLM {model} exception: {e}")
            continue

    return {
        "verdict": "INSUFFICIENT_EVIDENCE",
        "reasoning": "All LLM models unavailable. Check HF_TOKEN in Railway Variables and ensure token has Inference API access.",
        "confidence": 0,
        "model_used": "none"
    }

# ══════════════════════════════════════════════════════════════════
# VERDICT FUSION
# ══════════════════════════════════════════════════════════════════
def fuse_verdict(distilbert: dict, matched: list, llm: dict) -> dict:
    nigerian  = [s for s in matched if s["is_nigerian"]]
    factcheck = [s for s in matched if s["is_factchecker"]]
    total     = len(matched)
    llm_v     = llm.get("verdict","INSUFFICIENT_EVIDENCE")
    llm_conf  = llm.get("confidence",0)
    llm_exp   = llm.get("explanation","")
    llm_ok    = llm_v in ("SUPPORTS_CLAIM","CONTRADICTS_CLAIM")

    fc_fake = [s for s in factcheck if any(w in s["title"].lower() for w in ["false","fake","misinformation","misleading","debunked","hoax"])]
    fc_real = [s for s in factcheck if any(w in s["title"].lower() for w in ["true","confirmed","accurate","verified","correct"])]

    if fc_fake:
        return {"verdict":"FAKE","confidence":99,"title":"Confirmed Misinformation","subtitle":f"Flagged by {fc_fake[0]['name']}: {fc_fake[0]['title'][:60]}"}
    if fc_real:
        return {"verdict":"REAL","confidence":99,"title":"Fact-Checker Verified","subtitle":f"Confirmed by {fc_real[0]['name']}"}
    if llm_ok and llm_v=="CONTRADICTS_CLAIM" and llm_conf>=0.5:
        return {"verdict":"FAKE","confidence":95,"title":"AI Reading Contradicts Claim","subtitle":llm_exp}
    if llm_ok and llm_v=="SUPPORTS_CLAIM" and llm_conf>=0.6 and total>=2:
        return {"verdict":"REAL","confidence":95,"title":"Confirmed — AI Read and Verified","subtitle":f"AI confirmed across {total} sources: {llm_exp}"}
    if llm_ok and llm_v=="SUPPORTS_CLAIM" and llm_conf>=0.6:
        return {"verdict":"REAL","confidence":85,"title":"AI Reading Confirms Claim","subtitle":llm_exp}
    if total>=3 and llm_v!="CONTRADICTS_CLAIM":
        regions=list(dict.fromkeys(s["region"] for s in matched[:4]))
        return {"verdict":"REAL","confidence":85,"title":"Confirmed by Multiple Outlets","subtitle":f"Found in {total} outlets: {', '.join(regions)}"}
    if len(nigerian)>=2 and llm_v!="CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":82,"title":"Confirmed by Nigerian Outlets","subtitle":f"{nigerian[0]['name']} and {nigerian[1]['name']}"}
    if total>=2 and llm_v!="CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":75,"title":"Likely Authentic News","subtitle":f"Found in {matched[0]['name']} and {matched[1]['name']}"}
    if total==1 and llm_v!="CONTRADICTS_CLAIM":
        return {"verdict":"REAL","confidence":60,"title":"Possibly Authentic","subtitle":f"1 outlet: {matched[0]['name']}. Verify further."}
    if distilbert["prediction"]==1 and distilbert["confidence"]>=80:
        return {"verdict":"FAKE","confidence":78,"title":"Likely Fake — No Sources","subtitle":"Misinformation patterns + zero credible sources found."}
    if distilbert["prediction"]==0 and distilbert["confidence"]>=80:
        return {"verdict":"SUSPICIOUS","confidence":55,"title":"Suspicious — Cannot Verify","subtitle":"Writing appears credible but no outlet confirms this."}
    return {"verdict":"MIXED","confidence":50,"title":"Uncertain — Verify Manually","subtitle":"Mixed signals. Use the fact-checker links below."}

# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════
class AnalyseRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"name":"TruthLens API v9.0","status":"running","tavily":bool(TAVILY_KEY),"llm":bool(HF_TOKEN),"llm_model":LLM_MODEL,"sources":len(SOURCES)}

@app.get("/health")
def health():
    return {"status":"healthy","tavily":bool(TAVILY_KEY),"llm":bool(HF_TOKEN)}

@app.get("/stats")
def stats():
    return COUNTER.copy()

@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    text = req.text.strip()
    if not text or len(text)<8: raise HTTPException(400,"Text too short")
    if len(text)>8000: raise HTTPException(400,"Text too long")

    clean        = re.sub(r'http\S+|www\S+|<.*?>|\s+',' ',text).strip()[:600]
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

    db         = run_distilbert(clean)
    claim_type = detect_claim_type(clean)
    queries    = generate_queries(clean, claim_type)
    results, tavily_summary = search_all(queries, claim_type)
    matched    = match_sources(clean, results)
    factcheck  = [s for s in matched if s["is_factchecker"]]

    articles_for_llm = []
    for src in matched[:6]:
        raw = src.get("raw_content","")
        full_text = raw if raw and len(raw)>200 else read_article(src["article_url"])
        articles_for_llm.append((src["title"],full_text,src["pubdate"],src["name"]))

    for r in results[:8]:
        rc = r.get("raw_content","")
        if rc and len(rc)>300:
            title = r.get("title","")
            if not any(t==title for t,_,_,_ in articles_for_llm):
                sn = next((SOURCES[sd]["name"] for sd in SOURCES if sd in (r.get("href") or "").lower()),"Web")
                articles_for_llm.append((title,rc,r.get("pubdate",""),sn))

    scraped = sum(1 for _,t,_,_ in articles_for_llm if len(t)>200)

    llm    = reason_with_llm(clean, articles_for_llm, tavily_summary, claim_type)
    fusion = fuse_verdict(db, matched, llm)
    verdict    = fusion["verdict"]
    confidence = fusion["confidence"]

    model_used = llm.get("model_used","unknown")
    upper     = sum(1 for w in clean.split() if w.isupper() and len(w)>2)
    excl      = clean.count("!")
    clickbait = any(w in clean.lower() for w in ["shocking","secret","exposed","leaked","viral","breaking"])

    llm_sig = {
        "SUPPORTS_CLAIM":        (f"AI read articles — SUPPORTS claim","pos"),
        "CONTRADICTS_CLAIM":     (f"AI read articles — CONTRADICTS claim","neg"),
        "INSUFFICIENT_EVIDENCE": ("Insufficient evidence in articles","neu"),
    }.get(llm.get("verdict",""),("LLM unavailable — check HF_TOKEN","neu"))

    signals = [
        {"icon":"🌐","label":"Web Sources","value":f"{len(matched)} credible outlets","type":"pos" if len(matched)>=2 else "neg" if len(matched)==0 else "neu"},
        {"icon":"🤖","label":"DistilBERT","value":f"{'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}%","type":"pos" if db['prediction']==0 else "neg"},
        {"icon":"🧠","label":f"AI Reasoning","value":llm_sig[0],"type":llm_sig[1]},
        {"icon":"📰","label":"Articles Read","value":f"{scraped} articles analysed","type":"pos" if scraped>0 else "neu"},
        {"icon":"🎯","label":"Claim Category","value":claim_type.upper(),"type":"neu"},
        {"icon":"🎣","label":"Clickbait","value":"Detected" if clickbait else "None","type":"neg" if clickbait else "pos"},
    ]

    llm_reasoning = llm.get("reasoning","") or llm.get("explanation","") or "No reasoning returned. Check HF_TOKEN in Railway Variables."
    src_list = ", ".join(s["name"] for s in matched[:3]) if matched else "none"

    if verdict=="REAL":
        analysis=(f"CREDIBLE — {claim_type.upper()} CLAIM\n\nLAYER 1 — DistilBERT: {'REAL' if db['prediction']==0 else 'FAKE'} {db['confidence']}% (raw score before web check).\n\nLAYER 2 — WEB SEARCH [{claim_type}]: Found {len(matched)} outlet(s) including {src_list}. Searched via {len(queries)} targeted queries via Tavily + Google News.\n\nLAYER 3 — AI READING ({model_used}): {llm_reasoning[:500]}\n\nRECOMMENDATION: Content appears credible. Click sources to verify.")
    elif verdict in ("FAKE","SUSPICIOUS"):
        analysis=(f"LIKELY FAKE — {claim_type.upper()} CLAIM\n\nLAYER 1 — DistilBERT: {db['confidence']}% {'REAL' if db['prediction']==0 else 'FAKE'}.\n\nLAYER 2 — WEB SEARCH [{claim_type}]: {'Zero credible sources found.' if not matched else f'{len(matched)} source(s) — content contradicts claim.'}\n\nLAYER 3 — AI READING ({model_used}): {llm_reasoning[:500]}\n\nRECOMMENDATION: Do not share. Verify through AfricaCheck or Dubawa.")
    else:
        analysis=(f"UNCERTAIN — {claim_type.upper()} CLAIM\n\nDistilBERT: {db['confidence']}%. Sources: {len(matched)}. AI: {llm_reasoning[:300]}\n\nRECOMMENDATION: Check AfricaCheck or Dubawa before sharing.")

    result = {
        "verdict":verdict,"confidence":confidence,
        "verdict_title":fusion["title"],"verdict_subtitle":fusion["subtitle"],
        "analysis":analysis,"signals":signals,
        "real_probability":round(db["real"],4),"fake_probability":round(db["fake"],4),
        "phi3":{
            "verdict":llm.get("verdict",""),
            "reasoning":llm_reasoning,
            "explanation":llm.get("explanation",""),
            "confidence":llm.get("confidence",0),
            "model":model_used
        },
        "claim_type":claim_type,
        "tavily_summary":tavily_summary[:300] if tavily_summary else "",
        "sources_found":[{
            "source":s["source"],"name":s["name"],"region":s["region"],
            "url":s["url"],"article_url":s["article_url"],
            "title":s["title"][:120],"snippet":s["snippet"],
            "pubdate":s["pubdate"],"is_factchecker":s["is_factchecker"],"is_nigerian":s["is_nigerian"]
        } for s in matched[:8]],
        "fact_checks":[{
            "name":s["name"],"url":s["url"],"article_url":s["article_url"],
            "title":s["title"][:120],"snippet":s["snippet"]
        } for s in factcheck[:3]],
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
        "signals":[
            {"icon":"📅","label":"Date","value":f"Future year {year}","type":"neg"},
            {"icon":"🚫","label":"Verdict","value":"Impossible","type":"neg"},
            {"icon":"🤖","label":"DistilBERT","value":"N/A","type":"neg"},
            {"icon":"🧠","label":"AI Reading","value":"N/A","type":"neg"},
            {"icon":"🌐","label":"Web","value":"N/A","type":"neg"},
            {"icon":"⚠️","label":"Warning","value":"Do not share","type":"neg"},
        ],
        "real_probability":0.01,"fake_probability":0.99,
        "phi3":{"verdict":"N/A","reasoning":"Future date","confidence":0,"model":"none"},
        "claim_type":"temporal","tavily_summary":"",
        "sources_found":[],"fact_checks":[],"queries_used":[],"articles_scraped":0,
        "search_timestamp":datetime.now().isoformat(),"from_cache":False,"daily_stats":{}
    }
