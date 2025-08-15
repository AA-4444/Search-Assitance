# app.py — compact prod backend v2 (single-file)
import os, re, json, time, uuid, csv, random, logging, sqlite3
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ddgs import DDGS

# ==== Optional (graceful fallback) ====
try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None
try:
    import chardet
except Exception:
    chardet = None

# =================== ENV / CONFIG (names preserved) ===================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend_dist")
DB_PATH = os.getenv("DB_PATH", "search_results.db")
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "").strip() or None

MIN_RESULTS = int(os.getenv("MIN_RESULTS", "25"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "6"))
PER_HOST_LIMIT = int(os.getenv("PER_HOST_LIMIT", "3"))
STRONG_BLOCK_ON_BAD = (os.getenv("STRONG_BLOCK_ON_BAD", "true").lower() == "true")

REQUEST_COUNT_FILE = os.getenv("REQUEST_COUNT_FILE", "request_count.json")
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1200"))
REQUEST_PAUSE_MIN = float(os.getenv("REQUEST_PAUSE_MIN", "0.08"))
REQUEST_PAUSE_MAX = float(os.getenv("REQUEST_PAUSE_MAX", "0.22"))

DEFAULT_UA = os.getenv(
    "SCRAPER_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36"
)
SCRAPE_HARD_DEADLINE_SEC = int(os.getenv("SCRAPE_HARD_DEADLINE_SEC", "110"))
PIPELINE_DEADLINE_SEC = int(os.getenv("PIPELINE_DEADLINE_SEC", "150"))
MAX_MICROCRAWL_PAGES = int(os.getenv("MAX_MICROCRAWL_PAGES", "0"))
ENABLE_MICROCRAWL = (os.getenv("ENABLE_MICROCRAWL", "false").lower() == "true")  # по умолчанию off для скорости

FETCH_CONCURRENCY = int(os.getenv("FETCH_CONCURRENCY", str(MAX_WORKERS)))
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", "5400"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("RAILWAY_PORT", "8080")))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# =================== Logging ===================
logger = logging.getLogger("host")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# =================== Flask ===================
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="/")
CORS(app, origins=[
    "http://localhost:8080","http://127.0.0.1:8080","https://search-assistance-production.up.railway.app"
], supports_credentials=True)

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin in {"http://localhost:8080","http://127.0.0.1:8080","https://search-assistance-production.up.railway.app"}:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

# =================== DB & schema ===================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY, name TEXT, website TEXT, description TEXT,
            specialization TEXT, country TEXT, source TEXT, status TEXT,
            suitability TEXT, score REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS queries (
            id TEXT PRIMARY KEY, text TEXT, intent_json TEXT, region TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS results_history (
            id TEXT PRIMARY KEY, query_id TEXT, url TEXT, domain TEXT, source TEXT,
            score REAL, intent_json TEXT, region TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY, query_id TEXT, url TEXT, domain TEXT, action TEXT,
            weight REAL, user_id TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS domain_blocks (
            domain TEXT PRIMARY KEY, blocked_until TEXT, reason TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS crawl_cache (
            key TEXT PRIMARY KEY, value BLOB, created_at REAL)""")
        conn.commit()
    logger.info("Database initialized")
init_db()

# =================== Small utils ===================
def short(s: str, n=220) -> str:
    s = s or "";  return (s[:n] + "…") if len(s) > n else s

def domain_of(url: str) -> str:
    try:
        d = (urlparse(url).netloc or "").lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def normalize_url(url: str) -> str:
    try:
        u = urlparse(url)
        query = [(k,v) for k,v in parse_qsl(u.query, keep_blank_values=False) if not k.lower().startswith("utm_")]
        u2 = u._replace(query=urlencode(query, doseq=True), fragment="")
        return urlunparse(u2)
    except Exception:
        return url

def guess_lang(text: str) -> str:
    t = (text or "").strip()
    if not t: return "en"
    if re.search(r"[іїєґ]", t.lower()): return "uk"
    if re.search(r"[а-яё]", t.lower()): return "ru"
    if langdetect_detect:
        try: return langdetect_detect(t)
        except Exception: pass
    return "en"

def cache_get(key: str) -> Optional[bytes]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT value, created_at FROM crawl_cache WHERE key=?", (key,))
            row = c.fetchone()
            if not row: return None
            value, created = row
            if (time.time() - float(created)) > CACHE_TTL_S:
                c.execute("DELETE FROM crawl_cache WHERE key=?", (key,)); conn.commit()
                return None
            return value
    except Exception:
        return None

def cache_set(key: str, value: bytes):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO crawl_cache (key, value, created_at) VALUES (?, ?, ?)",
                      (key, value, time.time()))
            conn.commit()
    except Exception:
        pass

# =================== Regions & rules (kept names) ===================
REGION_MAP = {
    "wt-wt":{"hl":"en","gl":"us","google_domain":"google.com","tld":"com"},
    "us-en":{"hl":"en","gl":"us","google_domain":"google.com","tld":"us"},
    "uk-en":{"hl":"en","gl":"gb","google_domain":"google.co.uk","tld":"uk"},
    "kz-ru":{"hl":"ru","gl":"kz","google_domain":"google.com","tld":"kz"},
    "ua-ua":{"hl":"uk","gl":"ua","google_domain":"google.com","tld":"ua"},
    "by-ru":{"hl":"ru","gl":"by","google_domain":"google.com","tld":"by"},
    "ru-ru":{"hl":"ru","gl":"ru","google_domain":"google.com","tld":"ru"},
}
REGION_ALIAS = {"kz-kk":"kz-ru","ua-ru":"ua-ua","ua-en":"ua-ua","kz-en":"kz-ru"}
REGION_LANG_ALLOW = {"ua-ua":("uk","ru","en"),"by-ru":("ru","en"),"kz-ru":("kk","ru","en"),"ru-ru":("ru","en"),"wt-wt":("en","ru","uk")}
REGION_HINTS = {
  "kz-ru":{"tokens":["Казахстан","Kazakhstan","Алматы","Almaty","Астана","Astana","Шымкент","KZ"]},
  "ua-ua":{"tokens":["Україна","Ukraine","Київ","Kyiv","Львів","Lviv","Одеса","Odesa","Dnipro","Kharkiv"]},
  "by-ru":{"tokens":["Беларусь","Belarus","Минск","Minsk"]},
  "ru-ru":{"tokens":["Россия","Russian Federation","Москва","Санкт-Петербург","RF","RU"]}
}

# =================== Query understanding ===================
INTENT_AFFILIATE = {"affiliate","партнерск","партнёрск","referral","cpa","opm","program management","affiliate management"}
CASINO_TOKENS = {"casino","igaming","казино","гемблинг","игейминг","игемблинг","беттинг","sportsbook","bookmaker","slots","poker"}
LEARN_TOKENS = {"what is","что такое","overview","обзор","guide","гайд","лекция","лекція","definition","курс"}

def detect_intent(q: str) -> Dict[str,bool]:
    t = (q or "").lower()
    return {
        "affiliate": any(k in t for k in INTENT_AFFILIATE) or any(k in t for k in CASINO_TOKENS),
        "learn": any(k in t for k in LEARN_TOKENS),
        "casino": any(k in t for k in CASINO_TOKENS),
        "business": True
    }

def build_queries(user_prompt: str, region: str) -> List[str]:
    base = user_prompt.strip()
    core = [
        base,
        f'{base} agency',
        "casino affiliate marketing agency",
        "igaming affiliate management company",
        "outsourced affiliate program management igaming",
        "affiliate management for casino operators",
        "игейминг аффилиат агентство",
        "управление партнерской программой казино",
        "OPM igaming",
    ]
    # региональные подсказки
    reg = REGION_MAP.get(REGION_ALIAS.get(region, region), REGION_MAP["wt-wt"])
    tld = reg["tld"]
    hints = {
        "kz-ru":"(Казахстан OR Kazakhstan OR Алматы OR Astana)",
        "ua-ua":"(Україна OR Ukraine OR Київ OR Lviv OR Odesa)",
        "by-ru":"(Беларусь OR Belarus OR Minsk)",
        "ru-ru":"(Россия OR Moscow OR Санкт-Петербург)"
    }.get(REGION_ALIAS.get(region, region), "")
    expanded = []
    for i,q in enumerate(core):
        q2 = q
        if i==0 and hints: q2 = f"{q} {hints}"
        if i in (2,3) and tld: q2 = f"{q} site:*.{tld}"
        expanded.append(q2)
    # минус-слова
    minus = "-jobs -job -vacancy -ваканси -rabota -careers -blog -news -guide -overview -press -definition -glossary -directory -listing -compare -forum -community -apk -download"
    return [f"{q} {minus}" for q in expanded]

# =================== Search adapters ===================
def _load_request_count():
    try:
        with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "engines" not in data:
                data = {"engines":{"ddg":{"count":data.get("count",0)},"serpapi":{"count":0}},
                        "last_reset": data.get("last_reset", datetime.now().strftime("%Y-%m-%d"))}
            if datetime.strptime(data["last_reset"], "%Y-%m-%d").date() < datetime.now().date():
                return {"engines":{"ddg":{"count":0},"serpapi":{"count":0}},"last_reset":datetime.now().strftime("%Y-%m-%d")}
            return data
    except Exception:
        return {"engines":{"ddg":{"count":0},"serpapi":{"count":0}},"last_reset":datetime.now().strftime("%Y-%m-%d")}

def _save_request_count(data):
    try:
        with open(REQUEST_COUNT_FILE,"w",encoding="utf-8") as f: json.dump(data,f)
    except Exception as e: logger.error(f"save req count: {e}")

def _engine_allowed(engine: str) -> bool:
    data = _load_request_count(); return int(data["engines"].get(engine,{}).get("count",0)) < DAILY_REQUEST_LIMIT
def _bump(engine:str):
    data=_load_request_count(); data["engines"].setdefault(engine,{"count":0}); data["engines"][engine]["count"]=int(data["engines"][engine]["count"])+1; _save_request_count(data)

BAD_DOMAINS_HARD = {
    "google.com","maps.google.com","baidu.com","zhihu.com",
    "xnxx.com","pornhub.com","addons.mozilla.org","microsoft.com","support.microsoft.com","edge.microsoft.com",
    "minecraft.net","curseforge.com","planetminecraft.com","softonic.com","apkpure.com","apkcombo.com","uptodown.com",
}
SOFT_BAD_DOMAINS = {"wikipedia.org","facebook.com","instagram.com","x.com","twitter.com","vk.com","ok.ru","indeed.com","glassdoor.com","work.ua","rabota.ua","hh.ru"}

def is_hard_bad_domain(dom: str) -> bool:
    if not dom: return False
    d = dom.lower().lstrip(".");  d = d[4:] if d.startswith("www.") else d
    return d in BAD_DOMAINS_HARD

def duckduckgo_search(query, max_results=20, region="wt-wt"):
    if not _engine_allowed("ddg"): return []
    urls=[]
    try:
        with DDGS() as dd:
            for r in dd.text(query, region=region, safesearch="moderate", timelimit="y", max_results=max_results):
                href=r.get("href");  if href: urls.append(href)
        _bump("ddg"); time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls=[normalize_url(u) for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"DDG fail: {e}"); return []

def serpapi_search(query, max_results=15, region="wt-wt"):
    if not SERPAPI_API_KEY or not _engine_allowed("serpapi"): return []
    reg = REGION_MAP.get(REGION_ALIAS.get(region, region), REGION_MAP["wt-wt"])
    params={"engine":"google","q":query,"num":max_results,"api_key":SERPAPI_API_KEY,"hl":reg["hl"],"gl":reg["gl"],"google_domain":reg["google_domain"]}
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT); r.raise_for_status()
        data=r.json(); urls=[it.get("link") for it in data.get("organic_results",[]) if it.get("link")]
        for blk in ("inline_results","local_results","top_stories"):
            for it in data.get(blk,[]) or []:
                link = it.get("link") or it.get("source");  if link: urls.append(link)
        _bump("serpapi"); time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls=[normalize_url(u) for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.warning(f"SerpAPI fail: {e}"); return []

# =================== Fetch & parse ===================
def _headers(): return {"User-Agent": DEFAULT_UA, "Accept-Language": "en,ru,uk;q=0.9"}

def _get(url: str) -> Optional[requests.Response]:
    try:
        resp=requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code>=400: return None
        ct=(resp.headers.get("Content-Type","") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct: return None
        return resp
    except Exception:
        return None

def _encoding(b: bytes)->str:
    if not b: return "utf-8"
    if chardet:
        try:
            enc=chardet.detect(b).get("encoding")
            if enc: return enc
        except Exception: pass
    return "utf-8"

def parse_html(html: bytes) -> Tuple[str,str,List[Tuple[str,str]]]:
    if not html: return "","",[]
    try:
        soup=BeautifulSoup(html.decode(_encoding(html), errors="ignore"), "lxml")
    except Exception:
        soup=BeautifulSoup(html, "html.parser")
    for bad in soup(["script","style","noscript","svg","picture","source"]): bad.decompose()
    title=(soup.title.string if soup.title and soup.title.string else "").strip()
    md=soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
    meta=md["content"].strip() if md and md.get("content") else ""
    heads=" ".join(h.get_text(" ", strip=True) for h in soup.select("h1,h2,h3")[:8])
    paras=" ".join(p.get_text(" ", strip=True) for p in soup.select("p")[:14])
    text=re.sub(r"\s+"," "," ".join([meta,heads,paras]).strip())
    links=[(a.get_text(" ",strip=True)[:100], a.get("href") or "") for a in soup.select("a[href]")[:160]]
    return short(title,180), text, links

# =================== Relevance / filters / scoring ===================
BLOG_HINTS_URL = ("/blog","/news","/article","/articles","/insights","/guide","/guides","/academy","/press","/glossary")
JOBS_HINTS_URL = ("/jobs","/job","/careers","/vacancies","/vacancy","/rabota","/vakans","/career")
EVENT_HINTS = ("conference","expo","summit","event","exhibition","agenda","speakers")
DIRECTORY_HINTS = ("directory","catalog","list","listing","rank","rating","top-","compare","comparison")
NETWORK_FOR_PUBLISHERS_HINTS = ("for affiliates","for publishers","become an affiliate","sign up as an affiliate","traffic sources","webmasters","start promoting","affiliate network")
COMPANY_SERVICE_HINTS = ("our services","services","solutions","platform","for operators","for brands","for advertisers","clients","case study","features","integrations","pricing","request a demo","program management","affiliate management","opm","outsourced")

def looks_like_job(url,text): u=url.lower(); t=(text or "").lower(); return any(k in u for k in JOBS_HINTS_URL) or any(x in t for x in ("vacanc","ваканси","работа","hiring","careers"))
def looks_like_event(url,text): u=url.lower(); t=(text or "").lower(); return any(e in u for e in EVENT_HINTS) or any(e in t for e in EVENT_HINTS)
def looks_like_directory(url,text): u=url.lower(); t=(text or "").lower(); return any(e in u for e in DIRECTORY_HINTS) or any(e in t for e in DIRECTORY_HINTS)
def looks_like_blog(url,text): u=url.lower(); t=(text or "").lower(); return any(e in u for e in BLOG_HINTS_URL) or any(e in t for e in ("what is","що таке","что такое","guide","обзор","overview","blog","news"))
def is_network_for_publishers(text): t=(text or "").lower(); return any(k in t for k in NETWORK_FOR_PUBLISHERS_HINTS)
def is_company_or_platform(text):
    t=(text or "").lower()
    aff= any(k in t for k in ("affiliate","affiliat","партнерск","партнёрск","referral","opm","program management","affiliate management"))
    ig = any(k in t for k in ("igaming","casino","казино","гемблинг","игейминг","игемблинг","беттинг","sportsbook","bookmaker","poker","slots"))
    svc= any(k in t for k in COMPANY_SERVICE_HINTS)
    return aff and ig and svc and not is_network_for_publishers(t)
def language_ok(text, region):
    allowed=REGION_LANG_ALLOW.get(REGION_ALIAS.get(region,region), REGION_LANG_ALLOW["wt-wt"])
    return guess_lang(text) in allowed

def region_affinity(url,text,region):
    reg=REGION_MAP.get(REGION_ALIAS.get(region,region), REGION_MAP["wt-wt"]); tld="."+reg["tld"] if reg.get("tld") else ""
    u=url.lower(); t=(text or "").lower(); score=0.0
    if tld and u.endswith(tld): score+=0.5
    toks=[x.lower() for x in REGION_HINTS.get(REGION_ALIAS.get(region,region),{}).get("tokens",[])]
    if toks and any(tok in t for tok in toks): score+=0.35
    if language_ok(t,region): score+=0.1
    else: score-=0.2
    return max(0.0,min(1.0,score))

def base_relevance(text):
    t=(text or "").lower(); rel=0.0
    if any(k in t for k in ("affiliate","affiliat","партнерск","opm","program management","affiliate management")): rel+=0.45
    if any(k in t for k in ("igaming","casino","беттинг","sportsbook","bookmaker","poker","slots","gambling")): rel+=0.4
    if any(k in t for k in ("for operators","for brands","for advertisers","clients","case study","platform","services","solutions")): rel+=0.2
    return max(0.0,min(1.0,rel))

def score_item(url,text,region,seed):
    d=domain_of(url); base=0.9*base_relevance(text) + 1.05*region_affinity(url,text,region)
    # quality hints
    qt=0.0
    for k in ("about","contact","services","solutions","platform","features","pricing","clients","case","request a demo"):
        if k in (text or "").lower(): qt+=0.02
    if re.search(r"20(2[3-6])",(text or "").lower()): qt+=0.04
    rnd=random.Random(hash(d+str(seed)) & 0xffffffff).uniform(-0.05,0.05)
    return max(0.0, min(2.2, base+qt+rnd))

# =================== Scrape pipeline ===================
def prefilter_kind(url,text):
    if looks_like_job(url,text): return "jobs"
    if looks_like_event(url,text): return "event"
    if looks_like_directory(url,text): return "directory"
    if looks_like_blog(url,text): return "blog"
    if is_network_for_publishers(text): return "publisher_network"
    return None

def fetch_one(url:str, region:str)->Optional[Dict[str,Any]]:
    try:
        if not url or is_hard_bad_domain(domain_of(url)): return None
        r=_get(url)
        if not r: return None
        title,text,links=parse_html(r.content)
        if not text: return None
        if prefilter_kind(url,text): return None
        if not is_company_or_platform(text): return None
        if not language_ok(text, region): return None
        return {"url":url,"name":title or "N/A","text":text}
    except Exception:
        return None

def scrape_urls(urls: List[str], region: str, seed:int, cap:int)->List[Dict[str,Any]]:
    # per-host limit
    counts={}; filtered=[]
    for u in urls:
        d=domain_of(u)
        if counts.get(d,0)>=PER_HOST_LIMIT: continue
        counts[d]=counts.get(d,0)+1; filtered.append(u)
        if len(filtered)>=cap: break
    out=[];
    with ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY) as ex:
        futs={ex.submit(fetch_one,u,region):u for u in filtered}
        for fut in as_completed(futs):
            item=fut.result()
            if not item: continue
            sc=score_item(item["url"], item["text"], region, seed)
            out.append({
                "id":str(uuid.uuid4()),"name":item["name"],"website":item["url"],
                "description":short(item["text"],400),"country":"N/A","source":"Web","score":sc
            })
            if len(out)>=MAX_RESULTS: break
    # dedupe by domain (keep top)
    bydom={}
    for r in out:
        bydom.setdefault(domain_of(r["website"]), []).append(r)
    res=[]
    for d,items in bydom.items():
        items.sort(key=lambda x:x["score"], reverse=True)
        res.append(items[0])
    res.sort(key=lambda x:x["score"], reverse=True)
    return res

# =================== Seeds: affcatalog (guaranteed blend) ===================
def harvest_affcatalog(limit:int=8)->List[Dict[str,Any]]:
    base="https://affcatalog.com/ru/"
    try:
        r=requests.get(base, headers=_headers(), timeout=REQUEST_TIMEOUT)
        if not r or r.status_code>=400: return []
        soup=BeautifulSoup(r.text,"lxml")
        urls=[]
        for a in soup.select("a[href]"):
            href=a.get("href") or ""; txt=(a.get_text(" ",strip=True) or "")[:200]
            if not href: continue
            if any(key in href.lower() for key in ("/partner","/partners","/affiliat","/program")) or any(x in txt.lower() for x in ("партнер","партнёр","affiliate")):
                full=urljoin(base,href); urls.append(full)
            if len(urls)>=120: break
        # fetch a few to build proper items (lightweight check)
        items=[]; seen=set()
        for u in urls:
            if len(items)>=limit: break
            d=domain_of(u)
            if d in seen: continue
            rr=_get(u)
            if not rr: continue
            title,text,_=parse_html(rr.content)
            if not text: continue
            # keep only company/plattform-like
            if not is_company_or_platform(text): continue
            items.append({
                "id":str(uuid.uuid4()),"name":title or d,"website":u,
                "description":short(text,400),"country":"N/A","source":"Seed","score":0.62
            })
            seen.add(d)
        return items
    except Exception:
        return []

# =================== Collect URLs ===================
def collect_urls(qs: List[str], region:str, engine:str, per_query:int)->List[str]:
    urls=[]
    for q in qs:
        urls+=duckduckgo_search(q, max_results=per_query, region=region)
    # fallback SerpAPI if available or explicitly requested
    if (engine in ("serpapi","both")) and SERPAPI_API_KEY and len(urls) < 600:
        for q in qs[:6]:
            urls+=serpapi_search(q, max_results=max(10, per_query//2), region=region)
    urls=[normalize_url(u) for u in urls if not is_hard_bad_domain(domain_of(u))]
    # uniq
    seen=set(); merged=[]
    for u in urls:
        if u not in seen:
            merged.append(u); seen.add(u)
    return merged

# =================== Feedback & domain blocking ===================
BAD_BLOCK_DAYS = 60
def _block_domain(dom:str, reason="user_bad", days=BAD_BLOCK_DAYS):
    try:
        until=(datetime.utcnow()+timedelta(days=days)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor(); c.execute("""INSERT OR REPLACE INTO domain_blocks (domain, blocked_until, reason) VALUES (?,?,?)""",(dom,until,reason)); conn.commit()
    except Exception as e: logger.error(f"block domain: {e}")

def domain_is_blocked(dom:str)->bool:
    if not STRONG_BLOCK_ON_BAD:  # только явная блокировка
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c=conn.cursor(); c.execute("SELECT blocked_until FROM domain_blocks WHERE domain=?",(dom,)); row=c.fetchone()
                return bool(row and datetime.fromisoformat(row[0])>datetime.utcnow())
        except Exception: return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("SELECT blocked_until FROM domain_blocks WHERE domain=?",(dom,)); row=c.fetchone()
            if row and datetime.fromisoformat(row[0])>datetime.utcnow(): return True
            since=(datetime.utcnow()-timedelta(days=BAD_BLOCK_DAYS)).isoformat()
            c.execute("""SELECT COUNT(*) FROM interactions WHERE domain=? AND action='bad' AND created_at>=?""",(dom,since))
            cnt=c.fetchone()[0] or 0
            return cnt>=1
    except Exception:
        return False

# =================== Persistence helpers ===================
def insert_query_record(text: str, intent: Dict[str,bool], region: str) -> str:
    qid=str(uuid.uuid4())
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("INSERT INTO queries (id,text,intent_json,region,created_at) VALUES (?,?,?,?,?)",(qid,text,json.dumps(intent,ensure_ascii=False),region,datetime.utcnow().isoformat()))
            conn.commit()
    except Exception as e: logger.error(f"insert query: {e}")
    return qid

def save_result_records(row: Dict[str,Any], intent: Dict[str,bool], region: str, query_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("""INSERT OR IGNORE INTO results
                (id,name,website,description,specialization,country,source,status,suitability,score)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (row["id"],row.get("name","N/A"),row["website"],row.get("description","N/A"),"",row.get("country","N/A"),row.get("source","Web"),"Active","Подходит",row.get("score",0.0)))
            c.execute("""INSERT OR IGNORE INTO results_history
                (id,query_id,url,domain,source,score,intent_json,region,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()),query_id,row["website"],domain_of(row["website"]),row.get("source","Web"),row.get("score",0.0),json.dumps(intent,ensure_ascii=False),region,datetime.utcnow().isoformat()))
            conn.commit()
    except Exception as e: logger.error(f"persist row: {e}")

def export_files():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor(); c.execute("SELECT * FROM results ORDER BY score DESC"); rows=c.fetchall()
        if not rows: return
        with open("search_results.csv","w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["ID","Name","Website","Description","Specialization","Country","Source","Status","Suitability","Score"]); w.writerows(rows)
        with open("search_results.txt","w",encoding="utf-8") as f:
            for r in rows:
                f.write(f"Название: {r[1][:120] or 'N/A'}\nВебсайт: {r[2] or 'N/A'}\nОписание: {short(r[3] or '', 300)}\nОценка: {r[9]:.2f}\n" + "-"*60 + "\n")
        logger.info("CSV/TXT exported")
    except Exception as e:
        logger.error(f"export files: {e}")

# =================== Backfill: guarantee >= MIN_RESULTS ===================
def backfill_guarantee(results: List[Dict[str,Any]], user_query:str, region:str)->List[Dict[str,Any]]:
    out=list(results)
    seen={domain_of(r["website"]) for r in out}
    # 1) always blend 2–5 seeds at top
    need_seed=min(5, max(0, MAX_RESULTS-len(out)))
    seeds=harvest_affcatalog(limit=max(need_seed,8))
    for s in seeds:
        d=domain_of(s["website"])
        if d in seen: continue
        out.append(s); seen.add(d)
        if len(out)>=MAX_RESULTS: break
    # 2) if still < MIN_RESULTS — extended seeds
    if len(out)<MIN_RESULTS:
        extra=harvest_affcatalog(limit=25)
        for s in extra:
            d=domain_of(s["website"])
            if d in seen: continue
            out.append(s); seen.add(d)
            if len(out)>=MIN_RESULTS: break
    # 3) if still short — pull from history (no network)
    if len(out)<MIN_RESULTS:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c=conn.cursor()
                since=(datetime.utcnow()-timedelta(days=90)).isoformat()
                c.execute("""SELECT url, MAX(score) FROM results_history WHERE created_at>=?
                             GROUP BY url ORDER BY MAX(score) DESC LIMIT 200""",(since,))
                for url,sc in c.fetchall() or []:
                    d=domain_of(url)
                    if d in seen: continue
                    out.append({
                        "id":str(uuid.uuid4()),"name":d or "Candidate","website":url,
                        "description":"History backfill","country":"N/A","source":"History","score":float(sc or 0.4)
                    })
                    seen.add(d)
                    if len(out)>=MIN_RESULTS: break
        except Exception as e:
            logger.warning(f"history backfill error: {e}")
    # sort & cap
    out.sort(key=lambda x:x.get("score",0.0), reverse=True)
    return out[:MAX_RESULTS]

# =================== API ===================
@app.route("/search", methods=["POST","OPTIONS"])
@app.route("/api/search", methods=["POST","OPTIONS"])
def search():
    if request.method=="OPTIONS": return ("",204)
    try:
        data=request.json or {}
        user_query=(data.get("query") or "").strip()
        region_in=(data.get("region") or "wt-wt").strip()
        engine=(data.get("engine") or os.getenv("SEARCH_ENGINE","ddg")).lower()
        per_query=int(data.get("per_query", 18))
        if not user_query: return jsonify({"error":"Query is required"}), 400

        # wipe current results table to keep latest view
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor(); c.execute("DELETE FROM results"); conn.commit()

        # region normalization and intent
        region=REGION_ALIAS.get(region_in, region_in)
        if region not in REGION_MAP: region="wt-wt"
        intent=detect_intent(user_query)
        query_id=insert_query_record(user_query, intent, region)
        jitter=int(uuid.UUID(query_id)) & 0xffffffff

        t0=time.time(); deadline=t0+PIPELINE_DEADLINE_SEC

        # --- Build queries & collect URLs
        queries=build_queries(user_query, region)
        urls=collect_urls(queries, region, engine, per_query)
        logger.info(f"URL pool: {len(urls)}")

        # --- Scrape (fast strict)
        strict_cap=250
        web_results = scrape_urls(urls[:strict_cap], region, jitter, cap=strict_cap)

        # --- If not enough and time remains, light relax scrape over next slice
        if len(web_results)<MIN_RESULTS and time.time()<deadline-25:
            relax_pool=urls[strict_cap: strict_cap+350]
            web_results += scrape_urls(relax_pool, region, jitter, cap=200)
            # dedupe after extend
            bydom={};
            for r in web_results: bydom.setdefault(domain_of(r["website"]), []).append(r)
            web_results = [sorted(v, key=lambda x:x["score"], reverse=True)[0] for v in bydom.values()]
            web_results.sort(key=lambda x:x["score"], reverse=True)

        # --- Guarantee ≥ MIN_RESULTS with seeds/history
        if len(web_results)<MIN_RESULTS:
            web_results = backfill_guarantee(web_results, user_query, region)

        # --- Final slice and persistence
        web_results=web_results[:MAX_RESULTS]
        for row in web_results: save_result_records(row, intent, region, query_id)

        export_files()
        took=time.time()-t0
        logger.info(f"Final results: {len(web_results)}")

        return jsonify({
            "results": web_results,
            "region": region,
            "engine": engine,
            "query_id": query_id,
            "took_sec": round(took,2)
        })
    except Exception as e:
        logger.error(f"/search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST","OPTIONS"])
@app.route("/api/feedback", methods=["POST","OPTIONS"])
def feedback():
    if request.method=="OPTIONS": return ("",204)
    try:
        data=request.json or {}
        query_id=(data.get("query_id") or "").strip()
        url=(data.get("url") or "").strip()
        raw=(data.get("action") or "").strip().lower()
        user_id=(data.get("user_id") or "").strip()
        action="bad" if raw=="flag" else raw
        if not (query_id and url and action in {"click","good","bad"}):
            return jsonify({"error":"query_id, url, action(click|good|bad|flag) required"}), 400
        weight=1.0 if action=="click" else 2.0
        dom=domain_of(url)
        with sqlite3.connect(DB_PATH) as conn:
            c=conn.cursor()
            c.execute("""INSERT INTO interactions (id,query_id,url,domain,action,weight,user_id,created_at)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (str(uuid.uuid4()),query_id,url,dom,action,weight,user_id,datetime.utcnow().isoformat()))
            conn.commit()
        if action=="bad":
            _block_domain(dom)
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"/feedback error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/download/<filetype>", methods=["GET"])
def download_file(filetype):
    if filetype not in ["csv","txt"]:
        return jsonify({"error":"Invalid file type, use 'csv' or 'txt'"}), 400
    fname=f"search_results.{filetype}"
    try:
        mimetype="text/plain" if filetype=="txt" else "text/csv"
        return send_file(fname, as_attachment=True, mimetype=mimetype)
    except FileNotFoundError:
        return jsonify({"error": f"{fname} not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error downloading {fname}: {str(e)}"}), 500

# static (keep compatibility)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    full_path=os.path.join(FRONTEND_DIST, path)
    if path and os.path.exists(full_path) and os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIST, path)
    index_path=os.path.join(FRONTEND_DIST,"index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIST,"index.html")
    return "frontend_dist is missing. Please upload your built frontend.", 404

if __name__=="__main__":
    logger.info(f"Starting Flask on http://{HOST}:{PORT} (debug={FLASK_DEBUG})")
    app.run(host=HOST, port=PORT, debug=FLASK_DEBUG, use_reloader=False)
