import os
import re
import json
import time
import uuid
import csv
import random
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ddgs import DDGS

# ========= Optional libs already present in your env
try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None
try:
    import chardet
except Exception:
    chardet = None

# ======================= Config & Env =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend_dist")
DB_PATH = os.getenv("DB_PATH", "search_results.db")

SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "").strip() or None

MIN_RESULTS = int(os.getenv("MIN_RESULTS", "25"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
PER_HOST_LIMIT = int(os.getenv("PER_HOST_LIMIT", "2"))
STRONG_BLOCK_ON_BAD = (os.getenv("STRONG_BLOCK_ON_BAD", "true").lower() == "true")

REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1200"))
REQUEST_PAUSE_MIN = float(os.getenv("REQUEST_PAUSE_MIN", "0.2"))
REQUEST_PAUSE_MAX = float(os.getenv("REQUEST_PAUSE_MAX", "0.5"))

DEFAULT_UA = os.getenv(
    "SCRAPER_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

# ======================= Logging =======================
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
logger = logging.getLogger("host")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("scraper.log", encoding="utf-8") if LOG_TO_FILE else logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ======================= Flask =======================
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="/")
ALLOWED_ORIGINS = {
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://search-assistance-production.up.railway.app",
}
CORS(
    app,
    origins=list(ALLOWED_ORIGINS),
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
)

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

# ======================= Helpers =======================
CODE_FENCE_RE = re.compile(r"^```(?:json|js|python|txt)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

def clean_text(s: str) -> str:
    if not s: return ""
    t = s.strip()
    t = CODE_FENCE_RE.sub("", t)
    t = t.strip(" \n\t\r\"'`")
    while t and (t[0] in "[{" and t[-1] in "]}"):
        t = t[1:-1].strip()
    return t

def domain_of(url: str) -> str:
    try:
        d = (urlparse(url).netloc or "").lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def is_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))

def short(s: str, n=220) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

def guess_lang(text: str) -> str:
    """simple fallback if langdetect missing/throws"""
    t = (text or "").strip()
    if not t:
        return "en"
    # fast heuristic
    if re.search(r"[іїєґ]", t.lower()):
        return "uk"
    if re.search(r"[а-яё]", t.lower()):
        return "ru"
    # try langdetect
    if langdetect_detect:
        try:
            return langdetect_detect(t)
        except Exception:
            pass
    return "en"

def detect_encoding(content: bytes) -> str:
    if not content:
        return "utf-8"
    if chardet:
        try:
            enc = chardet.detect(content).get("encoding")
            if enc:
                return enc
        except Exception:
            pass
    return "utf-8"

# ======================= DB =======================
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    website TEXT,
                    description TEXT,
                    specialization TEXT,
                    country TEXT,
                    source TEXT,
                    status TEXT,
                    suitability TEXT,
                    score REAL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    intent_json TEXT,
                    region TEXT,
                    created_at TEXT
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS results_history (
                    id TEXT PRIMARY KEY,
                    query_id TEXT,
                    url TEXT,
                    domain TEXT,
                    source TEXT,
                    score REAL,
                    intent_json TEXT,
                    region TEXT,
                    created_at TEXT
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    query_id TEXT,
                    url TEXT,
                    domain TEXT,
                    action TEXT,
                    weight REAL,
                    created_at TEXT
                )"""
            )
            conn.commit()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

init_db()

# ======================= Regions =======================
REGION_MAP = {
    "wt-wt": {"hl": "en", "gl": "us", "google_domain": "google.com", "tld": "com"},
    "us-en": {"hl": "en", "gl": "us", "google_domain": "google.com", "tld": "us"},
    "uk-en": {"hl": "en", "gl": "gb", "google_domain": "google.co.uk", "tld": "uk"},
    "de-de": {"hl": "de", "gl": "de", "google_domain": "google.de", "tld": "de"},
    "fr-fr": {"hl": "fr", "gl": "fr", "google_domain": "google.fr", "tld": "fr"},
    "ru-ru": {"hl": "ru", "gl": "ru", "google_domain": "google.ru", "tld": "ru"},
    "ua-ua": {"hl": "uk", "gl": "ua", "google_domain": "google.com.ua", "tld": "ua"},
    "kz-ru": {"hl": "ru", "gl": "kz", "google_domain": "google.kz", "tld": "kz"},
    "by-ru": {"hl": "ru", "gl": "by", "google_domain": "google.by", "tld": "by"},
    "tr-tr": {"hl": "tr", "gl": "tr", "google_domain": "google.com.tr", "tld": "tr"},
    "ae-en": {"hl": "en", "gl": "ae", "google_domain": "google.ae", "tld": "ae"},
    "in-en": {"hl": "en", "gl": "in", "google_domain": "google.co.in", "tld": "in"},
    "sg-en": {"hl": "en", "gl": "sg", "google_domain": "google.com.sg", "tld": "sg"},
    "es-es": {"hl": "es", "gl": "es", "google_domain": "google.es", "tld": "es"},
    "it-it": {"hl": "it", "gl": "it", "google_domain": "google.it", "tld": "it"},
    "nl-nl": {"hl": "nl", "gl": "nl", "google_domain": "google.nl", "tld": "nl"},
    "se-sv": {"hl": "sv", "gl": "se", "google_domain": "google.se", "tld": "se"},
    "no-no": {"hl": "no", "gl": "no", "google_domain": "google.no", "tld": "no"},
    "fi-fi": {"hl": "fi", "gl": "fi", "google_domain": "google.fi", "tld": "fi"},
    "cz-cs": {"hl": "cs", "gl": "cz", "google_domain": "google.cz", "tld": "cz"},
    "sk-sk": {"hl": "sk", "gl": "sk", "google_domain": "google.sk", "tld": "sk"},
    "ro-ro": {"hl": "ro", "gl": "ro", "google_domain": "google.ro", "tld": "ro"},
    "hu-hu": {"hl": "hu", "gl": "hu", "google_domain": "google.hu", "tld": "hu"},
    "ch-de": {"hl": "de", "gl": "ch", "google_domain": "google.ch", "tld": "ch"},
    "br-pt": {"hl": "pt", "gl": "br", "google_domain": "google.com.br", "tld": "br"},
    "mx-es": {"hl": "es", "gl": "mx", "google_domain": "google.com.mx", "tld": "mx"},
    "au-en": {"hl": "en", "gl": "au", "google_domain": "google.com.au", "tld": "au"},
    "nz-en": {"hl": "en", "gl": "nz", "google_domain": "google.co.nz", "tld": "nz"},
}

# Языковые ограничения по региону
REGION_LANG_ALLOW = {
    "ua-ua": ("uk", "ru", "en"),
    "by-ru": ("ru", "en"),
    "kz-ru": ("ru", "en"),
    "ru-ru": ("ru", "en"),
    "wt-wt": ("en", "ru", "uk"),
}

# Токены региона для матчей внутри текста (укрепляют региональную привязку)
REGION_HINTS = {
    "ua-ua": {"tokens": ["Україна","Ukraine","Київ","Lviv","Одеса","Дніпро","Харків","UA"], "org": ["ТОВ","ФОП","LLC"]},
    "by-ru": {"tokens": ["Беларусь","Belarus","Минск","BY"], "org": ["ООО","ЧУП","ЗАО"]},
    "kz-ru": {"tokens": ["Казахстан","Kazakhstan","Алматы","Алма-Ата","Астана","Нур-Султан","KZ"], "org": ["ТОО","LLP","ИП","ЖШС"]},
    "ru-ru": {"tokens": ["Россия","Russian Federation","Москва","Санкт-Петербург","RF"], "org": ["ООО","АО","ИП"]},
}

# ======================= Intent / Filters =======================
INTENT_AFFILIATE = {
    "affiliate","аффилиат","аффилиэйт","партнерка","партнёрка","партнерская программа","партнёрская программа",
    "реферальная","referral","cpa","игемблинг","игейминг","igaming","casino affiliate","affiliate network",
    "партнеры казино","партнёры казино","аффилейт","афилейт","opm","outsourced affiliate program management"
}
INTENT_LEARN = {
    "что такое","what is","определение","definition","гайд","guide","обзор","overview","курс","course","как работает","how to",
    "що таке","лекция","лекція"
}
CASINO_TOKENS = {"казино","casino","igaming","гемблинг","игемблинг","игейминг","беттинг","sportsbook","bookmaker","ставки"}

BAD_DOMAINS_HARD = {
    "google.com","maps.google.com","baidu.com","zhihu.com","commentcamarche.net",
    "xnxx.com","pornhub.com","hometubeporn.com","porn7.xxx","fuckvideos.xxx",
    "addons.mozilla.org","microsoft.com","support.microsoft.com","edge.microsoft.com",
    "minecraft.net","curseforge.com","planetminecraft.com","softonic.com","apkpure.com","apkcombo.com","uptodown.com",
}

SOFT_BAD_DOMAINS = {
    "wikipedia.org","en.wikipedia.org","ru.wikipedia.org","uk.wikipedia.org",
    "work.ua","rabota.ua","hh.ru","jobs.ua","djinni.co","indeed.com","glassdoor.com",
    "tut.by","onliner.by","tumba.kz",
    "facebook.com","instagram.com","x.com","twitter.com","vk.com","ok.ru",
}

BLOG_HINTS_URL = ("/blog","/news","/article","/articles","/insights","/guide","/guides","/academy","/press","/glossary")
JOBS_HINTS_URL = ("/jobs","/job","/careers","/vacancies","/vacancy","/rabota","/vakans")
EVENT_HINTS = ("conference","expo","summit","event","exhibition","agenda","speakers")
DIRECTORY_HINTS = ("directory","catalog","list","listing","rank","rating","top-","compare","comparison")
CASINO_TLDS = (".casino",".bet",".betting",".poker",".slots",".bingo")

# ======================= Utility filters =======================
def looks_like_job(url: str, text: str) -> bool:
    u = url.lower()
    if any(tok in u for tok in JOBS_HINTS_URL): return True
    d = domain_of(u)
    if d in SOFT_BAD_DOMAINS:
        if "job" in u or "career" in u or "vacanc" in u or "ваканси" in u or "работа" in u:
            return True
    t = text.lower()
    return any(x in t for x in ("vacanc","ваканси","работа","hiring"))

def looks_like_event(url: str, text: str) -> bool:
    u = url.lower()
    if any(e in u for e in EVENT_HINTS): return True
    t = text.lower()
    return any(e in t for e in EVENT_HINTS)

def looks_like_directory(url: str, text: str) -> bool:
    u = url.lower()
    if any(e in u for e in DIRECTORY_HINTS): return True
    t = text.lower()
    return any(e in t for e in DIRECTORY_HINTS)

def looks_like_blog(url: str, text: str) -> bool:
    u = url.lower()
    if any(e in u for e in BLOG_HINTS_URL): return True
    t = text.lower()
    return any(e in t for e in ("what is","що таке","что такое","guide","обзор","overview","гайд","курс","лекция","лекція"))

def is_hard_bad_domain(dom: str) -> bool:
    if not dom: return False
    d = dom.lower().lstrip(".")
    if d.startswith("www."): d = d[4:]
    return d in BAD_DOMAINS_HARD

def detect_intent(q: str) -> Dict[str, bool]:
    t = (q or "").lower()
    affiliate = any(k in t for k in INTENT_AFFILIATE)
    learn = any(k in t for k in INTENT_LEARN)
    casino = any(k in t for k in CASINO_TOKENS)
    return {"affiliate": affiliate, "learn": learn, "casino": casino, "business": not learn}

# ======================= Request counter =======================
def _load_request_count():
    try:
        with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            last_reset = data.get("last_reset", "")
            if last_reset:
                last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d")
                if last_reset_date.date() < datetime.now().date():
                    return {"count": 0, "last_reset": datetime.now().strftime("%Y-%m-%d")}
            return data
    except Exception:
        return {"count": 0, "last_reset": datetime.now().strftime("%Y-%m-%d")}

def _save_request_count(count):
    try:
        with open(REQUEST_COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump({"count": count, "last_reset": datetime.now().strftime("%Y-%m-%d")}, f)
    except Exception as e:
        logger.error(f"Failed to save request count: {e}")

# ======================= Query building =======================
def with_intent_filters(q: str, intent: Dict[str,bool]) -> str:
    parts = [q]
    parts += ["-site:wikipedia.org", "-site:facebook.com", "-site:instagram.com", "-site:x.com", "-site:twitter.com", "-site:vk.com"]
    parts += ["-jobs", "-job", "-vacancy", "-вакансия", "-вакансии", "-rabota", "-careers", "-hh.ru"]
    parts += ["-blog", "-news", "-guide", "-overview", "-press", "-что", "-що", "-definition", "-glossary"]
    parts += ["-download", "-apk", "-edge", "-minecraft", "-browser"]
    return " ".join(parts)

def build_core_queries(user_prompt: str) -> List[str]:
    base = user_prompt.strip()
    core = [
        base,
        "casino affiliate marketing agency",
        "igaming affiliate agency",
        "casino affiliate program management",
        "igaming affiliate management company",
        "outsourced affiliate program management igaming",
        "casino affiliate network for operators",
        "igaming affiliate platform for operators",
        "casino affiliate services for operators",
        "игейминг аффилиат агентство",
        "агентство по аффилиат маркетингу казино",
        "управление партнерской программой казино",
    ]
    seen = set(); out = []
    for q in core:
        q = clean_text(q)
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

def apply_geo_bias(queries: List[str], region: str) -> List[str]:
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = reg["tld"]
    country_hint = {
        "ua-ua": "Україна OR Ukraine",
        "by-ru": "Беларусь OR Belarus",
        "kz-ru": "Казахстан OR Kazakhstan",
        "ru-ru": "Россия OR Russia",
        "wt-wt": "",
    }.get(region, "")
    out = []
    for i, q in enumerate(queries):
        if i == 0 and country_hint:
            out.append(f"{q} ({country_hint})")
        elif i in (1,2) and tld:
            hint = "site:.{t}".format(t=tld) if i == 1 else "site:*.{t}".format(t=tld)
            out.append(f"{q} {hint}")
        else:
            out.append(q)
    return out

# ======================= Learning (feedback) =======================
def domain_is_blocked(domain: str) -> bool:
    if not STRONG_BLOCK_ON_BAD: return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=60)).isoformat()
            c.execute("""SELECT COUNT(*) FROM interactions
                         WHERE domain=? AND action='bad' AND created_at>=?""", (domain, since))
            cnt = c.fetchone()[0] or 0
            return cnt >= 3
    except Exception:
        return False

def domain_feedback_adjustment(domain: str) -> float:
    """+boost за good/click, -penalty за bad; каппируем."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT
                  SUM(CASE WHEN action='good' THEN weight ELSE 0 END),
                  SUM(CASE WHEN action='click' THEN weight ELSE 0 END),
                  SUM(CASE WHEN action='bad' THEN weight ELSE 0 END)
                FROM interactions WHERE domain=?
            """, (domain,))
            row = c.fetchone() or (0,0,0)
            good, click, bad = (row[0] or 0.0), (row[1] or 0.0), (row[2] or 0.0)
            boost = 0.08*min(good,10) + 0.03*min(click,20) - 0.10*min(bad,10)
            return max(-0.5, min(0.4, boost))
    except Exception:
        return 0.0

# ======================= Search engines =======================
def duckduckgo_search(query, max_results=25, region="wt-wt"):
    data = _load_request_count()
    if data["count"] >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        return []
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, safesearch="moderate", timelimit="y", max_results=max_results)
            for r in results:
                href = r.get("href")
                if href: urls.append(href)
        _save_request_count(data["count"] + 1)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"DDG failed for '{query}': {e}")
        return []

def serpapi_search(query, max_results=25, region="wt-wt"):
    if not SERPAPI_API_KEY:
        return []
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": SERPAPI_API_KEY,
        "hl": reg["hl"],
        "gl": reg["gl"],
        "google_domain": reg["google_domain"],
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        urls = [it.get("link") for it in data.get("organic_results", []) if it.get("link")]
        for block in ("inline_results","local_results","top_stories"):
            for it in data.get(block, []) or []:
                link = it.get("link") or it.get("source")
                if link: urls.append(link)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.warning(f"SerpAPI error: {e}")
        return []

# ======================= Scraping & classification =======================
def extract_text_for_classification(html: bytes) -> Tuple[str, str]:
    """Возвращаем title и компактный видимый текст (meta + h1–h3 + первые p)."""
    if not html:
        return "", ""
    enc = detect_encoding(html)
    try:
        soup = BeautifulSoup(html.decode(enc, errors="ignore"), "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for bad in soup(["script", "style", "noscript", "svg", "picture", "source"]):
        bad.decompose()

    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    meta_desc = ""
    md = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
    if md and md.get("content"):
        meta_desc = md["content"].strip()

    # заголовки и первые параграфы
    heads = " ".join(h.get_text(" ", strip=True) for h in soup.select("h1, h2, h3")[:6])
    paras = " ".join(p.get_text(" ", strip=True) for p in soup.select("p")[:8])

    text = " ".join([meta_desc, heads, paras])
    text = re.sub(r"\s+", " ", text).strip()
    return short(title, 180), text

def fetch_one(url: str) -> Optional[Dict[str,str]]:
    try:
        if not url or is_hard_bad_domain(domain_of(url)): return None
        headers = {"User-Agent": DEFAULT_UA, "Accept-Language": "en,ru,uk;q=0.9"}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400: return None
        ct = (resp.headers.get("Content-Type","") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None
        title, text = extract_text_for_classification(resp.content)
        if not text:
            return None
        return {"url": url, "name": title or "N/A", "text": text}
    except Exception:
        return None

def is_affiliate_igaming_company(url: str, text: str) -> bool:
    u = url.lower(); t = text.lower()

    # отсекаем типовой мусор
    if looks_like_job(u, t): return False
    if looks_like_event(u, t): return False
    if looks_like_directory(u, t): return False
    if looks_like_blog(u, t): return False

    # must-have токены
    has_aff = any(k in t for k in (
        "affiliate","affiliat","партнерск","партнёрск","рефераль","partner program","opm","outsourced"
    ))
    has_ig = any(k in t for k in (
        "igaming","i-gaming","casino","казино","гемблинг","игейминг","игемблинг","беттинг","sportsbook","bookmaker"
    ))
    if not (has_aff and has_ig):
        return False

    # признаки «компании/услуг/платформы»
    company_like = any(k in u for k in ("/services","/solutions","/platform","/program","/partners","/about","/contact","/affiliat")) \
        or any(k in t for k in ("our services","services","solutions","platform","for operators","для операторов","о нас","контакты","кейсы","clients","features","integrations","pricing","demo"))
    return bool(company_like)

def region_affinity(url: str, text: str, region: str) -> float:
    """Сильный приоритет региону: boost локальному tld; для .com — требуется упоминание страны/города/орг-форм."""
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = "." + reg["tld"] if reg.get("tld") else ""
    u = url.lower(); t = text.lower()
    score = 0.0

    if tld and u.endswith(tld):
        score += 0.6  # сильный приоритет локального tld

    hints = REGION_HINTS.get(region, {})
    tokens = [x.lower() for x in hints.get("tokens", [])]
    orgs = [x.lower() for x in hints.get("org", [])]
    if tokens and any(tok.lower() in t for tok in tokens):
        score += 0.25
    if orgs and any(o.lower() in t for o in orgs):
        score += 0.15

    # если домен НЕ локальный (.com и прочее), но нет региональных токенов — штраф
    if tld and (not u.endswith(tld)) and not (tokens and any(tok in t for tok in tokens)):
        score -= 0.35

    return score

def language_ok(text: str, region: str) -> bool:
    allowed = REGION_LANG_ALLOW.get(region, REGION_LANG_ALLOW["wt-wt"])
    lang = guess_lang(text)
    return lang in allowed

# ======================= Persistence =======================
def insert_query_record(text: str, intent: Dict[str,bool], region: str) -> str:
    qid = str(uuid.uuid4())
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO queries (id, text, intent_json, region, created_at) VALUES (?, ?, ?, ?, ?)",
                      (qid, text, json.dumps(intent, ensure_ascii=False), region, datetime.utcnow().isoformat()))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to insert query: {e}")
    return qid

def save_result_records(row: Dict[str,Any], intent: Dict[str,bool], region: str, query_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                """INSERT OR IGNORE INTO results
                   (id, name, website, description, specialization, country, source, status, suitability, score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"], row.get("name","N/A"), row["website"], row.get("description","N/A"),
                    "", row.get("country","N/A"), row.get("source","Web"),
                    "Active", "Подходит", row.get("score",0.0),
                ),
            )
            c.execute(
                """INSERT OR IGNORE INTO results_history
                   (id, query_id, url, domain, source, score, intent_json, region, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()), query_id, row["website"], domain_of(row["website"]), row.get("source","Web"),
                    row.get("score",0.0), json.dumps(intent, ensure_ascii=False), region, datetime.utcnow().isoformat()
                )
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving records: {e}")

def save_to_csv_txt():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY score DESC")
            rows = cursor.fetchall()
            if not rows: return
            with open("search_results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID","Name","Website","Description","Specialization","Country","Source","Status","Suitability","Score"])
                writer.writerows(rows)
            with open("search_results.txt", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(f"Название: {row[1][:100] or 'N/A'}\n")
                    f.write(f"Вебсайт: {row[2] or 'N/A'}\n")
                    f.write(f"Описание: {short(row[3] or '', 300)}\n")
                    f.write(f"Оценка: {row[9]:.2f}\n")
                    f.write("-"*60 + "\n")
        logger.info("CSV/TXT exported")
    except Exception as e:
        logger.error(f"Export failed: {e}")

# ======================= Pipeline =======================
def generate_search_queries(user_prompt: str, region="wt-wt") -> Tuple[List[str], List[str], str, Dict[str,bool]]:
    if region not in REGION_MAP:
        logger.warning(f"Invalid region {region}, defaulting to wt-wt")
        region = "wt-wt"
    intent = detect_intent(user_prompt)
    base_queries = build_core_queries(user_prompt)
    queries = apply_geo_bias(base_queries, region)
    queries = [with_intent_filters(q, intent) for q in queries]
    return queries, base_queries, region, intent

def collect_urls(web_queries: List[str], region: str, engine: str, per_query_k: int) -> List[str]:
    all_urls: List[str] = []
    for q in web_queries:
        all_urls.extend(duckduckgo_search(q, max_results=per_query_k, region=region))
        if engine in ("serpapi","both") and SERPAPI_API_KEY:
            all_urls.extend(serpapi_search(q, max_results=per_query_k, region=region))
    deduped = []
    seen = set()
    for u in all_urls:
        if not u or u in seen: continue
        d = domain_of(u)
        if is_hard_bad_domain(d): continue
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(d): continue
        deduped.append(u); seen.add(u)
    return deduped

def scrape_parallel(urls: List[str], region: str) -> List[Dict[str,Any]]:
    results: List[Dict[str,Any]] = []

    # per-host limit
    per_host_counter: Dict[str,int] = {}
    filtered_urls = []
    for u in urls:
        dom = domain_of(u)
        if per_host_counter.get(dom, 0) >= PER_HOST_LIMIT:
            continue
        per_host_counter[dom] = per_host_counter.get(dom, 0) + 1
        filtered_urls.append(u)

    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = "." + reg["tld"] if reg.get("tld") else ""

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut2url = {ex.submit(fetch_one, u): u for u in filtered_urls}
        for fut in as_completed(fut2url):
            item = fut.result()
            if not item: continue
            url = item["url"]; name = item["name"]; text = item["text"]
            d = domain_of(url)

            # жёсткие отсеки типов
            if looks_like_job(url, text): continue
            if looks_like_event(url, text): continue
            if looks_like_directory(url, text): continue
            if looks_like_blog(url, text): continue

            # тематический классификатор
            if not is_affiliate_igaming_company(url, text):
                continue

            # языковой фильтр
            if not language_ok(text, region):
                continue

            # региональная «аффинность»
            score = 0.9
            score += region_affinity(url, text, region)

            # фидбэк домена
            score += domain_feedback_adjustment(d)

            # штраф операторских TLD (не агентства)
            if any(url.lower().endswith(ct) for ct in CASINO_TLDS):
                score -= 0.2

            score = max(0.0, min(score, 1.6))

            results.append({
                "id": str(uuid.uuid4()),
                "name": name or "N/A",
                "website": url,
                "description": short(text, 400),
                "country": "N/A",
                "source": "Web",
                "score": score
            })

    # приоритет: сначала локальные TLD, потом остальное по убыванию score
    if tld:
        local = [r for r in results if r["website"].lower().endswith(tld)]
        global_ = [r for r in results if not r["website"].lower().endswith(tld)]
        local.sort(key=lambda x: x["score"], reverse=True)
        global_.sort(key=lambda x: x["score"], reverse=True)
        results = local + global_
    else:
        results.sort(key=lambda x: x["score"], reverse=True)

    # обрезаем до MAX_RESULTS
    results = results[:MAX_RESULTS]
    return results

def save_to_db_and_files(web_results: List[Dict[str,Any]], intent: Dict[str,bool], region: str, query_id: str):
    for r in web_results:
        save_result_records(r, intent, region, query_id)
    if web_results:
        save_to_csv_txt()

# ======================= API =======================
@app.route("/search", methods=["POST", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.json or {}
        user_query = data.get("query", "").strip()
        region = data.get("region", "wt-wt")
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE","both")).lower()  # ddg | serpapi | both
        per_query_k = int(data.get("per_query", 25))  # берём побольше с каждого запроса

        if not user_query:
            return jsonify({"error":"Query is required"}), 400

        logger.info(f"API request: query='{user_query}', region={region}, engine={engine}")

        # очистка текущей таблицы результатов
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM results")
            conn.commit()

        # генерим подзапросы
        web_queries, phrases, region, intent = generate_search_queries(user_query, region)

        # пишем запрос
        query_id = insert_query_record(user_query, intent, region)

        # собираем URL’ы
        urls = collect_urls(web_queries, region, engine, per_query_k=per_query_k)
        logger.info(f"Pre-filtered URLs (unique & not blocked): {len(urls)}")

        # 1-й проход
        t0 = time.time()
        web_results = scrape_parallel(urls[:400], region)

        # если мало — 2-й проход по хвосту (без ослабления фильтров)
        if len(web_results) < MIN_RESULTS and len(urls) > 400:
            more = scrape_parallel(urls[400:1000], region)
            known = set(r["website"] for r in web_results)
            for r in more:
                if r["website"] not in known:
                    web_results.append(r); known.add(r["website"])
            web_results.sort(key=lambda x: x["score"], reverse=True)
            web_results = web_results[:MAX_RESULTS]

        dt = time.time() - t0
        logger.info(f"Scraped {len(web_results)} results in {dt:.2f}s")

        # запись в БД + экспорт
        save_to_db_and_files(web_results, intent, region, query_id)

        return jsonify({
            "results": web_results,
            "region": region,
            "engine": engine,
            "query_id": query_id,
            "took_sec": round(dt,2)
        })
    except Exception as e:
        logger.error(f"/search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.json or {}
        query_id = (data.get("query_id") or "").strip()
        url = (data.get("url") or "").strip()
        raw_action = (data.get("action") or "").strip().lower()
        action = "bad" if raw_action == "flag" else raw_action
        if not (query_id and url and action in {"click","good","bad"}):
            return jsonify({"error":"query_id, url, action(click|good|bad|flag) required"}), 400
        weight = 1.0 if action=="click" else (2.0 if action in {"good","bad"} else 1.0)
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO interactions (id, query_id, url, domain, action, weight, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (str(uuid.uuid4()), query_id, url, domain_of(url), action, weight, datetime.utcnow().isoformat()))
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"/feedback error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/download/<filetype>", methods=["GET"])
def download_file(filetype):
    if filetype not in ["csv", "txt"]:
        return jsonify({"error": "Invalid file type, use 'csv' or 'txt"}), 400
    filename = f"search_results.{filetype}"
    try:
        mimetype = "text/plain" if filetype == "txt" else "text/csv"
        return send_file(filename, as_attachment=True, mimetype=mimetype)
    except FileNotFoundError:
        return jsonify({"error": f"{filename} not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Error downloading {filename}: {str(e)}"}), 500

# ========= SPA =========
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    full_path = os.path.join(FRONTEND_DIST, path)
    if path and os.path.exists(full_path) and os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIST, path)
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return "frontend_dist is missing. Please upload your built frontend.", 404

# ========= Entry =========
if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("RAILWAY_PORT", "8080")))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Flask on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
