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
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ddgs import DDGS

# ========= Optional libs =========
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
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "12"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "8"))
PER_HOST_LIMIT = int(os.getenv("PER_HOST_LIMIT", "3"))
STRONG_BLOCK_ON_BAD = (os.getenv("STRONG_BLOCK_ON_BAD", "true").lower() == "true")

REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1200"))
REQUEST_PAUSE_MIN = float(os.getenv("REQUEST_PAUSE_MIN", "0.10"))
REQUEST_PAUSE_MAX = float(os.getenv("REQUEST_PAUSE_MAX", "0.30"))

DEFAULT_UA = os.getenv(
    "SCRAPER_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

# SerpAPI резерв + cooldown при 429
SERPAPI_COOLDOWN_SEC = int(os.getenv("SERPAPI_COOLDOWN_SEC", "900"))  # 15 минут
_last_serpapi_429_at = 0.0

# Жёсткий дедлайн на скрапинг
SCRAPE_HARD_DEADLINE_SEC = int(os.getenv("SCRAPE_HARD_DEADLINE_SEC", "150"))

# Микрокраул
MAX_MICROCRAWL_PAGES = int(os.getenv("MAX_MICROCRAWL_PAGES", "1"))
ENABLE_MICROCRAWL = (os.getenv("ENABLE_MICROCRAWL", "true").lower() == "true")

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

PHONE_CODE_RE = {
    "kz-ru": re.compile(r"(\+?7[\s\-]?\(?7\d\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})"),
    "ua-ua": re.compile(r"(\+?380[\s\-]?\(?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})"),
    "ru-ru": re.compile(r"(\+?7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})"),
    "by-ru": re.compile(r"(\+?375[\s\-]?\(?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})"),
}
CURRENCY_HINTS = {
    "kz-ru": (("₸","KZT"), 0.25),
    "ua-ua": (("₴","UAH","грн"), 0.25),
    "ru-ru": (("₽","RUB","руб"), 0.2),
    "by-ru": (("Br","BYN","руб"), 0.2),
}
ACCEPT_LANG_POOL = ["en,ru,uk;q=0.9", "ru,en;q=0.9", "en;q=0.8"]

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

def root_domain(d: str) -> str:
    """Грубое нормализованное имя домена для анти-повторов."""
    d = (d or "").lower()
    if d.startswith("www."): d = d[4:]
    parts = d.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else d

def normalize_url(url: str) -> str:
    try:
        u = urlparse(url)
        query = [(k,v) for k,v in parse_qsl(u.query, keep_blank_values=False) if not k.lower().startswith("utm_")]
        u2 = u._replace(query=urlencode(query, doseq=True), fragment="")
        return urlunparse(u2)
    except Exception:
        return url

def short(s: str, n=220) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

def guess_lang(text: str) -> str:
    t = (text or "").strip()
    if not t: return "en"
    if re.search(r"[іїєґ]", t.lower()): return "uk"
    if re.search(r"[а-яё]", t.lower()): return "ru"
    if langdetect_detect:
        try:
            return langdetect_detect(t)
        except Exception:
            pass
    return "en"

def detect_encoding(content: bytes) -> str:
    if not content: return "utf-8"
    if chardet:
        try:
            enc = chardet.detect(content).get("encoding")
            if enc: return enc
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
            c.execute(
                """CREATE TABLE IF NOT EXISTS domain_blocks (
                    domain TEXT PRIMARY KEY,
                    blocked_until TEXT,
                    reason TEXT
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS seen_recent (
                    domain TEXT PRIMARY KEY,
                    last_seen TEXT
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

# Алиасы регионов (фикс ошибки kz-kk и т.п.)
REGION_ALIASES = {
    "kz-kk": "kz-ru",
    "ua-ru": "ua-ua",
    "ua-en": "ua-ua",
    "ru-en": "ru-ru",
    "by-by": "by-ru",
    "kz-en": "kz-ru",
}

REGION_LANG_ALLOW = {
    "ua-ua": ("uk", "ru", "en"),
    "by-ru": ("ru", "en"),
    "kz-ru": ("ru", "en"),
    "ru-ru": ("ru", "en"),
    "wt-wt": ("en", "ru", "uk"),
}

REGION_HINTS = {
    "ua-ua": {
        "tokens": ["Україна","Ukraine","Київ","Kyiv","Львів","Lviv","Одеса","Odesa","Дніпро","Dnipro","Харків","Kharkiv","UA"],
        "org": ["ТОВ","ФОП","LLC","ПП","ПрАТ","АТ"],
        "cities": ["Kyiv","Київ","Lviv","Львів","Odesa","Одеса","Kharkiv","Харків","Dnipro","Дніпро"]
    },
    "by-ru": {
        "tokens": ["Беларусь","Belarus","Минск","Minsk","BY","Гродно","Брест","Гомель","Витебск","Могилев"],
        "org": ["ООО","ЧУП","ЗАО","ОАО"],
        "cities": ["Минск","Minsk","Гродно","Брест","Гомель","Витебск","Могилев"]
    },
    "kz-ru": {
        "tokens": ["Казахстан","Kazakhstan","Алматы","Almaty","Алма-Ата","Астана","Astana","Нур-Султан","Шымкент","Karaganda","Караганда","KZ"],
        "org": ["ТОО","LLP","ИП","ЖШС"],
        "cities": ["Алматы","Almaty","Астана","Astana","Шымкент","Караганда"]
    },
    "ru-ru": {
        "tokens": ["Россия","Russian Federation","Москва","Санкт-Петербург","Новосибирск","Екатеринбург","RF","RU"],
        "org": ["ООО","АО","ИП","ПАО","ЗАО","ОАО"],
        "cities": ["Москва","Санкт-Петербург","Новосибирск","Екатеринбург","Казань","Нижний Новгород"]
    },
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
CASINO_TOKENS = {"казино","casino","igaming","гемблинг","игемблинг","игейминг","беттинг","sportsbook","bookmaker","slots","poker"}

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
JOBS_HINTS_URL = ("/jobs","/job","/careers","/vacancies","/vacancy","/rabota","/vakans","/career")
EVENT_HINTS = ("conference","expo","summit","event","exhibition","agenda","speakers")
DIRECTORY_HINTS = ("directory","catalog","list","listing","rank","rating","top-","compare","comparison")
CASINO_TLDS = (".casino",".bet",".betting",".poker",".slots",".bingo")

OPERATOR_PROGRAM_HINTS = (
    "join our affiliate program","affiliates program","affiliate portal","commission",
    "revshare","cpa","commission tiers","promotional materials","tracking platform",
    "program terms","payouts","banners","affiliates terms","sign up and start promoting"
)
NETWORK_FOR_PUBLISHERS_HINTS = (
    "for affiliates","for publishers","become an affiliate","sign up as an affiliate",
    "traffic sources","webmasters","start promoting","affiliate network"
)

COMPANY_SERVICE_HINTS = (
    "our services","services","solutions","platform","for operators","for brands","for advertisers",
    "clients","case study","case studies","features","integrations","pricing",
    "request a demo","get a demo","contact sales","schedule a call","request proposal",
    "program management","affiliate management","opm","outsourced program management"
)

# ======================= Utility filters =======================
def looks_like_job(url: str, text: str) -> bool:
    u = url.lower()
    if any(tok in u for tok in JOBS_HINTS_URL): return True
    d = domain_of(u)
    if d in SOFT_BAD_DOMAINS:
        if "job" in u or "career" in u or "vacanc" in u or "ваканси" in u or "работа" in u:
            return True
    t = text.lower()
    return any(x in t for x in ("vacanc","ваканси","работа","hiring","we are hiring","careers"))

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
    return any(e in t for e in ("what is","що таке","что такое","guide","обзор","overview","гайд","курс","лекция","лекція","blog","news"))

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
    return {"affiliate": affiliate or casino, "learn": learn, "casino": casino, "business": not learn}

# ======================= Request counter =======================
def _load_request_count():
    try:
        with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "engines" not in data:
                data = {"engines": {"ddg": {"count": data.get("count", 0)}, "serpapi": {"count": 0}},
                        "last_reset": data.get("last_reset", datetime.now().strftime("%Y-%m-%d"))}
            last_reset_date = datetime.strptime(data["last_reset"], "%Y-%m-%d")
            if last_reset_date.date() < datetime.now().date():
                return {"engines": {"ddg":{"count":0},"serpapi":{"count":0}}, "last_reset": datetime.now().strftime("%Y-%m-%d")}
            return data
    except Exception:
        return {"engines": {"ddg":{"count":0},"serpapi":{"count":0}}, "last_reset": datetime.now().strftime("%Y-%m-%d")}

def _save_request_count(data):
    try:
        with open(REQUEST_COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save request count: {e}")

def _bump_counter(engine: str):
    data = _load_request_count()
    if engine not in data["engines"]:
        data["engines"][engine] = {"count": 0}
    data["engines"][engine]["count"] = int(data["engines"][engine].get("count", 0)) + 1
    _save_request_count(data)

def _engine_allowed(engine: str) -> bool:
    data = _load_request_count()
    cnt = int(data["engines"].get(engine, {}).get("count", 0))
    return cnt < DAILY_REQUEST_LIMIT

# ======================= Query building =======================
def with_intent_filters(q: str, intent: Dict[str,bool]) -> str:
    parts = [q]
    parts += ["-site:wikipedia.org", "-site:facebook.com", "-site:instagram.com", "-site:x.com", "-site:twitter.com", "-site:vk.com", "-site:ok.ru"]
    parts += ["-jobs", "-job", "-vacancy", "-вакансия", "-вакансии", "-rabota", "-careers", "-hh.ru"]
    parts += ["-blog", "-news", "-guide", "-overview", "-press", "-что", "-що", "-definition", "-glossary", "-directory", "-listing", "-compare"]
    parts += ["-download", "-apk", "-edge", "-minecraft", "-forum", "-community"]
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
        "casino affiliate services for operators",
        "igaming affiliate platform for operators",
        "affiliate management for casino operators",
        "игейминг аффилиат агентство",
        "агентство по аффилиат маркетингу казино",
        "управление партнерской программой казино",
        "OPM igaming",
        "igaming acquisition agency for operators",
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
        "ua-ua": "(Україна OR Ukraine OR Київ OR Lviv OR Одеса OR Dnipro OR Kharkiv)",
        "by-ru": "(Беларусь OR Belarus OR Минск)",
        "kz-ru": "(Казахстан OR Kazakhstan OR Алматы OR Астана OR Шымкент)",
        "ru-ru": "(Россия OR Moscow OR Санкт-Петербург)",
        "wt-wt": "",
    }.get(region, "")
    out = []
    for i, q in enumerate(queries):
        if i == 0 and country_hint:
            out.append(f"{q} {country_hint}")
        elif i in (1,2) and tld:
            hint = "site:.{t}".format(t=tld) if i == 1 else "site:*.{t}".format(t=tld)
            out.append(f"{q} {hint}")
        else:
            out.append(q)
    return out

# ======================= Learning (feedback) =======================
BAD_BLOCK_DAYS = 60
RECENT_SEEN_DAYS = int(os.getenv("RECENT_SEEN_DAYS", "14"))

def _block_domain(domain: str, reason: str = "user_bad", days: int = BAD_BLOCK_DAYS):
    try:
        until = (datetime.utcnow() + timedelta(days=days)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO domain_blocks (domain, blocked_until, reason) VALUES (?, ?, ?)""",
                      (domain, until, reason))
            conn.commit()
    except Exception as e:
        logger.error(f"Block domain failed: {e}")

def _domain_explicitly_blocked(domain: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""SELECT blocked_until FROM domain_blocks WHERE domain=?""",(domain,))
            row = c.fetchone()
            if not row: return False
            until = row[0]
            return datetime.fromisoformat(until) > datetime.utcnow()
    except Exception:
        return False

def domain_is_blocked(domain: str) -> bool:
    if not STRONG_BLOCK_ON_BAD:
        return _domain_explicitly_blocked(domain)
    try:
        if _domain_explicitly_blocked(domain): return True
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=BAD_BLOCK_DAYS)).isoformat()
            c.execute("""SELECT COUNT(*) FROM interactions
                         WHERE domain=? AND action='bad' AND created_at>=?""", (domain, since))
            cnt = c.fetchone()[0] or 0
            if cnt >= 1:
                return True
            # анти-повторы: недавние seen
            since2 = (datetime.utcnow() - timedelta(days=RECENT_SEEN_DAYS)).isoformat()
            c.execute("""SELECT last_seen FROM seen_recent WHERE domain=?""",(domain,))
            row = c.fetchone()
            if row and row[0] and row[0] >= since2:
                return True
            return False
    except Exception:
        return False

def update_seen_recent(domains: List[str]):
    try:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            for d in domains:
                c.execute("""INSERT OR REPLACE INTO seen_recent (domain, last_seen) VALUES (?, ?)""",(d, now))
            conn.commit()
    except Exception as e:
        logger.error(f"update_seen_recent failed: {e}")

def domain_feedback_adjustment(domain: str) -> float:
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
            boost = 0.2*min(good,10) + 0.06*min(click,20) - 0.6*min(bad,10)
            return max(-0.6, min(0.4, boost))
    except Exception:
        return 0.0

# ======================= Search engines =======================
def duckduckgo_search(query, max_results=25, region="wt-wt"):
    if not _engine_allowed("ddg"):
        logger.error("Daily request limit reached for DDG")
        return []
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, safesearch="moderate", timelimit="y", max_results=max_results)
            for r in results:
                href = r.get("href")
                if href: urls.append(href)
        _bump_counter("ddg")
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [normalize_url(u) for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"DDG failed for '{query}': {e}")
        return []

def serpapi_search(query, max_results=25, region="wt-wt"):
    global _last_serpapi_429_at
    if not SERPAPI_API_KEY:
        return []
    if (time.time() - _last_serpapi_429_at) < SERPAPI_COOLDOWN_SEC:
        return []
    if not _engine_allowed("serpapi"):
        logger.warning("Daily request limit reached for SerpAPI")
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
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        _bump_counter("serpapi")
        urls = [normalize_url(u) for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            logger.warning("SerpAPI 429: enabling cooldown")
            _last_serpapi_429_at = time.time()
        else:
            logger.warning(f"SerpAPI error: {e}")
        return []
    except Exception as e:
        logger.warning(f"SerpAPI error: {e}")
        return []

# ======================= Scraping =======================
def _request_headers():
    return {"User-Agent": DEFAULT_UA, "Accept-Language": random.choice(ACCEPT_LANG_POOL)}

def _http_get(url: str) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=_request_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400: return None
        ct = (resp.headers.get("Content-Type","") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct: return None
        return resp
    except Exception:
        return None

def extract_text_for_classification(html: bytes) -> Tuple[str, str, List[Tuple[str,str]]]:
    if not html: return "", "", []
    enc = detect_encoding(html)
    try:
        soup = BeautifulSoup(html.decode(enc, errors="ignore"), "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "svg", "picture", "source"]):
        bad.decompose()
    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    md = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
    meta_desc = md["content"].strip() if (md and md.get("content")) else ""
    heads = " ".join(h.get_text(" ", strip=True) for h in soup.select("h1, h2, h3")[:8])
    paras = " ".join(p.get_text(" ", strip=True) for p in soup.select("p")[:12])
    text = re.sub(r"\s+", " ", " ".join([meta_desc, heads, paras]).strip())
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        txt = a.get_text(" ", strip=True)[:120]
        if href and len(links) < 200:
            links.append((txt, href))
    return short(title, 180), text, links

def _absolute_url(base_url: str, href: str) -> Optional[str]:
    try:
        if not href or href.startswith(("javascript:", "mailto:")): return None
        if href.startswith("//"): return "https:" + href
        if bool(urlparse(href).netloc): return href
        base = urlparse(base_url)
        base_path = base.path if base.path.endswith("/") else os.path.dirname(base.path) + "/"
        return urlunparse((base.scheme, base.netloc, os.path.normpath(os.path.join(base_path, href)), "", "", ""))
    except Exception:
        return None

def pick_microcrawl_targets(base_url: str, links: List[Tuple[str,str]]) -> List[str]:
    targets, seen = [], set()
    for txt, href in links:
        t = (txt or "").lower(); u = (href or "").lower()
        if any(k in u for k in ("/contact","/contacts","/about","/about-us","/services","/solutions","/platform","/company")) \
           or any(k in t for k in ("contact","contacts","about","services","solutions","platform","company")):
            absu = _absolute_url(base_url, href)
            if absu:
                p = urlparse(absu).path
                if p not in seen:
                    targets.append(absu); seen.add(p)
        if len(targets) >= MAX_MICROCRAWL_PAGES: break
    return targets

# ======================= Classification & Scoring =======================
def _looks_garbled(text: str) -> bool:
    if not text:
        return True
    bad = text.count("\uFFFD")
    return (bad / max(1, len(text))) > 0.02

def is_operator_program(url: str, text: str) -> bool:
    u = url.lower(); t = text.lower()
    if any(tok in u for tok in CASINO_TLDS): return True
    if any(tok in t for tok in OPERATOR_PROGRAM_HINTS): return True
    if "/affiliate" in u or "/affiliates" in u:
        if "for operators" in t or "for brands" in t or "for advertisers" in t or "platform" in t:
            return False
        return True
    return False

def is_network_for_publishers(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in NETWORK_FOR_PUBLISHERS_HINTS)

def is_company_or_platform(url: str, text: str) -> bool:
    t = text.lower()
    has_aff = any(k in t for k in ("affiliate","affiliat","партнерск","партнёрск","рефераль","opm","outsourced","program management","affiliate management"))
    has_ig  = any(k in t for k in ("igaming","i-gaming","casino","казино","гемблинг","игейминг","игемблинг","беттинг","sportsbook","bookmaker","poker","slots"))
    if not (has_aff and has_ig): return False
    has_company = any(k in t for k in COMPANY_SERVICE_HINTS)
    if not has_company: return False
    if is_network_for_publishers(t): return False
    return True

def is_lenient_company(url: str, text: str) -> bool:
    t = text.lower()
    if is_operator_program(url, t) or is_network_for_publishers(t):
        return False
    ig = any(k in t for k in ("igaming","casino","гембл","игейминг","беттинг","sportsbook","bookmaker","slots","покер"))
    svc = any(k in t for k in ("marketing","growth","acquisition","crm","retention","consulting","services","solutions","agency","оптимизац","консалт","агентств"))
    b2b = any(k in t for k in ("for operators","for brands","for advertisers","clients","case study","request a demo","contact sales","request proposal","для оператор"))
    return ig and (svc or b2b)

def language_ok(text: str, region: str) -> bool:
    allowed = REGION_LANG_ALLOW.get(region, REGION_LANG_ALLOW["wt-wt"])
    lang = guess_lang(text)
    if lang in allowed:
        return True
    t = text.lower()
    ig = any(k in t for k in ("igaming","casino","бетт","казин","sportsbook","bookmaker","слот","покер"))
    aff = any(k in t for k in ("affiliate","аффил","партнерск","партнёрск","referral","opm"))
    if ig and aff and lang in ("en","ru"):
        return True
    return False

def quality_score(url: str, text: str) -> float:
    t = text.lower()
    score = 0.0
    if url.lower().startswith("https://"): score += 0.06
    for k in ("about","contact","services","solutions","platform","features","pricing","clients","case"):
        if k in t: score += 0.02
    if re.search(r"20(2[3-5])", t): score += 0.05
    return min(score, 0.4)

def agency_boost(text: str) -> float:
    t = text.lower()
    boost = 0.0
    if "agency" in t or "агентств" in t or "opm" in t or "program management" in t: boost += 0.12
    if "for operators" in t or "for brands" in t or "for advertisers" in t: boost += 0.08
    if "platform" in t and "features" in t and "pricing" in t: boost -= 0.05
    if "affiliate network" in t or is_network_for_publishers(t): boost -= 0.25
    return max(-0.3, min(0.2, boost))

def region_affinity_quick(url: str, t_low: str, region: str) -> float:
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"]); tld = "." + reg["tld"] if reg.get("tld") else ""
    u = url.lower(); score = 0.0
    if tld and u.endswith(tld): score += 0.5
    toks = [x.lower() for x in REGION_HINTS.get(region, {}).get("tokens", [])]
    if toks and any(tok in t_low for tok in toks): score += 0.2
    return max(0.0, min(1.0, score))

def region_affinity(url: str, text: str, region: str) -> float:
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"]); tld = "." + reg["tld"] if reg.get("tld") else ""
    u = url.lower(); t = text.lower(); score = 0.0
    if tld and u.endswith(tld): score += 0.55
    hints = REGION_HINTS.get(region, {})
    tokens = [x.lower() for x in hints.get("tokens", [])]
    orgs = [x.lower() for x in hints.get("org", [])]
    cities = [x.lower() for x in hints.get("cities", [])]
    if tokens and any(tok in t for tok in tokens): score += 0.25
    if orgs and any(o in t for o in orgs): score += 0.20
    if cities and any(c in t for c in cities): score += 0.15
    rex = PHONE_CODE_RE.get(region);  cur = CURRENCY_HINTS.get(region)
    if rex and rex.search(t): score += 0.30
    if cur and any(sym.lower() in t for sym in [c.lower() for c in cur[0]]): score += cur[1]
    if language_ok(t, region): score += 0.15
    else: score -= 0.30
    if tld and (not u.endswith(tld)) and not (tokens and any(tok in t for tok in tokens)): score -= 0.35
    return max(0.0, min(1.0, score))

def base_relevance(text: str) -> float:
    t = text.lower(); rel = 0.0
    if any(k in t for k in ("affiliate","affiliat","партнерск","opm","outsourced","program management","affiliate management")): rel += 0.45
    if any(k in t for k in ("igaming","casino","беттинг","sportsbook","bookmaker","poker","slots","gambling")): rel += 0.4
    if any(k in t for k in ("for operators","for brands","for advertisers","clients","case study","platform","services","solutions")): rel += 0.25
    return max(0.0, min(1.0, rel))

def score_item(url: str, text: str, region: str) -> float:
    d = domain_of(url)
    score = 0.85*base_relevance(text) + 1.1*region_affinity(url, text, region) + quality_score(url, text) \
            + domain_feedback_adjustment(d) + agency_boost(text)
    if any(url.lower().endswith(ct) for ct in CASINO_TLDS): score -= 0.2
    return max(0.0, min(2.2, score))

# ======================= Harvesters =======================
def harvest_affcatalog(base="https://affcatalog.com/ru/") -> List[str]:
    """Парсим каталожные страницы и берём ссылки на партнёрки/платформы/агентства."""
    out = []
    try:
        r = requests.get(base, headers=_request_headers(), timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400: return out
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            txt = a.get_text(" ", strip=True).lower()
            if not href: continue
            if "aff" in href or "affiliate" in href or "partners" in href or "casino" in txt or "игр" in txt:
                if href.startswith("/"): href = base.rstrip("/") + href
                if href.startswith("http"):
                    out.append(normalize_url(href))
        out = list(dict.fromkeys(out))
    except Exception:
        return out
    return out

# ======================= Collect URLs =======================
def generate_search_queries(user_prompt: str, region="wt-wt") -> Tuple[List[str], List[str], str, Dict[str,bool]]:
    region = REGION_ALIASES.get(region, region)
    if region not in REGION_MAP:
        logger.warning(f"Invalid region {region}, defaulting to wt-wt")
        region = "wt-wt"
    intent = detect_intent(user_prompt)
    base_queries = build_core_queries(user_prompt)
    queries = apply_geo_bias(base_queries, region)
    queries = [with_intent_filters(q, intent) for q in queries]
    return queries, base_queries, region, intent

def collect_urls(web_queries: List[str], region: str, engine: str, per_query_k: int) -> List[str]:
    # ВОЛНА 1: DDG с гео-бiais
    all_urls: List[str] = []
    for q in web_queries:
        all_urls.extend(duckduckgo_search(q, max_results=per_query_k, region=region))
    logger.info(f"Wave1 URLs: {len(all_urls)}")

    # ВОЛНА 2: те же запросы без site:.tld (больше охват)
    q2 = []
    for q in web_queries:
        q2.append(re.sub(r"site:\*?\.[a-z]{2,3}\b", "", q, flags=re.I).strip())
    add2 = []
    for q in q2:
        add2.extend(duckduckgo_search(q, max_results=max(10, per_query_k//2), region=region))
    merged = list(dict.fromkeys(all_urls + add2))
    logger.info(f"Wave2 URLs add: {len(add2)}, merged: {len(merged)}")

    # ВОЛНА 3: каталог affcatalog как «семена»
    try:
        seeds = harvest_affcatalog()
        merged = list(dict.fromkeys(merged + seeds))
        logger.info(f"Harvested from directories: {len(seeds)}")
    except Exception:
        pass

    # SerpAPI fallback — только если совсем пусто
    if engine in ("serpapi","both") and SERPAPI_API_KEY and (len(merged) < MIN_RESULTS*3):
        more = []
        for q in web_queries:
            more.extend(serpapi_search(q, max_results=per_query_k, region=region))
        merged = list(dict.fromkeys(merged + more))
        logger.info(f"SerpAPI fallback URLs: {len(more)}")

    # Пред-фильтр: выкинуть жёстко плохие домены и недавние seen
    filtered = []
    seen_roots = set()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        since = (datetime.utcnow() - timedelta(days=RECENT_SEEN_DAYS)).isoformat()
        recent = {row[0] for row in c.execute("SELECT domain FROM seen_recent WHERE last_seen>=?", (since,))}
    for u in merged:
        d = domain_of(u)
        if is_hard_bad_domain(d): continue
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(d): continue
        rd = root_domain(d)
        if rd in recent:  # анти-повторы между запросами
            continue
        if rd in seen_roots:
            continue
        seen_roots.add(rd)
        filtered.append(normalize_url(u))
    return filtered

# ======================= Pipeline =======================
def _prefilter_reason(url: str, text: str) -> Optional[str]:
    if looks_like_job(url, text): return "jobs"
    if looks_like_event(url, text): return "event"
    if looks_like_directory(url, text): return "directory"
    if looks_like_blog(url, text): return "blog"
    if is_network_for_publishers(text): return "network_for_publishers"
    if is_operator_program(url, text): return "operator_program"
    return None

def fetch_one(url: str, region: str) -> Optional[Dict[str,Any]]:
    try:
        if not url or is_hard_bad_domain(domain_of(url)): return None
        if domain_is_blocked(domain_of(url)): return None
        resp = _http_get(url)
        if not resp: return None
        title, text, links = extract_text_for_classification(resp.content)
        if not text: return None
        if _looks_garbled(text): return None

        t_low = text.lower()
        # быстрая отбраковка сетей "для вебмастеров"
        if any(h in t_low for h in NETWORK_FOR_PUBLISHERS_HINTS):
            return None

        # микрокраул (экономно)
        need_more_region = region_affinity_quick(url, t_low, region) < 0.6
        need_company = not is_company_or_platform(url, t_low)
        extras_text = ""
        if ENABLE_MICROCRAWL and (need_more_region or need_company):
            for ex_url in pick_microcrawl_targets(url, links):
                ex_resp = _http_get(ex_url)
                if not ex_resp: continue
                _, ex_text, _ = extract_text_for_classification(ex_resp.content)
                if ex_text: extras_text += " " + short(ex_text, 1500)

        full_text = (text + " " + extras_text).strip()
        return {"url": url, "name": title or "N/A", "text": full_text}
    except Exception:
        return None

def scrape_parallel(urls: List[str], region: str, allow_directory=False, lenient=False,
                    penalty_map: Optional[Dict[str,float]]=None, relax_language: bool=False
                   ) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]]]:
    results: List[Dict[str,Any]] = []
    leftovers: List[Dict[str,Any]] = []
    operators: List[Dict[str,Any]] = []

    per_host_counter: Dict[str,int] = {}
    filtered_urls = []
    for u in urls:
        dom = domain_of(u)
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(dom): continue
        if per_host_counter.get(dom, 0) >= PER_HOST_LIMIT: continue
        per_host_counter[dom] = per_host_counter.get(dom, 0) + 1
        filtered_urls.append(u)

    start_ts = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut2url = {ex.submit(fetch_one, u, region): u for u in filtered_urls}
        for fut in as_completed(fut2url):
            if (time.time() - start_ts) > SCRAPE_HARD_DEADLINE_SEC:
                logger.warning("Scrape deadline reached, stopping further processing")
                break
            item = fut.result()
            if not item: continue
            url = item["url"]; name = item["name"]; text = item["text"]

            reason = _prefilter_reason(url, text)
            if reason:
                if reason == "operator_program":
                    operators.append(item)
                else:
                    leftovers.append(item)
                continue

            ok_company = is_company_or_platform(url, text) or (lenient and is_lenient_company(url, text))
            if not ok_company:
                leftovers.append(item)
                continue

            if not (relax_language or language_ok(text, region)):
                leftovers.append(item)
                continue

            score = score_item(url, text, region)
            if penalty_map:
                d = root_domain(domain_of(url))
                score += penalty_map.get(d, 0.0)

            results.append({
                "id": str(uuid.uuid4()),
                "name": name or "N/A",
                "website": url,
                "description": short(text, 400),
                "country": "N/A",
                "source": "Web",
                "score": score
            })

    # Дедуп по корневому домену: держим 1 запись
    by_root: Dict[str, Dict[str,Any]] = {}
    for r in results:
        rd = root_domain(domain_of(r["website"]))
        if rd not in by_root or r["score"] > by_root[rd]["score"]:
            by_root[rd] = r
    compact: List[Dict[str,Any]] = list(by_root.values())

    # Лёгкая рандомизация в рамках близких скора (убирает «одни и те же»)
    compact.sort(key=lambda x: x["score"], reverse=True)
    i = 0
    while i < len(compact):
        j = i
        while j < len(compact) and abs(compact[j]["score"] - compact[i]["score"]) < 0.08:
            j += 1
        random.shuffle(compact[i:j])
        i = j

    return compact, leftovers, operators

def ensure_minimum(primary: List[Dict[str,Any]], leftovers: List[Dict[str,Any]], operators: List[Dict[str,Any]],
                   region: str) -> List[Dict[str,Any]]:
    results = list(primary)

    # Если хватает — просто отсортировать
    if len(results) >= MIN_RESULTS:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:MAX_RESULTS]

    # 1) Добираем из operators (легче допуск)
    op_urls = [it["url"] for it in operators if it.get("url")]
    if op_urls:
        op_res, _, _ = scrape_parallel(op_urls[:800], region, lenient=True, relax_language=True)
        existing = {r["website"] for r in results}
        for r in op_res:
            if r["website"] not in existing:
                results.append(r); existing.add(r["website"])
            if len(results) >= MAX_RESULTS: break

    if len(results) >= MIN_RESULTS:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:MAX_RESULTS]

    # 2) Добираем из leftovers c мягкими правилами
    lo_urls = [it["url"] for it in leftovers if it.get("url")]
    if lo_urls:
        penalty = {}
        for it in leftovers:
            rd = root_domain(domain_of(it["url"]))
            penalty[rd] = penalty.get(rd, 0.0) - 0.05  # чуть накажем качество
        lo_res, _, _ = scrape_parallel(lo_urls[:1200], region, lenient=True, relax_language=True, penalty_map=penalty)
        existing = {r["website"] for r in results}
        for r in lo_res:
            if r["website"] not in existing:
                results.append(r); existing.add(r["website"])
            if len(results) >= MAX_RESULTS: break

    if len(results) >= MIN_RESULTS:
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:MAX_RESULTS]

    # 3) FINISH HIM: affcatalog bulk
    try:
        aff_urls = harvest_affcatalog()
    except Exception:
        aff_urls = []
    if aff_urls:
        aff_res, _, _ = scrape_parallel(aff_urls[:1500], region, lenient=True, relax_language=True)
        existing = {r["website"] for r in results}
        for r in aff_res:
            if r["website"] not in existing:
                results.append(r); existing.add(r["website"])
            if len(results) >= MAX_RESULTS: break

    # Итог: отсортировать и обрезать
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:MAX_RESULTS]

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
                (row["id"], row.get("name","N/A"), row["website"], row.get("description","N/A"),
                 "", row.get("country","N/A"), row.get("source","Web"),
                 "Active", "Подходит", row.get("score",0.0))
            )
            c.execute(
                """INSERT OR IGNORE INTO results_history
                   (id, query_id, url, domain, source, score, intent_json, region, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), query_id, row["website"], domain_of(row["website"]), row.get("source","Web"),
                 row.get("score",0.0), json.dumps(intent, ensure_ascii=False), region, datetime.utcnow().isoformat())
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

# ======================= API =======================
@app.route("/search", methods=["POST", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.json or {}
        user_query = data.get("query", "").strip()
        region_req = (data.get("region") or "wt-wt").strip()
        region_req = REGION_ALIASES.get(region_req, region_req)
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE","ddg")).lower()
        per_query_k = int(data.get("per_query", 25))

        if not user_query:
            return jsonify({"error":"Query is required"}), 400

        logger.info(f"API request: query='{user_query}', region={region_req}, engine={engine}")

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM results")
            conn.commit()

        web_queries, phrases, region, intent = generate_search_queries(user_query, region_req)
        query_id = insert_query_record(user_query, intent, region)

        urls = collect_urls(web_queries, region, engine, per_query_k=per_query_k)

        # Скрапим строго, потом добираем
        t0 = time.time()
        strict_res, leftovers, operators = scrape_parallel(urls[:600], region, lenient=False, relax_language=False)
        final = ensure_minimum(strict_res, leftovers, operators, region)
        dt = time.time() - t0

        logger.info(f"Final results: {len(final)}")

        # Пометить домены как недавно виденные для анти-повторов следующих запросов
        update_seen_recent([root_domain(domain_of(it["website"])) for it in final])

        # Сохранить и отдать
        for r in final:
            save_result_records(r, intent, region, query_id)
        if final:
            save_to_csv_txt()

        return jsonify({
            "results": final,
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
        dom = domain_of(url)
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO interactions (id, query_id, url, domain, action, weight, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (str(uuid.uuid4()), query_id, url, dom, action, weight, datetime.utcnow().isoformat()))
            conn.commit()
        if action == "bad":
            _block_domain(dom, reason="user_bad", days=BAD_BLOCK_DAYS)
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
