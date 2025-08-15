
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

PHONE_CODE_RE = {
    "kz-ru": re.compile(r"(\+?7[\s\-]?\(?7\d\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})"),   # +7 7xx ...
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

def normalize_url(url: str) -> str:
    try:
        u = urlparse(url)
        # drop fragments and most tracking params
        query = [(k,v) for k,v in parse_qsl(u.query, keep_blank_values=False) if not k.lower().startswith("utm_")]
        u2 = u._replace(query=urlencode(query, doseq=True), fragment="")
        return urlunparse(u2)
    except Exception:
        return url

def is_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))

def short(s: str, n=220) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

def guess_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"
    if re.search(r"[іїєґ]", t.lower()):  # UA
        return "uk"
    if re.search(r"[а-яё]", t.lower()):
        return "ru"
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
            # optional helper table for explicit blocks; not required
            c.execute(
                """CREATE TABLE IF NOT EXISTS domain_blocks (
                    domain TEXT PRIMARY KEY,
                    blocked_until TEXT,
                    reason TEXT
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
    "ua-ua": {
        "tokens": ["Україна","Ukraine","Київ","Kyiv","Львів","Lviv","Одеса","Odesa","Дніпро","Dnipro","Харків","Kharkiv","UA","Чернівці","Чернігів","Запоріжжя","Миколаїв","Полтава","Вінниця","Суми","Рівне","Івано-Франківськ"],
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
    "program terms","payouts","banners","affiliates terms"
)

COMPANY_SERVICE_HINTS = (
    "our services","services","solutions","platform","for operators","for brands","for advertisers",
    "clients","case study","case studies","features","integrations","pricing",
    "request a demo","get a demo","contact sales","schedule a call","request proposal"
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
    # бизнесовый запрос по умолчанию
    return {"affiliate": affiliate or casino, "learn": learn, "casino": casino, "business": not learn}

# ======================= Request counter (пер-движок, back-compatible) =======================
def _load_request_count():
    try:
        with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # back-compat: если старая форма, оборачиваем
            if "engines" not in data:
                data = {
                    "engines": {"ddg": {"count": data.get("count", 0)}, "serpapi": {"count": 0}},
                    "last_reset": data.get("last_reset", datetime.now().strftime("%Y-%m-%d"))
                }
            last_reset = data.get("last_reset", "")
            if last_reset:
                last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d")
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
        "casino affiliate network for operators",
        "igaming affiliate platform for operators",
        "casino affiliate services for operators",
        "игейминг аффилиат агентство",
        "агентство по аффилиат маркетингу казино",
        "управление партнерской программой казино",
        "OPM igaming",
        "affiliate management for casino operators",
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
    # country/cities hints
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
    """STRONG: блокируем если есть явный блок или любой bad за последние 60 дней."""
    if not STRONG_BLOCK_ON_BAD:
        return _domain_explicitly_blocked(domain)
    try:
        if _domain_explicitly_blocked(domain):
            return True
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=BAD_BLOCK_DAYS)).isoformat()
            c.execute("""SELECT COUNT(*) FROM interactions
                         WHERE domain=? AND action='bad' AND created_at>=?""", (domain, since))
            cnt = c.fetchone()[0] or 0
            return cnt >= 1
    except Exception:
        return False

def domain_feedback_adjustment(domain: str) -> float:
    """+boost за good/click, -penalty за bad; каппируем пожёстче."""
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
    data_ok = _engine_allowed("ddg")
    if not data_ok:
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
    if not SERPAPI_API_KEY:
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
        for block in ("inline_results","local_results","top_stories"):
            for it in data.get(block, []) or []:
                link = it.get("link") or it.get("source")
                if link: urls.append(link)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        _bump_counter("serpapi")
        urls = [normalize_url(u) for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.warning(f"SerpAPI error: {e}")
        return []

# ======================= Scraping & Deep analysis =======================
def _request_headers():
    return {
        "User-Agent": DEFAULT_UA,
        "Accept-Language": random.choice(ACCEPT_LANG_POOL)
    }

def _http_get(url: str) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=_request_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            return None
        ct = (resp.headers.get("Content-Type","") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None
        return resp
    except Exception:
        return None

def extract_text_for_classification(html: bytes) -> Tuple[str, str, List[Tuple[str,str]]]:
    """Возвращаем title, компактный видимый текст, и список внутренних ссылок (text, href)."""
    if not html:
        return "", "", []
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

    heads = " ".join(h.get_text(" ", strip=True) for h in soup.select("h1, h2, h3")[:8])
    paras = " ".join(p.get_text(" ", strip=True) for p in soup.select("p")[:12])

    text = " ".join([meta_desc, heads, paras])
    text = re.sub(r"\s+", " ", text).strip()

    # collect internal links (anchor text, href)
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        txt = a.get_text(" ", strip=True)[:120]
        if href and len(links) < 200:
            links.append((txt, href))
    return short(title, 180), text, links

def _absolute_url(base_url: str, href: str) -> Optional[str]:
    try:
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            return None
        if href.startswith("//"):
            return "https:" + href
        if bool(urlparse(href).netloc):
            return href
        # relative
        base = urlparse(base_url)
        base_path = base.path if base.path.endswith("/") else os.path.dirname(base.path) + "/"
        joined = urlunparse((base.scheme, base.netloc, os.path.normpath(os.path.join(base_path, href)), "", "", ""))
        return joined
    except Exception:
        return None

def pick_microcrawl_targets(base_url: str, links: List[Tuple[str,str]]) -> List[str]:
    targets = []
    cand = []
    for txt, href in links:
        t = (txt or "").lower()
        u = (href or "").lower()
        if any(k in u for k in ("/contact","/contacts","/about","/o-nas","/about-us","/services","/solutions","/platform","/company")) \
           or any(k in t for k in ("contact","contacts","about","о нас","services","solutions","platform","company")):
            absu = _absolute_url(base_url, href)
            if absu:
                cand.append(absu)
    # dedupe per path
    seen = set()
    for u in cand:
        p = urlparse(u).path
        if p not in seen:
            targets.append(u); seen.add(p)
        if len(targets) >= 2:  # жесткий лимит
            break
    return targets

def fetch_one(url: str) -> Optional[Dict[str,Any]]:
    try:
        if not url or is_hard_bad_domain(domain_of(url)): return None
        if domain_is_blocked(domain_of(url)): return None
        resp = _http_get(url)
        if not resp: return None
        title, text, links = extract_text_for_classification(resp.content)
        if not text:
            return None

        # микрокраул (до 2 страниц)
        extras_text = ""
        for ex_url in pick_microcrawl_targets(url, links):
            ex_resp = _http_get(ex_url)
            if not ex_resp:
                continue
            _, ex_text, _ = extract_text_for_classification(ex_resp.content)
            if ex_text:
                extras_text += " " + short(ex_text, 1500)

        full_text = (text + " " + extras_text).strip()
        return {"url": url, "name": title or "N/A", "text": full_text}
    except Exception:
        return None

# ======================= Classification & Scoring =======================
def is_operator_program(url: str, text: str) -> bool:
    u = url.lower()
    t = text.lower()
    if any(tok in u for tok in CASINO_TLDS):
        return True
    if any(tok in t for tok in OPERATOR_PROGRAM_HINTS):
        return True
    # явные домены операторов часто содержат /affiliates
    if "/affiliate" in u or "/affiliates" in u:
        # различаем платформы/агентства vs программа оператора по контенту
        if "for operators" in t or "for brands" in t or "for advertisers" in t or "platform" in t:
            return False
        return True
    return False

def is_company_or_platform(url: str, text: str) -> bool:
    t = text.lower()
    # must-have тематические сигналы
    has_aff = any(k in t for k in ("affiliate","affiliat","партнерск","партнёрск","рефераль","opm","outsourced"))
    has_ig  = any(k in t for k in ("igaming","i-gaming","casino","казино","гемблинг","игейминг","игемблинг","беттинг","sportsbook","bookmaker","poker","slots"))
    if not (has_aff and has_ig):
        return False
    # признаки компании/сервиса
    has_company = any(k in t for k in COMPANY_SERVICE_HINTS)
    return has_company

def classify(url: str, text: str) -> str:
    if looks_like_job(url, text): return "jobs"
    if looks_like_event(url, text): return "event"
    if looks_like_directory(url, text): return "directory"
    if looks_like_blog(url, text): return "blog"
    if is_operator_program(url, text): return "operator_program"
    if is_company_or_platform(url, text): return "company_or_platform"
    return "junk"

def language_ok(text: str, region: str) -> bool:
    allowed = REGION_LANG_ALLOW.get(region, REGION_LANG_ALLOW["wt-wt"])
    lang = guess_lang(text)
    return lang in allowed

def quality_score(url: str, text: str) -> float:
    t = text.lower()
    score = 0.0
    # https / status assumed ok if we parsed
    try:
        if url.lower().startswith("https://"): score += 0.06
    except Exception:
        pass
    # presence of key pages (we inferred via text)
    for k in ("about","contact","services","solutions","platform","features","pricing","clients","case"):
        if k in t: score += 0.02
    score = min(score, 0.4)
    # freshness (скромная эвристика)
    if re.search(r"20(2[3-5])", t):  # 2023..2025
        score += 0.05
    return min(score, 0.4)

def region_affinity(url: str, text: str, region: str) -> float:
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = "." + reg["tld"] if reg.get("tld") else ""
    u = url.lower(); t = text.lower()
    score = 0.0

    # локальный tld (сильный приоритет)
    if tld and u.endswith(tld):
        score += 0.55

    hints = REGION_HINTS.get(region, {})
    tokens = [x.lower() for x in hints.get("tokens", [])]
    orgs = [x.lower() for x in hints.get("org", [])]
    cities = [x.lower() for x in hints.get("cities", [])]

    if tokens and any(tok in t for tok in tokens):
        score += 0.25
    if orgs and any(o in t for o in orgs):
        score += 0.20
    if cities and any(c in t for c in cities):
        score += 0.15

    # телефонные коды в контактной странице/тексте
    rex = PHONE_CODE_RE.get(region)
    if rex and rex.search(t):
        score += 0.30

    # валюта
    cur = CURRENCY_HINTS.get(region)
    if cur and any(sym.lower() in t for sym in [c.lower() for c in cur[0]]):
        score += cur[1]

    # язык
    if language_ok(t, region):
        score += 0.15
    else:
        score -= 0.30

    # штраф за не-локальные без токенов
    if tld and (not u.endswith(tld)) and not (tokens and any(tok in t for tok in tokens)):
        score -= 0.35

    return max(0.0, min(1.0, score))

def base_relevance(text: str) -> float:
    t = text.lower()
    rel = 0.0
    if any(k in t for k in ("affiliate","affiliat","партнерск","opm","outsourced")):
        rel += 0.4
    if any(k in t for k in ("igaming","casino","беттинг","sportsbook","bookmaker","poker","slots","gambling")):
        rel += 0.4
    if any(k in t for k in ("for operators","for brands","for advertisers","clients","case study","platform","services","solutions")):
        rel += 0.3
    return max(0.0, min(1.0, rel))

def score_item(url: str, text: str, region: str) -> float:
    d = domain_of(url)
    base = base_relevance(text)
    reg = region_affinity(url, text, region)
    q = quality_score(url, text)
    fb = domain_feedback_adjustment(d)
    # penalties already handled by classifier, but keep casino tlds
    tld_pen = -0.2 if any(url.lower().endswith(ct) for ct in CASINO_TLDS) else 0.0
    score = 0.9*base + 1.1*reg + q + fb + tld_pen
    return max(0.0, min(2.2, score))

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

# ======================= Pipeline helpers =======================
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
    # dedupe and preblock
    deduped = []
    seen = set()
    for u in all_urls:
        if not u or u in seen:
            continue
        d = domain_of(u)
        if is_hard_bad_domain(d):
            continue
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(d):
            continue
        deduped.append(normalize_url(u)); seen.add(u)
    return deduped

def _prefilter_url(url: str, name: str, text: str) -> Optional[str]:
    # Мгновенные отсеки мусора
    if looks_like_job(url, text): return "jobs"
    if looks_like_event(url, text): return "event"
    if looks_like_directory(url, text): return "directory"
    if looks_like_blog(url, text): return "blog"
    if is_operator_program(url, text): return "operator_program"
    return None

def scrape_parallel(urls: List[str], region: str) -> List[Dict[str,Any]]:
    results: List[Dict[str,Any]] = []

    # per-host limit
    per_host_counter: Dict[str,int] = {}
    filtered_urls = []
    for u in urls:
        dom = domain_of(u)
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(dom):
            continue
        if per_host_counter.get(dom, 0) >= PER_HOST_LIMIT:
            continue
        per_host_counter[dom] = per_host_counter.get(dom, 0) + 1
        filtered_urls.append(u)

    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = "." + reg["tld"] if reg.get("tld") else ""

    stage_counts = {"fetched":0,"prefilter_pass":0,"classified_ok":0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut2url = {ex.submit(fetch_one, u): u for u in filtered_urls}
        for fut in as_completed(fut2url):
            item = fut.result()
            if not item:
                continue
            url = item["url"]; name = item["name"]; text = item["text"]
            stage_counts["fetched"] += 1

            # предфильтр
            bad_reason = _prefilter_url(url, name, text)
            if bad_reason:
                continue

            # тематический классификатор (жёстко)
            cls = classify(url, text)
            if cls not in ("company_or_platform",):
                # операторские и информация/мусор — исключаем
                continue

            # языковой фильтр
            if not language_ok(text, region):
                continue

            stage_counts["classified_ok"] += 1

            # скоринг
            score = score_item(url, text, region)

            results.append({
                "id": str(uuid.uuid4()),
                "name": name or "N/A",
                "website": url,
                "description": short(text, 400),
                "country": "N/A",
                "source": "Web",
                "score": score
            })

    # дедуп по домену (не более 1 результата на домен, изредка 2 если сильная разница в score)
    by_domain: Dict[str, List[Dict[str,Any]]] = {}
    for r in results:
        d = domain_of(r["website"])
        by_domain.setdefault(d, []).append(r)
    compact: List[Dict[str,Any]] = []
    for d, items in by_domain.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        keep = [items[0]]
        if len(items) > 1 and (items[0]["score"] - items[1]["score"] > 0.3):
            keep.append(items[1])
        compact.extend(keep)

    # приоритет: сначала локальные (region-heavy mixing)
    if tld:
        local = [r for r in compact if r["website"].lower().endswith(tld) or region_affinity(r["website"], r["description"], region) >= 0.6]
        global_ = [r for r in compact if r not in local]
        local.sort(key=lambda x: x["score"], reverse=True)
        global_.sort(key=lambda x: x["score"], reverse=True)
        # enforce ≥70% региональных, если есть
        want_local = max(int(0.7 * MAX_RESULTS), MIN_RESULTS // 2)
        head = local[:want_local]
        tail = (local[want_local:] + global_)
        results_final = head + tail
    else:
        results_final = sorted(compact, key=lambda x: x["score"], reverse=True)

    # обрезаем до MAX_RESULTS
    results_final = results_final[:MAX_RESULTS]
    return results_final

def escalate_if_needed(results: List[Dict[str,Any]], urls: List[str], region: str) -> List[Dict[str,Any]]:
    """Если < MIN_RESULTS, делаем второй проход c ослаблением регион-порога и добор из хвоста."""
    if len(results) >= MIN_RESULTS:
        return results
    more = scrape_parallel(urls[400:1000], region)  # хвост
    known = set(r["website"] for r in results)
    for r in more:
        if r["website"] not in known:
            results.append(r); known.add(r["website"])
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:MAX_RESULTS]

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

        # если мало — эскалация по хвосту
        if len(web_results) < MIN_RESULTS and len(urls) > 400:
            web_results = escalate_if_needed(web_results, urls, region)

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
        dom = domain_of(url)
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO interactions (id, query_id, url, domain, action, weight, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (str(uuid.uuid4()), query_id, url, dom, action, weight, datetime.utcnow().isoformat()))
            conn.commit()
        # немедленная блокировка при bad
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
