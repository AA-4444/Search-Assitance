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

# ======================= Config & Env =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend_dist")
DB_PATH = os.getenv("DB_PATH", "search_results.db")

SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "").strip() or None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or None

MIN_RESULTS = int(os.getenv("MIN_RESULTS", "25"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "8"))
PER_HOST_LIMIT = int(os.getenv("PER_HOST_LIMIT", "2"))
STRONG_BLOCK_ON_BAD = (os.getenv("STRONG_BLOCK_ON_BAD", "true").lower() == "true")

# Гео-настройки
STRICT_GEO = (os.getenv("STRICT_GEO", "true").lower() == "true")
FOREIGN_FALLBACK_RATIO = float(os.getenv("FOREIGN_FALLBACK_RATIO", "0.35"))  # доля иностранщины при доборе
NONLOCAL_DROP_CJK = (os.getenv("NONLOCAL_DROP_CJK", "true").lower() == "true")

REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1000"))
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
CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")  # ja/zh/ko

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
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""

def host_tld(domain: str) -> str:
    d = (domain or "").lower()
    parts = d.split(".")
    if len(parts) < 2: return ""
    # поддержим двухуровневые ccTLD: com.kz, com.ua, etc.
    if len(parts) >= 3 and len(parts[-1]) == 2:
        return parts[-2] + "." + parts[-1]  # com.kz
    return parts[-1]  # kz

def is_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))

def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))

def clean_description(html_or_text) -> str:
    if not html_or_text: return "N/A"
    soup = BeautifulSoup(str(html_or_text), "html.parser")
    text = soup.get_text(" ").strip()
    return " ".join(text.split()[:220])

def short(s: str, n=220) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

def contains_any(t: str, toks: List[str]) -> bool:
    if not t or not toks: return False
    low = t.lower()
    return any(tok.lower() in low for tok in toks if tok)

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
                """CREATE TABLE IF NOT EXISTS gemini_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TEXT
                )"""
            )
            conn.commit()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

init_db()

# ======================= Intent / Filters =======================
INTENT_AFFILIATE = {
    "affiliate","аффилиат","аффилиэйт","партнерка","партнёрка","партнерская программа","партнёрская программа",
    "реферальная","referral","cpa","игемблинг","игейминг","igaming","casino affiliate","affiliate network",
    "партнеры казино","партнёры казино","аффилейт","афилейт"
}
INTENT_LEARN = {
    "что такое","what is","определение","definition","гайд","guide","обзор","overview","курс","course","как работает","how to"
}
CASINO_TOKENS = {"казино","casino","igaming","гемблинг","игемблинг","игейминг","беттинг","sportsbook","bookmaker","ставки"}

def detect_intent(q: str) -> Dict[str, bool]:
    t = (q or "").lower()
    affiliate = any(k in t for k in INTENT_AFFILIATE)
    learn = any(k in t for k in INTENT_LEARN)
    casino = any(k in t for k in CASINO_TOKENS)
    return {"affiliate": affiliate, "learn": learn, "casino": casino, "business": not learn}

BAD_DOMAINS_HARD = {
    "google.com","maps.google.com","baidu.com","zhihu.com","commentcamarche.net",
    "xnxx.com","pornhub.com","hometubeporn.com","porn7.xxx","fuckvideos.xxx",
}

# системный хлам и нерелевант
SOFT_BAD_DOMAINS = {
    "support.google.com","youtube.com","m.youtube.com","bing.com","yandex.com","yandex.ru",
    "microsoft.com","support.microsoft.com","aka.ms","xbox.com","minecraft.net","mojang.com",
    "store.steampowered.com","help.steampowered.com",
    "coinspaid.com","coinspaid.media",
    "hostadvice.com","wikipedia.org","en.wikipedia.org","ru.wikipedia.org",
    "vc.ru","habr.com","marketer.ua","influencermarketinghub.com","timesofcasino.com",
    "thebetting.net","cashradar.ru","softonic.com","uptodown.com","apkpure.com",
}

KNOWLEDGE_DOMAINS = {
    "hubspot.com","coursera.org","ibm.com","sproutsocial.com","digitalmarketinginstitute.com",
    "harvard.edu","professional.dce.harvard.edu","medium.com","sendx.io","news.ycombinator.com",
}

COMPANY_URL_TOKENS = [
    "agency","network","partners","program","programs","platform","services","solutions",
    "management","consulting","company","about","contact","affiliates","opm"
]

CASINO_TLDS = (".casino",".bet",".betting",".poker",".slots",".bingo")

LISTING_HINTS = {"directory","directories","list","listing","catalog","каталог","список","провайдеры","агентства"}
BLOG_HINTS_URL = ["/blog","/news","/article","/articles","/insights","/guide","/guides","/academy","/press"]
JOBS_HINTS_URL = ["/jobs","/job","/careers","/vacancies","/vacancy","/rabota","/vakansii"]
TRASH_PATH_TOKENS = ["/download","/update","/support","/help","/status"]

SPORTS_TRASH_TOKENS = {"премьер лига","лига чемпионов","таблица","расписание"}
def looks_like_sports_trash(text: str) -> bool:
    t = (text or "").lower()
    return any(tok in t for tok in SPORTS_TRASH_TOKENS)

def is_hard_bad_domain(dom: str) -> bool:
    if not dom: return False
    d = dom.lower().lstrip(".")
    if d.startswith("www."): d = d[4:]
    return d in BAD_DOMAINS_HARD

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

def domain_penalty(domain: str) -> float:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=60)).isoformat()
            c.execute("""SELECT COUNT(*) FROM interactions
                         WHERE domain=? AND action='bad' AND created_at>=?""", (domain, since))
            cnt = c.fetchone()[0] or 0
            if cnt >= 3: return 0.5
            return min(cnt * (0.5 / 3), 0.5)
    except Exception:
        return 0.0

def history_boosts(intent: Dict[str,bool], region: str) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT domain, COUNT(*) as cnt
                FROM results_history
                WHERE intent_json = ? AND region = ?
                GROUP BY domain
                HAVING cnt >= 2
            """, (json.dumps(intent, ensure_ascii=False), region))
            for domain, cnt in c.fetchall():
                boosts[domain] = min(0.2, 0.05 * (1 + (cnt ** 0.5)))
            c.execute("""
                SELECT domain,
                       SUM(CASE WHEN action='good' THEN weight WHEN action='click' THEN weight*0.5 ELSE 0 END) as pos,
                       SUM(CASE WHEN action='bad' THEN -weight ELSE 0 END) as neg
                FROM interactions GROUP BY domain
            """)
            for domain, pos, neg in c.fetchall():
                p = pos or 0.0; n = neg or 0.0
                adj = min(0.15 * (1 + (p ** 0.5)) - 0.1 * (1 + (n ** 0.5)), 0.3)
                pen = domain_penalty(domain)
                boosts[domain] = max(min(boosts.get(domain, 0.0) + adj - pen, 0.3), -0.5)
    except Exception as e:
        logger.warning(f"history_boosts error: {e}")
    return boosts

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

# ======================= Search Engines =======================
from ddgs import DDGS

# Расширенная карта регионов с google_domain и tld
REGION_MAP = {
    "wt-wt": {"hl": "en", "gl": "us", "google_domain": "google.com", "tld": "com", "cr": "", "lr": ""},
    "us-en": {"hl": "en", "gl": "us", "google_domain": "google.com", "tld": "us", "cr": "countryUS", "lr": "lang_en"},
    "uk-en": {"hl": "en", "gl": "gb", "google_domain": "google.co.uk", "tld": "uk", "cr": "countryGB", "lr": "lang_en"},
    "de-de": {"hl": "de", "gl": "de", "google_domain": "google.de", "tld": "de", "cr": "countryDE", "lr": "lang_de"},
    "fr-fr": {"hl": "fr", "gl": "fr", "google_domain": "google.fr", "tld": "fr", "cr": "countryFR", "lr": "lang_fr"},
    "ru-ru": {"hl": "ru", "gl": "ru", "google_domain": "google.ru", "tld": "ru", "cr": "countryRU", "lr": "lang_ru"},
    "ua-ua": {"hl": "uk", "gl": "ua", "google_domain": "google.com.ua", "tld": "ua", "cr": "countryUA", "lr": "lang_uk"},
    "kz-ru": {"hl": "ru", "gl": "kz", "google_domain": "google.kz", "tld": "kz", "cr": "countryKZ", "lr": "lang_ru"},
    "by-ru": {"hl": "ru", "gl": "by", "google_domain": "google.by", "tld": "by", "cr": "countryBY", "lr": "lang_ru"},
    "tr-tr": {"hl": "tr", "gl": "tr", "google_domain": "google.com.tr", "tld": "tr", "cr": "countryTR", "lr": "lang_tr"},
    "ae-en": {"hl": "en", "gl": "ae", "google_domain": "google.ae", "tld": "ae", "cr": "countryAE", "lr": "lang_en"},
    "in-en": {"hl": "en", "gl": "in", "google_domain": "google.co.in", "tld": "in", "cr": "countryIN", "lr": "lang_en"},
    "sg-en": {"hl": "en", "gl": "sg", "google_domain": "google.com.sg", "tld": "sg", "cr": "countrySG", "lr": "lang_en"},
    "es-es": {"hl": "es", "gl": "es", "google_domain": "google.es", "tld": "es", "cr": "countryES", "lr": "lang_es"},
    "it-it": {"hl": "it", "gl": "it", "google_domain": "google.it", "tld": "it", "cr": "countryIT", "lr": "lang_it"},
    "nl-nl": {"hl": "nl", "gl": "nl", "google_domain": "google.nl", "tld": "nl", "cr": "countryNL", "lr": "lang_nl"},
    "se-sv": {"hl": "sv", "gl": "se", "google_domain": "google.se", "tld": "se", "cr": "countrySE", "lr": "lang_sv"},
    "no-no": {"hl": "no", "gl": "no", "google_domain": "google.no", "tld": "no", "cr": "countryNO", "lr": "lang_no"},
    "fi-fi": {"hl": "fi", "gl": "fi", "google_domain": "google.fi", "tld": "fi", "cr": "countryFI", "lr": "lang_fi"},
    "cz-cs": {"hl": "cs", "gl": "cz", "google_domain": "google.cz", "tld": "cz", "cr": "countryCZ", "lr": "lang_cs"},
    "sk-sk": {"hl": "sk", "gl": "sk", "google_domain": "google.sk", "tld": "sk", "cr": "countrySK", "lr": "lang_sk"},
    "ro-ro": {"hl": "ro", "gl": "ro", "google_domain": "google.ro", "tld": "ro", "cr": "countryRO", "lr": "lang_ro"},
    "hu-hu": {"hl": "hu", "gl": "hu", "google_domain": "google.hu", "tld": "hu", "cr": "countryHU", "lr": "lang_hu"},
    "ch-de": {"hl": "de", "gl": "ch", "google_domain": "google.ch", "tld": "ch", "cr": "countryCH", "lr": "lang_de"},
    "br-pt": {"hl": "pt", "gl": "br", "google_domain": "google.com.br", "tld": "br", "cr": "countryBR", "lr": "lang_pt"},
    "mx-es": {"hl": "es", "gl": "mx", "google_domain": "google.com.mx", "tld": "mx", "cr": "countryMX", "lr": "lang_es"},
    "au-en": {"hl": "en", "gl": "au", "google_domain": "google.com.au", "tld": "au", "cr": "countryAU", "lr": "lang_en"},
    "nz-en": {"hl": "en", "gl": "nz", "google_domain": "google.co.nz", "tld": "nz", "cr": "countryNZ", "lr": "lang_en"},
}

# DDG понимаемые регионы; fallback для остальных (например, kz-ru -> ru-ru)
DDG_REGION_MAP = {
    "wt-wt": "wt-wt",
    "us-en": "us-en",
    "uk-en": "uk-en",
    "de-de": "de-de",
    "fr-fr": "fr-fr",
    "ru-ru": "ru-ru",
    "ua-ua": "ru-ru",   # ближний аналог
    "kz-ru": "ru-ru",
    "by-ru": "ru-ru",
    "tr-tr": "tr-tr",
    "ae-en": "ae-en",
    "in-en": "in-en",
    "sg-en": "sg-en",
    "es-es": "es-es",
    "it-it": "it-it",
    "nl-nl": "nl-nl",
    "se-sv": "se-sv",
    "no-no": "no-no",
    "fi-fi": "fi-fi",
    "cz-cs": "cz-cs",
    "sk-sk": "sk-sk",
    "ro-ro": "ro-ro",
    "hu-hu": "hu-hu",
    "ch-de": "ch-de",
    "br-pt": "br-pt",
    "mx-es": "mx-es",
    "au-en": "au-en",
    "nz-en": "nz-en",
}

# Гео-экстра: токены городов/ОПФ
REGION_EXTRAS = {
    "kz-ru": {"tokens": ["Казахстан","Алматы","Алма-Ата","Astana","Нур-Султан","KZ","Kazakhstan"], "tld": "kz", "org_tokens": ["ТОО","LLP","ИП","ЖШС"]},
    "ru-ru": {"tokens": ["Россия","Москва","Санкт-Петербург","RF"], "tld": "ru", "org_tokens": ["ООО","АО","ИП"]},
    "ua-ua": {"tokens": ["Україна","Київ","Lviv","Odessa","Ukraine","Kyiv"], "tld": "ua", "org_tokens": ["ТОВ","ФОП"]},
    "by-ru": {"tokens": ["Беларусь","Минск","Belarus","Minsk"], "tld": "by", "org_tokens": ["ООО","ЧУП"]},
    "tr-tr": {"tokens": ["Türkiye","Istanbul","Ankara","Izmir"], "tld": "tr", "org_tokens": ["AŞ","Ltd. Şti."]},
    "ae-en": {"tokens": ["UAE","Dubai","Abu Dhabi","Sharjah"], "tld": "ae", "org_tokens": ["LLC","FZ-LLC"]},
    "in-en": {"tokens": ["India","Delhi","Bengaluru","Mumbai"], "tld": "in", "org_tokens": ["Pvt Ltd","LLP"]},
    "uk-en": {"tokens": ["United Kingdom","London","UK","England"], "tld": "uk", "org_tokens": ["Ltd","LLP"]},
    "us-en": {"tokens": ["USA","United States","New York","San Francisco","Austin"], "tld": "us", "org_tokens": ["LLC","Inc"]},
}

def with_intent_filters(q: str, intent: Dict[str,bool]) -> str:
    parts = [q]
    if intent.get("business"):
        parts.append("-site:wikipedia.org")
        parts += ["-news", "-forum", "-форум", "-guide", "-docs", "-documentation"]
    if intent.get("affiliate") and intent.get("casino"):
        parts += ["-site:askgamblers.com", "-site:casino.org", "-site:bonusfinder.com"]
        parts += ["-вакансии","-вакансия","-работа","-hh.ru","-job","-jobs","-career"]
        parts += ["-\"what is\"", "-\"что такое\"", "-glossary", "-definition"]
    query = " ".join(parts)
    if len(query) > 500:
        query = " ".join(parts[:6])
    return query

def ddg_region(region: str) -> str:
    return DDG_REGION_MAP.get(region, "wt-wt")

def duckduckgo_search(query, max_results=15, region="wt-wt", intent=None, force_ru_ddg=False):
    data = _load_request_count()
    if data["count"] >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        return []
    chosen_region = ddg_region(region)
    if force_ru_ddg and region == "wt-wt":
        chosen_region = "ru-ru"
    q = with_intent_filters(query, intent or {"business": True})
    logger.info(f"DDG search: '{q}' region_used={chosen_region} (requested={region})")
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(q, region=chosen_region, safesearch="moderate", timelimit="y", max_results=max_results)
            for r in results:
                href = r.get("href")
                if href: urls.append(href)
        _save_request_count(data["count"] + 1)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"DDG failed for '{q}': {e}")
        return []

def serpapi_search(query, max_results=15, region="wt-wt", intent=None):
    if not SERPAPI_API_KEY:
        return []
    q = with_intent_filters(query, intent or {"business": True})
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    params = {
        "engine": "google",
        "q": q,
        "num": max_results,
        "api_key": SERPAPI_API_KEY,
        "hl": reg["hl"],
        "gl": reg["gl"],
        "google_domain": reg["google_domain"],
        "safe": "active",
    }
    if reg.get("cr"): params["cr"] = reg["cr"]        # country restrict
    if reg.get("lr"): params["lr"] = reg["lr"]        # language restrict
    if region in REGION_EXTRAS:
        params["location"] = REGION_EXTRAS[region]["tokens"][0]
    logger.info(f"SerpAPI search: '{q}' region={region} domain={reg['google_domain']} cr={params.get('cr')} lr={params.get('lr')}")
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 429:
            logger.warning("SerpAPI 429 Too Many Requests — skipping this batch")
            return []
        r.raise_for_status()
        data = r.json()
        urls = [it.get("link") for it in data.get("organic_results", []) if it.get("link")]
        for block in ("inline_results","local_results","shopping_results","top_stories"):
            for it in data.get(block, []) or []:
                link = it.get("link") or it.get("source")
                if link: urls.append(link)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if not is_hard_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.warning(f"SerpAPI error: {e}")
        return []

# ======================= Query expansion =======================
def apply_geo_bias(queries: List[str], region: str) -> List[str]:
    hint = REGION_EXTRAS.get(region)
    cfg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    if not hint:
        return queries
    tld = hint.get("tld", cfg.get("tld", "com"))
    tokens = hint.get("tokens", [])
    out = []
    for i, q in enumerate(queries):
        if i == 0:
            out.append(f"{q} {' '.join(tokens)}")
        elif i == 1:
            out.append(f"{q} site:.{tld}")
        elif i == 2:
            out.append(f"{q} site:*.{tld}")
        else:
            out.append(q)
    return out

def fallback_expand_queries(user_prompt: str, intent: Dict[str,bool], region: str = "wt-wt") -> List[str]:
    base = user_prompt.strip()
    qs = [base]
    casino_aff = intent.get("affiliate") and intent.get("casino")
    if casino_aff:
        core = [
            "casino affiliate marketing agency",
            "igaming affiliate management company",
            "casino affiliate program management",
            "igaming affiliate agency services",
            "outsourced affiliate program management igaming",
            "casino affiliate network management",
            "igaming growth agency affiliates",
            "casino performance marketing agency affiliates",
            "casino affiliate platform for operators",
            "partner program setup for casino",
        ]
        local = [
            "агентство по аффилиат маркетингу казино",
            "управление партнерской программой казино",
            "игейминг аффилиат агентство",
        ]
        if region in REGION_EXTRAS:
            loc = REGION_EXTRAS[region]
            tkns = loc.get("tokens", [])
            tld = loc.get("tld", "")
            cities = " ".join(tkns[:2]) if tkns else ""
            local += [
                f"casino affiliate agency {cities}".strip(),
                f"affiliate marketing agency site:.{tld}".strip() if tld else "",
                f"igaming affiliate agency {cities}".strip(),
            ]
        qs += [x for x in core + local if x]
    else:
        qs += [
            f"{base} agency", f"{base} services", f"{base} company", f"{base} companies",
            f"{base} directory", f"{base} providers",
        ]
    return list(dict.fromkeys(qs))

def generate_search_queries(user_prompt: str, region="wt-wt") -> Tuple[List[str], List[str], str, Dict[str,bool]]:
    if region not in REGION_MAP:
        logger.warning(f"Invalid region {region}, defaulting to wt-wt")
        region = "wt-wt"
    intent = detect_intent(user_prompt)
    queries: List[str] = []
    if GEMINI_API_KEY:
        try:
            body = {
                "system_instruction": {"parts": [{"text":
                    "Ты помощник-поисковик. Верни JSON-массив из 10-14 коротких поисковых фраз по теме. "
                    "Если тема — аффилиатки/казино — добавляй agency/opm/management. Только массив строк."
                }]},
                "contents": [{"parts": [{"text": f"Запрос: {user_prompt}\nРегион: {region}"}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512, "responseMimeType": "application/json"}
            }
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            r = requests.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = ""
            try:
                cands = data.get("candidates", [])
                if cands and "content" in cands[0]:
                    parts = cands[0]["content"].get("parts", [])
                    text = "".join(p.get("text","") for p in parts if "text" in p)
            except Exception:
                pass
            arr = []
            try:
                txt = clean_text(text)
                arr = json.loads(txt) if txt else []
            except Exception:
                arr = []
            if isinstance(arr, list) and arr:
                queries = [clean_text(str(x)) for x in arr if isinstance(x,(str,int,float))][:14]
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
    if not queries:
        queries = fallback_expand_queries(user_prompt, intent, region)
    queries = apply_geo_bias(queries, region)
    logger.info(f"Region applied: {region}; geo-bias tld=.{REGION_MAP.get(region,{}).get('tld','com')}; queries={len(queries)}")
    return queries, queries, region, intent

# ======================= Ranking & Extraction =======================
AGENCY_TOKENS_TEXT = {
    "agency","агентство","services","услуги","clients","клиенты","cases","кейсы",
    "affiliate management","opm","partner program","igaming","casino affiliate","acquisition",
    "performance marketing","traffic","media buying","рост","growth","agency partners","managers",
    "solutions","partners","platform","networks"
}

SOCIAL_OK = {
    "linkedin.com","www.linkedin.com",
    "instagram.com","www.instagram.com",
    "facebook.com","www.facebook.com",
    "x.com","twitter.com","www.twitter.com"
}

def is_company_like(url: str, text: str) -> bool:
    u = (url or "").lower(); t = (text or "").lower()
    dom = domain_of(u)
    if dom in SOCIAL_OK:
        strong = ["affiliate","igaming","casino","agency","partners","network","опм","агентство","аффилиат","партнерская","partner program"]
        if any(tok in t for tok in strong) or any(tok in u for tok in strong):
            return True
    if any(tok in u for tok in COMPANY_URL_TOKENS): return True
    if any(tok in u for tok in ["/about","/services","/partners","/program","/contact"]): return True
    if any(tok in t for tok in AGENCY_TOKENS_TEXT): return True
    return False

def looks_like_blog_like(url: str) -> bool:
    u = (url or "").lower()
    return any(tok in u for tok in BLOG_HINTS_URL)

def looks_like_job_like(url: str) -> bool:
    u = (url or "").lower()
    return any(tok in u for tok in JOBS_HINTS_URL)

def local_geo_signals(url: str, name: str, desc: str, region: str) -> bool:
    extras = REGION_EXTRAS.get(region, {})
    if not extras: return False
    tokens = extras.get("tokens", [])
    org = extras.get("org_tokens", [])
    dom = domain_of(url)
    tld = extras.get("tld")
    # домены .kz | .com.kz и т.п.
    if tld:
        if dom.endswith("." + tld) or dom.endswith(".com." + tld) or host_tld(dom) in {tld, "com."+tld, "net."+tld, "org."+tld}:
            return True
    text = f"{name} {desc}".lower()
    if contains_any(text, tokens): return True
    if contains_any(text, org): return True
    # путь с городами
    path = (urlparse(url).path or "").lower()
    if contains_any(path, [x.lower() for x in tokens]): return True
    # соцсети с локальными упоминаниями
    if domain_of(url) in SOCIAL_OK and contains_any(text, tokens): return True
    return False

def is_junk_domain(domain: str) -> bool:
    d = domain.lower()
    base = d[4:] if d.startswith("www.") else d
    return base in SOFT_BAD_DOMAINS or base in KNOWLEDGE_DOMAINS

def has_trash_path(url: str) -> bool:
    u = (url or "").lower()
    return any(tok in u for tok in TRASH_PATH_TOKENS)

def rank_result(description: str, phrases: List[str], url: Optional[str], region: str, boosts: Dict[str,float]) -> float:
    d = (description or "").lower()
    score = 0.0
    for p in phrases:
        p = p.lower()
        if p and p in d: score += 0.22
        for w in p.split():
            if len(w) > 3 and w in d: score += 0.10

    if url:
        u = url.lower()
        dom = domain_of(url)

        if is_junk_domain(dom): score -= 0.4
        if has_trash_path(u): score -= 0.25

        if is_company_like(u, d): score += 0.42
        if looks_like_blog_like(u): score -= 0.22
        if looks_like_job_like(u): score -= 0.60

        score += boosts.get(dom, 0.0)
        score -= domain_penalty(dom)

        if any(u.endswith(t) for t in CASINO_TLDS): score -= 0.25

        # Локальные сигналы
        if local_geo_signals(url, "", d, region):
            score += 0.28

    score = max(0.0, min(score, 1.0))
    return score

def extract_basic(soup: BeautifulSoup) -> Dict[str,str]:
    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    name = title
    meta_name = soup.select_one("meta[property='og:site_name']") or soup.select_one("meta[name='application-name']")
    if meta_name and meta_name.get("content"):
        name = meta_name["content"].strip() or name
    meta_desc = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
    description = ""
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()
    if not description:
        p = soup.find("p")
        if p: description = p.get_text(" ").strip()
    return {"name": short(name, 120) or "N/A", "description": clean_description(description) or "N/A"}

def fetch_one(url: str) -> Optional[Dict[str,str]]:
    try:
        if not url or is_hard_bad_domain(domain_of(url)): return None
        headers = {"User-Agent": DEFAULT_UA, "Accept-Language": "en,ru;q=0.9"}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400: return None
        ct = (resp.headers.get("Content-Type","") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None
        soup = BeautifulSoup(resp.content, "html.parser")
        base = extract_basic(soup)
        text = (base["name"] + " " + base["description"]).lower()
        if looks_like_sports_trash(text): return None
        return {"url": url, "name": base["name"], "description": base["description"]}
    except Exception:
        return None

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
                    f.write(f"Описание: {short(row[3] or '', 200)}\n")
                    f.write(f"Оценка: {row[9]:.2f}\n")
                    f.write("-"*60 + "\n")
        logger.info("CSV/TXT exported")
    except Exception as e:
        logger.error(f"Export failed: {e}")

# ======================= Search pipeline =======================
def collect_urls(web_queries: List[str], region: str, intent: Dict[str,bool], engine: str, force_ru_ddg: bool, per_query_k: int) -> List[str]:
    all_urls: List[str] = []
    for q in web_queries:
        all_urls.extend(duckduckgo_search(q, max_results=per_query_k, region=region, intent=intent, force_ru_ddg=force_ru_ddg))
        if engine in ("serpapi","both") and SERPAPI_API_KEY:
            all_urls.extend(serpapi_search(q, max_results=per_query_k, region=region, intent=intent))
    # дедуп + блокировки
    deduped = []
    seen = set()
    for u in all_urls:
        if not u or u in seen: continue
        d = domain_of(u)
        if is_hard_bad_domain(d): continue
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(d): continue
        deduped.append(u); seen.add(u)
    return deduped

def scrape_parallel(urls: List[str], phrases: List[str], region: str, boosts: Dict[str,float]) -> List[Dict[str,Any]]:
    local_results: List[Dict[str,Any]] = []
    foreign_results: List[Dict[str,Any]] = []
    maybe_results: List[Tuple[float, Dict[str,Any]]] = []

    # per host limit
    per_host_counter: Dict[str,int] = {}
    filtered_urls = []
    for u in urls:
        dom = domain_of(u)
        if per_host_counter.get(dom, 0) >= PER_HOST_LIMIT:
            continue
        per_host_counter[dom] = per_host_counter.get(dom, 0) + 1
        filtered_urls.append(u)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut2url = {ex.submit(fetch_one, u): u for u in filtered_urls}
        for fut in as_completed(fut2url):
            item = fut.result()
            if not item: continue
            url = item["url"]; name = item["name"]; desc = item["description"]
            dom = domain_of(url).lower()
            u = url.lower()

            # быстрые отсечки по доменам/путям/языку (в т.ч. чтобы выкинуть Edge JP / Minecraft и т.п.)
            if is_junk_domain(dom) or has_trash_path(u):
                continue
            if NONLOCAL_DROP_CJK and has_cjk(name + " " + desc):
                continue

            sc = rank_result(desc, phrases, url, region, boosts)

            # мягкая отсечка списков/обзоров
            is_dir_like = any(tok in u for tok in ["directory","/list","/listing","/rank","/top","/best","/reviews","/review","/rating","/comparison","/compare"])
            is_blog = looks_like_blog_like(u)
            is_job = looks_like_job_like(u)
            if is_job:
                continue
            if is_dir_like and sc < 0.62:  # каталоги сильно режем
                continue
            if is_blog and sc < 0.58:
                continue
            if looks_like_sports_trash(name + " " + desc):
                continue

            row = {
                "id": str(uuid.uuid4()),
                "name": name or "N/A",
                "website": url,
                "description": desc or "N/A",
                "country": "N/A",
                "source": "Web",
                "score": sc
            }

            # гео-классификация
            is_local = local_geo_signals(url, name, desc, region)
            if STRICT_GEO:
                if is_local:
                    local_results.append(row)
                else:
                    # иностранщину складываем отдельно, но сохраним как "maybe" для добора
                    maybe_results.append((sc, row))
            else:
                (local_results if is_local else foreign_results).append(row)

    # сортировка
    local_results.sort(key=lambda x: x["score"], reverse=True)
    foreign_results.sort(key=lambda x: x["score"], reverse=True)
    maybe_results.sort(key=lambda t: t[0], reverse=True)

    # если strict_geo — сначала берём локальные, затем дозаполняем иностранными по рейтингу
    results: List[Dict[str,Any]] = []
    results.extend(local_results[:MAX_RESULTS])

    if len(results) < MIN_RESULTS:
        # добор: сначала из maybe (иностранные, но топовые), затем — foreign_results (если STRICT_GEO=false)
        need = MIN_RESULTS - len(results)
        for _, r in maybe_results:
            results.append(r)
            if len(results) >= MIN_RESULTS: break

    # если STRICT_GEO=false — можно добавить foreign в пределах MAX_RESULTS и FOREIGN_FALLBACK_RATIO
    if not STRICT_GEO and len(results) < MAX_RESULTS and foreign_results:
        quota = int(MAX_RESULTS * FOREIGN_FALLBACK_RATIO)
        results.extend(foreign_results[:max(0, quota)])

    # финальный срез
    results = results[:MAX_RESULTS]
    logger.info(f"Geo select: local={len(local_results)}; foreign_maybe={len(maybe_results)}; final={len(results)} (strict_geo={STRICT_GEO})")
    return results

def ensure_minimum(results: List[Dict[str,Any]], target_min: int) -> List[Dict[str,Any]]:
    return results

# ======================= API =======================
@app.route("/search", methods=["POST", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.json or {}
        user_query = data.get("query", "").strip()
        region = data.get("region", "wt-wt")
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE","both")).lower()
        per_query_k = int(data.get("per_query", 20))

        if not user_query:
            return jsonify({"error":"Query is required"}), 400

        logger.info(f"API request: query='{user_query}', region={region}, engine={engine}, strict_geo={STRICT_GEO}")

        # очистка текущей таблицы результатов
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM results")
            conn.commit()

        # генерим подзапросы
        web_queries, phrases, region, intent = generate_search_queries(user_query, region)

        # пишем запрос
        query_id = insert_query_record(user_query, intent, region)

        # история бустов/штрафов
        boosts = history_boosts(intent, region)

        # если кириллица и регион дефолтный — используем ru-ru для DDG
        force_ru_ddg = is_cyrillic(user_query) and region == "wt-wt"
        logger.info(f"force_ru_ddg={force_ru_ddg} (query_has_cyrillic={is_cyrillic(user_query)})")

        # собираем URL’ы
        urls = collect_urls(web_queries, region, intent, engine, force_ru_ddg, per_query_k=10)
        logger.info(f"Pre-filtered URLs (unique & not blocked): {len(urls)}")

        # параллельный скрейп
        t0 = time.time()
        web_results = scrape_parallel(urls[:300], phrases, region, boosts)
        dt = time.time() - t0
        logger.info(f"Scraped {len(web_results)} results in {dt:.2f}s")

        # минимум/максимум
        web_results = web_results[:MAX_RESULTS]
        web_results = ensure_minimum(web_results, MIN_RESULTS)

        # запись в БД + экспорт
        for r in web_results:
            save_result_records(r, intent, region, query_id)
        if web_results:
            save_to_csv_txt()

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
