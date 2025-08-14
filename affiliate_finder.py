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
from typing import List, Tuple, Dict, Any

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ======================= Helpers: text cleaning =======================
CODE_FENCE_RE = re.compile(r"^```(?:json|js|python|txt)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = CODE_FENCE_RE.sub("", t)
    t = t.strip(" \n\t\r\"'`")
    while t and (t[0] in "[{" and t[-1] in "]}"):
        t = t[1:-1].strip()
    return t

def clean_description(description):
    if not description:
        return "N/A"
    soup = BeautifulSoup(str(description), "html.parser")
    text = soup.get_text(" ").strip()
    return " ".join(text.split()[:200])

def domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""

# ======================= Optional HF classifier =======================
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ========= Feature flags / API keys =========
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # укажи в Railway

# ========= Flask app + static frontend =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend_dist")

app = Flask(
    __name__,
    static_folder=FRONTEND_DIST,
    static_url_path="/"
)

# ========= CORS =========
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

# ========= Logging =========
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("scraper.log", encoding="utf-8") if LOG_TO_FILE else logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ========= Limits / pauses =========
REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1000"))
REQUEST_PAUSE_MIN = float(os.getenv("REQUEST_PAUSE_MIN", "0.8"))
REQUEST_PAUSE_MAX = float(os.getenv("REQUEST_PAUSE_MAX", "1.6"))

# ========= Search engines & helpers =========
from ddgs import DDGS

REGION_MAP = {
    "wt-wt": {"hl": "en", "gl": "us"},
    "us-en": {"hl": "en", "gl": "us"},
    "uk-en": {"hl": "en", "gl": "gb"},
    "de-de": {"hl": "de", "gl": "de"},
    "fr-fr": {"hl": "fr", "gl": "fr"},
    "ru-ru": {"hl": "ru", "gl": "ru"},
    "ua-ua": {"hl": "uk", "gl": "ua"},
    "kz-ru": {"hl": "ru", "gl": "kz"},
    "kz-kk": {"hl": "kk", "gl": "kz"},
    "by-ru": {"hl": "ru", "gl": "by"},
    "uz-ru": {"hl": "ru", "gl": "uz"},
    "az-ru": {"hl": "ru", "gl": "az"},
    "kg-ru": {"hl": "ru", "gl": "kg"},
    "am-ru": {"hl": "ru", "gl": "am"},
    "ge-ka": {"hl": "ka", "gl": "ge"},
    "pl-pl": {"hl": "pl", "gl": "pl"},
    "es-es": {"hl": "es", "gl": "es"},
    "it-it": {"hl": "it", "gl": "it"},
    "nl-nl": {"hl": "nl", "gl": "nl"},
    "se-sv": {"hl": "sv", "gl": "se"},
    "no-no": {"hl": "no", "gl": "no"},
    "fi-fi": {"hl": "fi", "gl": "fi"},
    "cz-cs": {"hl": "cs", "gl": "cz"},
    "sk-sk": {"hl": "sk", "gl": "sk"},
    "ro-ro": {"hl": "ro", "gl": "ro"},
    "hu-hu": {"hl": "hu", "gl": "hu"},
    "ch-de": {"hl": "de", "gl": "ch"},
    "tr-tr": {"hl": "tr", "gl": "tr"},
    "ae-en": {"hl": "en", "gl": "ae"},
    "in-en": {"hl": "en", "gl": "in"},
    "id-id": {"hl": "id", "gl": "id"},
    "ph-en": {"hl": "en", "gl": "ph"},
    "sg-en": {"hl": "en", "gl": "sg"},
    "my-ms": {"hl": "ms", "gl": "my"},
    "th-th": {"hl": "th", "gl": "th"},
    "vn-vi": {"hl": "vi", "gl": "vn"},
    "jp-ja": {"hl": "ja", "gl": "jp"},
    "kr-ko": {"hl": "ko", "gl": "kr"},
    "cn-zh": {"hl": "zh", "gl": "cn"},
    "ca-en": {"hl": "en", "gl": "ca"},
    "br-pt": {"hl": "pt", "gl": "br"},
    "mx-es": {"hl": "es", "gl": "mx"},
    "au-en": {"hl": "en", "gl": "au"},
    "nz-en": {"hl": "en", "gl": "nz"},
}

# ========= Proxy config =========
PROXY_CACHE_FILE = "proxies.json"
PROXY_API_URL = "https://www.proxy-list.download/api/v1/get?type=https&anon=elite"
MAX_PROXY_ATTEMPTS = int(os.getenv("MAX_PROXY_ATTEMPTS", "2"))

# ========= Classifier (optional) =========
def init_classifier():
    if pipeline is None:
        logger.warning("transformers pipeline not available; classifier disabled")
        return None
    model_main = os.getenv("CLASSIFIER_MODEL", "facebook/bart-large-mnli")
    model_fallback = os.getenv("CLASSIFIER_FALLBACK", "distilbert-base-uncased")
    device = int(os.getenv("CLASSIFIER_DEVICE", "-1"))
    try:
        clf = pipeline("zero-shot-classification", model=model_main, device=device)
        logger.info(f"Classifier initialized: {model_main} (device={device})")
        return clf
    except Exception as e:
        logger.warning(f"Failed to init {model_main}: {e}. Falling back to {model_fallback}")
        try:
            clf = pipeline("zero-shot-classification", model=model_fallback, device=device)
            logger.info(f"Fallback classifier initialized: {model_fallback} (device={device})")
            return clf
        except Exception as e2:
            logger.error(f"Failed to init fallback classifier: {e2}")
            return None

classifier = init_classifier()

# ========= DB =========
DB_PATH = "search_results.db"

def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()

            # --- основные таблицы ---
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
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}")

# ВАЖНО: под gunicorn блок __main__ не срабатывает — инициируем БД при импорте
init_db()

# ========= Utility: counters & proxies =========
def load_request_count():
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

def save_request_count(count):
    try:
        with open(REQUEST_COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump({"count": count, "last_reset": datetime.now().strftime("%Y-%m-%d")}, f)
    except Exception as e:
        logger.error(f"Failed to save request count: {e}")

def load_proxies():
    try:
        with open(PROXY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_proxies(proxies):
    try:
        with open(PROXY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(proxies, f)
    except Exception as e:
        logger.error(f"Failed to save proxies: {e}")

def fetch_free_proxies():
    logger.info("Fetching free proxies from API")
    proxies = []
    try:
        response = requests.get(PROXY_API_URL, timeout=10)
        response.raise_for_status()
        for proxy in response.text.strip().split("\n"):
            if ":" in proxy:
                proxies.append({"https": f"https://{proxy}"})
        save_proxies(proxies)
        logger.info(f"Fetched {len(proxies)} proxies")
        return proxies
    except Exception as e:
        logger.error(f"Failed to fetch proxies: {e}")
        return []

def get_proxy():
    proxies = load_proxies()
    if not proxies:
        proxies = fetch_free_proxies()
    return random.choice(proxies) if proxies else None

# ========= Trash filters =========
BAD_DOMAINS = {
    "zhihu.com","baidu.com","commentcamarche.net","google.com","d4drivers.uk","dvla.gov.uk",
    "youtube.com","reddit.com","affpapa.com","getlasso.co","wiktionary.org","rezka.ag",
    "linguee.com","bab.la","reverso.net","sinonim.org","wordhippo.com","microsoft.com",
    "romeo.com","xnxx.com","hometubeporn.com","porn7.xxx","fuckvideos.xxx","sport.ua",
    "openai.com","community.openai.com","discourse-cdn.com","stackoverflow.com",
}
def is_bad_domain(dom: str) -> bool:
    if not dom:
        return False
    d = dom.lower().lstrip(".")
    if d.endswith("wikipedia.org"):
        return True
    if d.startswith("google.") or d.endswith(".google.com") or d == "google.com":
        return True
    if d.startswith("www."):
        d = d[4:]
    if d in BAD_DOMAINS:
        return True
    return False

KNOWLEDGE_DOMAINS = {
    "wikipedia.org","en.wikipedia.org","ru.wikipedia.org",
    "hubspot.com","coursera.org","ibm.com","sproutsocial.com",
    "digitalmarketinginstitute.com","marketermilk.com","harvard.edu","professional.dce.harvard.edu"
}

SPORTS_TRASH_TOKENS = {
    "футбол","теннис","биатлон","хокей","баскетбол","волейбол","снукер",
    "премьер лига","лига чемпионов","таблица","расписание","тв-программа"
}

def looks_like_sports_garbage(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(tok in t for tok in SPORTS_TRASH_TOKENS)

# ========= Intent detection =========
INTENT_AFFILIATE = {
    "affiliate","аффилиат","аффилиэйт","партнерка","партнёрка","партнерская программа","партнёрская программа",
    "реферальная","referral","cpa","игемблинг","игейминг","igaming","casino affiliate","affiliate network",
    "партнеры казино","партнёры казино","аффилейт","афилейт","аффилированный"
}
INTENT_LEARN = {
    "что такое","what is","определение","definition","гайд","guide","обзор","overview","курс","course","как работает","how to"
}
CASINO_TOKENS = {
    "казино","casino","igaming","гемблинг","игемблинг","игейминг","беттинг","ставки","bookmaker","sportsbook"
}

def detect_intent(q: str) -> Dict[str, bool]:
    t = (q or "").lower()
    affiliate = any(k in t for k in INTENT_AFFILIATE)
    learn = any(k in t for k in INTENT_LEARN)
    casino = any(k in t for k in CASINO_TOKENS)
    return {"affiliate": affiliate, "learn": learn, "casino": casino, "business": not learn}

# ========= Cyrillic helpers =========
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

def is_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))

def maybe_add_ru_site_dupes(queries: List[str]) -> List[str]:
    add = []
    for q in queries:
        add.append(q)
        add.append(f"{q} site:.ru")
        add.append(f"{q} site:.ua")
        add.append(f"{q} site:.by")
    seen = set()
    out = []
    for x in add:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ========= iGaming-specific classifiers & tokens =========
CASINO_OPERATOR_TOKENS = {
    "play now","играть","играйте","получить бонус","бонус","депозит","cashback","slots","слоты",
    "пополнение","вывод средств","jackpot","free spins","казино обзор","casino review","лучшие казино",
    "sportsbook","ставки на спорт","букмекер"
}
CASINO_TLDS = (".casino",".bet",".betting",".poker",".slots",".bingo")

LISTING_TOKENS = {"рейтинг","топ","лучшие","каталог","список","обзор","обзоры","reviews","rating","top","best"}

JOB_TOKENS = {"вакансии","вакансия","работа","удаленно","зарплата","hh.ru","job","jobs","career","indeed","jooble","work"}

AGENCY_TOKENS_URL = {"agency","management","opm","consulting","services","solutions","partners","company"}
AGENCY_TOKENS_TEXT = {
    "agency","агентство","услуги","services","clients","клиенты","cases","кейсы",
    "program management","affiliate management","opm","partner program","igaming","casino affiliate"
}

def looks_like_casino_operator(text: str, url: str) -> bool:
    u = (url or "").lower()
    if u.endswith(CASINO_TLDS) or any(t in u for t in ["/casino","/slots","/bonuses","/bonus"]):
        return True
    t = (text or "").lower()
    return any(tok in t for tok in CASINO_OPERATOR_TOKENS)

def looks_like_listing(text: str, url: str) -> bool:
    u = (url or "").lower()
    if any(x in u for x in ["/top","/rating","/ratings","/best","/reviews","/review","/rank"]):
        return True
    t = (text or "").lower()
    return any(tok in t for tok in LISTING_TOKENS)

def looks_like_job_board(text: str, url: str) -> bool:
    u = (url or "").lower()
    if any(x in u for x in ["/jobs","/job","/careers","/vacancies","/vacancy","/career","/rabota","/vakansii"]):
        return True
    t = (text or "").lower()
    return any(tok in t for tok in JOB_TOKENS)

def looks_like_agency(text: str, url: str) -> bool:
    u = (url or "").lower()
    t = (text or "").lower()
    url_hit = any(tok in u for tok in AGENCY_TOKENS_URL)
    text_hit = any(tok in t for tok in AGENCY_TOKENS_TEXT)
    return url_hit or text_hit

# ========= History helpers (learning) =========
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

def cache_get(key: str, max_age_hours: int = 12) -> Any:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT data, created_at FROM gemini_cache WHERE key=?", (key,))
            row = c.fetchone()
            if not row:
                return None
            data, created_at = row
            try:
                if datetime.utcnow() - datetime.fromisoformat(created_at) > timedelta(hours=max_age_hours):
                    return None
            except Exception:
                return None
            return json.loads(data)
    except Exception:
        return None

def cache_set(key: str, data: Any):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO gemini_cache (key, data, created_at) VALUES (?, ?, ?)",
                      (key, json.dumps(data, ensure_ascii=False), datetime.utcnow().isoformat()))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to write cache: {e}")

# === NEW: penalty helpers (domain/user feedback → scoring/filters) ===
NEG_BAD_THRESHOLD = int(os.getenv("NEG_BAD_THRESHOLD", "3"))  # сколько «bad/flag» нужно, чтобы домен попал в блок-лист
NEG_BAD_WINDOW_DAYS = int(os.getenv("NEG_BAD_WINDOW_DAYS", "60"))  # окно в днях
NEG_BAD_SCORE_PENALTY = float(os.getenv("NEG_BAD_SCORE_PENALTY", "0.5"))  # штраф к score для домена с bad'ами

def domain_penalty(domain: str) -> float:
    """Вернёт штраф для домена на основе недавнего негативного фидбэка."""
    if not domain:
        return 0.0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=NEG_BAD_WINDOW_DAYS)).isoformat()
            c.execute("""
                SELECT COUNT(*) FROM interactions
                WHERE domain = ? AND action = 'bad' AND created_at >= ?
            """, (domain, since))
            cnt = c.fetchone()[0] or 0
            if cnt >= NEG_BAD_THRESHOLD:
                return NEG_BAD_SCORE_PENALTY
            # мягкий штраф, растущий с количеством bad
            return min(cnt * (NEG_BAD_SCORE_PENALTY / NEG_BAD_THRESHOLD), NEG_BAD_SCORE_PENALTY)
    except Exception as e:
        logger.warning(f"domain_penalty failed for {domain}: {e}")
        return 0.0

def domain_is_blocked(domain: str) -> bool:
    """Жёсткое исключение домена из выдачи при превышении порога bad/flag за период."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            since = (datetime.utcnow() - timedelta(days=NEG_BAD_WINDOW_DAYS)).isoformat()
            c.execute("""
                SELECT COUNT(*) FROM interactions
                WHERE domain = ? AND action = 'bad' AND created_at >= ?
            """, (domain, since))
            cnt = c.fetchone()[0] or 0
            return cnt >= NEG_BAD_THRESHOLD
    except Exception as e:
        logger.warning(f"domain_is_blocked failed for {domain}: {e}")
        return False

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
                boosts[domain] = boosts.get(domain, 0.0) + min(0.05 * (1 + (cnt ** 0.5)), 0.2)

            c.execute("""
                SELECT domain,
                       SUM(CASE WHEN action='good' THEN weight WHEN action='click' THEN weight*0.5 ELSE 0 END) as pos,
                       SUM(CASE WHEN action='bad' THEN -weight ELSE 0 END) as neg
                FROM interactions
                GROUP BY domain
            """)
            for domain, pos, neg in c.fetchall():
                p = pos or 0.0
                n = neg or 0.0
                adj = min(0.15 * (1 + (p ** 0.5)) - 0.1 * (1 + (n ** 0.5)), 0.3)
                # учитываем динамический штраф
                pen = domain_penalty(domain)
                boosts[domain] = max(min(boosts.get(domain, 0.0) + adj - pen, 0.3), -0.5)
    except Exception as e:
        logger.error(f"history_boosts error: {e}")
    return boosts

# ========= Intent-agnostic analysis =========
COMPANY_URL_TOKENS = [
    "agency","network","partners","program","programs","platform","services","solutions",
    "management","consulting","company","about","contact","affiliates"
]

def is_relevant_url(url, prompt_phrases):
    """Строгий URL-фильтр: нужен агентский ключ в URL."""
    if is_bad_domain(domain_of(url)):
        return False
    u = (url or "").lower()
    return any(tok in u for tok in ["agency","management","opm","consulting","services","solutions","partners","company"])

GEO_HINTS: Dict[str, Dict[str, Any]] = {
    "kz-ru": {"tld": "kz", "tokens": ["Казахстан", "KZ", "Алматы", "Астана", "Astana", "Almaty"]},
    "kz-kk": {"tld": "kz", "tokens": ["Қазақстан", "KZ", "Алматы", "Астана"]},
    "by-ru": {"tld": "by", "tokens": ["Беларусь", "BY", "Минск"]},
    "ua-ua": {"tld": "ua", "tokens": ["Україна", "Украина", "UA", "Київ", "Киев", "Львів", "Одеса"]},
    "ru-ru": {"tld": "ru", "tokens": ["Россия", "RF", "Москва", "Санкт-Петербург"]},
    "pl-pl": {"tld": "pl", "tokens": ["Polska", "Poland", "Warszawa"]},
    "de-de": {"tld": "de", "tokens": ["Deutschland", "Berlin", "München"]},
    "fr-fr": {"tld": "fr", "tokens": ["France", "Paris"]},
    "tr-tr": {"tld": "tr", "tokens": ["Türkiye", "Istanbul", "Ankara"]},
    "ae-en": {"tld": "ae", "tokens": ["UAE", "Dubai", "Abu Dhabi"]},
    "in-en": {"tld": "in", "tokens": ["India", "Delhi", "Mumbai"]},
    "sg-en": {"tld": "sg", "tokens": ["Singapore", "SG"]},
    "jp-ja": {"tld": "jp", "tokens": ["日本", "東京", "Tokyo"]},
    "kr-ko": {"tld": "kr", "tokens": ["대한민국", "서울", "Seoul"]},
    "es-es": {"tld": "es", "tokens": ["España", "Madrid", "Barcelona"]},
    "it-it": {"tld": "it", "tokens": ["Italia", "Roma", "Milano"]},
    "nl-nl": {"tld": "nl", "tokens": ["Nederland", "Amsterdam"]},
    "se-sv": {"tld": "se", "tokens": ["Sverige", "Stockholm"]},
    "no-no": {"tld": "no", "tokens": ["Norge", "Oslo"]},
    "fi-fi": {"tld": "fi", "tokens": ["Suomi", "Helsinki"]},
    "cz-cs": {"tld": "cz", "tokens": ["Česko", "Praha"]},
    "sk-sk": {"tld": "sk", "tokens": ["Slovensko", "Bratislava"]},
    "ro-ro": {"tld": "ro", "tokens": ["România", "București"]},
    "hu-hu": {"tld": "hu", "tokens": ["Magyarország", "Budapest"]},
    "ch-de": {"tld": "ch", "tokens": ["Schweiz", "Zürich"]},
    "us-en": {"tld": "us", "tokens": ["USA", "United States"]},
    "uk-en": {"tld": "uk", "tokens": ["United Kingdom", "London"]},
    "br-pt": {"tld": "br", "tokens": ["Brasil", "São Paulo"]},
    "mx-es": {"tld": "mx", "tokens": ["México", "CDMX"]},
    "au-en": {"tld": "au", "tokens": ["Australia", "Sydney", "Melbourne"]},
    "nz-en": {"tld": "nz", "tokens": ["New Zealand", "Auckland"]},
}

def apply_geo_bias(queries: List[str], region: str) -> List[str]:
    hint = GEO_HINTS.get(region)
    if not hint:
        return queries
    tld = hint["tld"]
    tokens = hint["tokens"]
    biased = []
    for i, q in enumerate(queries):
        if i == 0:
            biased.append(f"{q} {' '.join(tokens[:2])}".strip())
        elif i == 1:
            biased.append(f"{q} site:.{tld}")
        else:
            biased.append(q)
    return biased

def geo_score_boost(url: str, text: str, region: str) -> float:
    hint = GEO_HINTS.get(region)
    if not hint:
        return 0.0
    boost = 0.0
    dom = domain_of(url)
    if dom.endswith(f".{hint['tld']}"):
        boost += 0.35
    t = (text or "").lower()
    if any(tok.lower() in t for tok in hint["tokens"]):
        boost += 0.15
    return min(boost, 0.5)

def rank_result(description: str, prompt_phrases: List[str], url: str = None, region: str = None, boosts: Dict[str,float]=None) -> float:
    score = 0.0
    d = (description or "").lower()
    words = [w.lower() for p in prompt_phrases for w in p.split() if len(w) > 3]
    for w in words:
        if w in d:
            score += 0.25 if len(w) > 6 else 0.15
    for p in prompt_phrases:
        if p.lower() in d:
            score += 0.3

    if url:
        uu = url.lower()
        if any(tok in uu for tok in COMPANY_URL_TOKENS):
            score += 0.25
        if looks_like_agency(d, uu):
            score += 0.35
        if boosts:
            score += boosts.get(domain_of(url), 0.0)

        # NEW: domain-level penalty from feedback
        pen = domain_penalty(domain_of(url))
        if pen:
            score -= pen

    if url and looks_like_blog(url, d):
        score -= 0.25
    if looks_like_listing(d, url or ""):
        score -= 0.5
    if looks_like_job_board(d, url or ""):
        score -= 0.7
    if looks_like_casino_operator(d, url or ""):
        score -= 0.8
    if looks_like_sports_garbage(d):
        score = min(score, 0.1)

    if url and region:
        score += geo_score_boost(url, d, region)

    return max(0.0, min(score, 1.0))

def analyze_result(description, prompt_phrases):
    specialization = ", ".join(prompt_phrases[:2]).title() if prompt_phrases else "General"
    cleaned = clean_description(description).lower()
    words = [w.lower() for p in prompt_phrases for w in p.split() if len(w) > 3]
    if classifier is None:
        status = "Active" if any(w in cleaned for w in words) else "Unknown"
        return True, specialization, status, f"Подходит: Связан с {specialization}"
    labels = [p.title() for p in prompt_phrases[:3]] + ["Social Media", "Other"] or ["Relevant", "Other"]
    try:
        result = classifier(cleaned, candidate_labels=labels, multi_label=False)
        is_rel = (result["labels"][0] != "Other") or any(w in cleaned for w in words) or any(p.lower() in cleaned for p in prompt_phrases)
        status = "Active" if any(w in cleaned for w in words) else "Unknown"
        suitability = f"Подходит: Соответствует {specialization}" if is_rel else f"Частично подходит: Связан с {specialization}"
        return is_rel, specialization, status, suitability
    except Exception as e:
        logger.warning(f"Classifier analysis failed: {e}, saving anyway")
        status = "Active" if any(w in cleaned for w in words) else "Unknown"
        return True, specialization, status, f"Подходит: Связан с {specialization}"

# ========= Query generation with Gemini + fallback =========
def _extract_json_array_maybe(text: str) -> List[str]:
    if not text:
        return []
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    m = re.search(r"$begin:math:display$[\\s\\S]*$end:math:display$", text)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, list):
                return val
        except Exception:
            return []
    return []

def gemini_expand_queries(user_prompt: str, region: str, intent: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    system_instruction = (
        "Ты помощник для поиска. Преобразуй запрос пользователя в массив из 8–12 КОРОТКИХ поисковых запросов по теме. "
        "Если запрос НЕ про аффилиатки/партнёрские программы — НЕ добавляй слова 'affiliate', 'CPA', 'referral'. "
        "Если запрос про аффилиатки — добавь релевантные англоязычные термины (agency, network, program, platform, services). "
        "Ответи строго JSON-массивом строк без комментариев."
    )
    guard = "AFFILIATE=YES" if intent.get("affiliate") else "AFFILIATE=NO"
    user_text = f"{guard}\nЗапрос: {user_prompt}\nРегион: {region}\nФормат ответа: JSON массив строк."

    cache_key = f"gemini:{region}:{json.dumps(intent, sort_keys=True, ensure_ascii=False)}:{user_prompt}".strip()
    cached = cache_get(cache_key, max_age_hours=12)
    if cached:
        return cached[:12], cached[:12]

    body = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": 512, "responseMimeType": "application/json"}
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    params = {"key": GEMINI_API_KEY}

    r = requests.post(url, params=params, json=body, timeout=20)
    r.raise_for_status()
    data = r.json()

    text = ""
    try:
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
    except Exception:
        pass

    text = clean_text(text)
    arr = _extract_json_array_maybe(text)
    if isinstance(arr, list) and arr:
        arr = [clean_text(str(x)) for x in arr if isinstance(x, (str, int, float))]
        arr = [x for x in arr if x and len(x) <= 200]
        if arr:
            cache_set(cache_key, arr)
            logger.info(f"Using Gemini for query expansion ({len(arr)} variants)")
            return arr[:12], arr[:12]

    raise RuntimeError("Gemini returned invalid JSON")

def fallback_expand_queries(user_prompt: str, region: str, intent: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    base = user_prompt.strip()
    queries: List[str] = [base]
    t = base.lower()
    cyr = is_cyrillic(t)

    if intent.get("affiliate"):
        en_boost = [
            "affiliate programs","partner program application","affiliate marketing","referral program",
            "CPA affiliate","best affiliate networks","affiliate agency","affiliate management agency",
        ]
        if intent.get("casino"):
            en_boost += [
                "casino affiliate programs","iGaming affiliate agency","casino affiliate marketing agency",
                "online casino affiliates","gambling affiliate networks","igaming growth agency",
                "casino affiliate management company","igaming affiliate network","casino affiliate platform services",
                "casino affiliate management agency","igaming affiliate marketing agency",
                "igaming affiliate program management","outsourced program management igaming",
                "affiliate OPM agency gambling","casino affiliate program agency",
                "igaming performance marketing agency affiliates",
                "affiliates program management company igaming",
                "partner program management agency gambling",
            ]
            ru_boost = [
                "аффилейт агентство казино","маркетинговое агентство iGaming","партнерская программа казино",
                "управление аффилиатами казино","агентство по трафику для казино",
                "компания по управлению партнерками казино","igaming партнерские сети",
            ]
            queries += ru_boост
        queries += en_boost
    else:
        queries += [
            f"{base} официальный сайт", f"{base} каталог", f"{base} список", f"{base} компании",
            f"{base} directory", f"{base} companies", f"{base} official site", f"{base} contacts", f"{base} услуги",
        ]

    if cyr and intent.get("affiliate") and intent.get("casino"):
        queries += [
            "affiliate marketing companies for casino","casino marketing affiliate agencies",
            "igaming affiliate marketing companies","casino affiliate management services",
        ]
        queries = maybe_add_ru_site_dupes(queries)

    queries = list(dict.fromkeys([q for q in queries if q]))[:30]
    logger.info(f"Using fallback for query expansion (affiliate={intent.get('affiliate')}, casino={intent.get('casino')}), produced {len(queries)} queries")
    return queries, queries

# ========= Build query with negative filters =========
NEGATIVE_SITES_FOR_BUSINESS = [
    "site:wikipedia.org","site:en.wikipedia.org","site:ru.wikipedia.org",
    "site:hubspot.com","site:coursera.org","site:ibm.com","site:sproutsocial.com",
    "site:digitalmarketinginstitute.com","site:marketermilk.com","site:harvard.edu","site:professional.dce.harvard.edu"
]
NEGATIVE_INURL_FOR_BLOGS = [
    "inurl:blog","inurl:news","inurl:guide","inurl:guides","inurl:insights","inurl:academy",
    "inurl:articles","inurl:article","inurl:glossary","inurl:resources","inurl:resource",
    "inurl:learn","inurl:library","inurl:case-study","inurl:case-studies","inurl:whitepaper","inurl:press"
]
NEGATIVE_INTITLE_FOR_BLOGS = [
    'intitle:"what is"','intitle:"how to"','intitle:"guide"','intitle:"overview"','intitle:"tips"'
]
NEGATIVE_SITES_CASINO = [
    "site:askgamblers.com","site:casino.org","site:bonusfinder.com","site:gambling.com",
    "site:slotsup.com","site:goodluckmate.com","site:kingcasinobonus.uk","site:casinotopsonline.com",
]
NEGATIVE_TOKENS_JOBS_RU = ["-вакансии","-вакансия","-работа","-карьера","-hh.ru","-rabota","-job","-jobs","-career"]
NEGATIVE_TOKENS_RATINGS_RU = ["-рейтинг","-топ","-обзор","-обзоры","-лучшие","-best","-review","-reviews"]
NEGATIVE_TOKENS_CASINO_RU = ["-онлайн","-казино","-слоты","-бонус","-игровые","-игровых","-депозит","-вывод","-casino","-slots","-bonus"]
NEGATIVE_INURL_LISTINGS = ["-inurl:rating","-inurl:ratings","-inurl:reviews","-inurl:top","-inurl:best"]

def with_intent_filters(q: str, intent: Dict[str,bool]) -> str:
    base = q
    if intent.get("business") and (intent.get("affiliate") and intent.get("casino")):
        negatives_sites = " ".join(f"-{s}" for s in NEGATIVE_SITES_FOR_BUSINESS + NEGATIVE_SITES_CASINO)
        negatives_jobs = " ".join(NEGATIVE_TOKENS_JOBS_RU)
        negatives_ratings = " ".join(NEGATIVE_TOKENS_RATINGS_RU)
        negatives_casino = " ".join(NEGATIVE_TOKENS_CASINO_RU)
        negatives_inurl = " ".join(NEGATIVE_INURL_FOR_BLOGS + NEGATIVE_INURL_LISTINGS)
        negatives_intitle = " ".join(NEGATIVE_INTITLE_FOR_BLOGS)
        return f"{base} {negatives_sites} {negatives_jobs} {negatives_ratings} {negatives_casino} {negatives_inurl} {negatives_intitle}".strip()
    elif intent.get("business") and not intent.get("learn"):
        negatives = " ".join(f"-{s}" for s in NEGATIVE_SITES_FOR_BUSINESS[:3])
        return f"{base} {negatives}".strip()
    return base

# ========= Blog/article cut =========
BLOG_URL_TOKENS = [
    "/blog", "/news", "/article", "/articles", "/guides", "/guide", "/insights", "/academy", "/glossary",
    "/resources", "/resource", "/learn", "/library", "/case-study", "/case-studies", "/whitepaper", "/press", "/press-release"
]
BLOG_TEXT_TOKENS_RU = [
    "статья", "советы", "обзор", "гайд", "руководство", "как начать", "что такое", "инсайты", "новости",
    "пример", "кейс", "кейс-стади", "исследование", "белая книга", "пресс-релиз", "учебник", "обучение", "курс"
]
BLOG_TEXT_TOKENS_EN = [
    "article", "guide", "what is", "how to", "tips", "overview", "insights", "news",
    "case study", "case studies", "research", "whitepaper", "press release", "academy", "resource", "library"
]

def looks_like_blog(url: str, text: str) -> bool:
    u = (url or "").lower()
    if any(tok in u for tok in BLOG_URL_TOKENS):
        return True
    t = (text or "").lower()
    return any(tok in t for tok in BLOG_TEXT_TOKENS_RU) or any(tok in t for tok in BLOG_TEXT_TOKENS_EN)

# ========= Search engines (DDG / SerpAPI) =========
def duckduckgo_search(query, max_results=15, region="wt-wt", intent: Dict[str,bool]=None, force_ru_ddg=False):
    data = load_request_count()
    if data["count"] >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        return []
    real_region = "ru-ru" if force_ru_ddg else region
    q = with_intent_filters(query, intent or {"business": True})
    logger.info(f"DDG search: '{q}' region={real_region}")
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(q, region=real_region, safesearch="moderate", timelimit="y", max_results=max_results)
            for r in results:
                href = r.get("href")
                if href:
                    urls.append(href)
        save_request_count(data["count"] + 1)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if not is_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"DDG failed for '{q}': {e}")
        return []

def serpapi_search(query, max_results=15, region="wt-wt", intent: Dict[str,bool]=None):
    if not SERPAPI_API_KEY:
        logger.info("SERPAPI_API_KEY is not set; skipping SerpAPI")
        return []
    q = with_intent_filters(query, intent or {"business": True})
    params = {"engine": "google", "q": q, "num": max_results, "api_key": SERPAPI_API_KEY}
    params.update(REGION_MAP.get(region, REGION_MAP["wt-wt"]))
    logger.info(f"SerpAPI search: '{q}' region={region}")
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        urls = [it.get("link") for it in data.get("organic_results", [])[:max_results] if it.get("link")]
        for it in data.get("inline_results", []):
            if it.get("link"):
                urls.append(it["link"])
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if not is_bad_domain(domain_of(u))]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"SerpAPI failed for '{q}': {e}")
        return []

# ========= Scraper & filters =========
def looks_like_definition_page(text: str, url: str, intent: Dict[str,bool]) -> bool:
    if not intent.get("business") or intent.get("learn"):
        return False
    t = (text or "").lower()
    u = (url or "").lower()
    if any(k in t for k in ["что такое", "what is", "определение", "definition", "гайд", "guide", "курс", "обзор", "overview"]):
        return True
    dom = domain_of(u)
    if dom in KNOWLEDGE_DOMAINS:
        return True
    return False

def should_cut_blog(url: str, text: str, intent: Dict[str,bool]) -> bool:
    if intent.get("business") and intent.get("affiliate") and intent.get("casino"):
        if looks_like_blog(url, text) or looks_like_listing(text, url):
            return True
    return False

def search_and_scrape_websites(urls: List[str], prompt_phrases: List[str], region: str, intent: Dict[str,bool], boosts: Dict[str,float], query_id: str):
    logger.info(f"Starting scrape of {len(urls)} URLs")
    results = []
    urls = list(dict.fromkeys(urls))[:80]
    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Scraping: {url}")
        dom = domain_of(url)
        if is_bad_domain(dom) or domain_is_blocked(dom):  # NEW: домены из блок-листа — сразу мимо
            logger.info(f"Skip bad/blocked domain: {url}")
            continue
        proxy_attempts = 0
        proxies_used = []
        success = False
        while proxy_attempts <= MAX_PROXY_ATTEMPTS:
            proxy = None
            if proxy_attempts > 0:
                proxy = get_proxy()
                if not proxy or proxy in proxies_used:
                    logger.warning(f"No more unique proxies for {url}")
                    break
                proxies_used.append(proxy)
            try:
                headers = {"User-Agent": os.getenv("SCRAPER_UA",
                          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")}
                resp = requests.get(url, headers=headers, timeout=20, proxies=proxy)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, "html.parser")

                name = "N/A"
                el = soup.select_one("h1[class*='brand'], h1[class*='partner'], .company-name, .brand-name, .site-title, .logo-text, title, h1, .header-title")
                if el:
                    name = clean_description(getattr(el, "text", "") or el)

                description = "N/A"
                for sel in [
                    ".description, .about, .content, .intro, div[class*='about'], section[class*='program'], "
                    "p[class*='description'], meta[name='description'], div[class*='overview'], p"
                ]:
                    element = soup.select_one(sel)
                    if element:
                        content = element.get("content") if hasattr(element, "attrs") and element.has_attr("content") else element.text
                        description = clean_description(content)
                        if len(description) > 10:
                            break

                country = "N/A"
                el = soup.select_one(".location, .country, .address, .footer-address, div[class*='location'], div[class*='address'], footer, .contact-info")
                if el:
                    country = clean_description(el.text)

                # --- Фильтры класса сайта ---
                if looks_like_sports_garbage(description):
                    success = True; break
                if looks_like_definition_page(description, url, intent):
                    success = True; break
                if should_cut_blog(url, description, intent):
                    success = True; break
                if looks_like_casino_operator(description, url):
                    success = True; break
                if looks_like_listing(description, url):
                    success = True; break
                if looks_like_job_board(description, url):
                    success = True; break

                score = rank_result(description, prompt_phrases, url=url, region=region, boosts=boosts)
                logger.info(f"Rank score={score:.2f} url={url}")

                if is_relevant_url(url, prompt_phrases) or score > 0.1:
                    row = {
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "website": url,
                        "description": description,
                        "country": country,
                        "source": "Web",
                        "score": score,
                    }
                    results.append(row)
                    save_result_records(row, intent, region, query_id)
                success = True
                time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
                break
            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["403", "429", "connection", "timeout", "ssl", "remote disconnected"]):
                    logger.warning(f"Error for {url}: {e}, retry with proxy (attempt {proxy_attempts+1})")
                    proxy_attempts += 1
                    if proxy_attempts > MAX_PROXY_ATTEMPTS:
                        logger.warning(f"Max proxy attempts for {url}, skipping")
                        break
                else:
                    logger.error(f"Scrape failed for {url}: {e}, skipping")
                    break
        if not success:
            logger.warning(f"Failed to scrape {url} after attempts")
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Total web results scraped: {len(results)}")
    return results

# ========= Persistence =========
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
                    ", ".join([]).title(), row.get("country","N/A"), row.get("source","Web"),
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
    except sqlite3.Error as e:
        logger.error(f"Error saving records: {e}")

def save_to_csv():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY score DESC")
            rows = cursor.fetchall()
            if not rows:
                logger.warning("No data to save to CSV")
                return
            with open("search_results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID","Name","Website","Description","Specialization","Country","Source","Status","Suitability","Score"])
                writer.writerows(rows)
        logger.info("CSV file created: search_results.csv")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

def save_to_txt():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY score DESC")
            rows = cursor.fetchall()
            with open("search_results.txt", "w", encoding="utf-8") as f:
                if not rows:
                    f.write("Активных результатов не найдено.\n")
                    return
                f.write("Найденные результаты:\n\n")
                for row in rows:
                    f.write(f"Название: {row[1][:100] or 'N/A'}\n")
                    f.write(f"Вебсайт: {row[2] or 'N/A'}\n")
                    f.write(f"Описание: {row[3][:200] or 'N/A'}...\n")
                    f.write(f"Категория: {row[4] or 'N/A'}\n")
                    f.write(f"Страна: {row[5] or 'N/A'}\n")
                    f.write(f"Источник: {row[6] or 'N/A'}\n")
                    f.write(f"Статус: {row[7] or 'N/A'}\n")
                    f.write(f"Пригодность: {row[8] or 'N/A'}\n")
                    f.write(f"Оценка: {row[9]:.2f}\n")
                    f.write(f"Дата добавления: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                    f.write("\n" + "-" * 50 + "\n")
        logger.info("TXT file created: search_results.txt")
    except Exception as e:
        logger.error(f"Error saving to TXT: {e}")

def prefer_country_results(rows: List[dict], region: str) -> List[dict]:
    hint = GEO_HINTS.get(region)
    if not hint:
        return rows
    tld = f".{hint['tld']}"
    a = [r for r in rows if domain_of(r.get("website","")).endswith(tld)]
    b = [r for r in rows if not domain_of(r.get("website","")).endswith(tld)]
    return a + b

# ========= Generate queries (wrapper) =========
def generate_search_queries(user_prompt: str, region="wt-wt") -> Tuple[List[str], List[str], str, Dict[str,bool]]:
    if region not in REGION_MAP:
        logger.warning(f"Invalid region {region}, defaulting to wt-wt")
        region = "wt-wt"

    user_prompt = (user_prompt or "").strip()
    if not user_prompt:
        return [user_prompt], [], region, {"affiliate": False, "learn": False, "business": True, "casino": False}

    intent = detect_intent(user_prompt)

    try:
        web_queries, phrases = gemini_expand_queries(user_prompt, region, intent)
    except Exception as e:
        logger.warning(f"Gemini failed or invalid output: {e}. Falling back.")
        web_queries, phrases = fallback_expand_queries(user_prompt, region, intent)

    # лёгкий гео-биас (оставляет и общие запросы)
    web_queries = apply_geo_bias(web_queries, region)

    return web_queries, phrases, region, intent

# ========= API =========
@app.route("/search", methods=["POST", "OPTIONS"])
def search():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        origin = request.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Max-Age"] = "86400"
        return resp

    try:
        data = request.json or {}
        user_query = data.get("query", "")
        region = data.get("region", "wt-wt")
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE", "both")).lower()  # ddg | serpapi | both
        max_results = int(data.get("max_results", 15))

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        logger.info(f"API request: query='{user_query}', region={region}, engine={engine}")

        # Очистим текущую выдачу
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
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
            cursor.execute("DELETE FROM results")
            conn.commit()

        # Генерация подзапросов
        web_queries, prompt_phrases, region, intent = generate_search_queries(user_query, region)

        # Запись запроса
        query_id = insert_query_record(user_query, intent, region)

        # Бусты по доменам из истории + штрафы
        boosts = history_boosts(intent, region)

        # Если кириллица и регион дефолтный — для DDG используем ru-ru
        force_ru_ddg = is_cyrillic(user_query) and region == "wt-wt"

        # Собираем URL’ы
        all_urls = []
        for q in web_queries:
            if engine in ("ddg", "both"):
                all_urls.extend(
                    duckduckgo_search(q, max_results=max_results, region=region, intent=intent, force_ru_ddg=force_ru_ddg)
                )
            if engine in ("serpapi", "both"):
                all_urls.extend(
                    serpapi_search(q, max_results=max_results, region=region, intent=intent)
                )

        # Дедуп и отсев мусора по доменам/блок-листу
        deduped = []
        seen = set()
        for u in all_urls:
            dom = domain_of(u)
            if not u or u in seen:
                continue
            if is_bad_domain(dom) or domain_is_blocked(dom):
                continue
            deduped.append(u); seen.add(u)
        all_urls = deduped
        logger.info(f"Collected {len(all_urls)} unique URLs (after blocklist filter)")

        # Скрейп
        web_results = search_and_scrape_websites(all_urls, prompt_phrases, region, intent, boosts, query_id)

        # Жёсткая финальная чистка
        filtered = []
        for r in web_results:
            txt = f"{r.get('name','')} {r.get('description','')} {r.get('website','')}".lower()
            dom = domain_of(r.get("website",""))
            url_ = r.get("website","")
            if is_bad_domain(dom) or domain_is_blocked(dom):  # NEW: финальный блок-лист
                continue
            if any(url_.lower().endswith(t) for t in CASINO_TLDS):
                continue
            if looks_like_sports_garbage(txt):
                continue
            if intent.get("business") and not intent.get("learn") and dom in KNOWLEDGE_DOMAINS:
                continue
            if should_cut_blog(url_, txt, intent):
                continue
            if looks_like_casino_operator(txt, url_):
                continue
            if looks_like_listing(txt, url_):
                continue
            if looks_like_job_board(txt, url_):
                continue
            filtered.append(r)

        # Гео-приоритизация и сортировка
        all_results = prefer_country_results(filtered, region)
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Экспорты (опционально)
        save_to_csv()
        save_to_txt()

        return jsonify({
            "results": all_results,
            "region": region,
            "engine": engine,
            "query_id": query_id,        # NEW: чтобы фронт отправлял фидбэк
        })
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

# ========= Feedback API =========
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    body: { "query_id": "...", "url": "...", "action": "click|good|bad|flag" }
    'flag' трактуем как 'bad' для ясности на фронте.
    """
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

# ========= API aliases /api/* =========
@app.route("/api/search", methods=["POST", "OPTIONS"])
def api_search():
    return search()

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

@app.route("/api/download/<filetype>", methods=["GET"])
def api_download(filetype):
    return download_file(filetype)

# ========= Serve SPA (frontend_dist) =========
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
    port = int(os.getenv("PORT", os.getenv("RAILWAY_PORT", "5000")))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Flask on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
