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
from datetime import datetime
from typing import List, Tuple, Dict, Any

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ======================= Helpers: text cleaning =======================
CODE_FENCE_RE = re.compile(r"^```(?:json|js|python|txt)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def clean_text(text: str) -> str:
    """Убираем ```json...```, крайние кавычки/апострофы и лишние крайние скобки."""
    if not text:
        return ""
    t = text.strip()
    t = CODE_FENCE_RE.sub("", t)
    t = t.strip(" \n\t\r\"'`")
    while t and (t[0] in "[{" and t[-1] in "]}"):
        t = t[1:-1].strip()
    return t

def clean_description(description):
    """Убираем HTML и сокращаем до ~200 слов."""
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
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
if TELEGRAM_ENABLED:
    from telethon.sync import TelegramClient
    from telethon.sessions import StringSession
    from telethon.tl.functions.contacts import SearchRequest
    from telethon.errors import SessionPasswordNeededError, FloodWaitError

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
    # Global / базовые
    "wt-wt": {"hl": "en", "gl": "us"},
    "us-en": {"hl": "en", "gl": "us"},
    "uk-en": {"hl": "en", "gl": "gb"},
    "de-de": {"hl": "de", "gl": "de"},
    "fr-fr": {"hl": "fr", "gl": "fr"},
    "ru-ru": {"hl": "ru", "gl": "ru"},
    "ua-ua": {"hl": "uk", "gl": "ua"},

    # СНГ
    "kz-ru": {"hl": "ru", "gl": "kz"},
    "kz-kk": {"hl": "kk", "gl": "kz"},
    "by-ru": {"hl": "ru", "gl": "by"},
    "uz-ru": {"hl": "ru", "gl": "uz"},
    "az-ru": {"hl": "ru", "gl": "az"},
    "kg-ru": {"hl": "ru", "gl": "kg"},
    "am-ru": {"hl": "ru", "gl": "am"},
    "ge-ka": {"hl": "ka", "gl": "ge"},

    # Европа
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

    # Азия / Ближний Восток
    "tr-tr": {"hl": "tr", "gl": "tr"},
    "ae-en": {"hl": "en", "gl": "ae"},
    "sa-ar": {"hl": "ar", "gl": "sa"},
    "il-he": {"hl": "he", "gl": "il"},
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

    # Америка/Океания
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
def init_db():
    try:
        with sqlite3.connect("search_results.db") as conn:
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
            conn.commit()
        logger.info("Database initialized")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}")

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

# домены справочников/обучалок, которые мешают, когда мы ищем компании/каталоги
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
AFFILIATE_VARIANTS = {
    "affiliate","affiliat","affilate","referral","cpa","revshare","partner program","affiliate program",
    "affiliate network","networks",
    "аффилиат","аффилиейт","аффилейт","афилейт","афелейт",
    "партнерка","партнёрка","партнерские программы","партнёрские программы",
    "партнерская программа","партнёрская программа","реферальная","реферал","рефералы"
}
CASINO_VARIANTS = {"казино","casino","igaming","игейминг","гемблинг","беттинг","ставки","bookmaker","букмекер"}

COMPANY_HINTS = {
    "company","companies","agency","agencies","platform","software","solution","solutions",
    "service","services","provider","providers","network","program","programs","system",
    "компания","компании","агентство","агентства","платформа","софт","решение","решения",
    "услуга","услуги","провайдер","поставщик","платформы","система","системы"
}

LEARN_HINTS = {
    "что такое","what is","определение","definition","гайд","guide","обзор","overview",
    "курс","course","как работает","how to","советы","tips","статья","article","блог","blog","новости","news"
}

def detect_intent(q: str) -> Dict[str, bool]:
    t = (q or "").lower()

    score = 0.0
    if any(v in t for v in AFFILIATE_VARIANTS): score += 1.0
    if any(v in t for v in CASINO_VARIANTS):    score += 0.5
    if any(v in t for v in COMPANY_HINTS):      score += 0.5

    affiliate = score >= 1.0 and any(v in t for v in AFFILIATE_VARIANTS)
    company_search = affiliate and any(v in t for v in COMPANY_HINTS)
    learn = any(k in t for k in LEARN_HINTS)

    return {
        "affiliate": affiliate,
        "company_search": company_search,
        "learn": learn,
        "business": not learn  # по умолчанию считаем, что ищем компании/каталоги
    }

# ========= Intent-agnostic analysis =========
def is_relevant_url(url, prompt_phrases):
    u = (url or "").lower()
    if any(bad in u for bad in BAD_DOMAINS):
        return False
    words = [w.lower() for p in prompt_phrases for w in p.split() if len(w) > 3]
    return "t.me" in u or "instagram.com" in u or any(w in u for w in words) or any(p.lower() in u for p in prompt_phrases)

# --- GEO hints for biasing
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
    """Добавляем к первым подзапросам страновой контекст и site:.tld."""
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

def rank_result(description: str, prompt_phrases: List[str], url: str = None, region: str = None) -> float:
    score = 0.0
    d = (description or "").lower()
    words = [w.lower() for p in prompt_phrases for w in p.split() if len(w) > 3]
    for w in words:
        if w in d:
            score += 0.3 if len(w) > 6 else 0.2
    for p in prompt_phrases:
        if p.lower() in d:
            score += 0.4
    if "t.me" in d or "instagram.com" in d:
        score += 0.2
    if looks_like_sports_garbage(d):
        score = min(score, 0.2)
    if url and region:
        score += geo_score_boost(url, d, region)
    return min(score, 1.0)

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
def gemini_expand_queries(user_prompt: str, region: str, intent: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    system_instruction = (
        "Ты помощник для поиска. Преобразуй запрос пользователя в массив из 4–6 коротких запросов по теме. "
        "Если запрос НЕ про аффилиатки/партнёрские программы — НЕ добавляй слова вроде 'affiliate', 'CPA', 'referral'. "
        "Если запрос про аффилиатки — добавь релевантные англоязычные термины. Не пиши комментарии. "
        "Верни ТОЛЬКО JSON-массив строк."
    )
    guard = "AFFILIATE=YES" if intent.get("affiliate") else "AFFILIATE=NO"
    body = {
        "contents": [{
            "parts": [
                {"text": system_instruction},
                {"text": f"{guard}\nЗапрос: {user_prompt}\nРегион: {region}\nФормат ответа: JSON массив строк."}
            ]
        }],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": 256}
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    params = {"key": GEMINI_API_KEY}

    r = requests.post(url, params=params, json=body, timeout=15)
    r.raise_for_status()
    data = r.json()

    text = ""
    try:
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                text = parts[0]["text"]
    except Exception:
        pass

    text = clean_text(text)
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            arr = [clean_text(str(x)) for x in arr if isinstance(x, (str, int, float))]
            arr = [x for x in arr if x and len(x) <= 200]
            if arr:
                arr = apply_geo_bias(arr[:6], region)
                logger.info("Using Gemini for query expansion")
                return arr[:6], arr[:6]
    except Exception:
        pass

    raise RuntimeError("Gemini returned invalid JSON")

def fallback_expand_queries(user_prompt: str, region: str, intent: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    base = user_prompt.strip()
    queries: List[str] = [base]

    if intent.get("affiliate"):
        # Если ищем именно компании в affiliate-контексте — усиливаем company-маркерами
        if intent.get("company_search"):
            positives = [
                "affiliate marketing agency casino",
                "iGaming affiliate agency",
                "casino affiliate platform",
                "affiliate tracking software casino",
                "affiliate program management casino",
                "affiliate network casino",
                "services provider iGaming affiliate"
            ]
            ru_pos = [
                "агентство аффилиат маркетинга казино",
                "компании affiliate для казино",
                "платформа партнерских программ казино",
                "софт для affiliate казино",
                "управление партнерской программой казино",
                "affiliate network казино"
            ]
            queries += positives + ru_pos
        else:
            en_boost = [
                "affiliate programs",
                "partner program application",
                "affiliate marketing",
                "referral program",
                "CPA affiliate",
                "best affiliate networks",
            ]
            lower = base.lower()
            if any(k in lower for k in ["казино","гемблинг","gambling","casino","беттинг","ставки","igaming"]):
                en_boost += ["casino affiliate programs", "iGaming affiliate", "online casino affiliates"]
            queries += en_boost
    else:
        generic = [
            f"{base} официальный сайт",
            f"{base} каталог",
            f"{base} список",
            f"{base} компании",
            f"{base} directory",
            f"{base} companies",
            f"{base} official site"
        ]
        queries += generic

    # remove dups, apply geo bias
    queries = list(dict.fromkeys([q for q in queries if q]))
    queries = apply_geo_bias(queries[:6], region)
    logger.info(f"Using fallback for query expansion (affiliate={intent.get('affiliate')}, company={intent.get('company_search')})")
    return queries[:6], queries[:6]

# ========= Build query with negative site filters by intent =========
NEGATIVE_SITES_FOR_BUSINESS = [
    "site:wikipedia.org","site:en.wikipedia.org","site:ru.wikipedia.org",
    "site:hubspot.com","site:coursera.org","site:ibm.com","site:sproutsocial.com",
    "site:digitalmarketinginstitute.com","site:marketermilk.com","site:harvard.edu","site:professional.dce.harvard.edu"
]

NEGATIVE_CONTENT_FOR_COMPANY = [
    "-\"что такое\"","-обзор","-обзоры","-гайд","-guides","-guide","-blog","-блог","-новости","-news",
    "-курс","-courses","-статья","-article","-wiki"
]

def with_intent_filters(q: str, intent: Dict[str,bool]) -> str:
    # 1) режем обучалки/вики, если пользователь ищет бизнес
    if intent.get("business") and not intent.get("learn"):
        negatives = " ".join(f"-{s}" for s in NEGATIVE_SITES_FOR_BUSINESS)
        q = f"{q} {negatives}".strip()
    # 2) если affiliate+company_search — срезаем контентные страницы
    if intent.get("affiliate") and intent.get("company_search"):
        q = f"{q} " + " ".join(NEGATIVE_CONTENT_FOR_COMPANY)
    return q.strip()

# ========= Search engines (DDG / SerpAPI) =========
def duckduckgo_search(query, max_results=15, region="wt-wt", intent: Dict[str,bool]=None):
    data = load_request_count()
    if data["count"] >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        return []
    q = with_intent_filters(query, intent or {"business": True})
    logger.info(f"DDG search: '{q}' region={region}")
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(q, region=region, safesearch="moderate", timelimit="y", max_results=max_results)
            for r in results:
                href = r.get("href")
                if href:
                    urls.append(href)
        save_request_count(data["count"] + 1)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        urls = [u for u in urls if domain_of(u) not in BAD_DOMAINS]
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
        urls = [u for u in urls if domain_of(u) not in BAD_DOMAINS]
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"SerpAPI failed for '{q}': {e}")
        return []

# ========= Telegram (client helper + endpoints + search) =========
# ENV, которые нужны:
# TELEGRAM_ENABLED=true
# TELEGRAM_API_ID=...
# TELEGRAM_API_HASH=...
# TELEGRAM_PHONE=+XXXXXXXXXXX
# TELEGRAM_STRING_SESSION= (после /telegram/confirm сохранить сюда)
# TELEGRAM_FORCE_SMS=true|false (опционально)
def get_tg_client():
    """Создаёт/возвращает Telethon-клиент со строковой сессией (если задана) или сессией-файлом."""
    if not TELEGRAM_ENABLED:
        return None, "TELEGRAM_ENABLED is false"

    API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
    API_HASH = os.getenv("TELEGRAM_API_HASH", "")
    if not (API_ID and API_HASH):
        return None, "TELEGRAM_API_ID/TELEGRAM_API_HASH not set"

    string_session = os.getenv("TELEGRAM_STRING_SESSION", "").strip()
    try:
        if string_session:
            client = TelegramClient(StringSession(string_session), API_ID, API_HASH)
        else:
            os.makedirs("data", exist_ok=True)
            client = TelegramClient(os.path.join("data", "tg_session"), API_ID, API_HASH)
        client.connect()
        return client, None
    except Exception as e:
        return None, f"Failed to init Telegram client: {e}"

def telegram_search(queries, prompt_phrases):
    if not TELEGRAM_ENABLED:
        logger.info("Telegram search disabled")
        return []

    results = []
    API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
    API_HASH = os.getenv("TELEGRAM_API_HASH", "")
    PHONE_NUMBER = os.getenv("TELEGRAM_PHONE", "").strip()

    if not (API_ID and API_HASH and PHONE_NUMBER):
        logger.error("Telegram env not fully configured; skipping")
        return results

    client, err = get_tg_client()
    if err:
        logger.error(err)
        return results

    try:
        if not client.is_user_authorized():
            force_sms = os.getenv("TELEGRAM_FORCE_SMS", "false").lower() == "true"
            try:
                client.send_code_request(PHONE_NUMBER, force_sms=force_sms)
                logger.info("Telegram code sent. Complete login via /telegram/confirm")
                client.disconnect()
                return results
            except Exception as e:
                logger.error(f"Telegram code request failed: {e}")
                client.disconnect()
                return results

        for q in queries:
            logger.info(f"Searching Telegram for: {q}")
            try:
                result = client(SearchRequest(q=q, limit=10))
                for chat in result.chats:
                    # берём публичные каналы/суперчаты (без мегагрупп)
                    if hasattr(chat, "megagroup") and not chat.megagroup:
                        name = chat.title or "N/A"
                        username = f"t.me/{getattr(chat, 'username', None)}" if getattr(chat, "username", None) else "N/A"
                        description = clean_description(getattr(chat, "description", "N/A"))
                        is_rel, spec, status, suit = analyze_result(description, prompt_phrases)
                        score = rank_result(description, prompt_phrases, url=username)
                        if is_relevant_url(username, prompt_phrases) or score > 0.1:
                            row = {
                                "id": str(uuid.uuid4()),
                                "name": name,
                                "website": username,
                                "description": description,
                                "country": "N/A",
                                "source": "Telegram",
                                "score": score,
                            }
                            results.append(row)
                            save_to_db(row, is_rel, spec, status, suit, score)
                        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
            except FloodWaitError as e:
                logger.warning(f"Telegram rate limit, waiting {e.seconds}s")
                time.sleep(e.seconds)
            except Exception as e:
                logger.error(f"Telegram search failed for '{q}': {e}")
        client.disconnect()
    except Exception as e:
        logger.error(f"Telegram client failed: {e}")
    return results

# ========= Scraper: page-type heuristics =========
ARTICLE_PATTERNS = [
    "что такое","what is","определение","definition","гайд","guide","как","как работать",
    "обзор","overview","новости","news","блог","blog","статья","article","academy","wiki"
]
COMPANY_POS_SIGNALS = [
    "услуги","services","решения","solutions","о нас","about","контакты","contact",
    "клиенты","clients","кейсы","cases","цены","pricing","демо","demo","заказать","request",
    "портфолио","portfolio","команда","team","наш адрес","email","e-mail","+","тел","phone"
]

def looks_like_article_page(text: str, url: str) -> bool:
    t = (text or "").lower()
    u = (url or "").lower()
    if any(p in u for p in ["/blog","/news","/wiki","/academy","/kb","/knowledge"]):
        return True
    return any(p in t for p in ARTICLE_PATTERNS)

def looks_like_company_page(text: str, url: str, soup: BeautifulSoup) -> bool:
    t = (text or "").lower()
    if any(p in t for p in COMPANY_POS_SIGNALS):
        return True
    # header/footer навигация — если на странице есть явные пункты меню
    nav = " ".join(el.get_text(" ").lower() for el in soup.select("nav, header, footer")[:3])
    if any(p in nav for p in COMPANY_POS_SIGNALS):
        return True
    # наличие e-mail / телефона
    if re.search(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", t):
        return True
    if re.search(r"\+\d[\d\s().-]{6,}", t):
        return True
    return False

def looks_like_definition_page(text: str, url: str, intent: Dict[str,bool]) -> bool:
    """Отсекаем обучалки/вики, если пользователь ищет компании/партнёров."""
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

def search_and_scrape_websites(urls: List[str], prompt_phrases: List[str], region: str, intent: Dict[str,bool]):
    logger.info(f"Starting scrape of {len(urls)} URLs")
    results = []
    urls = list(dict.fromkeys(urls))[:50]
    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Scraping: {url}")
        if domain_of(url) in BAD_DOMAINS:
            logger.info(f"Skip bad domain: {url}")
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

                # 1) спорт-мусор
                if looks_like_sports_garbage(description):
                    logger.info(f"Skip sports-like garbage: {url}")
                    success = True
                    break

                # 2) контентные страницы, если ищем компании для affiliate
                if intent.get("affiliate") and intent.get("company_search") and looks_like_article_page(description, url):
                    logger.info(f"Skip article/blog page by intent: {url}")
                    success = True
                    break

                # 3) обучалки/вики, когда ищем бизнес
                if looks_like_definition_page(description, url, intent):
                    logger.info(f"Skip knowledge page by intent: {url}")
                    success = True
                    break

                # 4) если ищем компании — проверим, что страница похожа на компанию
                if intent.get("affiliate") and intent.get("company_search"):
                    if not looks_like_company_page(description, url, soup):
                        logger.info(f"Skip non-company-looking page for affiliate-company intent: {url}")
                        success = True
                        break

                is_rel, spec, status, suit = analyze_result(description, prompt_phrases)
                score = rank_result(description, prompt_phrases, url=url, region=region)
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
                    save_to_db(row, is_rel, spec, status, suit, score)
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
def save_to_db(result, is_relevant, specialization, status, suitability, score):
    try:
        with sqlite3.connect("search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO results
                   (id, name, website, description, specialization, country, source, status, suitability, score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result["id"], result["name"], result["website"], result["description"],
                    specialization, result["country"], result["source"], status, suitability, score
                ),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving to SQLite: {e}")

def save_to_csv():
    try:
        with sqlite3.connect("search_results.db") as conn:
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
        with sqlite3.connect("search_results.db") as conn:
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
        use_telegram = bool(data.get("telegram", False)) and TELEGRAM_ENABLED
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE", "both")).lower()  # ddg | serpapi | both
        max_results = int(data.get("max_results", 15))

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        logger.info(f"API request: query='{user_query}', region={region}, telegram={use_telegram}, engine={engine}")

        # Reset DB for each request
        with sqlite3.connect("search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS results")
            cursor.execute(
                """CREATE TABLE results (
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
            conn.commit()

        # Build & collect URLs
        web_queries, prompt_phrases, region, telegram_queries, intent = generate_search_queries(user_query, region)

        all_urls = []
        for q in web_queries:
            if engine in ("ddg", "both"):
                all_urls.extend(duckduckgo_search(q, max_results=max_results, region=region, intent=intent))
            if engine in ("serpapi", "both"):
                all_urls.extend(serpapi_search(q, max_results=max_results, region=region, intent=intent))

        all_urls = [u for u in list(dict.fromkeys(all_urls)) if domain_of(u) not in BAD_DOMAINS]
        logger.info(f"Collected {len(all_urls)} unique URLs")

        # Scrape
        web_results = search_and_scrape_websites(all_urls, prompt_phrases, region, intent)

        # Telegram (optional)
        telegram_results = telegram_search(telegram_queries, prompt_phrases) if use_telegram else []
        all_results = web_results + telegram_results

        # Hard cleanup pass
        filtered = []
        for r in all_results:
            txt = f"{r.get('name','')} {r.get('description','')} {r.get('website','')}".lower()
            dom = domain_of(r.get("website",""))
            if dom in BAD_DOMAINS:
                continue
            if looks_like_sports_garbage(txt):
                continue
            if intent.get("business") and not intent.get("learn") and dom in KNOWLEDGE_DOMAINS:
                continue
            filtered.append(r)

        # Гео-приоритизация и сортировка
        all_results = prefer_country_results(filtered, region)
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Exports
        save_to_csv()
        save_to_txt()

        return jsonify({
            "results": all_results,
            "telegram_enabled": use_telegram,
            "region": region,
            "engine": engine
        })
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

# ========= API aliases /api/* =========
@app.route("/api/search", methods=["POST", "OPTIONS"])
def api_search():
    return search()

@app.route("/download/<filetype>", methods=["GET"])
def download_file(filetype):
    if filetype not in ["csv", "txt"]:
        return jsonify({"error": "Invalid file type, use 'csv' or 'txt'"}), 400
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

# ========= Telegram endpoints =========
@app.route("/telegram/status", methods=["GET"])
def tg_status():
    if not TELEGRAM_ENABLED:
        return jsonify({"enabled": False, "authorized": False, "reason": "TELEGRAM_ENABLED is false"}), 200
    client, err = get_tg_client()
    if err:
        return jsonify({"enabled": True, "authorized": False, "error": err}), 200
    try:
        ok = client.is_user_authorized()
        me = None
        if ok:
            me = client.get_me()
            me = {"id": me.id, "username": getattr(me, "username", None), "phone": getattr(me, "phone", None)}
        client.disconnect()
        return jsonify({"enabled": True, "authorized": ok, "me": me}), 200
    except Exception as e:
        return jsonify({"enabled": True, "authorized": False, "error": str(e)}), 200

@app.route("/telegram/send_code", methods=["POST"])
def tg_send_code():
    if not TELEGRAM_ENABLED:
        return jsonify({"error": "TELEGRAM_ENABLED is false"}), 400
    PHONE_NUMBER = os.getenv("TELEGRAM_PHONE", "").strip()
    if not PHONE_NUMBER:
        return jsonify({"error": "TELEGRAM_PHONE not set"}), 400

    client, err = get_tg_client()
    if err:
        return jsonify({"error": err}), 500
    try:
        force_sms = os.getenv("TELEGRAM_FORCE_SMS", "false").lower() == "true"
        client.send_code_request(PHONE_NUMBER, force_sms=force_sms)
        client.disconnect()
        return jsonify({"ok": True, "sent_to": "sms" if force_sms else "telegram_app"}), 200
    except Exception as e:
        client.disconnect()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/telegram/confirm", methods=["POST"])
def tg_confirm():
    """
    Подтверждаем вход: body = { "code": "12345", "password": "опц. 2FA" }
    В ответ отдаём сгенерированный StringSession — скопируй в ENV TELEGRAM_STRING_SESSION.
    """
    if not TELEGRAM_ENABLED:
        return jsonify({"error": "TELEGRAM_ENABLED is false"}), 400

    API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
    API_HASH = os.getenv("TELEGRAM_API_HASH", "")
    PHONE_NUMBER = os.getenv("TELEGRAM_PHONE", "").strip()
    if not (API_ID and API_HASH and PHONE_NUMBER):
        return jsonify({"error": "TELEGRAM_API_ID/TELEGRAM_API_HASH/TELEGRAM_PHONE not set"}), 400

    data = request.json or {}
    code = (data.get("code") or "").strip()
    password = data.get("password")  # 2FA
    if not code:
        return jsonify({"error": "code is required"}), 400

    # используем файловую сессию для первичного логина, затем сохраним StringSession
    os.makedirs("data", exist_ok=True)
    client = TelegramClient(os.path.join("data", "tg_session"), API_ID, API_HASH)
    try:
        client.connect()
        if client.is_user_authorized():
            s = StringSession.save(client.session)
            me = client.get_me()
            client.disconnect()
            return jsonify({"ok": True, "string_session": s, "me": {"id": me.id, "username": getattr(me,"username",None)}}), 200

        try:
            client.sign_in(PHONE_NUMBER, code)
        except SessionPasswordNeededError:
            if not password:
                client.disconnect()
                return jsonify({"ok": False, "error": "2FA enabled: password required"}), 400
            client.sign_in(password=password)

        s = StringSession.save(client.session)
        me = client.get_me()
        client.disconnect()
        return jsonify({"ok": True, "string_session": s, "me": {"id": me.id, "username": getattr(me,"username",None)}}), 200
    except Exception as e:
        try:
            client.disconnect()
        except:
            pass
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/telegram/logout", methods=["POST"])
def tg_logout():
    if not TELEGRAM_ENABLED:
        return jsonify({"error": "TELEGRAM_ENABLED is false"}), 400
    client, err = get_tg_client()
    if err:
        return jsonify({"error": err}), 500
    try:
        client.log_out()
        client.disconnect()
        try:
            p = os.path.join("data", "tg_session.session")
            if os.path.exists(p):
                os.remove(p)
        except:
            pass
        return jsonify({"ok": True}), 200
    except Exception as e:
        client.disconnect()
        return jsonify({"ok": False, "error": str(e)}), 500

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
