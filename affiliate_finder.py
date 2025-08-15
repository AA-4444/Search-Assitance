import os, re, json, time, uuid, csv, random, logging, sqlite3, warnings, requests
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin

from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ddgs import DDGS

# ---------- silence noisy warnings ----------
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# ======================= Config & Env =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend_dist")
DB_PATH = os.getenv("DB_PATH", "search_results.db")

SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "").strip() or None

MIN_RESULTS = int(os.getenv("MIN_RESULTS", "25"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))

# Скорость/масштаб
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "24"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "8"))
PER_HOST_LIMIT = int(os.getenv("PER_HOST_LIMIT", "4"))

STRONG_BLOCK_ON_BAD = (os.getenv("STRONG_BLOCK_ON_BAD", "true").lower() == "true")

REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1600"))
REQUEST_PAUSE_MIN = float(os.getenv("REQUEST_PAUSE_MIN", "0.08"))
REQUEST_PAUSE_MAX = float(os.getenv("REQUEST_PAUSE_MAX", "0.22"))

DEFAULT_UA = os.getenv(
    "SCRAPER_UA",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

# SerpAPI — резерв с кулингом при 429
SERPAPI_COOLDOWN_SEC = int(os.getenv("SERPAPI_COOLDOWN_SEC", "900"))  # 15 мин
_last_serpapi_429_at = 0.0

# Жёсткий дедлайн на скрапинг
SCRAPE_HARD_DEADLINE_SEC = int(os.getenv("SCRAPE_HARD_DEADLINE_SEC", "300"))  # 5 мин

# Микро-краул
MAX_MICROCRAWL_PAGES = int(os.getenv("MAX_MICROCRAWL_PAGES", "2"))
ENABLE_MICROCRAWL = (os.getenv("ENABLE_MICROCRAWL", "true").lower() == "true")

# Лимиты «урожая» из каталогов/листингов
DIR_HARVEST_PER_PAGE = int(os.getenv("DIR_HARVEST_PER_PAGE", "40"))
DIR_HARVEST_GLOBAL_CAP = int(os.getenv("DIR_HARVEST_GLOBAL_CAP", "400"))

# Квота на смешивание операторских партнёрок (если по смыслу запроса уместно)
MIX_OPERATOR_MAX = int(os.getenv("MIX_OPERATOR_MAX", "10"))

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
CORS(app,
     origins=list(ALLOWED_ORIGINS),
     methods=["GET","POST","OPTIONS"],
     allow_headers=["Content-Type","Authorization"],
     supports_credentials=True)

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

SOCIAL_HOSTS = {
    "facebook.com","instagram.com","x.com","twitter.com","vk.com","ok.ru",
    "linkedin.com","t.me","telegram.me","telegram.org","youtube.com","youtu.be"
}

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
        query = [(k,v) for k,v in parse_qsl(u.query, keep_blank_values=False) if not k.lower().startswith("utm_")]
        u2 = u._replace(query=urlencode(query, doseq=True), fragment="")
        return urlunparse(u2)
    except Exception:
        return url

def short(s: str, n=220) -> str:
    s = s or ""
    return (s[:n] + "…") if len(s) > n else s

# ===== language / encoding =====
try:
    from langdetect import detect as langdetect_detect
except Exception:
    langdetect_detect = None
try:
    import chardet
except Exception:
    chardet = None

def guess_lang(text: str) -> str:
    t = (text or "").strip()
    if not t: return "en"
    if re.search(r"[іїєґ]", t.lower()): return "uk"
    if re.search(r"[а-яё]", t.lower()): return "ru"
    # грубая эвристика для kk
    if re.search(r"[әіңғүұқөһ]", t.lower()): return "kk"
    if langdetect_detect:
        try:
            return langdetect_detect(t)
        except Exception:
            pass
    return "en"

def detect_encoding(content: bytes, resp: Optional[requests.Response]=None) -> str:
    if resp and resp.encoding:
        return resp.encoding
    if resp and hasattr(resp, "apparent_encoding") and resp.apparent_encoding:
        return resp.apparent_encoding
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
            conn.commit()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

init_db()

# ======================= Regions =======================
REGION_MAP = {
    "wt-wt": {"hl":"en","gl":"us","google_domain":"google.com","tld":"com"},
    "us-en": {"hl":"en","gl":"us","google_domain":"google.com","tld":"us"},
    "uk-en": {"hl":"en","gl":"gb","google_domain":"google.co.uk","tld":"uk"},
    "de-de": {"hl":"de","gl":"de","google_domain":"google.de","tld":"de"},
    "fr-fr": {"hl":"fr","gl":"fr","google_domain":"google.fr","tld":"fr"},
    "ru-ru": {"hl":"ru","gl":"ru","google_domain":"google.ru","tld":"ru"},
    "ua-ua": {"hl":"uk","gl":"ua","google_domain":"google.com.ua","tld":"ua"},
    "kz-ru": {"hl":"ru","gl":"kz","google_domain":"google.kz","tld":"kz"},
    "kz-kk": {"hl":"kk","gl":"kz","google_domain":"google.kz","tld":"kz"},  # добавил
    "by-ru": {"hl":"ru","gl":"by","google_domain":"google.by","tld":"by"},
    "tr-tr": {"hl":"tr","gl":"tr","google_domain":"google.com.tr","tld":"tr"},
    "ae-en": {"hl":"en","gl":"ae","google_domain":"google.ae","tld":"ae"},
    "in-en": {"hl":"en","gl":"in","google_domain":"google.co.in","tld":"in"},
    "sg-en": {"hl":"en","gl":"sg","google_domain":"google.com.sg","tld":"sg"},
    "es-es": {"hl":"es","gl":"es","google_domain":"google.es","tld":"es"},
    "it-it": {"hl":"it","gl":"it","google_domain":"google.it","tld":"it"},
    "nl-nl": {"hl":"nl","gl":"nl","google_domain":"google.nl","tld":"nl"},
    "se-sv": {"hl":"sv","gl":"se","google_domain":"google.se","tld":"se"},
    "no-no": {"hl":"no","gl":"no","google_domain":"google.no","tld":"no"},
    "fi-fi": {"hl":"fi","gl":"fi","google_domain":"google.fi","tld":"fi"},
    "cz-cs": {"hl":"cs","gl":"cz","google_domain":"google.cz","tld":"cz"},
    "sk-sk": {"hl":"sk","gl":"sk","google_domain":"google.sk","tld":"sk"},
    "ro-ro": {"hl":"ro","gl":"ro","google_domain":"google.ro","tld":"ro"},
    "hu-hu": {"hl":"hu","gl":"hu","google_domain":"google.hu","tld":"hu"},
    "ch-de": {"hl":"de","gl":"ch","google_domain":"google.ch","tld":"ch"},
    "br-pt": {"hl":"pt","gl":"br","google_domain":"google.com.br","tld":"br"},
    "mx-es": {"hl":"es","gl":"mx","google_domain":"google.com.mx","tld":"mx"},
    "au-en": {"hl":"en","gl":"au","google_domain":"google.com.au","tld":"au"},
    "nz-en": {"hl":"en","gl":"nz","google_domain":"google.co.nz","tld":"nz"},
}

# для DDG у них фиксированный набор region-кодов
DDG_SUPPORTED = {
    "wt-wt","us-en","uk-en","de-de","fr-fr","ru-ru","ua-ua","kz-ru","by-ru","tr-tr","ae-en",
    "in-en","sg-en","es-es","it-it","nl-nl","se-sv","no-no","fi-fi","cz-cs","sk-sk",
    "ro-ro","hu-hu","ch-de","br-pt","mx-es","au-en","nz-en"
}

def ddg_region_for(region: str) -> str:
    """Если DDG не знает код (например, kz-kk), берём ближайший валидный."""
    if region in DDG_SUPPORTED:
        return region
    # нормализуем по стране
    if region.startswith("kz-"):
        return "kz-ru"
    if region.startswith("ua-"):
        return "ua-ua"
    if region.startswith("ru-"):
        return "ru-ru"
    if region.startswith("by-"):
        return "by-ru"
    return "wt-wt"

REGION_LANG_ALLOW = {
    "ua-ua": ("uk","ru","en"),
    "by-ru": ("ru","en"),
    "kz-ru": ("ru","en","kk"),
    "kz-kk": ("kk","ru","en"),  # добавил казахский
    "ru-ru": ("ru","en"),
    "wt-wt": ("en","ru","uk","tr","de","fr","es","it","pt","kk")
}

REGION_HINTS = {
    "ua-ua": {
        "tokens":["Україна","Ukraine","Київ","Kyiv","Львів","Lviv","Одеса","Odesa","Дніпро","Dnipro","Харків","Kharkiv","UA"],
        "org":["ТОВ","ФОП","LLC","ПП","ПрАТ","АТ"],
        "cities":["Kyiv","Київ","Lviv","Львів","Odesa","Одеса","Kharkiv","Харків","Dnipro","Дніпро"]
    },
    "by-ru": {
        "tokens":["Беларусь","Belarus","Минск","Minsk","BY","Гродно","Брест","Гомель","Витебск","Могилев"],
        "org":["ООО","ЧУП","ЗАО","ОАО"],
        "cities":["Минск","Minsk","Гродно","Брест","Гомель","Витебск","Могилев"]
    },
    "kz-ru": {
        "tokens":["Казахстан","Kazakhstan","Алматы","Almaty","Астана","Astana","Шымкент","Karaganda","Караганда","Нур-Султан","KZ"],
        "org":["ТОО","LLP","ИП","ЖШС"],
        "cities":["Алматы","Almaty","Астана","Astana","Шымкент","Караганда"]
    },
    "kz-kk": {
        "tokens":["Қазақстан","Kazakhstan","Алматы","Алмалы","Астана","Шымкент","Қарағанды","KZ"],
        "org":["ЖШС","ТОО","LLP","ИП"],
        "cities":["Алматы","Астана","Шымкент","Қарағанды","Орал"]
    },
    "ru-ru": {
        "tokens":["Россия","Russian Federation","Москва","Санкт-Петербург","Новосибирск","Екатеринбург","RF","RU"],
        "org":["ООО","АО","ИП","ПАО","ЗАО","ОАО"],
        "cities":["Москва","Санкт-Петербург","Новосибирск","Екатеринбург","Казань","Нижний Новгород"]
    },
}

# ======================= Intent / Filters =======================
INTENT_AFFILIATE = {
    "affiliate","аффилиат","аффилиэйт","партнерка","партнёрка","партнерская программа","партнёрская программа",
    "реферальная","referral","cpa","игемблинг","игейминг","igaming","casino affiliate","affiliate network",
    "партнеры казино","партнёры казино","аффилейт","афилейт","opm","outsourced affiliate program management",
    "партнер","партнеров","партнерскую","программа","program"
}
INTENT_LEARN = {
    "что такое","what is","определение","definition","гайд","guide","обзор","overview","курс","course","как работает","how to",
    "що таке","лекция","лекція"
}
CASINO_TOKENS = {"казино","casino","igaming","гемблинг","игемблинг","игейминг","беттинг","sportsbook","bookmaker","slots","poker"}

BAD_DOMAINS_HARD = {
    "google.com","maps.google.com","baidu.com","zhihu.com","commentcamarche.net",
    "xnxx.com","pornhub.com",
    "addons.mozilla.org","microsoft.com","support.microsoft.com","edge.microsoft.com",
    "minecraft.net","curseforge.com","planetminecraft.com","softonic.com","apkpure.com","apkcombo.com","uptodown.com",
}
SOFT_BAD_DOMAINS = {
    "wikipedia.org","en.wikipedia.org","ru.wikipedia.org","uk.wikipedia.org",
    "work.ua","rabota.ua","hh.ru","jobs.ua","djinni.co","indeed.com","glassdoor.com",
    "facebook.com","instagram.com","x.com","twitter.com","vk.com","ok.ru",
}

BLOG_HINTS_URL = ("/blog","/news","/article","/articles","/insights","/guide","/guides","/academy","/press","/glossary")
JOBS_HINTS_URL = ("/jobs","/job","/careers","/vacancies","/vacancy","/rabota","/vakans","/career")
EVENT_HINTS = ("conference","expo","summit","event","exhibition","agenda","speakers")
DIRECTORY_HINTS = ("directory","catalog","list","listing","rank","rating","top-","compare","comparison")
CASINO_TLDS = (".casino",".bet",".betting",".poker",".slots",".bingo")

# анти-сигналы
OPERATOR_PROGRAM_HINTS = (
    "join our affiliate program","affiliates program","affiliate portal","commission",
    "revshare","cpa","commission tiers","promotional materials","tracking platform",
    "program terms","payouts","banners","affiliates terms","sign up and start promoting"
)
NETWORK_FOR_PUBLISHERS_HINTS = (
    "for affiliates","for publishers","become an affiliate","sign up as an affiliate",
    "traffic sources","webmasters","start promoting","affiliate network"
)

# сигналы услуг/управления для операторов (агентства/OPM/платформы)
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
    if d in SOFT_BAD_DOMAINS and any(w in u for w in ("job","career","vacanc","rabota","vakans")):
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
    want_programs = any(k in t for k in ("program","программ","партнерк","партнёрк","partners","affiliate program"))
    return {"affiliate": affiliate or casino, "learn": learn, "casino": casino, "business": not learn, "want_programs": want_programs}

# ======================= Request counter (по движкам) =======================
def _load_request_count():
    try:
        with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "engines" not in data:
                data = {"engines": {"ddg":{"count":data.get("count",0)}, "serpapi":{"count":0}},
                        "last_reset": data.get("last_reset", datetime.now().strftime("%Y-%m-%d"))}
            last = datetime.strptime(data["last_reset"], "%Y-%m-%d")
            if last.date() < datetime.now().date():
                return {"engines":{"ddg":{"count":0},"serpapi":{"count":0}}, "last_reset": datetime.now().strftime("%Y-%m-%d")}
            return data
    except Exception:
        return {"engines":{"ddg":{"count":0},"serpapi":{"count":0}}, "last_reset": datetime.now().strftime("%Y-%m-%d")}

def _save_request_count(data):
    try:
        with open(REQUEST_COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save request count: {e}")

def _bump_counter(engine: str):
    data = _load_request_count()
    if engine not in data["engines"]:
        data["engines"][engine] = {"count":0}
    data["engines"][engine]["count"] = int(data["engines"][engine].get("count",0)) + 1
    _save_request_count(data)

def _engine_allowed(engine: str) -> bool:
    data = _load_request_count()
    cnt = int(data["engines"].get(engine, {}).get("count",0))
    return cnt < DAILY_REQUEST_LIMIT

# ======================= Query building =======================
def with_intent_filters(q: str, intent: Dict[str,bool]) -> str:
    parts = [q]
    parts += ["-site:wikipedia.org","-site:facebook.com","-site:instagram.com","-site:x.com","-site:twitter.com","-site:vk.com","-site:ok.ru"]
    parts += ["-jobs","-job","-vacancy","-вакансия","-вакансии","-rabota","-careers","-hh.ru"]
    parts += ["-blog","-news","-guide","-overview","-press","-что","-що","-definition","-glossary","-directory","-listing","-compare"]
    parts += ["-download","-apk","-edge","-minecraft","-forum","-community"]
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
        "igaming acquisition agency for operators",
        "affiliate management for casino operators",
        "igaming affiliate platform for operators",
        # ru/ua
        "игейминг аффилиат агентство",
        "агентство по аффилиат маркетингу казино",
        "управление партнерской программой казино",
        "OPM igaming",
    ]
    seen, out = set(), []
    for q in core:
        q = clean_text(q)
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

REGION_CITY_TOKENS = {
    "ua-ua": ["Київ","Kyiv","Львів","Lviv","Харків","Kharkiv","Одеса","Odesa","Дніпро","Dnipro","Україна","Ukraine"],
    "kz-ru": ["Алматы","Almaty","Астана","Astana","Шымкент","Караганда","Kazakhstan","Казахстан"],
    "kz-kk": ["Алматы","Астана","Шымкент","Қарағанды","Қазақстан","Kazakhstan"],
    "ru-ru": ["Москва","Санкт-Петербург","Екатеринбург","Казань","Новосибирск","Россия","Russian Federation"],
    "by-ru": ["Минск","Гродно","Брест","Беларусь","Belarus"],
}

def apply_geo_bias(queries: List[str], region: str) -> List[str]:
    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = reg["tld"]
    country_hint = {
        "ua-ua": "(Україна OR Ukraine OR Київ OR Lviv OR Одеса OR Dnipro OR Kharkiv)",
        "by-ru": "(Беларусь OR Belarus OR Минск)",
        "kz-ru": "(Казахстан OR Kazakhstan OR Алматы OR Астана OR Шымкент)",
        "kz-kk": "(Қазақстан OR Kazakhstan OR Алматы OR Астана OR Шымкент)",
        "ru-ru": "(Россия OR Moscow OR Санкт-Петербург)",
        "wt-wt": "",
    }.get(region, "")
    out = []
    cities = REGION_CITY_TOKENS.get(region, [])
    for i, q in enumerate(queries):
        if i == 0 and country_hint:
            out.append(f"{q} {country_hint}")
        elif i in (1,2) and tld:
            hint = "site:.{t}".format(t=tld) if i == 1 else "site:*.{t}".format(t=tld)
            out.append(f"{q} {hint}")
        else:
            out.append(q)
        if i < 3 and cities:
            out.append(f"{q} " + "(" + " OR ".join(cities[:6]) + ")")
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
            return datetime.fromisoformat(row[0]) > datetime.utcnow()
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
            return cnt >= 1
    except Exception:
        return False

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

def recent_domain_penalty(region: str, hours: int = 168) -> Dict[str, float]:
    """Понижаем домены, которые уже показывали в последние N часов (по региону)."""
    penalty: Dict[str,float] = {}
    try:
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
              SELECT domain, COUNT(*) FROM results_history
              WHERE region=? AND created_at>=?
              GROUP BY domain
            """, (region, since))
            for dom, cnt in c.fetchall() or []:
                if not dom: continue
                # чем больше раз всплывал, тем сильнее штраф (до -0.6)
                penalty[dom] = -min(0.6, 0.15 * int(cnt))
    except Exception:
        pass
    return penalty

# ======================= Search engines =======================
def duckduckgo_search(query, max_results=25, region="wt-wt"):
    if not _engine_allowed("ddg"):
        logger.error("Daily request limit reached for DDG")
        return []
    urls = []
    try:
        with DDGS() as ddgs:
            ddg_region = ddg_region_for(region)
            results = ddgs.text(query, region=ddg_region, safesearch="moderate", timelimit="y", max_results=max_results)
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
        for block in ("inline_results","local_results","top_stories"):
            for it in data.get(block, []) or []:
                link = it.get("link") or it.get("source")
                if link: urls.append(link)
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

# ======================= Scraping & Deep analysis =======================
def _request_headers():
    return {"User-Agent": DEFAULT_UA, "Accept-Language": random.choice(["ru,en;q=0.9","kk,ru;q=0.9","uk,ru;q=0.9","en;q=0.8"])}

def _http_get(url: str) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=_request_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400: return None
        ct = (resp.headers.get("Content-Type","") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct: return None
        return resp
    except Exception:
        return None

def extract_text_for_classification(resp: requests.Response) -> Tuple[str, str, List[Tuple[str,str]]]:
    if not resp: return "", "", []
    content = resp.content or b""
    enc = detect_encoding(content, resp=resp)
    try:
        soup = BeautifulSoup(content.decode(enc, errors="replace"), "lxml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")
    for bad in soup(["script","style","noscript","svg","picture","source"]):
        bad.decompose()

    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    md = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
    meta_desc = md["content"].strip() if (md and md.get("content")) else ""
    heads = " ".join(h.get_text(" ", strip=True) for h in soup.select("h1, h2, h3")[:8])
    paras = " ".join(p.get_text(" ", strip=True) for p in soup.select("p")[:12])
    text = re.sub(r"\s+", " ", " ".join([meta_desc, heads, paras]).strip())

    links = []
    for a in soup.select("a[href]")[:800]:
        href = a.get("href") or ""
        txt = a.get_text(" ", strip=True)[:160]
        if href:
            links.append((txt, href))
    return short(title, 180), text, links

def _absolute_url(base_url: str, href: str) -> Optional[str]:
    try:
        if not href or href.startswith(("javascript:", "mailto:", "#")): return None
        return urljoin(base_url, href)
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

def fetch_one(url: str, region: str) -> Optional[Dict[str,Any]]:
    try:
        if not url or is_hard_bad_domain(domain_of(url)): return None
        if domain_is_blocked(domain_of(url)): return None
        resp = _http_get(url)
        if not resp: return None
        title, text, links = extract_text_for_classification(resp)
        if not text: return None

        t_low = text.lower()
        if any(h in t_low for h in NETWORK_FOR_PUBLISHERS_HINTS):
            return None

        need_more_region = region_affinity_quick(url, t_low, region) < 0.6
        need_company = not is_company_or_platform(url, t_low)
        extras_text = ""
        if ENABLE_MICROCRAWL and (need_more_region or need_company):
            for ex_url in pick_microcrawl_targets(url, links):
                ex_resp = _http_get(ex_url)
                if not ex_resp: continue
                _, ex_text, _ = extract_text_for_classification(ex_resp)
                if ex_text: extras_text += " " + short(ex_text, 1500)

        full_text = (text + " " + extras_text).strip()
        return {"url": url, "name": title or "N/A", "text": full_text, "links": links}
    except Exception:
        return None

# ======================= Classification & Scoring =======================
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
    if is_operator_program(url, t): return False
    if is_network_for_publishers(t): return False
    ig = any(k in t for k in ("igaming","casino","гембл","игейминг","беттинг","sportsbook","bookmaker"))
    svc = any(k in t for k in ("marketing","growth","acquisition","crm","retention","consulting","services","solutions","agency","оператор"))
    b2b = any(k in t for k in ("for operators","for brands","for advertisers","clients","case study","request a demo","contact sales","request proposal","опер"))
    return ig and svc and b2b

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
    if url.lower().startswith("https://"): score += 0.06
    for k in ("about","contact","services","solutions","platform","features","pricing","clients","case"):
        if k in t: score += 0.02
    if re.search(r"20(2[3-6])", t): score += 0.05
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
    # Доп. региональные сигналы можно добавить здесь (телефоны/валюты), если нужно.
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

def score_item(url: str, text: str, region: str, penalty_map: Optional[Dict[str,float]]=None) -> float:
    d = domain_of(url)
    score = 0.85*base_relevance(text) + 1.1*region_affinity(url, text, region) + quality_score(url, text) \
            + domain_feedback_adjustment(d) + agency_boost(text)
    if any(url.lower().endswith(ct) for ct in CASINO_TLDS): score -= 0.2
    if penalty_map: score += penalty_map.get(d, 0.0)
    return max(0.0, min(2.2, score))

# ======================= Directory harvesting =======================
def _is_external_company_link(base_domain: str, href_abs: str) -> bool:
    d = domain_of(href_abs)
    if not d or d == base_domain: return False
    if d in SOCIAL_HOSTS: return False
    if is_hard_bad_domain(d): return False
    if any(href_abs.lower().endswith(ext) for ext in (".pdf",".zip",".rar",".7z",".doc",".docx",".xls",".xlsx",".ppt",".pptx")):
        return False
    return True

def harvest_from_directory(url: str, text: str, links: List[Tuple[str,str]]) -> List[str]:
    base_domain = domain_of(url)
    out: List[str] = []
    for txt, href in links:
        absu = _absolute_url(url, href)
        if not absu: continue
        if _is_external_company_link(base_domain, absu):
            out.append(normalize_url(absu))
        if len(out) >= DIR_HARVEST_PER_PAGE:
            break
    # дедуп по домену
    seen, dedup = set(), []
    for u in out:
        d = domain_of(u)
        if d not in seen:
            seen.add(d); dedup.append(u)
    return dedup

# Спец-сборщик для affcatalog.com (включая /ru/)
def harvest_affcatalog(base_url: str = "https://affcatalog.com/ru/") -> List[str]:
    urls: List[str] = []
    resp = _http_get(base_url)
    if not resp:
        return urls
    title, text, links = extract_text_for_classification(resp)
    base_dom = domain_of(base_url)
    for txt, href in links:
        absu = _absolute_url(base_url, href)
        if not absu: continue
        d = domain_of(absu)
        # берём только внешние сайты партнёрок/компаний
        if d and d != base_dom and d not in SOCIAL_HOSTS and not is_hard_bad_domain(d):
            urls.append(normalize_url(absu))
        if len(urls) >= DIR_HARVEST_GLOBAL_CAP:
            break
    # дедуп по домену
    seen: Set[str] = set(); out: List[str] = []
    for u in urls:
        dd = domain_of(u)
        if dd in seen: continue
        seen.add(dd); out.append(u)
    return out

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
    # 1) DDG (основной)
    for q in web_queries:
        all_urls.extend(duckduckgo_search(q, max_results=per_query_k, region=region))
    # 2) SerpAPI — если сильно не хватает
    need_fallback = (engine in ("serpapi","both")) and SERPAPI_API_KEY and (len(all_urls) < MIN_RESULTS * 3)
    if need_fallback:
        for q in web_queries:
            all_urls.extend(serpapi_search(q, max_results=per_query_k, region=region))
            if len(all_urls) >= MAX_RESULTS * 20:
                break
    # dedup + блоклисты
    deduped, seen = [], set()
    for u in all_urls:
        if not u or u in seen: continue
        d = domain_of(u)
        if is_hard_bad_domain(d): continue
        if STRONG_BLOCK_ON_BAD and domain_is_blocked(d): continue
        deduped.append(normalize_url(u)); seen.add(u)
    return deduped

def _prefilter_reason(url: str, text: str) -> Optional[str]:
    if looks_like_job(url, text): return "jobs"
    if looks_like_event(url, text): return "event"
    if looks_like_directory(url, text): return "directory"
    if looks_like_blog(url, text): return "blog"
    if is_network_for_publishers(text): return "network_for_publishers"
    if is_operator_program(url, text): return "operator_program"
    return None

def scrape_parallel(urls: List[str], region: str, allow_directory=False, lenient=False, penalty_map: Optional[Dict[str,float]]=None) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Возвращает: (good_results, leftovers_with_text, operator_program_candidates)
    """
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

    reg = REGION_MAP.get(region, REGION_MAP["wt-wt"])
    tld = "." + reg["tld"] if reg.get("tld") else ""

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
            if not language_ok(text, region):
                leftovers.append(item)
                continue

            score = score_item(url, text, region, penalty_map=penalty_map)
            results.append({
                "id": str(uuid.uuid4()),
                "name": name or "N/A",
                "website": url,
                "description": short(text, 400),
                "country": "N/A",
                "source": "Web",
                "score": score
            })

    # Дедуп по домену
    by_domain: Dict[str, List[Dict[str,Any]]] = {}
    for r in results:
        d = domain_of(r["website"])
        by_domain.setdefault(d, []).append(r)
    compact: List[Dict[str,Any]] = []
    for d, items in by_domain.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        compact.append(items[0])

    # Сборка локальных сначала
    if tld:
        local = [r for r in compact if r["website"].lower().endswith(tld) or region_affinity(r["website"], r["description"], region) >= 0.6]
        global_ = [r for r in compact if r not in local]
        local.sort(key=lambda x: x["score"], reverse=True)
        global_.sort(key=lambda x: x["score"], reverse=True)
        want_local_min = max(MIN_RESULTS // 2, 8)
        local_head = local[:max(want_local_min, min(len(local), int(0.7*MAX_RESULTS)))]
        results_final = local_head + (local[len(local_head):] + global_)
    else:
        results_final = sorted(compact, key=lambda x: x["score"], reverse=True)

    return results_final[:MAX_RESULTS], leftovers, operators

def escalate_harvest_from_directories(leftovers: List[Dict[str,Any]]) -> List[str]:
    harvested: List[str] = []
    for it in leftovers:
        url = it.get("url"); text = it.get("text",""); links = it.get("links") or []
        if not looks_like_directory(url or "", text):
            continue
        new_links = harvest_from_directory(url, text, links)
        harvested.extend(new_links)
        if len(harvested) >= DIR_HARVEST_GLOBAL_CAP:
            break
    # дедуп по домену
    seen: Set[str] = set(); out: List[str] = []
    for u in harvested:
        d = domain_of(u)
        if d in seen: continue
        seen.add(d); out.append(u)
    return out

def mix_operator_programs(operators: List[Dict[str,Any]], region: str, penalty_map: Dict[str,float], limit: int) -> List[Dict[str,Any]]:
    picked: List[Dict[str,Any]] = []
    seen: Set[str] = set()
    for it in operators:
        url = it["url"]; text = it["text"]; name = it["name"]
        d = domain_of(url)
        if d in seen: continue
        score = 0.55*base_relevance(text) + 0.9*region_affinity(url, text, region) + quality_score(url, text) + penalty_map.get(d,0.0)
        if any(tok in text.lower() for tok in ("casino","igaming","бетт","казино","sportsbook","bookmaker")):
            picked.append({
                "id": str(uuid.uuid4()),
                "name": name or "N/A",
                "website": url,
                "description": short(text, 400),
                "country": "N/A",
                "source": "Web",
                "score": max(0.0, min(1.6, score))
            })
            seen.add(d)
        if len(picked) >= limit:
            break
    return picked

def ensure_minimum(user_prompt: str, region: str, engine: str, per_query_k: int) -> List[Dict[str,Any]]:
    # намерения и штрафы за повторы
    intent = detect_intent(user_prompt)
    penalty_map = recent_domain_penalty(region)

    # ВОЛНА 1
    queries, _, region, _ = generate_search_queries(user_prompt, region)
    urls = collect_urls(queries, region, engine, per_query_k=per_query_k)
    logger.info(f"Wave1 URLs: {len(urls)}")

    results, leftovers, operators = scrape_parallel(urls[:900], region, allow_directory=True, lenient=False, penalty_map=penalty_map)

    # ВОЛНА 2 (расширенные)
    if len(results) < MIN_RESULTS:
        queries2 = [q for q in apply_geo_bias(build_core_queries(user_prompt), region)]
        queries2 = [with_intent_filters(q, intent) for q in queries2]
        more_urls = collect_urls(queries2, region, engine, per_query_k=max(20, per_query_k))
        merged = list(dict.fromkeys(urls + more_urls))
        logger.info(f"Wave2 URLs add: {len(more_urls)}, merged: {len(merged)}")
        more, leftovers2, operators2 = scrape_parallel(merged[len(urls):len(urls)+1500], region, allow_directory=True, lenient=False, penalty_map=penalty_map)
        known = {r["website"] for r in results}
        for r in more:
            if r["website"] not in known:
                results.append(r); known.add(r["website"])
        leftovers += leftovers2
        operators += operators2

    # ВОЛНА 3: Harvest из каталогов/листингов
    if len(results) < MIN_RESULTS:
        harvested_urls = escalate_harvest_from_directories(leftovers)
        if harvested_urls:
            logger.info(f"Harvested from directories: {len(harvested_urls)}")
            harvest_res, _, ops3 = scrape_parallel(harvested_urls[:1500], region, allow_directory=False, lenient=True, penalty_map=penalty_map)
            known = {r["website"] for r in results}
            for r in harvest_res:
                if r["website"] not in known:
                    results.append(r); known.add(r["website"])
            operators += ops3

    # СПЕЦ-ВОЛНА: affcatalog.com (подмешивание внешних ссылок)
    if len(results) < MAX_RESULTS:
        try:
            aff_urls = harvest_affcatalog("https://affcatalog.com/ru/")
            if aff_urls:
                logger.info(f"Affcatalog harvested: {len(aff_urls)}")
                aff_res, _, ops_aff = scrape_parallel(aff_urls[:800], region, allow_directory=False, lenient=True, penalty_map=penalty_map)
                known = {r["website"] for r in results}
                for r in aff_res:
                    if r["website"] not in known:
                        results.append(r); known.add(r["website"])
                operators += ops_aff
        except Exception as e:
            logger.warning(f"Affcatalog harvest failed: {e}")

    # Микс операторских программ (если запрос этого хочет)
    if intent.get("want_programs") and len(results) < MAX_RESULTS:
        mix = mix_operator_programs(operators, region, penalty_map, limit=min(MIX_OPERATOR_MAX, MAX_RESULTS - len(results)))
        known = {r["website"] for r in results}
        for r in mix:
            if r["website"] not in known:
                results.append(r); known.add(r["website"])

    # если всё ещё мало — SerpAPI soft
    if len(results) < MIN_RESULTS and SERPAPI_API_KEY and (time.time() - _last_serpapi_429_at) >= SERPAPI_COOLDOWN_SEC:
        serp_urls = []
        for q in queries[:6]:
            serp_urls.extend(serpapi_search(q, max_results=25, region=region))
        serp_urls = list(dict.fromkeys(serp_urls))
        logger.info(f"SerpAPI fallback URLs: {len(serp_urls)}")
        soft_res, _, ops_soft = scrape_parallel(serp_urls, region, allow_directory=True, lenient=True, penalty_map=penalty_map)
        known = {r["website"] for r in results}
        for r in soft_res:
            if r["website"] not in known:
                results.append(r); known.add(r["website"])
        if intent.get("want_programs") and len(results) < MAX_RESULTS:
            mix2 = mix_operator_programs(ops_soft, region, penalty_map, limit=min(6, MAX_RESULTS - len(results)))
            for r in mix2:
                if r["website"] not in known:
                    results.append(r); known.add(r["website"])

    # итог: сортировка и обрезка
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
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE","ddg")).lower()
        per_query_k = int(data.get("per_query", 25))

        if not user_query:
            return jsonify({"error":"Query is required"}), 400

        logger.info(f"API request: query='{user_query}', region={region}, engine={engine}")

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM results")
            conn.commit()

        # intent + запись запроса
        intent = detect_intent(user_query)
        if region not in REGION_MAP:
            logger.warning(f"Invalid region {region}, defaulting to wt-wt")
            region = "wt-wt"
        query_id = insert_query_record(user_query, intent, region)

        web_results = ensure_minimum(user_query, region, engine, per_query_k=per_query_k)
        logger.info(f"Final results: {len(web_results)}")

        save_to_db_and_files(web_results, intent, region, query_id)

        return jsonify({
            "results": web_results,
            "region": region,
            "engine": engine,
            "query_id": query_id
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
