import os
import requests
from bs4 import BeautifulSoup
import sqlite3
from transformers import pipeline
import time
import csv
import logging
import uuid
import random
import json
from ddgs import DDGS
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# ========== Optional: Telegram (disabled by default in Railway) ==========
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
if TELEGRAM_ENABLED:
    from telethon.sync import TelegramClient
    from telethon.tl.functions.contacts import SearchRequest
    from telethon.errors import SessionPasswordNeededError, FloodWaitError

# ========== SerpAPI ==========
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY")

# ========== Flask App / CORS ==========
app = Flask(__name__, template_folder=os.getenv("TEMPLATE_FOLDER", "templates"))

# Явно разрешаем localhost и твой Railway-домен. Этого достаточно для дев/прода.
CORS(
    app,
    origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://search-assistance-production.up.railway.app",
    ],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
)

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "Backend is running."

# ========== Logging ==========
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("scraper.log", encoding="utf-8") if LOG_TO_FILE else logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ========== Limits / pauses ==========
REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "1000"))
REQUEST_PAUSE_MIN = float(os.getenv("REQUEST_PAUSE_MIN", "0.8"))
REQUEST_PAUSE_MAX = float(os.getenv("REQUEST_PAUSE_MAX", "1.6"))

# ========== Proxy config ==========
PROXY_CACHE_FILE = "proxies.json"
PROXY_API_URL = "https://www.proxy-list.download/api/v1/get?type=https&anon=elite"
MAX_PROXY_ATTEMPTS = int(os.getenv("MAX_PROXY_ATTEMPTS", "2"))

# ========== Telegram creds (only if enabled) ==========
API_ID = int(os.getenv("TELEGRAM_API_ID", "0")) if TELEGRAM_ENABLED else 0
API_HASH = os.getenv("TELEGRAM_API_HASH", "") if TELEGRAM_ENABLED else ""
PHONE_NUMBER = os.getenv("TELEGRAM_PHONE") if TELEGRAM_ENABLED else None
TELEGRAM_2FA_PASSWORD = os.getenv("TELEGRAM_2FA_PASSWORD") if TELEGRAM_ENABLED else None

# ========== Classifier (CPU by default) ==========
def init_classifier():
    model_main = os.getenv("CLASSIFIER_MODEL", "facebook/bart-large-mnli")
    model_fallback = os.getenv("CLASSIFIER_FALLBACK", "distilbert-base-uncased")
    device = int(os.getenv("CLASSIFIER_DEVICE", "-1"))  # -1 = CPU
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

# ========== DB Init ==========
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

# ========== Helpers ==========
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
    except Exception as e:
        logger.error(f"Error loading request count: {e}")
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
        proxy_list = response.text.strip().split("\n")
        for proxy in proxy_list:
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

def clean_description(description):
    if not description:
        return "N/A"
    soup = BeautifulSoup(str(description), "html.parser")
    text = soup.get_text().strip()
    return " ".join(text.split()[:200])

def is_relevant_url(url, prompt_phrases):
    irrelevant = [
        "zhihu.com","baidu.com","commentcamarche.net","google.com","d4drivers.uk","dvla.gov.uk",
        "youtube.com","reddit.com","affpapa.com","getlasso.co","wiktionary.org","rezka.ag",
        "linguee.com","bab.la","reverso.net","sinonim.org","wordhippo.com","microsoft.com",
        "romeo.com","xnxx.com","hometubeporn.com","porn7.xxx","fuckvideos.xxx"
    ]
    u = (url or "").lower()
    if any(d in u for d in irrelevant):
        return False
    words = [w.lower() for p in prompt_phrases for w in p.split() if len(w) > 3]
    return "t.me" in u or "instagram.com" in u or any(w in u for w in words) or any(p.lower() in u for p in prompt_phrases)

def rank_result(description, prompt_phrases):
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
    return min(score, 1.0)

def analyze_result(description, prompt_phrases):
    specialization = ", ".join(prompt_phrases[:2]).title() if prompt_phrases else "General"
    cleaned = clean_description(description).lower()
    words = [w.lower() for p in prompt_phrases for w in p.split() if len(w) > 3]
    if not classifier:
        status = "Active" if any(w in cleaned for w in words) else "Unknown"
        return True, specialization, status, f"Подходит: Связан с {specialization}"
    labels = [p.title() for p in prompt_phrases[:3]] + ["Social Media", "Other"] or ["Relevant","Other"]
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

# ========== Query generation ==========
def generate_search_queries(prompt, region="wt-wt"):
    valid = ["wt-wt","ua-ua","ru-ru","us-en","de-de","fr-fr","uk-en"]
    prompt = (prompt or "").strip()
    if region not in valid:
        logger.warning(f"Invalid region {region}, defaulting to wt-wt")
        region = "wt-wt"
    phrases = [p.strip() for p in prompt.split(",") if p.strip()]
    if not phrases:
        return [prompt], [], region, [prompt]
    return [prompt], phrases, region, [prompt]

# ========== DuckDuckGo ==========
def duckduckgo_search(query, max_results=15, region="wt-wt"):
    data = load_request_count()
    if data["count"] >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        return []
    logger.info(f"DDG search: '{query}' region={region}")
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region=region, safesearch="moderate", timelimit="y", max_results=max_results)
            for r in results:
                href = r.get("href")
                if href:
                    urls.append(href)
        save_request_count(data["count"] + 1)
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"DDG failed for '{query}': {e}")
        return []

# ========== SerpAPI ==========
REGION_MAP = {
    "wt-wt": {"hl": "en", "gl": "us"},
    "ua-ua": {"hl": "uk", "gl": "ua"},
    "ru-ru": {"hl": "ru", "gl": "ru"},
    "us-en": {"hl": "en", "gl": "us"},
    "de-de": {"hl": "de", "gl": "de"},
    "fr-fr": {"hl": "fr", "gl": "fr"},
    "uk-en": {"hl": "en", "gl": "gb"},
}

def serpapi_search(query, max_results=15, region="wt-wt"):
    if not SERPAPI_API_KEY:
        logger.warning("SERPAPI_API_KEY is not set; skipping SerpAPI")
        return []
    params = {"engine": "google", "q": query, "num": max_results, "api_key": SERPAPI_API_KEY}
    params.update(REGION_MAP.get(region, REGION_MAP["wt-wt"]))
    logger.info(f"SerpAPI search: '{query}' region={region}")
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        urls = [it.get("link") for it in data.get("organic_results", [])[:max_results] if it.get("link")]
        for it in data.get("inline_results", []):
            if it.get("link"):
                urls.append(it["link"])
        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
        return list(dict.fromkeys(urls))[:max_results]
    except Exception as e:
        logger.error(f"SerpAPI failed for '{query}': {e}")
        return []

# ========== Telegram ==========
def telegram_search(queries, prompt_phrases):
    if not TELEGRAM_ENABLED:
        logger.info("Telegram search disabled")
        return []
    results = []
    if not (API_ID and API_HASH and PHONE_NUMBER):
        logger.error("Telegram env not fully configured; skipping")
        return results
    try:
        client = TelegramClient("session_name", API_ID, API_HASH)
        client.connect()
        if not client.is_user_authorized():
            try:
                client.send_code_request(PHONE_NUMBER)
                code = os.getenv("TELEGRAM_LOGIN_CODE")
                if not code:
                    logger.error("TELEGRAM_LOGIN_CODE not provided")
                    return results
                client.sign_in(PHONE_NUMBER, code)
            except SessionPasswordNeededError:
                if not TELEGRAM_2FA_PASSWORD:
                    logger.error("2FA enabled but TELEGRAM_2FA_PASSWORD not set")
                    return results
                client.sign_in(password=TELEGRAM_2FA_PASSWORD)
            except Exception as e:
                logger.error(f"Telegram auth failed: {e}")
                return results
        for q in queries:
            logger.info(f"Searching Telegram for: {q}")
            try:
                result = client(SearchRequest(q=q, limit=10))
                for chat in result.chats:
                    if hasattr(chat, "megagroup") and not chat.megagroup:
                        name = chat.title or "N/A"
                        username = f"t.me/{getattr(chat, 'username', None)}" if getattr(chat, "username", None) else "N/A"
                        description = clean_description(getattr(chat, "description", "N/A"))
                        is_rel, spec, status, suit = analyze_result(description, prompt_phrases)
                        score = rank_result(description, prompt_phrases)
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

# ========== Core scraping ==========
def search_and_scrape_websites(urls, prompt_phrases):
    logger.info(f"Starting scrape of {len(urls)} URLs")
    results = []
    urls = list(dict.fromkeys(urls))[:50]
    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Scraping: {url}")
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

                is_rel, spec, status, suit = analyze_result(description, prompt_phrases)
                score = rank_result(description, prompt_phrases)
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

# ========== API ==========
@app.route("/search", methods=["POST", "OPTIONS"])
def search():
    # Preflight
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.json or {}
        query = data.get("query", "")
        region = data.get("region", "wt-wt")
        use_telegram = bool(data.get("telegram", False)) and TELEGRAM_ENABLED
        engine = (data.get("engine") or os.getenv("SEARCH_ENGINE", "both")).lower()  # ddg | serpapi | both
        max_results = int(data.get("max_results", 15))

        if not query:
            return jsonify({"error": "Query is required"}), 400

        logger.info(f"API request: query='{query}', region={region}, telegram={use_telegram}, engine={engine}")

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

        # Generate & collect URLs
        web_queries, prompt_phrases, region, telegram_queries = generate_search_queries(query, region)
        all_urls = []
        for q in web_queries:
            if engine in ("ddg", "both"):
                all_urls.extend(duckduckgo_search(q, max_results=max_results, region=region))
            if engine in ("serpapi", "both"):
                all_urls.extend(serpapi_search(q, max_results=max_results, region=region))
        all_urls = list(dict.fromkeys(all_urls))
        logger.info(f"Collected {len(all_urls)} unique URLs")

        # Scrape
        web_results = search_and_scrape_websites(all_urls, prompt_phrases)

        # Telegram
        telegram_results = telegram_search(telegram_queries, prompt_phrases) if use_telegram else []
        all_results = web_results + telegram_results
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

# ========== Entry ==========
if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Flask on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=False)

