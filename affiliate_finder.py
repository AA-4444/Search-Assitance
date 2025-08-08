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
import urllib.parse
import json
from duckduckgo_search import DDGS
from telethon.sync import TelegramClient
from telethon.tl.functions.contacts import SearchRequest
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
import fcntl  # For file locking on Unix-like systems (Railway)
from datetime import datetime

# Setup Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests from frontend

@app.route('/')
def index():
    return render_template('index.html')

# Setup logging with UTF-8 encoding
logging.basicConfig(
    filename="scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

# Track API requests
REQUEST_COUNT_FILE = "request_count.json"
DAILY_REQUEST_LIMIT = 100  # Adjust based on SerpAPI plan (free tier: 100 searches/month)
REQUEST_PAUSE_MIN = 0.5
REQUEST_PAUSE_MAX = 1

# Proxy settings for DuckDuckGo
PROXY_CACHE_FILE = "proxies.json"
PROXY_API_URL = "https://www.proxy-list.download/api/v1/get?type=https&anon=elite"
MAX_PROXY_ATTEMPTS = 3

def load_request_count():
    try:
        with open(REQUEST_COUNT_FILE, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
            last_reset = data.get("last_reset", "")
            if last_reset:
                last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d")
                if last_reset_date.date() < datetime.now().date():
                    data = {"count": 0, "last_reset": datetime.now().strftime("%Y-%m-%d")}
                    save_request_count(0)  # Create the file with default values
            return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        data = {"count": 0, "last_reset": datetime.now().strftime("%Y-%m-%d")}
        save_request_count(0)
        logger.info(f"Created new {REQUEST_COUNT_FILE} with count=0")
        print(f"Created new {REQUEST_COUNT_FILE} with count=0")
        return data

def save_request_count(count):
    try:
        with open(REQUEST_COUNT_FILE, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
            f.seek(0)
            json.dump({"count": count, "last_reset": datetime.now().strftime("%Y-%m-%d")}, f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
    except Exception as e:
        logger.error(f"Failed to save request count: {e}")
        print(f"Failed to save request count: {e}")

def load_proxies():
    try:
        with open(PROXY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info(f"{PROXY_CACHE_FILE} not found, initializing empty proxy list")
        print(f"{PROXY_CACHE_FILE} not found, initializing empty proxy list")
        return []

def save_proxies(proxies):
    try:
        with open(PROXY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(proxies, f)
    except Exception as e:
        logger.error(f"Failed to save proxies: {e}")
        print(f"Failed to save proxies: {e}")

def fetch_free_proxies():
    logger.info("Fetching free proxies from API")
    print("Fetching free proxies from API")
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
        print(f"Fetched {len(proxies)} proxies")
        return proxies
    except Exception as e:
        logger.error(f"Failed to fetch proxies: {e}")
        print(f"Failed to fetch proxies: {e}")
        return []

def get_proxy():
    proxies = load_proxies()
    if not proxies:
        proxies = fetch_free_proxies()
    return random.choice(proxies) if proxies else None

# Setup classifier
try:
    if torch.cuda.is_available():
        device = 0  # CUDA
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple GPU
    else:
        device = -1  # CPU

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        clean_up_tokenization_spaces=True
    )
    logger.info(f"Classifier initialized: facebook/bart-large-mnli on device {device}")
    print(f"Classifier initialized: facebook/bart-large-mnli on device {device}")

except Exception as e:
    logger.error(f"Failed to initialize facebook/bart-large-mnli: {e}")
    print(f"Failed to initialize facebook/bart-large-mnli: {e}")
    classifier = pipeline(
        "zero-shot-classification",
        model="distilbert-base-uncased",
        device=-1,
        clean_up_tokenization_spaces=True
    )
    logger.info("Fallback classifier initialized: distilbert-base-uncased on CPU")
    print("Fallback classifier initialized: distilbert-base-uncased on CPU")

# Initialize SQLite database
def init_db():
    try:
        with sqlite3.connect("/tmp/search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS results (
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
            )''')
            conn.commit()
        logger.info("Database initialized at /tmp/search_results.db")
        print("Database initialized at /tmp/search_results.db")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        print(f"Failed to initialize database: {e}")

# Clean description
def clean_description(description):
    if not description:
        return "N/A"
    soup = BeautifulSoup(str(description), "html.parser")
    text = soup.get_text().strip()
    return ' '.join(text.split()[:200])

# Filter URLs for relevance
def is_relevant_url(url, prompt_phrases):
    irrelevant_domains = ["zhihu.com", "baidu.com", "commentcamarche.net", "google.com", "d4drivers.uk", "dvla.gov.uk", "youtube.com", "reddit.com", "affpapa.com", "getlasso.co", "wiktionary.org", "rezka.ag", "linguee.com", "bab.la", "reverso.net", "sinonim.org", "wordhippo.com", "microsoft.com", "romeo.com", "xnxx.com", "hometubeporn.com", "porn7.xxx", "fuckvideos.xxx"]
    if any(domain in url.lower() for domain in irrelevant_domains):
        logger.info(f"URL {url} skipped due to irrelevant domain")
        return False
    prompt_words = [word.lower() for phrase in prompt_phrases for word in phrase.split() if len(word) > 3]
    return "t.me" in url.lower() or "instagram.com" in url.lower() or any(word in url.lower() for word in prompt_words) or any(phrase.lower() in url.lower() for phrase in prompt_phrases)

# Rank results
def rank_result(description, prompt_phrases):
    score = 0
    description = description.lower()
    prompt_words = [word.lower() for phrase in prompt_phrases for word in phrase.split() if len(word) > 3]
    for word in prompt_words:
        if word in description:
            score += 0.3 if len(word) > 6 else 0.2
    for phrase in prompt_phrases:
        if phrase.lower() in description:
            score += 0.4
    if any(x in description for x in ["t.me", "instagram.com"]):
        score += 0.2
    return min(score, 1.0)

# Analyze result
def analyze_result(description, prompt_phrases):
    specialization = ", ".join(prompt_phrases[:2]).title()
    cleaned_description = clean_description(description).lower()
    prompt_words = [word.lower() for phrase in prompt_phrases for word in phrase.split() if len(word) > 3]
    
    if not classifier:
        logger.warning("No classifier available, saving result anyway")
        print("No classifier available, saving result anyway")
        status = "Active" if any(word in cleaned_description for word in prompt_words) else "Unknown"
        suitability = f"Подходит: Связан с {specialization}"
        return True, specialization, status, suitability
    
    labels = [phrase.title() for phrase in prompt_phrases[:3]] + ["Social Media", "Other"]
    if not labels:
        labels = ["Relevant", "Other"]
    
    try:
        result = classifier(cleaned_description, candidate_labels=labels, multi_label=False)
        logger.info(f"Classifier result for description '{cleaned_description[:50]}...': {result['labels'][0]}")
        is_relevant = result["labels"][0] not in ["Other"] or any(word in cleaned_description for word in prompt_words) or any(phrase.lower() in cleaned_description for phrase in prompt_phrases)
        
        status = "Active" if any(word in cleaned_description for word in prompt_words) else "Unknown"
        suitability = f"Подходит: Соответствует {specialization}" if is_relevant else f"Частично подходит: Связан с {specialization}"
        
        return is_relevant, specialization, status, suitability
    except Exception as e:
        logger.warning(f"Classifier analysis failed: {e}, saving result anyway")
        print(f"Classifier analysis failed: {e}, saving result anyway")
        status = "Active" if any(word in cleaned_description for word in prompt_words) else "Unknown"
        return True, specialization, status, f"Подходит: Связан с {specialization}"

# Generate search queries
def generate_search_queries(prompt, region="wt-wt"):
    valid_regions = ["wt-wt", "ua-ua", "ru-ru", "us-en", "de-de", "fr-fr", "uk-en"]
    prompt = prompt.strip()
    
    if region not in valid_regions:
        logger.warning(f"Invalid region {region}, defaulting to wt-wt")
        print(f"Invalid region {region}, defaulting to wt-wt")
        region = "wt-wt"
    
    prompt_phrases = [p.strip() for p in prompt.split(",")]
    if not prompt_phrases:
        logger.info("No prompt provided, using original prompt")
        print("No prompt provided, using original prompt")
        return [prompt], prompt_phrases, region, [prompt]
    
    logger.info(f"Generating queries from prompt: {prompt}, region: {region}")
    print(f"Generating queries from prompt: {prompt}, region: {region}")
    
    web_queries = [prompt]
    telegram_queries = [prompt]
    
    logger.info(f"Generated 1 web query: {web_queries}")
    logger.info(f"Generated 1 Telegram query: {telegram_queries}")
    print(f"Generated 1 web query: {web_queries}")
    print(f"Generated 1 Telegram query: {telegram_queries}")
    return web_queries, prompt_phrases, region, telegram_queries

# SerpAPI search
def serpapi_search(query, max_results=5, region="wt-wt"):
    request_data = load_request_count()
    request_count = request_data["count"]
    
    if request_count >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        print("Daily request limit reached")
        return None, []
    
    logger.info(f"Performing SerpAPI search for query: {query}, region: {region}, max_results: {max_results}")
    print(f"Performing SerpAPI search for query: {query}, region: {region}, max_results: {max_results}")
    
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        logger.error("SERPAPI_KEY environment variable not set, falling back to DuckDuckGo")
        print("SERPAPI_KEY environment variable not set, falling back to DuckDuckGo")
        return "no_api_key", []
    
    urls = []
    params = {
        "q": query,
        "api_key": api_key,
        "num": max_results,
        "hl": region.split("-")[1] if region != "wt-wt" else "en",
        "gl": region.split("-")[0] if region != "wt-wt" else None,
        "location": "United States" if region == "us-en" else None,
        "tbm": "lcl" if region == "us-en" else None
    }
    params = {k: v for k, v in params.items() if v is not None}
    
    for attempt in range(3):
        try:
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("organic_results", []) or data.get("local_results", [])
            for result in results[:max_results]:
                url = result.get("link") or result.get("website")
                if url:
                    urls.append(url)
                    logger.info(f"Found URL: {url}")
                    print(f"Found URL: {url}")
            
            request_count += 1
            save_request_count(request_count)
            logger.info(f"Request count: {request_count}/{DAILY_REQUEST_LIMIT}")
            print(f"Request count: {request_count}/{DAILY_REQUEST_LIMIT}")
            
            time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
            return "success", list(set(urls))[:max_results]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI search failed for query {query}: {e}")
            print(f"SerpAPI search failed for query {query}: {e}")
            if hasattr(e, 'response') and e.response.status_code in [403, 429]:
                logger.error(f"SerpAPI error {e.response.status_code}: {'Invalid API key' if e.response.status_code == 403 else 'Rate limit exceeded'}, falling back to DuckDuckGo")
                print(f"SerpAPI error {e.response.status_code}: {'Invalid API key' if e.response.status_code == 403 else 'Rate limit exceeded'}, falling back to DuckDuckGo")
                return "error_" + str(e.response.status_code), []
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    logger.warning(f"Max retries reached for SerpAPI query {query}, falling back to DuckDuckGo")
    print(f"Max retries reached for SerpAPI query {query}, falling back to DuckDuckGo")
    return "max_retries", []

# DuckDuckGo search with proxies and retry logic
def duckduckgo_search(query, max_results=5, region="wt-wt"):
    request_data = load_request_count()
    request_count = request_data["count"]
    
    if request_count >= DAILY_REQUEST_LIMIT:
        logger.error("Daily request limit reached")
        print("Daily request limit reached")
        return []
    
    logger.info(f"Performing DuckDuckGo search for query: {query}, region: {region}, max_results: {max_results}")
    print(f"Performing DuckDuckGo search for query: {query}, region: {region}, max_results: {max_results}")
    urls = []
    
    retries = 3
    backoff_factor = 1  # Initial wait of 1 second, doubles each retry
    proxy_attempts = 0
    proxies_used = []
    
    while proxy_attempts <= MAX_PROXY_ATTEMPTS:
        proxy = get_proxy() if proxy_attempts > 0 else None
        if proxy and proxy in proxies_used:
            logger.warning("No more unique proxies available")
            print("No more unique proxies available")
            break
        if proxy:
            proxies_used.append(proxy)
        
        for attempt in range(retries):
            try:
                with DDGS(proxies=proxy) as ddgs:
                    results = ddgs.text(
                        query,
                        region=region,
                        safesearch="moderate",
                        timelimit="y",
                        max_results=max_results
                    )
                    for result in results:
                        url = result.get("href")
                        if url:
                            urls.append(url)
                            logger.info(f"Found URL: {url}, Proxy: {proxy}")
                            print(f"Found URL: {url}, Proxy: {proxy}")
                
                request_count += 1
                save_request_count(request_count)
                logger.info(f"Request count: {request_count}/{DAILY_REQUEST_LIMIT}")
                print(f"Request count: {request_count}/{DAILY_REQUEST_LIMIT}")
                
                time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
                return list(set(urls))[:max_results]
            except Exception as e:
                if "202 Ratelimit" in str(e):
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Rate limit hit for query {query}, retrying in {wait_time} seconds (attempt {attempt + 1}/{retries}, proxy: {proxy})")
                    print(f"Rate limit hit for query {query}, retrying in {wait_time} seconds (attempt {attempt + 1}/{retries}, proxy: {proxy})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"DuckDuckGo search failed for query {query}: {e}, proxy: {proxy}")
                    print(f"DuckDuckGo search failed for query {query}: {e}, proxy: {proxy}")
                    break
        proxy_attempts += 1
        logger.warning(f"Retrying with new proxy (attempt {proxy_attempts}/{MAX_PROXY_ATTEMPTS})")
        print(f"Retrying with new proxy (attempt {proxy_attempts}/{MAX_PROXY_ATTEMPTS})")
    
    logger.warning(f"Max retries and proxies reached for query {query}, returning empty results")
    print(f"Max retries and proxies reached for query {query}, returning empty results")
    return []

# Telegram search
def telegram_search(queries, prompt_phrases):
    results = []
    
    if not queries:
        logger.info("No Telegram queries provided, skipping Telegram search")
        print("No Telegram queries provided, skipping Telegram search")
        return results
    
    # Check for Telegram credentials
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    phone_number = os.environ.get("TELEGRAM_PHONE_NUMBER")
    
    if not all([api_id, api_hash, phone_number]):
        logger.error("Missing Telegram credentials (TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_PHONE_NUMBER), skipping Telegram search")
        print("Missing Telegram credentials, skipping Telegram search")
        return results
    
    try:
        api_id = int(api_id)  # Convert to int here to avoid TypeError
        client = TelegramClient("session_name", api_id, api_hash)
        client.connect()
        
        if not client.is_user_authorized():
            try:
                client.send_code_request(phone_number)
                code = input("Please enter the code you received: ")
                client.sign_in(phone_number, code)
            except SessionPasswordNeededError:
                password = input("Введите пароль для двухфакторной аутентификации: ")
                client.sign_in(password=password)
            except Exception as e:
                logger.error(f"Telegram authentication failed: {e}")
                print(f"Telegram authentication failed: {e}")
                return results
        
        for query in queries:
            logger.info(f"Searching Telegram for query: {query}")
            print(f"Searching Telegram for query: {query}")
            try:
                result = client(SearchRequest(q=query, limit=10))
                for chat in result.chats:
                    if hasattr(chat, "megagroup") and not chat.megagroup:
                        name = chat.title or "N/A"
                        username = f"t.me/{chat.username}" if chat.username else "N/A"
                        description = clean_description(chat.description) if hasattr(chat, "description") else "N/A"
                        country = "N/A"
                        
                        is_relevant, specialization, status, suitability = analyze_result(description, prompt_phrases)
                        score = rank_result(description, prompt_phrases)
                        
                        if is_relevant_url(username, prompt_phrases) or score > 0.1:
                            result = {
                                "id": str(uuid.uuid4()),
                                "name": name,
                                "website": username,
                                "description": description,
                                "country": country,
                                "source": "Telegram",
                                "score": score
                            }
                            results.append(result)
                            logger.info(f"Scraped Telegram: Name={name[:50]}, Channel={username}, Classified as Relevant ({specialization}), Score={score}")
                            print(f"Scraped Telegram: Name={name[:50]}, Channel={username}, Classified as Relevant ({specialization}), Score={score}")
                            save_to_db(result, is_relevant, specialization, status, suitability, score)
                        else:
                            logger.info(f"Skipped Telegram: Name={name[:50]}, Channel={username}, Not Relevant (Score={score})")
                            print(f"Skipped Telegram: Name={name[:50]}, Channel={username}, Not Relevant (Score={score})")
                        
                        time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
            except FloodWaitError as e:
                logger.warning(f"Telegram rate limit hit, waiting {e.seconds} seconds")
                print(f"Telegram rate limit hit, waiting {e.seconds} seconds")
                time.sleep(e.seconds)
            except Exception as e:
                logger.error(f"Telegram search failed for query {query}: {e}")
                print(f"Telegram search failed for query {query}: {e}")
        
        client.disconnect()
    except Exception as e:
        logger.error(f"Telegram client failed: {e}")
        print(f"Telegram client failed: {e}")
    
    return results

# Scrape and analyze websites
def search_and_scrape_websites(queries, prompt_phrases, max_results=5, region="wt-wt"):
    logger.info(f"Starting web search for {len(queries)} queries in region: {region}")
    print(f"Starting web search for {len(queries)} queries in region: {region}")
    results = []
    urls = []
    
    total_queries = len(queries)
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing web query {i}/{total_queries}: {query}")
        print(f"Processing web query {i}/{total_queries}: {query}")
        for attempt in range(2):
            try:
                status, serp_urls = serpapi_search(query, max_results, region)
                if status == "success":
                    urls.extend(serp_urls)
                    break
                elif status in ["no_api_key", "error_403", "error_429", "max_retries"]:
                    logger.info(f"SerpAPI failed ({status}), falling back to DuckDuckGo for query: {query}")
                    print(f"SerpAPI failed ({status}), falling back to DuckDuckGo for query: {query}")
                    urls.extend(duckduckgo_search(query, max_results, region))
                    break
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1} failed for query {query}: {e}")
                print(f"Search attempt {attempt + 1} failed for query {query}: {e}")
                time.sleep(random.uniform(0.5, 1))
    
    if not urls:
        logger.warning("No URLs found from search")
        print("No URLs found from search")
    else:
        urls = list(set(urls))[:30]
        logger.info(f"Total unique URLs found: {len(urls)}")
        print(f"Total unique URLs found: {len(urls)}")
        
        total_urls = len(urls)
        for i, url in enumerate(urls, 1):
            logger.info(f"Scraping URL {i}/{total_urls}: {url}")
            print(f"Scraping URL {i}/{total_urls}: {url}")
            proxy_attempts = 0
            proxies_used = []
            success = False
            
            while proxy_attempts <= MAX_PROXY_ATTEMPTS:
                proxy = get_proxy()
                if proxy and proxy in proxies_used:
                    logger.warning(f"No more unique proxies available for {url}")
                    print(f"No more unique proxies available for {url}")
                    break
                if proxy:
                    proxies_used.append(proxy)
                
                try:
                    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"}
                    response = requests.get(url, headers=headers, timeout=10, verify=False, proxies=proxy)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, "html.parser")
                    
                    name_selectors = ["h1[class*='brand'], h1[class*='partner'], .company-name, .brand-name, .site-title, .logo-text, title, h1, .header-title"]
                    description_selectors = [".description, .about, .content, .intro, div[class*='about'], section[class*='program'], p[class*='description'], meta[name='description'], div[class*='overview'], p"]
                    country_selectors = [".location, .country, .address, .footer-address, div[class*='location'], div[class*='address'], footer, .contact-info"]
                    
                    name = None
                    for selector in name_selectors:
                        element = soup.select_one(selector)
                        if element:
                            name = clean_description(element.text)
                            if len(name) > 5:
                                break
                    name = name or "N/A"
                    
                    description = None
                    for selector in description_selectors:
                        element = soup.select_one(selector)
                        if element:
                            description = clean_description(element.get("content") if element.get("content") else element.text)
                            if len(description) > 10:
                                break
                    description = description or "N/A"
                    
                    country = None
                    for selector in country_selectors:
                        element = soup.select_one(selector)
                        if element:
                            country = clean_description(element.text)
                            if len(country) > 2:
                                break
                    country = country or "N/A"
                    
                    is_relevant, specialization, status, suitability = analyze_result(description, prompt_phrases)
                    score = rank_result(description, prompt_phrases)
                    source = "SerpAPI" if url in serp_urls else "DuckDuckGo"
                    if is_relevant_url(url, prompt_phrases) or score > 0.1:
                        result = {
                            "id": str(uuid.uuid4()),
                            "name": name,
                            "website": url,
                            "description": description,
                            "country": country,
                            "source": source,
                            "score": score
                        }
                        results.append(result)
                        logger.info(f"Scraped: Name={name[:50]}, Website={url}, Classified as Relevant ({specialization}), Score={score}, Source={source}, Proxy={proxy}")
                        print(f"Scraped: Name={name[:50]}, Website={url}, Classified as Relevant ({specialization}), Score={score}, Source={source}, Proxy={proxy}")
                        save_to_db(result, is_relevant, specialization, status, suitability, score)
                    else:
                        logger.info(f"Skipped: Name={name[:50]}, Website={url}, Not Relevant (Score={score}), Source={source}, Proxy={proxy}")
                        print(f"Skipped: Name={name[:50]}, Website={url}, Not Relevant (Score={score}), Source={source}, Proxy={proxy}")
                    
                    success = True
                    time.sleep(random.uniform(REQUEST_PAUSE_MIN, REQUEST_PAUSE_MAX))
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if any(err in error_str for err in ["403", "429", "connection", "timeout", "ssl"]):
                        logger.warning(f"Error for {url}: {e}, attempting with proxy (attempt {proxy_attempts + 1})")
                        print(f"Error for {url}: {e}, attempting with proxy (attempt {proxy_attempts + 1})")
                        proxy_attempts += 1
                        if proxy_attempts > MAX_PROXY_ATTEMPTS:
                            logger.warning(f"Max proxy attempts reached for {url}, skipping")
                            print(f"Max proxy attempts reached for {url}, skipping")
                            break
                    else:
                        logger.error(f"Scraping failed for {url}: {e}, skipping")
                        print(f"Scraping failed for {url}: {e}, skipping")
                        break
                
                if success:
                    break
            
            if not success:
                logger.warning(f"Failed to scrape {url} after all attempts")
                print(f"Failed to scrape {url} after all attempts")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Total web results scraped: {len(results)}")
    print(f"Total web results scraped: {len(results)}")
    return results

# Save to SQLite
def save_to_db(result, is_relevant, specialization, status, suitability, score):
    try:
        with sqlite3.connect("/tmp/search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT OR IGNORE INTO results (id, name, website, description, specialization, country, source, status, suitability, score)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (result["id"], result["name"], result["website"], result["description"], specialization, result["country"], result["source"], status, suitability, score))
            conn.commit()
        logger.info(f"Saved to DB: Name={result['name'][:50]}, Specialization={specialization}, Status={status}, Score={score}")
        print(f"Saved to DB: Name={result['name'][:50]}, Specialization={specialization}, Status={status}, Score={score}")
    except sqlite3.Error as e:
        logger.error(f"Error saving to SQLite: {e}")
        print(f"Error saving to SQLite: {e}")

# Save to CSV
def save_to_csv():
    try:
        with sqlite3.connect("/tmp/search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY score DESC")
            rows = cursor.fetchall()
            if not rows:
                logger.warning("No data to save to CSV")
                print("No data to save to CSV")
                return
            with open("search_results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "Website", "Description", "Specialization", "Country", "Source", "Status", "Suitability", "Score"])
                writer.writerows(rows)
            logger.info("CSV file created: search_results.csv")
            print("CSV file created: search_results.csv")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        print(f"Error saving to CSV: {e}")

# Save to TXT
def save_to_txt():
    try:
        with sqlite3.connect("/tmp/search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY score DESC")
            results = cursor.fetchall()
            if not results:
                logger.warning("No data to save to TXT")
                print("No data to save to TXT")
                with open("search_results.txt", "w", encoding="utf-8") as f:
                    f.write("Активных результатов не найдено.\n")
                return
            with open("search_results.txt", "w", encoding="utf-8") as f:
                f.write("Найденные результаты:\n\n")
                for row in results:
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
                    f.write("\n" + "-"*50 + "\n")
            logger.info("TXT file created: search_results.txt")
            print("TXT file created: search_results.txt")
    except Exception as e:
        logger.error(f"Error saving to TXT: {e}")
        print(f"Error saving to TXT: {e}")

# Flask API endpoints
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query', '')
        region = data.get('region', 'wt-wt')
        use_telegram = data.get('telegram', False)
        
        if not query:
            return jsonify({"error": "Query is required", "results": [], "telegram_enabled": use_telegram, "region": region, "message": "No query provided"}), 400
        
        logger.info(f"Received API request: query={query}, region={region}, telegram={use_telegram}")
        print(f"Received API request: query={query}, region={region}, telegram={use_telegram}")
        
        with sqlite3.connect("/tmp/search_results.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS results")
            cursor.execute('''CREATE TABLE results (
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
            )''')
            conn.commit()
        
        logger.info("Database reset")
        print("Database reset")
        
        web_queries, prompt_phrases, region, telegram_queries = generate_search_queries(query, region)
        
        web_results = search_and_scrape_websites(web_queries, prompt_phrases, max_results=5, region=region)
        
        telegram_results = []
        if use_telegram:
            telegram_results = telegram_search(telegram_queries, prompt_phrases)
        else:
            logger.info("Telegram search skipped")
            print("Telegram search skipped")
        
        all_results = web_results + telegram_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        save_to_csv()
        save_to_txt()
        
        response = {
            "results": all_results,
            "telegram_enabled": use_telegram,
            "region": region,
            "message": "No results found, possibly due to rate limits or no matching content" if not all_results else "Search completed successfully"
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"API error: {e}")
        print(f"API error: {e}")
        return jsonify({"error": str(e), "results": [], "telegram_enabled": use_telegram, "region": region, "message": "Search failed due to an error"}), 500

@app.route('/download/<filetype>', methods=['GET'])
def download_file(filetype):
    if filetype not in ['csv', 'txt']:
        logger.error(f"Invalid file type requested: {filetype}")
        print(f"Invalid file type requested: {filetype}")
        return jsonify({"error": "Invalid file type, use 'csv' or 'txt'"}), 400
    filename = f"search_results.{filetype}"
    try:
        return send_file(filename, as_attachment=True, mimetype=f'text/{filetype}')
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        print(f"File {filename} not found")
        return jsonify({"error": f"{filename} not found"}), 404
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        print(f"Error downloading {filename}: {e}")
        return jsonify({"error": f"Error downloading {filename}: {str(e)}"}), 500

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5002))
    print(f"Starting Flask server on http://0.0.0.0:{port} (Press CTRL+C to quit)", flush=True)
    try:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask startup failed: {e}")
        print(f"Flask startup failed: {e}", flush=True)
        raise
