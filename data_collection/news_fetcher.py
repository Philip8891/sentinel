#!/usr/bin/env python3
"""
news_fetcher.py
---------------
Aszinkron hírcsatorna-letöltés, feldolgozás és per-hír sentiment elemzés.
Minden hírhez külön elemzés és szimbólumonkénti mentés.
"""

import sys, os, time, asyncio, ssl, socket, hashlib, calendar, random, heapq
from urllib.parse import urlparse
from datetime import datetime
from collections import defaultdict
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import aiohttp, feedparser
from aiolimiter import AsyncLimiter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.traceback import install
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# Logger és Config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from my_project.common.logger import logger
from my_project.common.config import Config, STOCK_KEYWORDS, CRYPTO_KEYWORDS
from my_project.utils.db_connector import get_db_engine
from my_project.data_processing.sentiment_analyzer import hybrid_sentiment_analysis, NewsSymbol

install()  # Rich színes traceback

# — Prometheus metrikák —
start_http_server(8000)
FEED_HEALTH   = Gauge('feed_health',   'Feed egészségi állapota', ['source'])
FEED_FAILURES = Counter('feed_failures_total', 'Hiba a feed letöltésénél', ['source'])
LATENCY       = Histogram('processing_latency', 'Feldolgozási késleltetés', ['stage'])
SENTIMENT_REQ = Counter('sentiment_requests', 'Sentiment kérések száma')
SENTIMENT_ERR = Counter('sentiment_errors',   'Sentiment hibák száma')

console = Console()
Base = declarative_base()

# — ORM osztályok —
class MarketNews(Base):
    __tablename__ = "market_news"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    symbol    = Column(String(50),  nullable=False)
    source    = Column(String(255), nullable=False)
    title     = Column(String(255), nullable=False)
    content   = Column(Text,        nullable=False)
    guid      = Column(String(255), unique=True)
    published = Column(DateTime,    nullable=False)

class NewsSymbol(Base):
    __tablename__  = "news_symbols"
    news_guid      = Column(String(255), primary_key=True)
    symbol         = Column(String(50),  primary_key=True)
    detected_at    = Column(DateTime,    default=datetime.utcnow)

class BannedChannel(Base):
    __tablename__  = "banned_channels"
    url            = Column(String(512), primary_key=True)
    banned_at      = Column(DateTime,    default=datetime.utcnow)

# Táblák létrehozása
engine = create_engine(Config.NEWS_DB_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# DNS-checker
class DNSChecker:
    @staticmethod
    async def resolve(url: str) -> bool:
        domain = urlparse(url).netloc
        resolver = __import__('dns.resolver').resolver.Resolver()
        resolver.nameservers = Config.DNS_SERVERS + resolver.nameservers
        try:
            await asyncio.to_thread(resolver.resolve, domain, "A")
            return True
        except Exception:
            return False

# Duplikációkezelők
class ContentDeduplicator:
    def __init__(self): self.hashes = set()
    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self.hashes: return True
        self.hashes.add(h); return False

from sentence_transformers import SentenceTransformer, util
class SemanticDeduplicator:
    def __init__(self):
        self.model  = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeds = []
    def is_duplicate(self, text: str, thresh: float = 0.85) -> bool:
        emb = self.model.encode(text)
        for e in self.embeds:
            if util.cos_sim(emb, e).item() > thresh: return True
        self.embeds.append(emb); return False

from langdetect import detect
from cachetools import TTLCache
from googletrans import Translator
class LanguageProcessor:
    def __init__(self):
        self.translator = Translator()
        self.cache      = TTLCache(maxsize=1000, ttl=3600)
    def detect_and_translate(self, text: str) -> str:
        lang = self.cache.get(text) or detect(text); self.cache[text] = lang
        return self.translator.translate(text, src=lang, dest='en').text if lang!='en' else text

# Relevancia és prioritás
class EnhancedRelevanceClassifier:
    def calculate_relevance(self, text: str) -> float:
        score = 0.0
        tl = text.lower()
        for kws in (Config.STOCK_KEYWORDS.values()):
            for kw in kws: score += tl.count(kw.lower())
        for kws in (Config.CRYPTO_KEYWORDS.values()):
            for kw in kws: score += tl.count(kw.lower())
        return score
    def is_relevant(self, text: str) -> bool:
        return self.calculate_relevance(text) >= Config.RELEVANCE_THRESHOLD

class NewsPriorityQueue:
    def __init__(self):
        self.queue   = []
        self.counter = 0  # tie‐breaker
    def add_item(self, item: dict):
        freshness = time.time() - item['published']
        prio = item['relevance_score']*0.7 + (1/(freshness+1))*0.3
        # hasítunk, sorszámmal, hogy ne dobjon TypeError‑t
        heapq.heappush(self.queue, (-prio, self.counter, item))
        self.counter += 1
    def get_next_batch(self, batch_size: int):
        batch = []
        for _ in range(min(batch_size, len(self.queue))):
            _, _, itm = heapq.heappop(self.queue)
            batch.append(itm)
        return batch
    def empty(self) -> bool:
        return not self.queue

class SourceReputationTracker:
    def __init__(self): self.rep = defaultdict(lambda: 1.0)
    def update(self, source: str, relevant: bool):
        delta = 0.1 if relevant else -0.2
        self.rep[source] = max(0.1, min(self.rep[source]+delta, 5.0))
    def get(self, source: str) -> float:
        return self.rep[source]

# RSS fetcher
limiter = AsyncLimiter(Config.REQUESTS_PER_MINUTE, 60)

@retry(stop=stop_after_attempt(Config.MAX_RETRIES),
       wait=wait_random_exponential(multiplier=1, max=60),
       retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    if not await DNSChecker.resolve(url):
        FEED_FAILURES.labels(source=url).inc()
        raise ValueError(f"📡 DNS hiba: {url}")
    async with limiter:
        try:
            async with session.get(url, timeout=Config.REQUEST_TIMEOUT,
                                   ssl=ssl.create_default_context()) as resp:
                resp.raise_for_status()
                FEED_HEALTH.labels(source=url).set(1)
                return await resp.text()
        except aiohttp.ClientResponseError as e:
            FEED_FAILURES.labels(source=url).inc()
            FEED_HEALTH.labels(source=url).set(0)
            raise

async def run_cycle():
    console.print(Panel("📥 [bold underline]Hírek feldolgozása[/bold underline]", expand=False))
    # DB session + banned channels betöltése
    db_session = Session()
    banned = {b.url: b.banned_at.timestamp() for b in db_session.query(BannedChannel).all()}
    banned = {u:ts for u,ts in banned.items() if time.time()-ts < 86400}

    urls = [u for u in Config.STOCK_RSS_URLS + Config.CRYPTO_RSS_URLS if u not in banned]

    # helper osztályok példányosítása
    dedup       = ContentDeduplicator()
    semdedup    = SemanticDeduplicator()
    langproc    = LanguageProcessor()
    relevance   = EnhancedRelevanceClassifier()
    queue       = NewsPriorityQueue()
    reptrack    = SourceReputationTracker()

    # Feed letöltés
    async with aiohttp.ClientSession() as session_http:
        progress = Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), 
                            BarColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn())
        with progress:
            task = progress.add_task("🔄 RSS letöltése...", total=len(urls))
            fetch_tasks = [fetch(session_http, u) for u in urls]
            results     = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            progress.update(task, completed=len(urls))

    # Feldolgozás
    for url, content in zip(urls, results):
        if isinstance(content, Exception):
            console.print(f"❌ [red]{url} hiba: {content}[/red]")
            banned[url] = time.time()
            continue

        feed = feedparser.parse(content)
        for entry in feed.entries[:Config.MAX_ENTRIES_PER_FEED]:
            raw = (entry.get("title","") + " " + entry.get("description","")).strip()
            if dedup.is_duplicate(raw) or semdedup.is_duplicate(raw):
                continue
            eng = langproc.detect_and_translate(raw)
            relevant = relevance.is_relevant(eng)
            reptrack.update(url, relevant)
            if not relevant:
                continue

            guid = (entry.get("id") or entry.get("link",""))[:255]
            pub_ts = calendar.timegm(entry.get("published_parsed", time.gmtime()))
            symbols = []
            tl = eng.lower()
            for sym,kws in Config.STOCK_KEYWORDS.items():
                if any(kw.lower() in tl for kw in kws): symbols.append(sym)
            for sym,kws in Config.CRYPTO_KEYWORDS.items():
                if any(kw.lower() in tl for kw in kws): symbols.append(sym)
            symbols = list(dict.fromkeys(symbols))[:Config.SYMBOL_DETECTION_THRESHOLD]
            if not symbols: continue

            base = {"guid":guid, "source":url,
                    "title":entry.get("title","")[:255],
                    "content":entry.get("description","")[:1024],
                    "published":pub_ts,
                    "relevance_score":relevance.calculate_relevance(eng)}
            for sym in symbols:
                itm = {**base, "symbol":sym}
                queue.add_item(itm)

    # Mentés és per‑hír sentiment
    console.print("⚙️  Mentés és elemzés...")
    while not queue.empty():
        batch = queue.get_next_batch(batch_size=Config.SENTIMENT_BATCH_SIZE)
        db   = Session()
        for itm in batch:
            try:
                mn = MarketNews(symbol=itm["symbol"], source=itm["source"],
                                title=itm["title"], content=itm["content"],
                                guid=itm["guid"],
                                published=datetime.utcfromtimestamp(itm["published"]))
                db.merge(mn)
                db.merge(NewsSymbol(news_guid=itm["guid"], symbol=itm["symbol"]))
                db.commit()
                console.print(f"💾 Mentve: {itm['guid']} / {itm['symbol']}")
            except Exception as e:
                db.rollback()
                console.print(f"❌ Adatbázis hiba: {e}")
                continue

            # Sentiment
            try:
                hybrid_sentiment_analysis({
                    "guid":itm["guid"], "symbol":itm["symbol"],
                    "title":itm["title"], "content":itm["content"]
                })
                SENTIMENT_REQ.inc()
                console.print(f"🧠 Elemzés elkészült: {itm['symbol']} – {itm['guid']}")
            except Exception as e:
                SENTIMENT_ERR.inc()
                console.print(f"❌ Elemzési hiba: {e}")
        db.close()

    # Tiltott csatornák frissítése
    db_session.query(BannedChannel).delete()
    for u in banned:
        db_session.add(BannedChannel(url=u))
    db_session.commit()
    db_session.close()

    console.print("[bold green]✅ Ciklus befejezve, 5 perc múlva újraindul.[/bold green]\n")

async def main():
    while True:
        await run_cycle()
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main())
