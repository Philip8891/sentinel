#!/usr/bin/env python3
"""
sentiment_analyzer.py
---------------------
Hibrid sentiment elemzés FinBERT + Advanced Loughran + AWS Comprehend.
Per‑szimbólum eredmények mentése DB‑be.
"""

import sys, os, logging
from typing import Dict
from datetime import datetime
from collections import Counter

# 1) Projekt gyökér hozzáadása
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

# 2) Config és naplózás
from my_project.common.config import Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 3) Sentiment eszközök
import boto3, torch, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tenacity import retry, stop_after_attempt, wait_exponential
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model     = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)
comprehend        = boto3.client("comprehend", region_name=Config.AWS_REGION)
CONFIDENCE_THRESHOLD = 0.7

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def aws_comprehend_analyze(text: str):
    resp = comprehend.detect_sentiment(Text=text[:5000], LanguageCode="en")
    sent = resp["Sentiment"].lower()
    conf = resp["SentimentScore"][sent.capitalize()]
    mapping = {"positive":"pozitív","negative":"negatív","neutral":"semleges","mixed":"semleges"}
    return mapping[sent], conf

def finbert_analyze(text: str):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = finbert_model(**inputs)
    probs   = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    labels  = ["pozitív","semleges","negatív"]
    idx     = int(np.argmax(probs))
    return labels[idx], float(probs[idx])

# 4) Advanced Loughran
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOUGHRAN_PATH = os.path.join(BASE_DIR, "data", "Loughran-McDonald_MasterDictionary_1993-2024.xlsx")

def load_loughran_dict():
    if not os.path.exists(LOUGHRAN_PATH):
        raise FileNotFoundError(f"Loughran fájl nem található: {LOUGHRAN_PATH}")
    df = pd.read_excel(LOUGHRAN_PATH)
    numeric = ["Average Proportion","Std Dev","Doc Count","Positive","Negative","Complexity","Syllables"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    ldict = {}
    for _,r in df.iterrows():
        w = str(r["Word"]).strip().lower()
        ldict[w] = {
            "avg": float(r.get("Average Proportion",0)),
            "std": float(r.get("Std Dev",0)),
            "cnt": float(r.get("Doc Count",0)),
            "pos": float(r.get("Positive",0)),
            "neg": float(r.get("Negative",0)),
            "comp": float(r.get("Complexity",1)),
            "syl": float(r.get("Syllables",1)),
        }
    return ldict

advanced_loughran = load_loughran_dict()

def advanced_loughran_analyze(text: str):
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha() and t not in stopwords.words("english")]
    tot_pos=tot_neg=tot_w=0.0
    for tok in tokens:
        f = advanced_loughran.get(tok)
        if not f: continue
        w = f["avg"]/(f["std"]+1) * np.log1p(f["cnt"]) / ((f["comp"]+1)*(f["syl"]+1))
        tot_pos += f["pos"]*w
        tot_neg += f["neg"]*w
        tot_w   += w
    if tot_w==0: return "semleges", 0.0
    score = (tot_pos - tot_neg)/tot_w
    if score>=0.1:  s="pozitív"
    elif score<=-0.1: s="negatív"
    else:            s="semleges"
    return s, abs(score)

# 5) DB setup és ORM
from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)

class NewsSymbol(Base):
    __tablename__ = "news_symbols"
    news_guid   = Column(String(255), ForeignKey("market_news.guid"), primary_key=True)
    symbol      = Column(String(50),                 primary_key=True)
    detected_at = Column(DateTime, default=datetime.utcnow)

class NewsSentimentAnalysis(Base):
    __tablename__ = "news_sentiment_analysis"
    news_guid   = Column(String(255), ForeignKey("market_news.guid"), primary_key=True)
    symbol      = Column(String(50),                primary_key=True)
    sentiment   = Column(String(20))
    confidence  = Column(Float)
    method      = Column(String(50))
    details     = Column(String)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# 6) Hibrid logika és per‑szimbólum mentés
def hybrid_sentiment_analysis(news_item: Dict, aws_fallback: bool = True) -> None:
    session = Session()
    try:
        guid = news_item.get("guid")
        if not guid:
            logger.error("Nincs guid!") 
            return

        text = (news_item.get("title","")+" "+news_item.get("content",""))[:5000]
        # FinBERT
        fin_res, fin_conf = finbert_analyze(text)
        # Loughran
        lou_res, lou_conf = advanced_loughran_analyze(text)

        if fin_res==lou_res:
            final, conf, method, details = fin_res, (fin_conf+lou_conf)/2, "FinBERT+Loughran", f"finbert={fin_res},loughran={lou_res}"
        else:
            aws_res, aws_conf = aws_comprehend_analyze(text)
            details = f"finbert={fin_res},loughran={lou_res},aws={aws_res}({aws_conf:.2f})"
            if aws_conf>=CONFIDENCE_THRESHOLD:
                final, conf, method = aws_res, aws_conf, "AWS Comprehend"
            else:
                votes = [fin_res,lou_res,aws_res]
                cnt = Counter(votes)
                maxv = cnt.most_common(1)[0][1]
                cands = [v for v,k in cnt.items() if k==maxv]
                confmap = {fin_res:fin_conf,lou_res:lou_conf,aws_res:aws_conf}
                choice = max(cands, key=lambda x:confmap[x])
                final, conf, method = choice, (fin_conf+lou_conf+aws_conf)/3, "Mixed Voting"

        # Szimbólumok lekérése
        symbols = session.query(NewsSymbol.symbol).filter_by(news_guid=guid).all()

        # Mentés
        for (symbol,) in symbols:
            rec = NewsSentimentAnalysis(
                news_guid=guid,
                symbol=symbol,
                sentiment=final,
                confidence=round(conf,3),
                method=method,
                details=details,
                analyzed_at=datetime.utcnow()
            )
            session.merge(rec)

        session.commit()

    except Exception as e:
        session.rollback()
        logger.error(f"E lemzési hiba: {e}", exc_info=True)
    finally:
        session.close()
