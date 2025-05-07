#!/usr/bin/env python3
"""
sentiment_analyzer.py
---------------------
Ez a modul végzi a hibrid sentiment elemzést, amely a FinBERT, a fejlett Loughran és az AWS Comprehend
elemzést használja. Az eredményt SQLAlchemy ORM objektumként adja vissza.
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import boto3
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tenacity import retry, stop_after_attempt, wait_exponential

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Projekt gyökere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from my_project.common.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
CONFIDENCE_THRESHOLD = 0.7
comprehend = boto3.client("comprehend", region_name=AWS_REGION)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def aws_comprehend_analyze(text: str) -> Tuple[str, float]:
    try:
        response = comprehend.detect_sentiment(Text=text[:5000], LanguageCode="en")
        sentiment = response["Sentiment"].lower()
        confidence = response["SentimentScore"][sentiment.capitalize()]
        mapping = {
            "positive": "pozitív",
            "negative": "negatív",
            "neutral": "semleges",
            "mixed": "semleges"
        }
        return mapping[sentiment], confidence
    except Exception as aws_error:
        logger.error(f"AWS Comprehend hiba: {str(aws_error)}", exc_info=True)
        return "semleges", 0.0

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)

def finbert_analyze(text: str) -> Tuple[str, float]:
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    labels = ["pozitív", "semleges", "negatív"]
    return labels[np.argmax(probs)], float(np.max(probs))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOUGHRAN_PATH = os.path.join(BASE_DIR, "data", "Loughran-McDonald_MasterDictionary_1993-2024.xlsx")

def load_loughran_dict() -> Dict[str, Dict]:
    if not os.path.exists(LOUGHRAN_PATH):
        raise FileNotFoundError(f"Loughran fájl nem található: {LOUGHRAN_PATH}")
    df = pd.read_excel(LOUGHRAN_PATH, sheet_name=0)
    logger.info("Excel oszlopok: %s", df.columns.tolist())
    numeric_cols = ["Word Count", "Word Proportion", "Average Proportion", "Std Dev", "Doc Count",
                    "Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal",
                    "Weak_Modal", "Constraining", "Complexity", "Syllables"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    l_dict = {}
    for _, row in df.iterrows():
        word = str(row["Word"]).strip().lower()
        pos_val = float(row.get("Positive", 0))
        neg_val = float(row.get("Negative", 0))
        if pos_val > neg_val:
            computed_sentiment = "Positive"
        elif neg_val > pos_val:
            computed_sentiment = "Negative"
        else:
            computed_sentiment = "Neutral"
        l_dict[word] = {
            "Seq_num": row.get("Seq_num"),
            "Word Count": row.get("Word Count"),
            "Word Proportion": row.get("Word Proportion"),
            "Average Proportion": row.get("Average Proportion"),
            "Std Dev": row.get("Std Dev"),
            "Doc Count": row.get("Doc Count"),
            "Negative": neg_val,
            "Positive": pos_val,
            "Uncertainty": row.get("Uncertainty"),
            "Litigious": row.get("Litigious"),
            "Strong_Modal": row.get("Strong_Modal"),
            "Weak_Modal": row.get("Weak_Modal"),
            "Constraining": row.get("Constraining"),
            "Complexity": row.get("Complexity", 1),
            "Syllables": row.get("Syllables", 1),
            "Source": row.get("Source"),
            "Computed Sentiment": computed_sentiment
        }
    return l_dict

advanced_loughran_dict = load_loughran_dict()

def advanced_loughran_analyze(text: str) -> Tuple[str, float]:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    total_pos = 0.0
    total_neg = 0.0
    total_weight = 0.0
    for token in tokens:
        if token in advanced_loughran_dict:
            features = advanced_loughran_dict[token]
            try:
                avg_prop = float(features.get("Average Proportion", 0))
                std_dev = float(features.get("Std Dev", 0))
                doc_count = float(features.get("Doc Count", 0))
                complexity = float(features.get("Complexity", 1))
                syllables = float(features.get("Syllables", 1))
                pos_val = float(features.get("Positive", 0))
                neg_val = float(features.get("Negative", 0))
            except Exception:
                continue
            weight = (avg_prop) * (1.0 / (std_dev + 1)) * np.log1p(doc_count) * (1.0 / (complexity + 1)) * (1.0 / (syllables + 1))
            total_pos += pos_val * weight
            total_neg += neg_val * weight
            total_weight += weight
    if total_weight == 0:
        return "semleges", 0.0
    normalized_score = (total_pos - total_neg) / total_weight
    if normalized_score >= 0.1:
        sentiment = "pozitív"
    elif normalized_score <= -0.1:
        sentiment = "negatív"
    else:
        sentiment = "semleges"
    confidence = abs(normalized_score)
    return sentiment, confidence

# SQLAlchemy ORM beállítás
Base = declarative_base()

class NewsSentimentAnalysis(Base):
    __tablename__ = "news_sentiment_analysis"
    id = Column(Integer, primary_key=True)
    news_guid = Column(String(255), unique=True, nullable=False)
    symbol = Column(String(50))
    sentiment = Column(String(20))
    confidence = Column(Float)
    method = Column(String(50))
    details = Column(String)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def update_or_add_analysis(session, result: NewsSentimentAnalysis) -> None:
    try:
        with session.begin():
            existing = session.query(NewsSentimentAnalysis).filter_by(news_guid=result.news_guid).first()
            if existing:
                existing.sentiment = result.sentiment
                existing.confidence = result.confidence
                existing.method = result.method
                existing.details = result.details
                existing.analyzed_at = result.analyzed_at
            else:
                session.add(result)
            session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Adatbázis hiba: {str(e)}", exc_info=True)

def hybrid_sentiment_analysis(news_item: Dict, aws_fallback: bool = True) -> Optional[NewsSentimentAnalysis]:
    try:
        title = news_item.get("title", "")
        content = news_item.get("content", "")
        text = (title + " " + content)[:5000]
        symbol = news_item.get("symbol", "UNKNOWN")
        guid = news_item.get("guid")
        if not guid:
            logger.error("Nincs guid a news_item-ben!")
            return None

        finbert_result, finbert_conf = finbert_analyze(text)
        advanced_loughran_result, advanced_loughran_conf = advanced_loughran_analyze(text)

        if finbert_result == advanced_loughran_result:
            final_result = finbert_result
            confidence = (finbert_conf + advanced_loughran_conf) / 2
            method = "FinBERT + Advanced Loughran"
            details = f"finbert={finbert_result}, advanced_loughran={advanced_loughran_result}"
        elif aws_fallback:
            try:
                aws_result, aws_conf = aws_comprehend_analyze(text)
            except Exception as aws_error:
                logger.error(f"AWS Comprehend hiba: {str(aws_error)}", exc_info=True)
                aws_result, aws_conf = "semleges", 0.0
            details = f"finbert={finbert_result}, advanced_loughran={advanced_loughran_result}, aws={aws_result}({aws_conf:.2f})"
            if aws_conf >= CONFIDENCE_THRESHOLD:
                final_result = aws_result
                confidence = aws_conf
                method = "AWS Comprehend"
            else:
                votes = [finbert_result, advanced_loughran_result, aws_result]
                vote_counts = Counter(votes)
                max_votes = max(vote_counts.values())
                candidates = [k for k, v in vote_counts.items() if v == max_votes]
                if len(candidates) > 1:
                    confs = {finbert_result: finbert_conf, advanced_loughran_result: advanced_loughran_conf, aws_result: aws_conf}
                    chosen = max(candidates, key=lambda x: confs.get(x, 0))
                else:
                    chosen = candidates[0]
                final_result = chosen
                confidence = (finbert_conf + advanced_loughran_conf + aws_conf) / 3
                method = "Mixed Voting"
        else:
            votes = [finbert_result, advanced_loughran_result]
            vote_counts = Counter(votes)
            max_votes = max(vote_counts.values())
            candidates = [k for k, v in vote_counts.items() if v == max_votes]
            if len(candidates) > 1:
                confs = {finbert_result: finbert_conf, advanced_loughran_result: advanced_loughran_conf}
                chosen = max(candidates, key=lambda x: confs.get(x, 0))
            else:
                chosen = candidates[0]
            final_result = chosen
            confidence = max(finbert_conf, advanced_loughran_conf)
            method = "Local Voting"
            details = f"finbert={finbert_result}, advanced_loughran={advanced_loughran_result}"

        return NewsSentimentAnalysis(
            news_guid=guid,
            symbol=symbol,
            sentiment=final_result,
            confidence=round(confidence, 3),
            method=method,
            details=details
        )
    except Exception as e:
        logger.error(f"❌ Elemzési hiba: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    session = Session()
    sample_news = {
        "guid": "example-apple-2025-01",
        "symbol": "AAPL",
        "title": "Apple reports record revenue",
        "content": "Apple reported record revenue in Q1, driven by strong iPhone sales and services growth."
    }
    result = hybrid_sentiment_analysis(sample_news)
    if result:
        update_or_add_analysis(session, result)
        print(f"✅ Elemzés mentve: {result.sentiment} ({result.confidence})")
    else:
        print("❌ Nem sikerült az elemzés.")
