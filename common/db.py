# common/db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine as create_sync_engine, text
from common.config import Config
from common.logger import logger

def get_async_engine(db_url):
    try:
        engine = create_async_engine(db_url, pool_pre_ping=True)
        logger.info("✅ Async SQLAlchemy engine létrejött.")
        return engine
    except Exception as e:
        logger.error(f"❌ Async engine hiba: {e}")
        raise

def get_async_session(db_url):
    engine = get_async_engine(db_url)
    return sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

def get_sync_engine(db_url):
    try:
        engine = create_sync_engine(db_url, pool_pre_ping=True)
        engine.connect().close()
        return engine
    except Exception as e:
        logger.error(f"❌ Sync engine hiba: {e}")
        raise
