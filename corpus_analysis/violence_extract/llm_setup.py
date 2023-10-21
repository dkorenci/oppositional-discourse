from settings import LLM_CACHE_FOLDER, OPENAI_API_KEY

# LANGCHAIN CACHING
from pathlib import Path
from langchain.cache import SQLiteCache

_LANGCHAIN_SQLITE_CACHE = None

def _create_langchain_sqlite_cache():
    return SQLiteCache(database_path=Path(LLM_CACHE_FOLDER)/"langchain-sqlite-cache.db")

def langchain_sqlite_cache():
    global _LANGCHAIN_SQLITE_CACHE
    if _LANGCHAIN_SQLITE_CACHE is None:
        _LANGCHAIN_SQLITE_CACHE = _create_langchain_sqlite_cache()
    return _LANGCHAIN_SQLITE_CACHE


# OPEANAI KEY
import openai
openai.api_key = OPENAI_API_KEY