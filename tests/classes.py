from collections import defaultdict
from typing import Any, Dict, List, Set, Union

class Thermometer:
    def __init__(self, temperature, location):
        self.temperature: float = temperature
        self.location: str = location
        
class OpenAiDriver:
    def __init__(self, max_tokens, model="gpt-4o"):
        self.max_tokens = max_tokens
        self.model = model

    def __repr__(self):
        return f"OpenAiDriver(max_tokens={self.max_tokens}, model={self.model})"

class AnthropicDriver:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"AnthropicDriver(max_tokens={self.max_tokens})"

class CacheManager:
    def __init__(self, cache):
        self.cache = cache

    def __repr__(self):
        return f"CacheManager(cache={self.cache})"

class DiskCache:
    def __init__(self, path="path", cache_dir="cache"):
        self.cache_dir = cache_dir
        self.path = path

    def __repr__(self):
        return f"DiskCache(path={self.path})"

class SqlCache:
    def __init__(self, table):
        self.table = table

    def __repr__(self):
        return f"SqlCache(table={self.table})"
