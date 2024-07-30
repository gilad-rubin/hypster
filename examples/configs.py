from hypster import Select, prep
from dataclasses import dataclass

class CacheInterface:
    pass

class DiskCache(CacheInterface):
    def __init__(self, path, cache_op: str):
        self.path = path
        self.cache_op = cache_op

class MemoryCache(CacheInterface):
    def __init__(self, max_size, cache_op: str):
        self.max_size = max_size
        self.cache_op = cache_op

class SqlCache(CacheInterface):
    def __init__(self, conn_str, table):
        self.conn_str = conn_str
        self.table = table

@dataclass
class CacheManager:
    cache: CacheInterface

# Fixed configurations
cache_manager = prep(CacheManager(cache=Select("cache"))) #this can also be None

#mutual
cache_op = "all"

#exclusive 1
max_size = 1000 #can be automatically inferred from mem definition that its dependent
cache__mem = prep(MemoryCache(max_size=max_size, cache_op=cache_op))
#cache__mem.variant("cache") #this should be automatically infered from name

#bundle 2
path = "data/cache" #TODO: change this to disk_cache_path and adjust the hypster code
cache__disk = prep(DiskCache(path=path, cache_op=cache_op))
#cache__mem.variant("cache")

cache__new = prep(SqlCache(conn_str="sqlite:///data/cache.db", 
                           table="cache"))

class OpenAiDriver:
    def __init__(self, model):
        self.model = model

class AnthropicDriver:
    def __init__(self, model):
        self.model = model

llm_driver = Select("llm_driver")
llm_driver__openai = prep(OpenAiDriver(model="gpt3.5"))
llm_driver__anthropic = prep(AnthropicDriver(model="claude3.5"))
