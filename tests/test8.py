# import os
# from src.azure_doc_ai import AzureAIDocIntelligenceLoader
# from src.cache_manager import CacheManager, DiskCache
from classes import CacheManager, DiskCache
from hypster import Options, lazy


class AzureAIDocIntelligenceLoader:
    def __init__(self, api_key, endpoint, cache_manager):
        self.api_key = api_key
        self.endpoint = endpoint
        self.cache_manager = cache_manager

    def __repr__(self):
        return f"AzureAIDocIntelligenceLoader(api_key={self.api_key}, endpoint={self.endpoint}, cache_manager={self.cache_manager})"

CacheManager = lazy(CacheManager)
DiskCache = lazy(DiskCache)

cache_dir="cache/azure_ai_doc" #TODO: add tags for potential prod changes
cache = DiskCache(cache_dir=cache_dir) #TODO: handle "reference" error
cache_manager_disk = CacheManager(cache=cache) #TODO: handle implicit assignments

api_key = "abc"
endpoint = "cdf"

azure_doc_ai_loader = lazy(AzureAIDocIntelligenceLoader)(
    api_key=api_key,
    endpoint=endpoint,
    cache_manager=Options({"disk" : cache_manager_disk}, default="disk")
)

final_vars = ["azure_doc_ai_loader"] #TODO: find this automatically?