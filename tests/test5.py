from classes import CacheManager, DiskCache
from hypster import Options, lazy

CacheManager = lazy(CacheManager)
DiskCache = lazy(DiskCache)

cache_manager = CacheManager(cache=DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp")))