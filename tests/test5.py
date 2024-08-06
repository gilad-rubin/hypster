from hypster import Options, lazy

from .classes import CacheManager, DiskCache

CacheManager = lazy(CacheManager)
DiskCache = lazy(DiskCache)

cache_manager = CacheManager(cache=DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp")))