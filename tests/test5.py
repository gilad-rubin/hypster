from hypster import CacheManager, DiskCache, Options, lazy

OpenAiDriver = lazy(CacheManager)
DiskCache = lazy(DiskCache)

cache_manager = CacheManager(cache=DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp")))