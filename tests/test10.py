from classes import CacheManager, DiskCache, SqlCache
from hypster import Options, lazy

lazy([CacheManager, DiskCache, SqlCache], update_globals=True)

disk_cache = DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp"))
sql_cache = SqlCache(table="cache")
cache_manager = CacheManager(cache=Options([disk_cache, sql_cache]))