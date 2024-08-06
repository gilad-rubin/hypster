from classes import CacheManager, DiskCache, SqlCache
from hypster import Options, lazy

CacheManager = lazy(CacheManager)
DiskCache = lazy(DiskCache)
SqlCache = lazy(SqlCache)

# Example 6: Multiple cache options
sql_cache = SqlCache(table="cache")
disk_cache = DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp"))
cache_manager = CacheManager(cache=Options([disk_cache, sql_cache]))
