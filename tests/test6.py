from hypster import CacheManager, DiskCache, Options, SqlCache, lazy

OpenAiDriver = lazy(CacheManager)
DiskCache = lazy(DiskCache)
SqlCache = lazy(SqlCache)

# Example 6: Multiple cache options
sql_cache = SqlCache(table="cache")
disk_cache = DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp"))
cache_manager = CacheManager(cache=Options([disk_cache, sql_cache]))
