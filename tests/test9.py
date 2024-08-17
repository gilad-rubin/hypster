from classes import CacheManager, DiskCache
from hypster import Options, lazy

# Single class, updating globals (returns None, modifies Database in place)
lazy(DiskCache, update_globals=True)

# Now Database is a lazy wrapper
cache = DiskCache(path=Options(["/tmp", "/var/tmp"], default="/tmp"))