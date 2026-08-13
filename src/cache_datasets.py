import pickle, os, logging
from . import config
logger = logging.getLogger("viper.cache")

_CACHE = {}

def load_cached(name, loader_fn, doc_limit, query_limit):
    key = f"{name}_{doc_limit}_{query_limit}"
    if key in _CACHE:
        logger.info(f"[cache] {name} from memory")
        return _CACHE[key]
    cache_file = os.path.join(config.CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_file):
        logger.info(f"[cache] {name} from disk")
        with open(cache_file,"rb") as f:
            data = pickle.load(f)
        _CACHE[key] = data
        return data
    logger.info(f"[cache] {name} downloading...")
    data = loader_fn(doc_limit, query_limit)
    with open(cache_file,"wb") as f:
        pickle.dump(data, f)
    _CACHE[key] = data
    return data
