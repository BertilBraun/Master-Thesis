from src.log import LogLevel, log
import time


def timeit(message: str, level: LogLevel = LogLevel.INFO):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            log(f'{func.__name__} - {message}: {time.time() - start} seconds', level=level)
            return res

        return wrapper

    return decorator