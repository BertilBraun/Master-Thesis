from src.logging import LogLevel, log
import time


def timeit(message: str, level: LogLevel = LogLevel.INFO):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            log(f'{message}: {time.time() - start} seconds', level=level)
            return res

        return wrapper

    return decorator
