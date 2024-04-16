from functools import wraps
import os
import random
import requests
from src.log import LogLevel, log
import time


def timeit(message: str, level: LogLevel = LogLevel.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            log(f'{func.__name__} - {message}: {time.time() - start} seconds', level=level)
            return res

        return wrapper

    return decorator


def download(url: str, file_name: str) -> bool:
    result = requests.get(url)

    if result.status_code != 200:
        log(f'Failed to download file from {url}')
        return False

    with open(file_name, 'wb') as f:
        f.write(result.content)

    return True


# A with temp file context manager
# Usage:
# with TempFile('.pdf') as file_name:
#     download(url, file_name)
# This must use different file names for each instance and should clean up the file after the context manager exits


class TempFile:
    def __init__(self, extension: str):
        self.extension = extension
        self.file_name = None

    def __enter__(self):
        self.file_name = f'temp{random.randint(0, 100000)}{self.extension}'
        return self.file_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_name and os.path.exists(self.file_name):
            os.remove(self.file_name)
        return False
