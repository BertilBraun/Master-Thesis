import os
import re
import time
import requests

from functools import wraps

from src.log import LogLevel, log


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


def sanitize_filename(filename: str) -> str:
    """
    Remove or replace characters that are illegal in filenames.
    """
    illegal_char_pattern = r'[\/:*?"<>|]'
    sanitized = re.sub(illegal_char_pattern, '_', filename)  # Replace illegal characters with underscore
    return sanitized


def generate_filename(url: str, extension: str) -> str:
    """
    Generate a filename based on the URL and current date.
    Ensure the filename is free of illegal characters.
    """
    base_name = url.split('/')[-1]  # Assumes the URL ends with the filename
    if not base_name.lower().endswith(extension):
        base_name += extension  # Ensures the file has a PDF extension
    return sanitize_filename(base_name)


def download(url: str, extension: str = '') -> tuple[bool, str]:
    # Download the file from `url` and save it locally under `file_name`. Return True if the file was successfully downloaded, False otherwise. The file_name is returned as the second element of the tuple.

    file_name = 'downloads/' + generate_filename(url, extension)

    if os.path.exists(file_name):
        log(f'File already exists: {file_name}', level=LogLevel.DEBUG)
        return True, file_name

    result = requests.get(url)

    if result.status_code != 200:
        log(f'Failed to download file from {url}')
        return False, ''

    os.makedirs('downloads', exist_ok=True)
    with open(file_name, 'wb') as f:
        f.write(result.content)

    log(f'Downloaded file from {url} to {file_name}', level=LogLevel.DEBUG)

    return True, file_name


def cache_to_file(file_name: str, return_type_to_be_able_to_parse_from_file):
    # This decorator should be usable like @cache to cache the result of a function. The cache mapping should be stored in a file with the given file_name. The cache should be loaded at the beginning of the function and saved at the end of the function. The cache should be a dictionary that maps the arguments to the result of the function.
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    cache = eval(
                        f.read(),
                        {
                            # add the return_type to globals so that eval can find it
                            return_type_to_be_able_to_parse_from_file.__name__: return_type_to_be_able_to_parse_from_file,
                            **globals(),
                            **locals(),
                        },
                    )
            else:
                cache = {}

            key = (args, frozenset(kwargs.items()))
            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result

            with open(file_name, 'w') as f:
                f.write(str(cache))

            return result

        return wrapper

    return decorator
