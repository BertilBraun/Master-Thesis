from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Generator, Generic, Protocol, Type, TypeVar, overload
import requests

from enum import Enum
from dataclasses import is_dataclass

from functools import wraps

from src.log import LogLevel, log


def timeit(message: str, level: LogLevel = LogLevel.INFO):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            # log function name(params) - message: time in seconds (rounded to 3 decimal places)
            params = ', '.join([str(arg) for arg in args] + [f'{k}={v}' for k, v in kwargs.items()])
            log(f'{func.__name__}({params}) - {message}: {time.time() - start:.3f} seconds', level=level)
            return res

        return wrapper

    return decorator


@contextmanager
def timeblock(message: str, level: LogLevel = LogLevel.INFO):
    """
    with timeblock('Sleeping') as timer:
        time.sleep(2)
        print(f'Slept for {timer.elapsed_time} seconds')
        time.sleep(1)

    # Output:
    # Starting Sleeping
    # Slept for 2.001 seconds
    # Timing Sleeping took: 3.002 seconds
    """
    start_time = time.time()  # Record the start time

    class Timer:
        # Nested class to allow access to elapsed time within the block
        @property
        def elapsed_time(self):
            # Calculate elapsed time whenever it's requested
            return time.time() - start_time

    timer = Timer()

    log(f'Starting {message}', level=level)
    try:
        yield timer  # Allow the block to access the timer
    finally:
        log(f'Timing {message} took: {timer.elapsed_time:.3f} seconds', level=level)


def generate_hashcode(data) -> str:
    # Serialisieren der Liste von Dictionaries in einen JSON-String
    # sort_keys sorgt für konsistente Reihenfolge der Schlüssel
    serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))

    # Erstellen eines Hash-Objekts mit MD5
    hash_object = hashlib.md5()
    hash_object.update(serialized_data.encode('utf-8'))  # Daten müssen als Bytes übergeben werden

    # Rückgabe des Hashcodes als Hexadezimal-String
    return hash_object.hexdigest()


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

    try:
        result = requests.get(url)
    except:  # noqa
        log(f'Failed to download file from {url}')
        return False, ''

    if result.status_code != 200:
        log(f'Failed to download file from {url}')
        return False, ''

    os.makedirs('downloads', exist_ok=True)
    with open(file_name, 'wb') as f:
        f.write(result.content)

    log(f'Downloaded file from {url} to {file_name}', level=LogLevel.DEBUG)

    return True, file_name


def text_similarity(a: str, b: str) -> float:
    # count the words in a that are in b including the number of times they appear -> all words of a are in b -> 1
    c = Counter(a.split())
    d = Counter(b.split())
    return sum((c & d).values()) / sum(c.values())


def cache_to_file(file_name: str, return_type_to_be_able_to_parse_from_file):
    # This decorator should be usable like @cache to cache the result of a function. The cache mapping should be stored in a file with the given file_name. The cache should be loaded at the beginning of the function and saved at the end of the function. The cache should be a dictionary that maps the arguments to the result of the function.
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}

            key = json.dumps((args, *kwargs.items()))
            if key in cache:
                return return_type_to_be_able_to_parse_from_file.from_json(cache[key])

            result = func(*args, **kwargs)
            cache[key] = result

            dump_json(cache, file_name)

            return result

        return wrapper

    return decorator


def write_to_file(file_name: str, content: str) -> None:
    dir_name = os.path.dirname(file_name)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(content)


def custom_asdict(obj):
    if is_dataclass(obj):
        result = {}
        for field_name, field_type in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            result[field_name] = custom_asdict(value)
        return result
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [custom_asdict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: custom_asdict(value) for key, value in obj.items()}
    elif callable(obj):
        return obj.__qualname__  # Save the function's qualname if it's a callable
    else:
        return obj


def dump_json(obj: Any, file_name: str) -> None:
    write_to_file(file_name, json.dumps(custom_asdict(obj), indent=4))


T = TypeVar('T')


class FromJsonProtocol(Protocol, Generic[T]):  # type: ignore
    @classmethod
    def from_json(cls: Any, data: dict) -> T:
        ...


@overload
def load_json(file_name: str) -> Any:
    ...


@overload
def load_json(file_name: str, obj_type: Type[FromJsonProtocol[T]]) -> list[T]:
    ...


def load_json(file_name: str, obj_type: Type[FromJsonProtocol[T]] | None = None) -> Any | list[T]:
    if not os.path.exists(file_name):
        log(f'File not found: {file_name}', level=LogLevel.ERROR)
        exit(1)

    # Datei lesen und JSON laden
    with open(file_name, 'r') as f:
        json_data = json.load(f)

    if obj_type is None:
        return json_data

    # Überprüfen, ob json_array eine Liste ist
    if not isinstance(json_data, list):
        raise ValueError('Das JSON-Objekt muss ein Array sein.')

    # Liste der Objekte erstellen
    obj_list = []
    for entry in json_data:
        # Erstellen einer Instanz des obj_type und Initialisieren mit den JSON-Daten
        obj = obj_type.from_json(entry)
        obj_list.append(obj)

    return obj_list


def create_backup(file_path: str) -> tuple[bool, str]:
    if not os.path.exists(file_path):
        log(f'No file to backup found at {file_path}', level=LogLevel.ERROR)
        return False, ''

    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        write_to_file(backup_path, open(file_path, 'r').read())
        return True, backup_path

    for i in range(1, 1000):
        backup_path = file_path + f'.bak({i})'
        if not os.path.exists(backup_path):
            write_to_file(backup_path, open(file_path, 'r').read())
            return True, backup_path

    log(f'Could not create backup for {file_path}', level=LogLevel.ERROR)
    return False, ''


@contextmanager
def json_dumper(file_name: str) -> Generator[Callable[[Any], None], None, None]:
    # with json_dumper('data.json') as dumper:
    #    for i in range(3):
    #        dumper({'a': i})
    # This will write the following content to data.json:
    # [ {"a": 0}, {"a": 1}, {"a": 2} ]
    dir_name = os.path.dirname(file_name)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    create_backup(file_name)

    with open(file_name, 'w') as f:
        f.write('[')
        first = True

        def write(obj: Any) -> None:
            nonlocal first
            if not first:
                f.write(',')
            f.write(json.dumps(custom_asdict(obj), indent=4))
            f.flush()
            first = False

        try:
            yield write
        finally:
            f.write(']')


@contextmanager
def log_all_exceptions(message: str = ''):
    try:
        yield
    except KeyboardInterrupt:
        # if e is keyboard interrupt, exit the program
        raise
    except Exception as e:
        log(f'Error occurred "{message}": {e}', level=LogLevel.ERROR)

        import traceback

        traceback.print_exc()
