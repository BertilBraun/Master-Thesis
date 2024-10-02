import hashlib
import json
import os


from functools import wraps

from src.util.json import custom_asdict, FromJsonProtocol, load_json, dump_json


def generate_hashcode(data) -> str:
    # Serialisieren der Liste von Dictionaries in einen JSON-String
    # sort_keys sorgt für konsistente Reihenfolge der Schlüssel

    serialized_data = json.dumps(custom_asdict(data), sort_keys=True, separators=(',', ':'))

    # Erstellen eines Hash-Objekts mit MD5
    hash_object = hashlib.md5()
    hash_object.update(serialized_data.encode('utf-8'))  # Daten müssen als Bytes übergeben werden

    # Rückgabe des Hashcodes als Hexadezimal-String
    return hash_object.hexdigest()


def cache_to_file(folder_name: str, return_type_to_be_able_to_parse_from_file: FromJsonProtocol):
    # This decorator should be usable like @cache to cache the result of a function. The cache mapping should be stored in a file with the given file_name. The cache should be loaded at the beginning of the function and saved at the end of the function. The cache should be a dictionary that maps the arguments to the result of the function.
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(folder_name, exist_ok=True)

            hash_code = generate_hashcode((func.__name__, args, *kwargs.items()))
            cache_file_name = os.path.join(folder_name, hash_code + '.json')

            if os.path.exists(cache_file_name):
                return return_type_to_be_able_to_parse_from_file.from_json(load_json(cache_file_name))

            result = func(*args, **kwargs)

            dump_json(result, cache_file_name)

            return result

        return wrapper

    return decorator
