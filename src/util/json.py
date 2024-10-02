from contextlib import contextmanager
import json
import os
from typing import Any, Callable, Generator, Generic, Protocol, Type, TypeVar, overload

from enum import Enum
from dataclasses import is_dataclass


from src.util.log import LogLevel, log
from src.util.files import create_backup, read_from_file, write_to_file


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
    write_to_file(file_name, dump_json_str(obj))


def dump_json_str(obj: Any) -> str:
    return json.dumps(custom_asdict(obj), indent=4, ensure_ascii=False)


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

    file_content = read_from_file(file_name)

    with open(file_name, 'r') as f:
        file_content = f.read()
        try:
            json_data = json.loads(file_content)
        except json.JSONDecodeError:
            json_data = json.loads(file_content + ']')

    if obj_type is None:
        return json_data

    # Überprüfen, ob json_array eine Liste ist
    if not isinstance(json_data, list):
        raise ValueError('Das JSON-Objekt muss ein Array sein.')

    # Liste der Objekte erstellen
    obj_list: list[T] = []
    for entry in json_data:
        # Erstellen einer Instanz des obj_type und Initialisieren mit den JSON-Daten
        obj = obj_type.from_json(entry)
        obj_list.append(obj)

    return obj_list


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
