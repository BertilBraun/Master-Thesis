import os
import time

from pprint import pprint
from enum import Enum


def datetime_str() -> str:
    return date_str() + ' ' + time_str()


def date_str() -> str:
    return time.strftime('%Y-%m-%d')


def time_str() -> str:
    return time.strftime('%H.%M.%S')


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


LOG_FOLDER = f'logs/{date_str()}'
LOG_FILE = LOG_FOLDER + f'/log {time_str()}.log'
LOG_LEVEL = LogLevel.INFO
os.makedirs(LOG_FOLDER, exist_ok=True)
GLOBAL_LOG_FILE = open(LOG_FILE, 'w')


def log(
    *args,
    level: LogLevel = LogLevel.INFO,
    use_pprint: bool = False,
    log_file_name: str = LOG_FILE,
    **kwargs,
) -> None:
    timestamp = f'[{time_str()}]'
    log_level = f'[{level.name}]'

    if log_file_name != LOG_FILE:
        # ensure that the log file folder exists
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        log_file = open(log_file_name, 'a')
    else:
        log_file = GLOBAL_LOG_FILE

    if use_pprint:
        print(timestamp, log_level, end=' ', file=log_file, flush=True)
        pprint(*args, **kwargs, stream=log_file, width=200)
        log_file.flush()
        if level.value >= LOG_LEVEL.value:
            print(timestamp, log_level, end=' ', flush=True)
            pprint(*args, **kwargs, width=120)
    else:
        print(timestamp, log_level, *args, **kwargs, file=log_file, flush=True)
        if level.value >= LOG_LEVEL.value:
            print(timestamp, log_level, *args, **kwargs)

    if log_file_name != LOG_FILE:
        log_file.close()
