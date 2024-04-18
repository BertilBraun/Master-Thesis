import os
import time

from enum import Enum


def datetime_str() -> str:
    return time.strftime('%Y-%m-%d %H %M %S')


def time_str() -> str:
    return time.strftime('%H:%M:%S')


class LogLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


LOG_FILE = f'logs/log {datetime_str()}.txt'
LOG_LEVEL = LogLevel.INFO
os.makedirs('logs', exist_ok=True)
log_file = open(LOG_FILE, 'w')


def log(*args, level: LogLevel = LogLevel.INFO, **kwargs) -> None:
    timestamp = f'[{time_str()}]'
    print(timestamp, *args, **kwargs, file=log_file)
    if level.value >= LOG_LEVEL.value:
        print(timestamp, *args, **kwargs)
