from contextlib import contextmanager
import time


from functools import wraps

from src.util.log import LogLevel, log


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
