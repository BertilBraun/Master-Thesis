from contextlib import contextmanager


from src.util.log import LogLevel, log


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
