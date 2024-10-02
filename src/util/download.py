import os
import requests


from src.util.log import LogLevel, log

from src.util.files import generate_filename


def download(url: str, extension: str = '') -> tuple[bool, str]:
    # Download the file from `url` and save it locally under `file_name`. Return True if the file was successfully downloaded, False otherwise. The file_name is returned as the second element of the tuple.

    file_name = 'downloads/' + generate_filename(url, extension)

    if os.path.exists(file_name):
        log(f'File already exists: {file_name}', level=LogLevel.DEBUG)
        return True, file_name

    try:
        result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    except Exception as e:
        log(f'Failed to download file from {url}: {e}')
        return False, ''

    if result.status_code != 200:
        log(f'Failed to download file from {url}')
        return False, ''

    os.makedirs('downloads', exist_ok=True)
    with open(file_name, 'wb') as f:
        f.write(result.content)

    log(f'Downloaded file from {url} to {file_name}', level=LogLevel.DEBUG)

    return True, file_name
