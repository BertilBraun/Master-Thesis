import os
import re
import shutil

from src.util.log import LogLevel, log


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


def write_to_file(file_name: str, content: str) -> None:
    dir_name = os.path.dirname(file_name)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_name, 'w', encoding='utf8') as f:
        f.write(content)


def read_from_file(file_name: str) -> str:
    with open(file_name, 'r', encoding='utf8') as f:
        return f.read()


def create_backup(file_path: str) -> tuple[bool, str]:
    if not os.path.exists(file_path):
        log(f'No file to backup found at {file_path}', level=LogLevel.ERROR)
        return False, ''

    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        write_to_file(backup_path, read_from_file(file_path))
        return True, backup_path

    for i in range(1, 1000):
        backup_path = file_path + f'.bak({i})'
        if not os.path.exists(backup_path):
            write_to_file(backup_path, read_from_file(file_path))
            return True, backup_path

    log(f'Could not create backup for {file_path}', level=LogLevel.ERROR)
    return False, ''


def create_folder_backup(folder_path: str) -> tuple[bool, str]:
    if not os.path.exists(folder_path):
        print(f'No folder to backup found at {folder_path}')
        return False, ''

    backup_path = folder_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copytree(folder_path, backup_path)
        return True, backup_path

    for i in range(1, 1000):
        backup_path = folder_path + f'.bak({i})'
        if not os.path.exists(backup_path):
            shutil.copytree(folder_path, backup_path)
            return True, backup_path

    print(f'Could not create backup for {folder_path}')
    return False, ''
