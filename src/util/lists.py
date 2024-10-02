from typing import Generator, TypeVar


T = TypeVar('T')


def chunked_iterate(data: list[T], chunk_size: int) -> Generator[list[T], None, None]:
    assert chunk_size >= 1
    chunk: list[T] = []
    for element in data:
        chunk.append(element)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
