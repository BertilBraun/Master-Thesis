from typing import Callable, TypeVar

import multiprocessing

import torch


T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')


def __process_batch(args: tuple[list[T], U, int, Callable[[list[T], U, int], list[S]]]) -> list[S]:
    elements_batch, extra_args, device_id, func_to_apply = args
    return func_to_apply(elements_batch, extra_args, device_id)


def map_over_devices(
    func_to_apply: Callable[[list[T], U, int], list[S]],
    all_elements: list[T],
    extra_args: U,
) -> list[S]:
    """
    Apply a function to a list of elements in parallel using multiple devices.
    Note: The func_to_apply function must be defined in the global scope.
    """

    num_devices = torch.cuda.device_count()
    elements_per_device = (len(all_elements) + num_devices - 1) // num_devices
    batches = [
        all_elements[i * elements_per_device : min(((i + 1) * elements_per_device), len(all_elements))]
        for i in range(num_devices)
    ]

    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=num_devices) as pool:
        args = [(batches[i], extra_args, i, func_to_apply) for i in range(num_devices)]
        results = pool.map(__process_batch, args)

    return [item for sublist in results for item in sublist]


def my_func(elementes: list['Profile'], extra_args: str, index: int) -> list['Profile']:
    from src.logic.types import Profile

    return elementes + [Profile(domain='This is a test 2: ' + extra_args, competencies=[])] * len(elementes)


if __name__ == '__main__':
    from src.logic.types import Profile

    res = map_over_devices(
        my_func,
        [Profile(domain='This is a test', competencies=[])],
        'Some Extra Arg',
    )

    print('All done', res)
