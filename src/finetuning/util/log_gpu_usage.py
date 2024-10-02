import GPUtil
from time import sleep

from src.util.log import LogLevel, log


def trace_gpu_usage(file_name: str = 'gpu_usage.log'):
    average_usage = 0
    average_memory_free = 0
    average_memory_used = 0
    average_memory_total = 0
    average_temperature = 0
    total_num_samples = 0

    while True:
        for gpu in GPUtil.getGPUs():
            log(f'GPU ID: {gpu.id}, Name: {gpu.name}', log_file_name=file_name, level=LogLevel.DEBUG)
            log(f'Belastung: {gpu.load * 100:.1f}%', log_file_name=file_name, level=LogLevel.DEBUG)
            log(f'Freier Speicher: {gpu.memoryFree:.2f} MB', log_file_name=file_name, level=LogLevel.DEBUG)
            log(f'Verwendeter Speicher: {gpu.memoryUsed:.2f} MB', log_file_name=file_name, level=LogLevel.DEBUG)
            log(f'Gesamtspeicher: {gpu.memoryTotal:.2f} MB', log_file_name=file_name, level=LogLevel.DEBUG)
            log(f'Temperatur: {gpu.temperature:.2f} C', log_file_name=file_name, level=LogLevel.DEBUG)
            log('-' * 40, log_file_name=file_name, level=LogLevel.DEBUG)
            average_usage += gpu.load * 100
            average_memory_free += gpu.memoryFree
            average_memory_used += gpu.memoryUsed
            average_memory_total += gpu.memoryTotal
            average_temperature += gpu.temperature
            total_num_samples += 1
        log(f'Average usage: {average_usage / total_num_samples:.1f}%', log_file_name=file_name, level=LogLevel.DEBUG)
        log(
            f'Average memory free: {average_memory_free / total_num_samples:.2f} MB',
            log_file_name=file_name,
            level=LogLevel.DEBUG,
        )
        log(
            f'Average memory used: {average_memory_used / total_num_samples:.2f} MB',
            log_file_name=file_name,
            level=LogLevel.DEBUG,
        )
        log(
            f'Average memory total: {average_memory_total / total_num_samples:.2f} MB',
            log_file_name=file_name,
            level=LogLevel.DEBUG,
        )
        log(
            f'Average temperature: {average_temperature / total_num_samples:.2f} C',
            log_file_name=file_name,
            level=LogLevel.DEBUG,
        )
        log('-' * 40, log_file_name=file_name, level=LogLevel.DEBUG)
        log('=' * 40, log_file_name=file_name, level=LogLevel.DEBUG)

        sleep(2)


if __name__ == '__main__':
    trace_gpu_usage()
