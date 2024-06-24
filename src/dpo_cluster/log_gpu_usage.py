import GPUtil
from time import sleep

average_usage = 0
average_memory_free = 0
average_memory_used = 0
average_memory_total = 0
average_temperature = 0
total_num_samples = 0

while True:
    for gpu in GPUtil.getGPUs():
        print(f'GPU ID: {gpu.id}, Name: {gpu.name}')
        print(f'Belastung: {gpu.load * 100:.1f}%')
        print(f'Freier Speicher: {gpu.memoryFree:.2f} MB')
        print(f'Verwendeter Speicher: {gpu.memoryUsed:.2f} MB')
        print(f'Gesamtspeicher: {gpu.memoryTotal:.2f} MB')
        print(f'Temperatur: {gpu.temperature:.2f} C')
        print('-' * 40)
        average_usage += gpu.load * 100
        average_memory_free += gpu.memoryFree
        average_memory_used += gpu.memoryUsed
        average_memory_total += gpu.memoryTotal
        average_temperature += gpu.temperature
        total_num_samples += 1
    print(f'Average usage: {average_usage / total_num_samples:.1f}%')
    print(f'Average memory free: {average_memory_free / total_num_samples:.2f} MB')
    print(f'Average memory used: {average_memory_used / total_num_samples:.2f} MB')
    print(f'Average memory total: {average_memory_total / total_num_samples:.2f} MB')
    print(f'Average temperature: {average_temperature / total_num_samples:.2f} C')
    print('-' * 40)

    sleep(2)
