import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Dummy logs, ersetzen Sie dies durch Ihre echten Log-Daten
LOGS = """
[01.02.01] [DEBUG] GPU ID: 0, Name: Tesla V100-SXM2-32GB
[01.02.01] [DEBUG] Belastung: 71.0%
[01.02.01] [DEBUG] Freier Speicher: 7539.00 MB
[01.02.01] [DEBUG] Verwendeter Speicher: 24954.00 MB
[01.02.01] [DEBUG] Gesamtspeicher: 32768.00 MB
[01.02.01] [DEBUG] Temperatur: 47.00 C
[01.02.01] [DEBUG] ----------------------------------------
[01.02.01] [DEBUG] GPU ID: 1, Name: Tesla V100-SXM2-32GB
[01.02.01] [DEBUG] Belastung: 73.0%
[01.02.01] [DEBUG] Freier Speicher: 4415.00 MB
[01.02.01] [DEBUG] Verwendeter Speicher: 28078.00 MB
[01.02.01] [DEBUG] Gesamtspeicher: 32768.00 MB
[01.02.01] [DEBUG] Temperatur: 55.00 C
[01.02.01] [DEBUG] ----------------------------------------
[01.02.01] [DEBUG] GPU ID: 2, Name: Tesla V100-SXM2-32GB
[01.02.01] [DEBUG] Belastung: 70.0%
[01.02.01] [DEBUG] Freier Speicher: 6465.00 MB
[01.02.01] [DEBUG] Verwendeter Speicher: 26028.00 MB
[01.02.01] [DEBUG] Gesamtspeicher: 32768.00 MB
[01.02.01] [DEBUG] Temperatur: 50.00 C
[01.02.01] [DEBUG] ----------------------------------------
[01.02.01] [DEBUG] GPU ID: 3, Name: Tesla V100-SXM2-32GB
[01.02.01] [DEBUG] Belastung: 67.0%
[01.02.01] [DEBUG] Freier Speicher: 4495.00 MB
[01.02.01] [DEBUG] Verwendeter Speicher: 27998.00 MB
[01.02.01] [DEBUG] Gesamtspeicher: 32768.00 MB
[01.02.01] [DEBUG] Temperatur: 54.00 C
[01.02.01] [DEBUG] ----------------------------------------
[01.02.01] [DEBUG] Average usage: 68.8%
[01.02.01] [DEBUG] Average memory free: 5288.87 MB
[01.02.01] [DEBUG] Average memory used: 27204.13 MB
[01.02.01] [DEBUG] Average memory total: 32768.00 MB
[01.02.01] [DEBUG] Average temperature: 50.92 C
[01.02.01] [DEBUG] ----------------------------------------
[01.02.01] [DEBUG] ========================================"""


# Regex, um GPU ID, Belastung und freien Speicher zu extrahieren
pattern = re.compile(r'GPU ID: (\d+), Name:.*?Belastung: ([0-9.]+)%.*?Freier Speicher: ([0-9.]+) MB', re.S)

# Datenstrukturen für GPU-Daten
gpu_data = defaultdict(lambda: {'usage': [], 'free_memory': []})

# Daten extrahieren und in Dictionaries speichern
for match in re.finditer(pattern, LOGS):
    gpu_id = match.group(1)
    load = float(match.group(2))
    free_memory = float(match.group(3))
    gpu_data[gpu_id]['usage'].append(load)
    gpu_data[gpu_id]['free_memory'].append(free_memory)

# Maximum freier Speicher für Y-Achsen-Skalierung ermitteln
max_free_memory = max(max(data['free_memory']) for data in gpu_data.values())

# Plot-Einstellungen
plt.figure(figsize=(14, 7))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Farben für die Plots
color_index = 0

# Subplot für GPU-Auslastung
plt.subplot(1, 2, 1)
for gpu_id, data in gpu_data.items():
    plt.plot(
        data['usage'], marker='o', linestyle='-', color=colors[color_index % len(colors)], label=f'GPU {gpu_id} Load'
    )
    color_index += 1
plt.title('GPU Load Over Time')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Load (%)')
plt.legend()
plt.grid(True)

# Subplot für freien Speicher
plt.subplot(1, 2, 2)
color_index = 0
for gpu_id, data in gpu_data.items():
    plt.plot(
        data['free_memory'],
        marker='x',
        linestyle='--',
        color=colors[color_index % len(colors)],
        label=f'GPU {gpu_id} Free Memory',
    )
    color_index += 1
plt.ylim(0, max_free_memory)  # Y-Achsen-Skalierung anpassen
plt.title('GPU Free Memory Over Time')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Free Memory (MB)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
