LOGS = """
TODO Insert the logs here
"""


# Parsing the logs
import re
import matplotlib.pyplot as plt


# Regex to find the load and free memory
pattern = re.compile(r'Belastung: ([0-9.]+)%\nFreier Speicher: ([0-9.]+) MB')

# Extracting data
usage = []
free_memory = []

# Go through each match and extract usage and free memory
for match in re.finditer(pattern, LOGS):
    usage.append(float(match.group(1)))
    free_memory.append(float(match.group(2)))

# Plotting the data
plt.figure(figsize=(10, 5))

# Plot for GPU load
plt.subplot(1, 2, 1)
plt.plot(usage, marker='o', linestyle='-', color='b')
plt.title('GPU Load Over Time')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Load (%)')
plt.grid(True)

# Plot for Free Memory
plt.subplot(1, 2, 2)
plt.plot(free_memory, marker='o', linestyle='-', color='r')
plt.title('GPU Free Memory Over Time')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Free Memory (MB)')
plt.grid(True)

plt.tight_layout()
plt.show()
