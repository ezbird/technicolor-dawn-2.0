import re
import matplotlib.pyplot as plt

# Path to your timings.txt
timings_file = '../output/timings.txt'

# Lists to store extracted data
steps = []
time_all = []
time_tree = []
time_wait = []

# Regular expression to match the lines
pattern = re.compile(r'Step:\s+(\d+).+?<all>=([\d\.eE\+\-]+)\s+<tree>=([\d\.eE\+\-]+)\s+<wait>=([\d\.eE\+\-]+)')

with open(timings_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            all_time = float(match.group(2))
            tree_time = float(match.group(3))
            wait_time = float(match.group(4))
            steps.append(step)
            time_all.append(all_time)
            time_tree.append(tree_time)
            time_wait.append(wait_time)

# Convert to arrays
steps = steps
time_all = time_all
time_tree = time_tree
time_wait = time_wait

# Plot
plt.figure(figsize=(10,6))
plt.plot(steps, time_all, label='Total Time (<all>)', color='black')
plt.plot(steps, time_tree, label='Gravity Tree Time (<tree>)', color='blue')
plt.plot(steps, time_wait, label='Waiting Time (<wait>)', color='red')
plt.xlabel('Simulation Step')
plt.ylabel('Time per Step (seconds)')
plt.title('Gadget-4 Timing Breakdown Over Simulation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
