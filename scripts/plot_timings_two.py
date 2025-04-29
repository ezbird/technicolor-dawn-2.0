import re
import matplotlib.pyplot as plt
import numpy as np

# Path to your timings.txt
timings_file = '../output//timings.txt'

# Lists to store extracted data
steps = []
time_all = []
time_tree = []
time_wait = []

current_step = None

# Regular expressions for both kinds of lines
step_pattern = re.compile(r'Step:\s+(\d+)')
time_pattern = re.compile(r'<all>=([\d\.eE\+\-]+)\s+<tree>=([\d\.eE\+\-]+)\s+<wait>=([\d\.eE\+\-]+)')

with open(timings_file, 'r') as f:
    for line in f:
        step_match = step_pattern.search(line)
        if step_match:
            current_step = int(step_match.group(1))
        else:
            time_match = time_pattern.search(line)
            if time_match and current_step is not None:
                all_time = float(time_match.group(1))
                tree_time = float(time_match.group(2))
                wait_time = float(time_match.group(3))
                steps.append(current_step)
                time_all.append(all_time)
                time_tree.append(tree_time)
                time_wait.append(wait_time)
                current_step = None  # reset after using it

steps = np.array(steps)
time_all = np.array(time_all)
time_tree = np.array(time_tree)
time_wait = np.array(time_wait)

# What remains after tree + wait is 'other' (small fetch, stack, etc)
time_other = time_all - (time_tree + time_wait)
time_other = np.clip(time_other, 0, None)  # no negative values

# Plot
plt.figure(figsize=(10,6))
plt.stackplot(steps, time_tree, time_wait, time_other, labels=['Tree Force', 'Waiting', 'Other'],
              colors=['blue', 'red', 'grey'], alpha=0.8)

plt.xlabel('Simulation Step')
plt.ylabel('Time per Step (seconds)')
plt.title('Gadget-4 Step Timing Breakdown (Stacked Area)')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
