import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("python_script_log.txt", "r") as f:
    text = f.read()

splits = {}
meta_data = {}
for line in text.split('\n'):
    if 'layer' in line:
        layer_num = int(line.split(' ')[1])
        if layer_num not in splits:
            splits[layer_num] = {}
            meta_data[layer_num] = {}
    elif 'head' in line:
        head_num = int(line.split(' ')[1])
        if head_num not in splits[layer_num]:
            splits[layer_num][head_num] = []
            meta_data[layer_num][head_num] = []
    else:
        left_val = float(line.split('tensor(')[1].split(',')[0])
        right_val = float(line.split('tensor(')[2].split(',')[0])
        splits[layer_num][head_num].append((left_val - right_val) / (left_val + right_val) * 100)
        meta_data[layer_num][head_num].append((left_val / (left_val + right_val) * 100, right_val / (left_val + right_val) * 100))

# Define state variables
plot_data = []

for threshold in range(110):

    # Count the (layer, head) which have atleast one instance of being below threshold
    violations = []
    for layer in range(40):
        for head in range(40):
            data_array = np.array(splits[layer][head])
            num_contra = (data_array < threshold).sum()
            if num_contra > 0:
                violations.append((layer, head, num_contra))

    # Update state variables
    plot_data.append((threshold, len(violations), violations))

threshold_values, num_violations, violations = zip(*plot_data)

plt.figure(figsize=(8, 6))
plt.plot(threshold_values, num_violations, marker='o', linestyle='-', color='blue')
plt.xlabel('Threshold Values')
plt.ylabel('Number of Violations')
plt.title('Threshold vs. Number of Violations')
plt.grid(True)

plt.tight_layout()
plt.savefig('Threshold_NumViolations.png')

######################

cutoff_threshold = 70
cutoff_num_violations = None
cutoff_violations = None
for thresh, num_viol, viol in plot_data:
    if thresh == cutoff_threshold:
        cutoff_num_violations = num_viol
        cutoff_violations = viol
        break
with open("cutoff_violations.pkl", "wb") as f:
    pickle.dump(cutoff_violations, f)

#####################

sorted_cutoff_violations = sorted(cutoff_violations, key=lambda x: (-x[2], x[0], x[1]))
for layer, head, num_contra in sorted_cutoff_violations:
    print(layer, head, num_contra)
    # for left_val, right_val in meta_data[layer][head]:
    #     if (left_val - right_val) < cutoff_threshold:
    #         print(layer, head, left_val, right_val)