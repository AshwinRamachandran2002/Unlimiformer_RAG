import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("python_script_log.txt", "r") as f:
    text = f.read()

splits = {}
for line in text.split('\n'):
    if 'layer' in line:
        layer_num = int(line.split(' ')[1])
        if layer_num not in splits:
            splits[layer_num] = {}
    elif 'head' in line:
        head_num = int(line.split(' ')[1])
        if head_num not in splits[layer_num]:
            splits[layer_num][head_num] = []
    else:
        left_val = float(line.split('tensor(')[1].split(',')[0])
        right_val = float(line.split('tensor(')[2].split(',')[0])
        splits[layer_num][head_num].append((left_val - right_val) / (left_val + right_val) * 100)

violations = []
cum_sum_added = 0
plot_data = []
util_cum_sum = []
util_data = []
for threshold in range(110):
    prev_violations = violations
    violations = []
    for layer in range(40):
        for head in range(40):
            data_array = np.array(splits[layer][head])
            num_contra = (data_array < threshold).sum()
            if num_contra > 0:
                violations.append((layer, head))
                # print("layer", layer, "head", head, "num", num_contra)
    new_added = []
    for violation in violations:
        if violation not in prev_violations:
            new_added.append(violation)
    cum_sum_added += len(new_added)
    plot_data.append((threshold, cum_sum_added))
    util_cum_sum += new_added
    util_data.append((threshold, util_cum_sum))

threshold_values, cumulative_sums = zip(*plot_data)

plt.figure(figsize=(8, 6))

plt.plot(threshold_values, cumulative_sums, marker='o', linestyle='-', color='blue')
plt.xlabel('Threshold Values')
plt.ylabel('Cumulative Sums of Violations')
plt.title('Threshold vs. Cumulative Sums of Violations')
plt.grid(True)

plt.tight_layout()
plt.savefig('threshol_cumulative.png')

######################
cutoff_threshold = 60
for thresh, cum_sum in util_data:
    if thresh == 60:
        with open("nonconform_layer_head.pkl", "wb") as f:
            pickle.dump(cum_sum, f)