import json
with open("tmp/test/test-yolo-distribution.json", "r") as f:
    data = json.load(f)['class']

import matplotlib.pyplot as plt
import numpy as np

# Get all unique keys across all groups
all_keys = set().union(*(group.keys() for group in data.values()))
# Prepare group_data, ensuring every key exists in each group, defaulting to 0
group_data = {group_name: [group.get(key, 0) for key in all_keys] for group_name, group in data.items()}

group_data_sum = {group_name: sum(group) for group_name, group in group_data.items()}
print(group_data_sum)

class_count = {n: 0 for n in all_keys}
for atkname, values in group_data.items():
    for i, class_name in enumerate(class_count.keys()):
        class_count[class_name] += values[i]

# 根据 class_count 的值排序key，生成all_keys
import operator

sorted_tuples = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
all_keys = [t[0] for t in sorted_tuples]
group_data = {group_name: [group.get(key, 0) for key in all_keys] for group_name, group in data.items()}

max_class_count = 10 # 取前几个类别，不然表格过长
all_keys = [k for k in all_keys[: max_class_count]]
all_keys = [k.replace(" ", "\n") for k in all_keys]
group_data = {k: v[: max_class_count] for k, v in group_data.items()}

# Setup
barWidth = 0.13
r = np.arange(len(all_keys))

plt.rcParams['font.family'] = ['Times New Roman']

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=300)

# : https://www.sojson.com/rgb.html
colors = [
    "#3CB371",     # clean
    "#D3E2B7",     # noise
    "#74AED4",     # DAS
    "#CFAFD4",     # FCA
    "#FF7256",     # MCC-dog
    "#FFB6C1",     # MCC-kite
    "#FFD700",     # MCC-s
]

for i, (group_name, values) in enumerate(group_data.items()):
    ax.bar(
        r + barWidth * i,
        values,
        color=colors[i % len(colors)],
        width=barWidth,
        edgecolor='gray',
        label=group_name,
    )

ax.set_xlabel('class')
ax.set_ylabel('predicted class count')
ax.set_xlabel('category', fontweight='bold', fontsize=11)
ax.set_ylabel('number of each predicted category', fontweight='bold', fontsize=11)
ax.set_xticks([r + barWidth for r in range(len(all_keys))])
ax.set_xticklabels(all_keys)
ax.legend()

plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.12)
plt.savefig("test-metrics-class.png")
plt.savefig("test-metrics-class.pdf")
