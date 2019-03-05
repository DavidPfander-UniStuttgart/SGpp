#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re

file_name = "refined_grid.csv"
dataset_file_name = None
df = pd.read_csv(file_name, comment='#') # names=['x', 'y'] header=None,
# print(df)
# print(df.columns.values)

with open(file_name) as f:
    reader = csv.reader(f)
    row1 = next(reader)
    row2 = next(reader)
    if not row2[0].startswith("#"):
        raise
    m = re.match(r"\#filename\: (.*?)$", row2[0])
    if m == None:
        raise
    dataset_file_name = m.group(1)

print("dataset_file_name:", dataset_file_name)

# df_dataset = pd.read_csv("ripley.arff", header=None, names=['x', 'y', 'target'], skiprows=7)
# df_dataset = pd.read_csv("gaussian_c2_size1000_dim2.arff", header=None, names=['x', 'y', 'target'], skiprows=7)
df_dataset = pd.read_csv(dataset_file_name, header=None, names=['x', 'y', 'target'], skiprows=7)
# print(df_dataset)

plt.scatter(df_dataset['x'], df_dataset['y'])
plt.scatter(df['x0'], df[' x1'])
plt.xlim((0.0, 1.0))
plt.ylim((0.0, 1.0))
plt.savefig("refined_grid.pdf")
# plt.show()
