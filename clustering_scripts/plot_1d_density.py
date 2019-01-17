#!/usr/bin/python3

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import common
plt.figure(figsize=(4,3))
# plt.title("Estimated density for varied regularization")
plt.tight_layout()

# # x = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
# # y = [0.393036, 3.45267, 6.87863, -0.861596, -0.959676, -0.615335, -0.517254]
# # lambda = 0.1
# x = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]
# y = [0.549672, 0.466857, 0.869728, 0.216256, 0.470872, 0.71577, 0.461153, 0.106043, 0.334971, 0.459825, 0.500434, 0.620522, 0.839161, 0.714308, 0.226131]
# zipped = list(zip(x, y))
# zipped.sort()
# x, y = zip(*zipped)
# # for row in zipped:
# #     print(row)

# plt.plot(x, y, label='lambda=0.1')
# # plt.savefig("graphs/density1d_lambda0.1.eps")
# # plt.clf()
# # plt.show()


# # lambda = 0.01
# x = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]
# y = [0.416808, 1.24759, 2.74669, 0.324397, 0.432774, 1.17022, 1.06185, 0.134169, 0.650167, 0.694991, 0.351383, 0.656387, 2.17301, 2.12819, 0.439173]

# # x = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
# # y = [0.395876, 3.45685, 6.88542, -0.864212, -0.963181, -0.620324, -0.521355]
# zipped = list(zip(x, y))
# zipped.sort()
# x, y = zip(*zipped)
# # for row in zipped:
# #     print(row)

# plt.plot(x, y, label='lambda=0.01')
# # plt.savefig("graphs/density1d_lambda0.01.eps")
# # plt.clf()
# # plt.show()

# # lambda = 0.001
# x = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]
# y = [0.00150215, 3.43334, 5.22545, 0.0199517, 0.0199604, 0.238536, 0.238527, 0.00151217, 0.261731, 0.261732, 0.00162667, 0.0181929, 3.12906, 3.12906, 0.0180784]

# # x = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
# # y = [0.395876, 3.45685, 6.88542, -0.864212, -0.963181, -0.620324, -0.521355]
# zipped = list(zip(x, y))
# zipped.sort()
# x, y = zip(*zipped)
# # for row in zipped:
# #     print(row)

# with open('results_WPDM18/lambda_experiments_tiny_0_1_density_eval.csv') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         print(', '.join(row))

fig, ax = plt.subplots(figsize=[0.9*4,0.9*3])
df_dataset = pd.read_csv('results_WPDM18/tiny.arff', names=['x'], skiprows=4)
print(df_dataset)
# print()
ax.scatter(df_dataset['x'], [0 for i in range(len(df_dataset['x']))], c='k', s=common.markersize_scatter, label="dataset")

lambda_values = ['0_05', '0_2', '1_0'] # , '0_001' '0_0001'
for lambda_value in lambda_values:
    df = pd.read_csv('results_WPDM18/lambda_experiments_tiny_' + lambda_value + '_density_eval.csv', names=['x', 'y'])
    # print(df)
    ax.plot(df['x'], df['y'], label="$\lambda = " + lambda_value.replace('_', '.') + "$")
ax.set_xlabel("Domain $[0, 1]$")
ax.set_ylabel("Estimated density")
ax.legend()
plt.savefig("results_WPDM18/density1d_lambda.eps")
# plt.show()
