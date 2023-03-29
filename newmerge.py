import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Lamm import *

plt.rcParams['figure.figsize'] = [8, 12]

importfile = 'IMU1_Anders_Gravity.csv'
importfile2 = 'IMU1_Anders_Quaternion.csv'

#importfile = 'testmed100hz.csv'
#importfile2 = 'testmed100hzq.csv'

df_acc = pd.read_csv(importfile)
df_gyro = pd.read_csv(importfile2)
# print(df_acc.head(5))
# print(df_gyro.head(5))

df_acc.pop('timestamp (+0200)')
df_acc.pop('epoc (ms)')
df_gyro.pop('timestamp (+0200)')
df_gyro.pop('epoc (ms)')

table = pd.merge_asof(df_acc, df_gyro, on='elapsed (s)')

table.iloc[:,5:]*=9.82

# Korrigering

for i, val in enumerate(table.columns):
    table.iloc[:, i] = table.iloc[:, i] - table.iloc[0, i]

fig, axs = plt.subplots(3, 1)
axs[0].grid()
axs[1].grid()
axs[2].grid()
#axs[0].set_xlim([8, 9])
#axs[1].set_xlim([8, 9])
#axs[2].set_xlim([8, 9])

axs[0].plot(table.iloc[:, 0], table.iloc[:, 5], 'r', label="ay")
axs[0].plot(table.iloc[:, 0], table.iloc[:, 6], 'b', label="ax")
axs[0].plot(table.iloc[:, 0], table.iloc[:, 7], 'g', label="az")

totnorm = np.linalg.norm(table.iloc[:, 5:], axis=1)

pos = pd.Series(table.iloc[:, 5]).idxmax()
posmin = pd.Series(table.iloc[:, 5]).idxmin()
val = max(table.iloc[:, 5])

between = range(pos - 5, pos + 7)

axs[0].plot(table.iloc[between[0], 0], table.iloc[between[0], 5], '*')
axs[0].annotate('Jump force starts', xy=(table.iloc[between[0], 0], table.iloc[between[0], 5]), xytext=(8, 3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
axs[0].plot(table.iloc[between[-1], 0], table.iloc[between[-1], 5], '*')
axs[0].annotate('Jump force ends', xy=(table.iloc[between[-1], 0], table.iloc[between[-1], 5]), xytext=(8.3, 3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

num1 = range(pos + 20, pos + 30)
# finner var värdet är närmast 1. Där kan man anse att man har hamnat i högsta läget av hoppet.
error = 1
for i in num1:
    value = sum(table.iloc[i, 5:])
    if (abs(value) - 1) < error:
        error = abs(value - 1)
        position = i

axs[0].axvline(table.iloc[position, 0])
axs[0].annotate(f'Highest position with force {error}', xy=(table.iloc[position, 0], table.iloc[position, 5]),
                xytext=(8.6, 3), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
plt.grid()
axs[0].legend()
#plt.xlim([8, 9])


def avgforce(between, yval):
    valmin = yval[between[0]]
    val = np.sum(abs(yval[between]))
    return val


force = avgforce(between, totnorm)

axs[1].plot(table.iloc[:, 0], totnorm)
axs[1].fill_between(table.iloc[between, 0], totnorm[between], color='blue', alpha=.5)
axs[1].annotate(f'Rörelsemängd = {force}', xy=(table.iloc[between[6], 0], table.iloc[between[6], 5]), xytext=(8.2, 3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# Gyroscope
v=[]
for i, val in table.iterrows():
    if sum(val[1:5])==0:
        val[1]+=0.001
    r = R.from_quat(val[1:5])
    v.append(r.as_euler('zyx', degrees=True))
#print(v)







# Integrering

time = table.iloc[:, 0]

dt = time[1] - time[0]

vx = [0]
vy = [0]
vz = [0]
vx_t = [0]
vy_t = [0]
vz_t = [0]

b = 0  # beta, rotation kring x-axel
y = 0  # keppa rotation kring y-axel
a = 0  # alfa, rotation kring z-axel

for i in np.arange((len(time)) - 1):
    b = b + table.iloc[:, 2][i] * (time[i + 1] - time[i])
    y = y + table.iloc[:, 3][i] * (time[i + 1] - time[i])
    a = a + table.iloc[:, 4][i] * (time[i + 1] - time[i])

    vx = vx + [vx[-1] + table.iloc[:, 5][i] * (time[i + 1] - time[i])]
    vy = vy + [vy[-1] + table.iloc[:, 6][i] * (time[i + 1] - time[i])]
    vz = vz + [vz[-1] + table.iloc[:, 7][i] * (time[i + 1] - time[i])]

    vx_t = vx_t + [vx[-1] * ((np.cos(b) * np.cos(y)) + (np.cos(b) * np.sin(y)) - np.sin(b))]
    vy_t = vy_t + [vy[-1] * ((np.sin(a) * np.sin(b) * np.cos(y) - np.cos(a) * np.sin(y)) + (
            np.sin(a) * np.sin(b) * np.sin(y) + np.cos(a) * np.cos(y)) + (np.sin(a) * np.cos(b)))]
    vz_t = vz_t + [vz[-1] * ((np.cos(a) * np.sin(b) * np.cos(y) + np.sin(a) * np.sin(y)) + (
            np.cos(a) * np.sin(b) * np.sin(y) - np.sin(a) * np.cos(y)) + (np.cos(a) * np.cos(b)))]

axs[2].plot(time, vx_t, 'r', label="vx")
axs[2].plot(time, vy_t, 'b', label="vy")
axs[2].plot(time, vz_t, 'g', label="vz")

error = 100
for i in num1:
    value = sum([vx_t[i], vy_t[i], vz_t[i]])
    if abs(value) < error:
        errorv = abs(value - 1)
        positionv = i

axs[2].plot(time[positionv], vy_t[positionv], 'o')

# get_positions(table, 0, 0, 0, 0, 0, 0, 100)

axs[2].grid()
axs[2].legend()
plt.show()
