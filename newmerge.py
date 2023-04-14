import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from crop import crop
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg') # Så att man kan debugga samtidigt som ploten är uppe.
plt.rcParams['figure.figsize'] = [10, 10]


basepath = 'needstobecropped/'
endpath = 'iscropped/'
entries = crop(basepath, endpath)
weight = 86


importfile = entries[0]
importfile2 = entries[1]
importfile3 = entries[2]
importfile4 = entries[3]


# ---------------------------------- Program starts -------------------------
df_acc = pd.read_csv(importfile)
df_gyro = pd.read_csv(importfile2)
df_acc2 = pd.read_csv(importfile3)
df_gyro2 = pd.read_csv(importfile4)

# Ny uppdatering 1.7.2
df_acc.pop(df_acc.columns[1])

df_gyro.pop(df_gyro.columns[1])


df_acc2.pop(df_acc2.columns[1])

df_gyro2.pop(df_gyro2.columns[1])


interpolacc = (df_acc.iloc[:]+df_acc2.iloc[:])/2
interpolQua = (df_gyro.iloc[:]+df_gyro2.iloc[:])/2
interpolacc = interpolacc.dropna()
interpolQua = interpolQua.dropna()

table = pd.merge_asof(interpolQua, interpolacc, on=df_acc.columns[0])
table.pop(table.columns[0])
table.pop(table.columns[5])

table = table.dropna()

# Gör om till m/s^2
table.iloc[:, 5:] *= 9.82 * weight

# Detta tror jag är fel
# table.iloc[:,5:]-=table.iloc[0,5:]

# Sätter w på rätt plats
# Fel efter 1.7.2 -->: 'z (number)'--> ' z (number)' SE MELLANSLAGET
table = table[
    [table.columns[0], 'x (number)', 'y (number)', ' z (number)', 'w (number)', 'x-axis (g)', 'y-axis (g)', 'z-axis (g)']]

# Degrees value from quaternion
acc = []
for i, val in table.iterrows():
    if sum(val[1:5]) == 0:
        val[1] += 0.001
    v = R.from_quat(val[1:5]).as_matrix()

    acc.append(np.dot(v, val[5:]))
acc = np.asarray(acc)

ax = acc[:, 0]  # Framåt, positivt är frmaåt
ay = acc[:, 1]  # Punktens sida är positiv (alltså höjdhopparens högra sida)
az = acc[:, 2]  # Uppåt

time = table.iloc[:, 0]

vx = [0]
vy = [0]
vz = [0]

x = [0]
y = [0]
z = [0]

for i in np.arange((len(time)) - 1):
    dt = time[i + 1] - time[i]

    vx.append(vx[i] + ax[i] * dt)
    vy.append(vy[i] + ay[i] * dt)
    vz.append(vz[i] + az[i] * dt)

    x.append(x[i] + vx[i] * dt + (ax[i] * dt**2)/2)
    y.append(y[i] + vy[i] * dt + (ay[i] * dt**2)/2)
    z.append(z[i] + vz[i] * dt + (az[i] * dt**2)/2)

# Functions ------------------------------------------------------------------------------------

def localMinUtil(arr, low, high, n):
    # Find index of middle element
    mid = low + (high - low) // 2  # (low + high) // 2

    # Compare middle element with its neighbours (if neighbours exist)
    if ((mid == 0 or arr[mid - 1] > arr[mid]) and
            (mid == n - 1 or arr[mid + 1] > arr[mid])):
        return mid

    # If middle element is not minima and its left neighbour is smaller than it, then left half must have a local minima.
    elif (mid > 0 and arr[mid - 1] < arr[mid]):
        return localMinUtil(arr, low, (mid - 1), n)

    # If middle element is not minima and its right neighbour is smaller than it, then right half must have a local minima.
    return localMinUtil(arr, (mid + 1), high, n)


# A wrapper over recursive function localMinUtil()
def localMin(arr, n):
    return localMinUtil(arr, 0, n - 1, n)


def avgforce(between, yval):
    valmin = yval[between[0]]
    val = np.sum(abs(yval[between]))
    return val


# ---------------------------------------------------------------------------------------------

# Plotting
fig, axs = plt.subplots(3, 1)
axs[0].grid()
axs[1].grid()
axs[2].grid()

axs[0].plot(time, ax, 'r', label="ax")
axs[0].plot(time, ay, 'b', label="ay")
axs[0].plot(time, az, 'g', label="az")

axs[2].plot(time, vx, 'r', label="vx")
axs[2].plot(time, vy, 'b', label="vy")
axs[2].plot(time, vz, 'g', label="vz")

pos = pd.Series(vz).idxmax()
posmin = pd.Series(vz).idxmin()

rangevallow = 10
locminofpos = localMin(az[pos-rangevallow:pos+5], len(az[pos-rangevallow:pos+5]))+pos-20
val = max(table.iloc[:, 5])

between = range(locminofpos-rangevallow, pos + 7)

axs[0].plot(time[between[0]], az[between[0]], '*')
axs[0].annotate('Jump force starts', xy=(time[between[0]], az[between[0]]), xycoords='data',
                xytext=(0.6, 0.11), textcoords='axes fraction',
                va='top', ha='left',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
axs[0].plot(time[between[-1]], az[between[-1]], '*')
axs[0].annotate('Jump force ends', xy=(time[between[-1]], az[between[-1]]), xycoords='data',
                xytext=(0.01, .99), textcoords='axes fraction',
                va='top', ha='left',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

num1 = range(pos + 20, pos + 30)
# finner var värdet är närmast 1. Där kan man anse att man har hamnat i högsta läget av hoppet.
error = 10
for i in num1:
    value = sum(table.iloc[i, 5:])
    if (abs(value) - 1) < error:
        error = abs(value - 1)
        position = i

# axs[0].axvline(table.iloc[position, 0])
# axs[0].annotate(f'Highest position with force {error}', xy=(table.iloc[position, 0], table.iloc[position, 5]),xytext=(8.6, 3), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
plt.grid()
axs[0].legend()

# plt.xlim([8, 9])

# A Python program to find a local minima in an array


totnorm = np.linalg.norm(table.iloc[:, 5:], axis=1)

force = avgforce(between, totnorm)

axs[1].plot(time, totnorm)
axs[1].fill_between(table.iloc[between, 0], totnorm[between], color='blue', alpha=.5)
axs[1].annotate(f'Impuls = {force}', xy=(time[between[3]], ax[between[3]]), xycoords='data',
                xytext=(0.01, .8), textcoords='axes fraction',
                va='top', ha='left',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))


axs[2].grid()
axs[2].legend()
plt.show()


ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z, 'r', label="Kurvlöpning")
plt.show()

pass
