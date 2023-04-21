import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from cropbackup import crop
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Qt5Agg')  # Så att man kan debugga samtidigt som ploten är uppe.
plt.rcParams['figure.figsize'] = [10, 10]

basepath = 'IMU1/'
endpath = 'iscropped/'

with os.scandir(basepath) as entr:
    entries = []
    for entry in entr:
        entries.append(entry)



importfile = entries[10]
importfile2 = entries[11]
importfile3 = entries[10]
importfile4 = entries[11]

weight = 89
calibration = 311


# ---------------------------------- Program starts -------------------------
df_acc = pd.read_csv(importfile)
df_qua = pd.read_csv(importfile2)
df_acc2 = pd.read_csv(importfile3)
df_qua2 = pd.read_csv(importfile4)

correct = pd.read_csv('erik2_acc.tsv', delimiter='\t')
correct = correct.iloc[:] / 1000

# Ny uppdatering 1.7.2
df_acc.pop(df_acc.columns[1])

df_qua.pop(df_qua.columns[1])

df_acc2.pop(df_acc2.columns[1])

df_qua2.pop(df_qua2.columns[1])

syncedvalues = pd.merge_asof(df_acc.iloc[:], df_acc2.iloc[:], on=df_acc.iloc[:].columns[0])

syncedvalues = syncedvalues.rename(columns={"x-axis (g)_x": "x-axis (g)", "y-axis (g)_x": "y-axis (g)", 'z-axis (g)_x': 'z-axis (g)', "x-axis (g)_y": "x-axis (g)", "y-axis (g)_y": "y-axis (g)", 'z-axis (g)_y':'z-axis (g)', 'elapsed (s)_x':'elapsed (s)'})


interpolacc = (syncedvalues.iloc[:, 2:5] + syncedvalues.iloc[:,6:]) / 2
interpolacc['epoch (ms)'] = syncedvalues.iloc[:,0]
interpolQua = (df_qua.iloc[:] + df_qua2.iloc[:]) / 2
interpolacc = interpolacc.dropna()
interpolQua = interpolQua.dropna()

interpolacc = interpolacc[['epoch (ms)', 'x-axis (g)', 'y-axis (g)', 'z-axis (g)']]
interpolQua = interpolQua[['epoch (ms)', 'elapsed (s)', 'w (number)', 'x (number)', 'y (number)',
       ' z (number)']]

interpolacc['epoch (ms)'] = pd.to_datetime(interpolacc['epoch (ms)'])
interpolQua['epoch (ms)'] = pd.to_datetime(interpolQua['epoch (ms)'])



table = pd.merge_asof(interpolQua, interpolacc, on=interpolacc.columns[0])
#table = pd.merge_asof(df_qua2, df_acc2, on=df_acc.columns[0])
table.pop('epoch (ms)')

table = table.dropna()

end = 406
start = 516

# table.iloc[:,5:] = syncedvalues.iloc[:,2:5]
# table.iloc[:,5:] = syncedvalues.iloc[:,6:]

table = table.drop(len(table)-table.index[range(end)]-1)
table = table.drop(table.index[range(start)])

table = table.reset_index(drop=True)

table.iloc[:,0] -= table.iloc[0,0]

# Gör om till m/s^2
table.iloc[:, 5:] *= 9.82

# Sätter w på rätt plats
# Fel efter 1.7.2 -->: 'z (number)'--> ' z (number)' SE MELLANSLAGET
table = table[
    [table.columns[0], 'x (number)', 'y (number)', ' z (number)', 'w (number)', 'x-axis (g)', 'y-axis (g)',
     'z-axis (g)']]

# Degrees value from quaternion
acc = []
for i, val in table.iterrows():
    if sum(val[1:5]) == 0:
        val[1] += 0.001
    v = R.from_quat(val[1:5]).as_matrix()

    acc.append(np.dot(v, val[5:]))
acc = np.asarray(acc)

ax = acc[:, 0] * np.cos(calibration) + acc[:, 1] * np.sin(calibration)  # Framåt, positivt är framåt
ay = acc[:, 1] * np.cos(calibration) + acc[:, 0] * np.sin(calibration)  # Punktens sida positiv (hopparens högra sida)
az = acc[:, 2]  # Uppåt

ax -= ax[0]  # Tar bort offset så att värdena alltid börjar på noll
ay -= ay[0]
az -= az[0]

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

    x.append(x[i] + vx[i] * dt + (ax[i] * dt ** 2) / 2)
    y.append(y[i] + vy[i] * dt + (ay[i] * dt ** 2) / 2)
    z.append(z[i] + vz[i] * dt + (az[i] * dt ** 2) / 2)

qtmvx = [0]
qtmvy = [0]
qtmvz = [0]
corabs = [0]
qtmx = [0]
qtmy = [0]
qtmz = [0]

cor = [correct.iloc[:, 0] + correct.iloc[:, 3], correct.iloc[:, 1] + correct.iloc[:, 4],
       correct.iloc[:, 2] + correct.iloc[:, 5]]
cor = pd.DataFrame(cor).T

for i in np.arange((len(correct)) - 1):
    dt = 1 / 100

    qtmvx.append(qtmvx[i] + cor.iloc[i, 0] * dt)
    qtmvy.append(qtmvy[i] - cor.iloc[i, 1] * dt)  # Eftersom kalibreringen blev åt andra hållet blir den negativ.
    qtmvz.append(qtmvz[i] + cor.iloc[i, 2] * dt)

    qtmx.append(qtmx[i] + qtmvx[i] * dt + (cor.iloc[i, 0] * dt ** 2) / 2)
    qtmy.append(qtmy[i] + qtmvy[i] * dt + (cor.iloc[i, 1] * dt ** 2) / 2)
    qtmz.append(qtmz[i] + qtmvz[i] * dt + (cor.iloc[i, 2] * dt ** 2) / 2)

    corabs.append(np.sqrt(cor.iloc[i, 0] ** 2 + cor.iloc[i, 1] ** 2 + cor.iloc[i, 2] ** 2))


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
fig, axs = plt.subplots(2, 3)
axs[0, 0].grid()
axs[1, 0].grid()
axs[1, 1].grid()

# axs[0].plot(time, ax, 'r', label="ax")
# axs[0].plot(time, ay, 'b', label="ay")
axs[0, 0].plot(time, az, 'g', label="az")
axs[0, 0].title.set_text('IMU')

t = np.linspace(0, len(correct) / 100, len(correct))

# axs[2].plot(correct.iloc[:,0], 'r', label="ax")
# axs[2].plot(correct.iloc[:,1], 'b', label="ay")
axs[1, 0].plot(t, correct.iloc[:, 2], 'g', label="az")

axs[0, 2].plot(time, vx, 'g', label="vx")
axs[0, 2].plot(time, vy, 'b', label="vy")
axs[0, 2].plot(time, vz, 'k', label="vz")

pos = pd.Series(az).idxmax()
flyingbuff = 60
pos -= flyingbuff

rangevallow = 10
locminofpos = localMin(az[pos - rangevallow:pos + 5], len(az[pos - rangevallow:pos + 5])) + pos - 20

between = range(locminofpos - rangevallow, pos + 7)

axs[0, 0].plot(time[between[0]], az[between[0]], '*')
axs[0, 0].annotate('Jump force starts', xy=(time[between[0]], az[between[0]]), xycoords='data',
                   xytext=(0.6, 0.11), textcoords='axes fraction',
                   va='top', ha='left',
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
axs[0, 0].plot(time[between[-1]], az[between[-1]], '*')
axs[0, 0].annotate('Jump force ends', xy=(time[between[-1]], az[between[-1]]), xycoords='data',
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
axs[0, 0].legend()

t = np.linspace(0, len(correct) / 100, len(correct))

axs[1, 2].plot(t, qtmvx, 'b', label="vx")
axs[1, 2].plot(t, qtmvy, 'g', label="vy")
axs[1, 2].plot(t, qtmvz, 'k', label="vz")

axs[1, 1].plot(t, corabs)

# A Python program to find a local minima in an array


totnorm = np.linalg.norm(table.iloc[:, 5:], axis=1)

force = avgforce(between, totnorm)

axs[0, 1].plot(time, totnorm)
axs[0, 1].fill_between(table.iloc[between, 0], totnorm[between], color='blue', alpha=.5)
axs[0, 1].annotate(f'Impuls = {force * weight}', xy=(time[between[3]], ax[between[3]]), xycoords='data',
                   xytext=(0.01, .8), textcoords='axes fraction',
                   va='top', ha='left',
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

axs[0, 1].grid()
axs[1, 0].legend()
axs[0, 2].legend()
axs[1, 2].legend()

axs[1, 0].title.set_text('QTM')
axs[0, 1].title.set_text('Summa av krafter')
axs[0, 2].title.set_text('Hastighet IMU')
axs[1, 1].title.set_text('Absolutacceleration QTM')
axs[1, 2].title.set_text('Hastighet QTM (Direkt interegering)')

axs[1, 0].set_xlabel('Tid [s]')
axs[0, 1].set_xlabel('Tid [s]')
axs[0, 0].set_xlabel('Tid [s]')
axs[1, 1].set_xlabel('Tid [s]')

axs[1, 0].set_ylabel('m/s^2')
axs[0, 1].set_ylabel('m/s^2')
axs[0, 0].set_ylabel('m/s^2')
axs[1, 1].set_ylabel('m/s')
plt.show()

felmarginal = (sum(acc[:,2]-correct.iloc[:,3]))/len(acc)
print(felmarginal)
pass
