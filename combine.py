import csv
from cv2 import normalize
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

#from vpython import *

importfile = 'XYZ_acc.csv'
importfile2 = 'IMU1_Gyroscope.csv'


#---------------------------------------- Accelerometer
with open(importfile, "r") as file:
    csvreader = csv.reader(file)

    time = []
    ax = []
    ay = []
    az = []
    brus = 0.05

    iterfile = iter(csvreader)
    next(iterfile)

    for row in csvreader:
        time.append(float(row[2]))

        if abs(float(row[3]) + 1.01615) < brus:
            ay.append(0)
        else:
            ay.append(-(float(row[3]) + 1.01615))

        if abs(float(row[4]) + 0.05) < brus:
            ax.append(0)
        else:
            ax.append(-(float(row[4]) + 0.05 ))

        if abs(float(row[5]) - 0.115) < brus:
            az.append(0) 
        else:
             az.append(float(row[5]) - 0.115) 

plt.xlabel("time [s]")
plt.ylabel("a [m/s^2]")

plt.plot(time, ax, 'r', label = "ay")
plt.plot(time, ay, 'b' , label = "ax")
plt.plot(time, az, 'g', label = "az")

plt.grid()
plt.legend()
plt.show()

#---------------------------------------- Integrering -> hastighet

dt = time[1] - time[0]
g = 9.8
vx = [0]
vy = [0]
vz = [0]

for i in np.arange((len(time)) - 1):
    vx = vx + [vx[-1] + ax[i] * (time[i+1] - time[i]) ] 
    vy = vy + [vy[-1] + ay[i] * (time[i+1] - time[i]) ] 
    vz = vz + [vz[-1] + az[i] * (time[i+1] - time[i]) ] 

plt.xlabel("time [s]")
plt.ylabel("v [m/s]")

plt.plot(time, vx, 'r', label = "vx")
plt.plot(time, ay, 'b' , label = "vy")
plt.plot(time, az, 'g', label = "vz")

plt.grid()
plt.legend()
plt.show()


#---------------------------------------- Gyroscope
with open(importfile2, "r") as file2:
    csvreader2 = csv.reader(file2)

    time2 = []
    gyro_x = []
    gyro_y = []
    gyro_z = []
    
    iterfile2 = iter(csvreader2)
    next(iterfile2)

    for row in csvreader2:
        #print(row)
        time2.append(float(row[2]))
        gyro_x.append(float(row[3]))
        gyro_y.append(float(row[4]))
        gyro_z.append(float(row[5]))

plt.xlabel("time [s]")
plt.ylabel("v [m/s]")

plt.plot(time2, gyro_x, 'r', label = "vx")
plt.plot(time2,  gyro_y, 'b' , label = "vy")
plt.plot(time2, gyro_z, 'g', label = "vz")

plt.grid()
plt.legend()
plt.show()


def merge_acc_gyro(acc_df,gyro_df):


    acc_df.pop('timestamp (+0100)')
    acc_df.pop('elapsed (s)')
    gyro_df.pop('timestamp (+0100)')
    gyro_df.pop('elapsed (s)')
    merged = pd.merge_asof(gyro_df,acc_df, on='epoc (ms)',direction='nearest')

    return merged