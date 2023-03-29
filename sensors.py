import csv
from cv2 import normalize
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

#from vpython import *

importfile = 'IMU1_LinearAcceleration.csv'
importfile2 = 'IMU1_Quaternion.csv'


#Accelerometer
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



# Gyroscope
with open(importfile2, "r") as file2:
    csvreader2 = csv.reader(file2)

    time2 = []
    wx = []
    wy = []
    wz = []
    
    iterfile2 = iter(csvreader2)
    next(iterfile2)

    for row in csvreader2:
        #print(row)
        time2.append(float(row[2]))
        wx.append(float(row[3]))
        wy.append(float(row[4]))
        wz.append(float(row[5]))



#Integrering 

dt = time[1] - time[0]

vx = [0]
vy = [0]
vz = [0]
vx_t = [0]
vy_t = [0]
vz_t = [0]

b = 0 # beta, rotation kring x-axel
y = 0 # keppa rotation kring y-axel
a = 0 # alfa, rotation kring z-axel

for i in np.arange((len(time)) - 1):
    b = b + wx[i] * (time[i+1] - time[i]) 
    y = y + wy[i] * (time[i+1] - time[i]) 
    a = a + wz[i] * (time[i+1] - time[i]) 

    vx = vx + [vx[-1] + ax[i] * (time[i+1] - time[i]) ] 
    vy = vy + [vy[-1] + ay[i] * (time[i+1] - time[i]) ] 
    vz = vz + [vz[-1] + az[i] * (time[i+1] - time[i]) ] 

    vx_t = vx_t + [vx[-1] * ((np.cos(b)*np.cos(y)) + (np.cos(b)*np.sin(y)) - np.sin(b))]
    vy_t = vy_t + [vy[-1] * ((np.sin(a)*np.sin(b)*np.cos(y) - np.cos(a)*np.sin(y)) + (np.sin(a)*np.sin(b)*np.sin(y) + np.cos(a)*np.cos(y)) + (np.sin(a)*np.cos(b)))]
    vz_t = vz_t+[vz[-1] * ((np.cos(a)*np.sin(b)*np.cos(y) + np.sin(a)*np.sin(y)) + (np.cos(a)*np.sin(b)*np.sin(y) - np.sin(a)*np.cos(y)) + (np.cos(a)*np.cos(b)))]

 

plt.xlabel("time [s]")
plt.ylabel("v [m/s]")

plt.plot(time, vx_t, 'r', label = "vx")
plt.plot(time, vy_t, 'b' , label = "vy")
plt.plot(time, vz_t, 'g', label = "vz")

plt.grid()
plt.legend()
plt.show()


