import csv
from cv2 import normalize
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

importfile = 'IMU1_Accelerometer.csv'
importfile2 = 'IMU1_Gyroscope.csv'

#---------------------------------------- Accelerometer
with open(importfile, "r") as file:
    csvreader = csv.reader(file)

    time = []
    xforce = []
    yforce = []
    zforce = []
    
    iterfile = iter(csvreader)
    next(iterfile)

    for row in csvreader:
        #print(row)
        time.append(float(row[2]))
        xforce.append(float(row[3]))
        yforce.append(float(row[4]))
        zforce.append(float(row[5]))

fig, axs = plt.subplots(2)


axs[0].plot(time, xforce, 'r')
axs[0].plot(time, yforce, 'b')
axs[0].plot(time, zforce, 'g')


#---------------------------------------- Gyroscope
with open(importfile2, "r") as file2:
    csvreader2 = csv.reader(file2)

    time2 = []
    xforce2 = []
    yforce2 = []
    zforce2 = []
    
    iterfile2 = iter(csvreader2)
    next(iterfile2)

    for row in csvreader2:
        #print(row)
        time2.append(float(row[2]))
        xforce2.append(float(row[3]))
        yforce2.append(float(row[4]))
        zforce2.append(float(row[5]))

axs[1].plot(time2, xforce2, 'r')
axs[1].plot(time2, yforce2, 'b')
axs[1].plot(time2, zforce2, 'g')

plt.show()

#---------------------------------------- Kraftvektor
with open(importfile, "r") as file:
    csvreader = csv.reader(file)

    vector = []

    iterfile = iter(csvreader)
    next(iterfile)

    for row in csvreader:
        
        vector.append([float(row[2]), float(row[3]), float(row[4]), float(row[5])])


ax = plt.figure().add_subplot(projection='3d')

ax.quiver(0, 0, 0, vector[-1][1], vector[-1][2], vector[-1][3], length = 0.1, normalize = True) # vektor i sista ögonblicket 

plt.show()


'''
#---------------------------------------- Integrering*2 -> position

x = [0]
y = [0]
z = [0]

for i in np.arange((len(time)) - 1):
    x = x + [x[-1] + vx[i] * dt ]
    y = y + [y[-1] + vy[i] * dt ]
    z = z + [z[-1] + vz[i] * dt ]

plt.xlabel("time [s]")
plt.ylabel("r [m]")

plt.plot(time, x, 'r', label = "x")
plt.plot(time, y, 'b' , label = "y")
plt.plot(time, z, 'g', label = "z")

plt.grid()
plt.legend()
plt.show()


#------------------------------------------------------------------------------------------ Simulator 3D rum
plane = sphere(pos=vector(0, 0, 0 ), radius = 5, color = color.yellow, make_trail = True)
for i in range(len(time)):
     rate(500)
     plane.pos = vector(x[i], y[i], z[i])
'''