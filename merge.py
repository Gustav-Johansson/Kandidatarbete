import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_acc = pd.read_csv('IMU1_LinearAcceleration_hopp.csv')
df_gyro = pd.read_csv('IMU1_Quaternion_hopp.csv')

df_acc.pop('timestamp (+0100)')
df_acc.pop('epoc (ms)')
df_gyro.pop('timestamp (+0100)')
df_gyro.pop('epoc (ms)')

merge = pd.merge_asof(df_gyro, df_acc, on='elapsed (s)')
print(merge)


g = 9.81

time = merge.iloc[:,0]
ax = merge.iloc[:,5] * g
ay = merge.iloc[:,6] * g
az = merge.iloc[:,7] * g 

wx = merge.iloc[:,2] 
wy = merge.iloc[:,3] 
wz = merge.iloc[:,4] 

b = 0 # beta, rotation kring x-axel
y = 0 # keppa rotation kring y-axel
a = 0 # alfa, rotation kring z-axel

vx = [0] 
vy = [0] 
vz = [0] 

vx_t = [0]
vy_t = [0]
vz_t = [0]

#korrigering - behövs ej för linjär accelecration
'''for i, val in enumerate(merge.columns):
    #merge.iloc[:,i] = merge.iloc[:,i] - merge.iloc[0,i]'''

#Integrering 
for i in np.arange((len(time) - 1)):
    b = b + wx[i] * ( time[i+1]- time[i]) 
    y = y + wy[i] * (time[i+1] - time[i]) 
    a = a + wz[i] * (time[i+1] - time[i]) 

    vx = vx + [vx[-1] + ax[i] * (time[i+1] - time[i]) ] 
    vy = vy + [vy[-1] + ay[i] * (time[i+1] - time[i]) ] 
    vz = vz + [vz[-1] + az[i] * (time[i+1] - time[i]) ] 

    vx_t = vx_t + [vx[-1] * ((np.cos(b)*np.cos(y)) + (np.cos(b)*np.sin(y)) - np.sin(b))]
    vy_t = vy_t + [vy[-1] * ((np.sin(a)*np.sin(b)*np.cos(y) - np.cos(a)*np.sin(y)) + (np.sin(a)*np.sin(b)*np.sin(y) + np.cos(a)*np.cos(y)) + (np.sin(a)*np.cos(b)))]
    vz_t = vz_t+[vz[-1] * ((np.cos(a)*np.sin(b)*np.cos(y) + np.sin(a)*np.sin(y)) + (np.cos(a)*np.sin(b)*np.sin(y) - np.sin(a)*np.cos(y)) + (np.cos(a)*np.cos(b)))]



# Plocka ut accelerationerna
totacc = np.sqrt(ax**2 + ay**2 + az**2) #normen

# tiden till max ay
time_max_ay = pd.Series(ax).idxmax()
val = max(ax)# Ta fram tidpunkten med mest acceleration
print ("Max value element : ", max(ax) )

# totacc plot
plt.xlabel("time [s]")
plt.ylabel("a [m/s^2]")
plt.plot(time, totacc , label = "totacc")
plt.grid()
plt.legend()
plt.show()




# acc plot
plt.xlabel("time [s]")
plt.ylabel("a [m/s^2]")
plt.plot(time, ax, 'r', label = "ax")
plt.plot(time, ay, 'b' , label = "ay")
plt.plot(time, az, 'g', label = "az")
plt.plot(merge.iloc[time_max_ay, 0], merge.iloc[time_max_ay, 5], '*')
plt.plot(merge.iloc[time_max_ay - 5, 0], merge.iloc[time_max_ay - 5, 5], '*')


plt.grid()
plt.legend()
plt.show()

print('Time at max acc: '+ str(merge.iloc[time_max_ay]) )


#velocity plot
plt.xlabel("time [s]")
plt.ylabel("v [m/s]")
plt.plot(time, vx_t, 'r', label = "vx")
plt.plot(time, vy_t, 'b' , label = "vy")
plt.plot(time, vz_t, 'g', label = "vz")
plt.grid()
plt.legend()
plt.show()