import csv
import os
import numpy as np
import shutil

"""
csvfiles = [f for f in os.listdir() if '.csv' in f.lower()]

for csvfile in csvfiles:
    new_path = 'iscropped/' + csvfile
    shutil.move(csvfile, new_path)
"""

basepath = 'needstobecropped/'
endpath = 'iscropped/'
aa = ''
bb = ''
cc = ''
dd = ''
ee = ''
ff = ''
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            inname = f'{basepath}{entry.name}'
            if len(entry.name.split('_')) == 6:
                a, b, c, d, e, f = entry.name.split('_')
                if a == aa and b == bb and e == 'Linear Acceleration':
                    acciterval = iterval
                elif e == 'Quaternion':
                    pass
                else:
                    acciterval = 0
                outname = endpath+a+b+e[:3]+'_c.csv'
            else:
                outname = endpath + entry.name + '_c.csv'
                acciterval = 0
            with open(inname, 'r', newline='') as file, open(outname, 'w',newline='') as outFile:
                reader = csv.reader(file, delimiter=',')
                writer = csv.writer(outFile, delimiter=',')
                header = next(reader)
                writer.writerow(header)
                accelerations=[]
                for row in reader:
                    accelerations.append(row)
                accelerations = np.asarray(accelerations)

                for i, val in enumerate(accelerations):
                    if i <= acciterval:
                        continue
                    if e == 'Linear Acceleration' and float(val[2]) > 4 and np.sqrt(float(val[3]) ** 2 + float(val[4]) ** 2 + float(val[5]) ** 2) > 0.1:
                        break
                    elif e != 'Linear Acceleration':
                        break
                if e == 'Quaternion':
                    i += 50
                iterval = i - 50 # Tar bort
                itervalend = len(accelerations)-200 # Tar bort 3s från slutet
                writer.writerows(accelerations[iterval:itervalend])

                if e == 'Linear Acceleration':
                    acciterval = iterval
