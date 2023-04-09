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
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            inname = f'{basepath}{entry.name}'
            outname = endpath+entry.name.split('_')[0]+inname.split('/')[1].split('_')[1][:4]+'_crop'
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
                    if float(val[2]) > 4 and np.sqrt(float(val[3]) ** 2 + float(val[4]) ** 2 + float(val[5]) ** 2) > 0.2:
                        break
                iterval = i - 100
                writer.writerows(accelerations[iterval:])







"""
                    if float(row[2]) > 5:
                        if np.sqrt(float(row[3])**2 + float(row[4])**2 + float(row[5])**2) > 1 and trueplacer:
                            trueplacer = False
                            truetime = float(row[2])

                reader.line_num = 0
                for rowi in reader:
                    if float(rowi[2]) >= truetime-4:
                        writer.writerow(rowi)
"""