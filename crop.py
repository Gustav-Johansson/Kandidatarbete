import os
import csv


with open('Datainsamling/Smilla/both/IMU2_2023-03-29T15.28.02.680_CFFE2E15FC84_Quaternion.csv','r') as file, open('quat2', "w") as outFile:
    reader = csv.reader(file, delimiter=',')
    writer = csv.writer(outFile, delimiter=',')
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        if float(row[2]) > 5:
            writer.writerow(row)