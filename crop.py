import csv


with open('Datainsamling/Smilla/both/IMU2_2023-03-29T15.28.02.680_CFFE2E15FC84_Quaternion.csv','r', newline='') as file, open('quat2', 'w', newline='') as outFile:
    reader = csv.reader(file, delimiter=',')
    writer = csv.writer(outFile, delimiter=',')
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        if float(row[2]) > 5:
            if row:
                writer.writerow(row)