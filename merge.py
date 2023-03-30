import csv


with open('test','r') as file1, open('test2','r') as file2, open('merged', "w") as outFile:
    reader = csv.reader(file1, delimiter=',')
    reader2 = csv.reader(file2, delimiter=',')
    writer = csv.writer(outFile, delimiter=',')
    header = next(reader)
    writer.writerow(header)
    for i, row in enumerate(reader):
        for j, row2 in enumerate(reader2):
            if i == j and any([3, 4, 5]) == i:
                row[j]+=row2[j]
            writer.writerow(row)