import numpy as np
import pandas as pd
import csv
from itertools import islice


# crops the time at start to minimize data
def crop(accdata, quatdata):
    with open(accdata, 'r') as inp, open(quatdata, 'w') as out:
        writer = csv.writer(out)
        iterval = iter(csv.reader(inp))
        next(iterval)
        for row in iterval:
            floats = [float(x) for x in row[3:]]
            if sum(list(map(abs, floats))) < 1.5:
                continue
            else:
                break
            writer.writerow(row)


# read the files and run the program
file1 = 'IMU1_Arvid_Gravity.csv'  # acceleration
file2 = '_merged.csv'

crop(file1, file2)
