import pandas as pd
import csv


# crops the time at start to minimize data
def crop(accdata, quatdata):

    accdata.pop('timestamp (+0200)')
    accdata.pop('epoc (ms)')
    quatdata.pop('timestamp (+0200)')
    quatdata.pop('epoc (ms)')

    table = pd.merge_asof(quatdata, accdata, on='elapsed (s)')

    with open(f'_mergerd', 'w') as fout:
        writer = csv.writer(fout)
        for i, val in table.iterrows():
            if sum(val[5:]) < 1:
                continue
            else:
                break

    for j, val in table.iterrows():
        writer.writerow(table.iloc[i:,:])

    table.iloc[:, 5:] *= 9.82


# read the files and run the program
file1 = pd.read_csv('IMU1_Arvid_Gravity.csv')  # acceleration
file2 = pd.read_csv('IMU1_Arvid_Quaternion.csv')  # Quaternion

crop(file1, file2)
