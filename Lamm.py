import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def get_global_coord_acceleration(orientation, x_acc, y_acc, z_acc):
    '''Local (IMU-coordinate system) acceleration to global coordinate system'''

    r_m = orientation.as_matrix()
    acc_local = np.array([[x_acc, y_acc, z_acc]]).T
    acc_global = np.matmul(r_m, acc_local)

    return acc_global.flatten()


def get_positions(data, X_rot, Y_rot, Z_rot, X_vel, Y_vel, Z_vel, sampling_freq):
    '''Returns global coordinate positions integrated from IMU data,
    With initial conditions X_rot, Y_rot, Z_rot, X_vel,Y_vel,Z_vel, [deg / ms^-1]
    and data with rows as:<x-axis (deg/s), y-axis (deg/s), z-axis(deg/s), x-axis(g), y-axis(g), x-axis (g)>
    ie from earlier sent merge function
    '''

    G = 9.81
    # initial position in global coordinate system
    X_pos = 0
    Y_pos = 0
    Z_pos = 0

    X_positions = []
    Y_positions = []
    Z_positions = []

    orientation = R.from_euler('xyz', [X_rot, Y_rot, Z_rot], degrees=True)

    for row in data:
        r = R.from_euler('xyz', [row[0] / sampling_freq, row[1] / sampling_freq, row[2] / sampling_freq], degrees=True)
        orientation *= r
        acc_global = get_global_coord_acceleration(orientation, x_acc=row[3], y_acc=row[4], z_acc=row[5])

        X_vel += acc_global[0] * G / sampling_freq
        Y_vel += acc_global[1] * G / sampling_freq
        Z_vel += (acc_global[2] - 1) * G / sampling_freq
        X_pos += X_vel / sampling_freq
        Y_pos += Y_vel / sampling_freq
        Z_pos += Z_vel / sampling_freq
        X_positions.append(X_pos)
        Y_positions.append(Y_pos)
        Z_positions.append(Z_pos)

    return [X_positions, Y_positions, Z_positions]