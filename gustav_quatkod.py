# Gyroscope


# Integrering

vx = [0]
vy = [0]
vz = [0]
vx_t = [0]
vy_t = [0]
vz_t = [0]

b = 0  # beta, rotation kring x-axel
y = 0  # keppa rotation kring y-axel
a = 0  # alfa, rotation kring z-axel

"""
for i in np.arange((len(time)) - 1):
    b = b + v[i][0] * (v[i+1][0] - v[i][0])
    y = y + v[i][1] * (v[i+1][1] - v[i][1])
    a = a + v[i][2] * (v[i+1][2] - v[i][2])

    # y = y + table.iloc[:, 3][i] * (time[i + 1] - time[i])
    # a = a + table.iloc[:, 4][i] * (time[i + 1] - time[i])

    vx = vx + [vx[-1] + table.iloc[:, 5][i] * (time[i + 1] - time[i])]
    vy = vy + [vy[-1] + table.iloc[:, 6][i] * (time[i + 1] - time[i])]
    vz = vz + [vz[-1] + table.iloc[:, 7][i] * (time[i + 1] - time[i])]

    vx_t = vx_t + [vx[-1] * ((np.cos(b) * np.cos(y)) + (np.cos(b) * np.sin(y)) - np.sin(b))]
    vy_t = vy_t + [vy[-1] * ((np.sin(a) * np.sin(b) * np.cos(y) - np.cos(a) * np.sin(y)) + (
            np.sin(a) * np.sin(b) * np.sin(y) + np.cos(a) * np.cos(y)) + (np.sin(a) * np.cos(b)))]
    vz_t = vz_t + [vz[-1] * ((np.cos(a) * np.sin(b) * np.cos(y) + np.sin(a) * np.sin(y)) + (
            np.cos(a) * np.sin(b) * np.sin(y) - np.sin(a) * np.cos(y)) + (np.cos(a) * np.cos(b)))]

axs[2].plot(time, vx_t, 'r', label="vx")
axs[2].plot(time, vy_t, 'b', label="vy")
axs[2].plot(time, vz_t, 'g', label="vz")

error = 1000
for i in num1:
    value = sum([vx_t[i], vy_t[i], vz_t[i]])
    if abs(value) < error:
        errorv = abs(value - 1)
        positionv = i

#axs[2].plot(time[positionv], vy_t[positionv], 'o')

# get_positions(table, 0, 0, 0, 0, 0, 0, 100)