import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import normalize

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

# path = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# path = "Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# path = "Vglut-cre C162 F1DLC_resnet50_EnclosedBehaviorMay27shuffle1_307000.csv"

# TODO: recognition of multiple files in folder to each generate their own plot
data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx', 'righteary',
                                                'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                'righthindpawlikelihood', 'tailbasex', 'tailbasey', 'taillikelihood'])



data_df['Time Elapsed'] = data_df["frameNo"]/30
xy = data_df[['snoutX', 'snoutY']]
b = np.roll(xy, -1, axis=0)[1:-1]

a = xy[1:-1]

# change in xy
dxy = np.linalg.norm(a - b, axis=1)

# change in time
dt = (np.roll(data_df['Time Elapsed'], -1) - data_df['Time Elapsed'])[1:-1]

# calculating the speed, change in displacement over time
speeds = np.divide(dxy, dt)

speed_df = pd.DataFrame(data={'Time Elapsed': data_df['Time Elapsed'][1:-1], 'Speed': speeds, 'Snout Likelihood': data_df['snoutLike'][1:-1]})

speed_normalized = (speed_df - speed_df.mean())/speed_df.std()

speed_df['CMA'] = speed_df['Speed'][1:-1].expanding(min_periods=2).mean()

speed_df['pandas_SMA_3'] = speed_df.iloc[:,1].rolling(window=100).mean()

x_val_numpy = speed_df['Time Elapsed'].to_numpy()
y_val_numpy = speed_df['pandas_SMA_3'].to_numpy()

y_der = np.gradient(y_val_numpy, x_val_numpy, edge_order=1)
y_der_0 = np.nan_to_num(y_der)
y_der_norm = normalize(y_der_0[:,np.newaxis], axis=0).ravel()
y_der_cm = y_der * (1/24.421)

speed_df['first der cm'] = y_der_cm
speed_df['first der cm'] = speed_df.iloc[:,1].rolling(window=100).mean()

# plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')
# plt.plot(speed_df['Time Elapsed'], y_der_cm,color='green', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')

plt.plot(speed_df['Time Elapsed'], speed_df['first der cm'],color='green', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')


#normalized first derivative
# plt.plot(speed_df['Time Elapsed'], y_der_norm,color='red', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')
plt.xlabel('Time (seconds)')
plt.ylabel('Accelaration (Pixels/second $^2$)')
animal = []
animal[:] = ' '.join(path.split()[2:3])
plt.title('Snout Acceleration vs. Time for: ' + ' '.join(path.split()[:2]) + " "+ ''.join(animal[:2]))
plt.show()


