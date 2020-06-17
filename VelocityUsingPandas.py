import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

# path = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
path = "Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# TODO: recognition of multiple files in folder to each generate their own plot
data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx', 'righteary',
                                                'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                'righthindpawlikelihood', 'tailbasex', 'tailbasey', 'taillikelihood'])


# # parsing snout likelihood higher than 60%
# dataParsed = pd.DataFrame(data = data_df[data_df.snoutLike >= 0.6])
#
#
# dataParsed['Time'] = dataParsed.index/30
# dist = dataParsed.diff().fillna(0.)
#
#
# dist['Dist'] = np.sqrt(dist.snoutX**2 + dist.snoutY**2)
# dist['speed'] = dist.Dist/(dist.Time)
#
# print(dist)

# data_df['Time Elapsed'] = data_df.index/30
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

# cummulative moving average
# data_df['CMA'] = speed_df['Speed'][1:-1].expanding(min_periods=10).mean()
speed_df['CMA'] = speed_df['Speed'][1:-1].expanding(min_periods=2).mean()

# speed_df['EMA'] = speed_normalized['Speed'].ewm(span=40,adjust=False).mean()
# for i in range(0,speed_df.shape[0]-2):
#     speed_df.loc[speed_df.index[i+2],'SMA_3'] = np.round(((speed_df.iloc[i,1]+ speed_df.iloc[i+1,1] +speed_df.iloc[i+2,1])/3),1)
# print(speed_df.head())

speed_df['pandas_SMA_3'] = speed_df.iloc[:,1].rolling(window=100).mean()

# plt.plot(speed_df['Time Elapsed'], speed_df['Speed'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='Raw Data')
plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')
# plt.plot(dist['Time'], dist['speed'],color='green', marker='o', markersize=0.1, linewidth=0.5, label='CMA')
#
plt.xlabel('time (seconds)')
plt.ylabel('velocity (Pixels/second)')
# plt.legend(loc=2)
animal = []
animal[:] = ' '.join(path.split()[2:3])
plt.title('Snout Velocity vs. Time for: ' + ' '.join(path.split()[:2]) + " "+ ''.join(animal[:3]))
plt.show()