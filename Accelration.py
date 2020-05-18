# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # prevent numpy exponential
# # notation on print, default False
# np.set_printoptions(suppress=True)
#
# path = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# # path = "Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
#
# # TODO: recognition of multiple files in folder to each generate their own plot
# data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
#                                                 'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx', 'righteary',
#                                                 'rightearlikelihood', 'leftforepawx', 'leftforepawy',
#                                                 'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
#                                                 'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
#                                                 'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
#                                                 'righthindpawlikelihood', 'tailbasex', 'tailbasey', 'taillikelihood'])
#
# # calculate the time elapsed per frame and append column
# data_df['Time Elapsed'] = data_df['frameNo'] / 30
#
# # calculate the difference from row under to row before
# # then calculate absolute value
# data_df['|diff X|'] = data_df['snoutX'].diff(-1)
# data_df['|diff X|'] = data_df['|diff X|'].abs()
#
# data_df['|diff Y|'] = data_df['snoutY'].diff(-1)
# data_df['|diff Y|'] = data_df['|diff Y|'].abs()
#
# # calculating the cummulative sum down the column
# data_df['sumX'] = data_df['|diff X|'].cumsum()
# data_df['sumY'] = data_df['|diff Y|'].cumsum()
#
# # squaring delta x and y values
# data_df['deltax^2'] = data_df['|diff X|']**2
# data_df['deltay^2'] = data_df['|diff Y|']**2
#
# # adding deltaX^2 + deltaY^2
# data_df['deltaSummed'] = data_df['deltax^2'] + data_df['deltay^2']
#
# # taking square root of deltaX^2 + deltaY^2
# data_df['eucDist'] = data_df['deltaSummed']**(1/2)
# data_df['eucDistSum'] = data_df['eucDist'].cumsum()
#
# # calculate the average value between x and y values from subsequent rows
# data_df['velocity'] = data_df['deltaSummed'][30:-1] / data_df['Time Elapsed'][30:-1]
#
# movements_over_timesteps = (np.roll(data_df, -1, axis=0)- data_df)[:-1]
# speeds = np.sqrt(data_df['snoutX'] ** 2 + data_df['snoutY'] ** 2) / (data_df['frameNo']/30)
#
# print(speeds)
# # plot formatting
#
# plt.xlabel('time (seconds)')
# plt.ylabel('velocity (pixels/sec)')
# plt.legend(loc=2)
# plt.title('velocity vs. time: ' + path)
# plt.axvspan(300, 600, alpha=0.25, color='blue')
# plt.show()

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
accFromSpeed = np.divide(speeds, dt)

speed_df = pd.DataFrame(data={'Time Elapsed': data_df['Time Elapsed'][1:-1], 'Speed': speeds, 'Snout Likelihood': data_df['snoutLike'][1:-1]})
acc_df = pd.DataFrame(data={'Time Elapsed': data_df['Time Elapsed'][1:-1], 'Acceleration': accFromSpeed, 'Snout Likelihood': data_df['snoutLike'][1:-1]})

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
plt.plot(acc_df['Time Elapsed'], acc_df['Acceleration'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='Raw Data')

# plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')
# plt.plot(dist['Time'], dist['speed'],color='green', marker='o', markersize=0.1, linewidth=0.5, label='CMA')
#
plt.xlabel('time (seconds)')
plt.ylabel('velocity (Pixels/second)')
plt.legend(loc=2)
plt.title('Snout Velocity vs. Time for: ' + path)
plt.show()

