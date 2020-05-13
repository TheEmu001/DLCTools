import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

path = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# TODO: recognition of multiple files in folder to each generate their own plot
data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx', 'righteary',
                                                'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                'righthindpawlikelihood', 'tailbasex', 'tailbasey', 'taillikelihood'])

# calculate the time elapsed per frame and append column
data_df['Time Elapsed'] = data_df['frameNo'] / 30

# dataframe only for xy values
xy = data_df[['snoutX', 'snoutY']]
print(xy)

# elements at the end of array come to beginning, will be x1 and y1
b = np.roll(xy, -1, axis=0)[1:-1]
print(b)

# skip first value to make it easy to subtract, this will basically be x2 and x2
a = xy[1:-1]
print(a)

dxy = np.linalg.norm(a - b, axis=1)
print(dxy)

dist = euclidean_distances(a, b, squared=True)
# distances = pdist(dxy.values, metric='euclidean')
# dist_matrix = squareform(distances)
# print(dist_matrix)

# dxy = np.linalg.norm(a - b, axis=1)
# print(dxy)

# #
# a = xy[1:-1]
# #
# # change in xy
# dxy = np.linalg.norm(a - b, axis=1)
# dxy_Dist = np.linalg.norm(a - b, axis=1)
#
# # total distance calculation
# # print()
# # distances = pdist(xy, metric='euclidean')
# # dist_matrix = squareform(distances)
# # print(dist_matrix)
#
# # change in time
# dt = (np.roll(data_df['Time Elapsed'], -1) - data_df['Time Elapsed'])[1:-1]
#
# # calculating the speed, change in displacement over time
# speeds = np.divide(dxy, dt)
#
# speed_df = pd.DataFrame(data={'Time Elapsed': data_df['Time Elapsed'][1:-1], 'Speed': speeds, 'Snout Likelihood': data_df['snoutLike'][1:-1]})
#
# speed_normalized = (speed_df - speed_df.mean())/speed_df.std()
#
#
# speed_df['pandas_SMA_3'] = speed_df.iloc[:,1].rolling(window=2).mean()
#
# # plt.plot(speed_df['Time Elapsed'], speed_df['Speed'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='Raw Data')
# # plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')
# #
# # plt.xlabel('time (seconds)')
# # plt.ylabel('distance (pixels)')
# # plt.legend(loc=2)
# # plt.title('Total Distance vs. Time for: ' + path)
# # plt.show()

