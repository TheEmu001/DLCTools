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

# calculate the difference from row under to row before
# then calculate absolute value
data_df['|diff X|'] = data_df['snoutX'].diff(-1)
data_df['|diff X|'] = data_df['|diff X|'].abs()

data_df['|diff Y|'] = data_df['snoutY'].diff(-1)
data_df['|diff Y|'] = data_df['|diff Y|'].abs()

# calculating the cummulative sum down the column
data_df['sumX'] = data_df['|diff X|'].cumsum()
data_df['sumY'] = data_df['|diff Y|'].cumsum()

print(data_df)

# what's being plotted
plt.plot(data_df['Time Elapsed'], data_df['sumX'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='xSum')
plt.plot(data_df['Time Elapsed'], data_df['sumY'],color='red', marker='o', markersize=0.1, linewidth=0.1, label='ySum')

# plot formatting
plt.xlabel('time (seconds)')
plt.ylabel('distance travelled (pixels)')
plt.legend(loc=2)
plt.title('total distance traveled vs. time: ' + path)
plt.show()
