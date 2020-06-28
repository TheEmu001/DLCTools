from scipy import signal
from scipy.signal import argrelmax, argrelmin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

# path = "Vglut-cre C137 F2+_2DLC_resnet50_EnclosedBehaviorMay27shuffle1_307000.csv"
# path = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# path = "Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
# TODO: recognition of multiple files in folder to each generate their own plot
data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx', 'righteary',
                                                'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                'righthindpawlikelihood', 'tailbasex', 'tailbasey', 'taillikelihood'])


y_coords = np.array(data_df['snoutY'])


# method
# this method finds the peak(or in this case trough) in a span of time, currently set to 150 meaning 150 frames or 5 seconds.
# data is then parsed by seting a y-value to filter data that may not be considered a "jump"
# bigger points indicate the points that the program considers to be a "jump" based on parameters set

peaks, _ = find_peaks(-data_df['snoutY'], distance=100)

data_df_peaks = pd.DataFrame(data=data_df.loc[data_df.frameNo.isin(peaks)]['snoutY'])

parse_peaks = pd.DataFrame(data=data_df_peaks[data_df_peaks.snoutY <= 175])
print(parse_peaks)
print(parse_peaks.size)

plt.plot(data_df['frameNo'], data_df['snoutY'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='snoutY vals')
plt.plot(parse_peaks.index, parse_peaks['snoutY'],color='green', marker='o', label='snout max')
plt.xlabel('frame')
plt.ylabel('y-coordinate')
plt.legend(loc=2)
plt.title('y-coordinate snout peak ' + path)
plt.show()


# other random functions
#
#
# ind_max = argrelmin(y_coords)
# maxvals = y_coords[ind_max]
# print(maxvals.size)
# y_min = min(y_coords)
# y_max = max(y_coords)
# comp_value = (y_min + y_max)/2.25
#
# somePercentile = np.percentile(y_coords, 2.25)
# initial_count = 0
# parsed_max = []
#
# for y in maxvals:
#     if y <= somePercentile:
#         initial_count = initial_count+1
#         parsed_max.append(y)
#
# index_max_parse = data_df.loc[data_df.snoutY.isin(parsed_max)].index
# # max_df = pd.DataFrame(data={'frame': index_max_parse, 'snoutValue': parsed_max})
#
# peaks, _ = find_peaks(-data_df['snoutY'], distance=150)
#
# index_max_parse_2 = data_df.loc[data_df.frameNo.isin(peaks)].index
# max_df_2 = pd.DataFrame(data={'frame': index_max_parse_2, 'snoutValue': data_df[peaks]})
#
# plt.plot(data_df['frameNo'], data_df['snoutY'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='snoutY vals')
# plt.plot(max_df['frame'], max_df['snoutValue'],color='green', marker='o', label='snout max')
#
# plt.xlabel('frame')
# plt.ylabel('y-coordinate')
# plt.legend(loc=2)
# plt.title('y-coordinate snout peak ' + path)
# plt.show()







