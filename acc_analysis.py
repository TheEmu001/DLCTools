import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

# global variables
# velocity DataFrame will store all velocity values for videos analyzed
# acceleration DataFrame will store all acceleration values for videos analyzed
velocity_df = pd.DataFrame(columns=['TimeElapsed'])
acceleration_df = pd.DataFrame(columns=['TimeElapsed'])

# conversion path *must* point to CSV and dataFrame is reading from CSV
# conversion path points to the location of the pixel conersion csv and will generate a Pandas Dataframe as a result
# every video will have a certain conversion value to convert x and y values from pixel to cm
# this accounts for video framing, an idea would be to add a tracking point in DLC and parse files accordingly
conv_path = "/Users/imehndiokho/PycharmProjects/DLCTools/pixel_conversion.csv"
# conv_path = open(r"C:\Users\ad-anestnorrislab\Desktop\pixel_conv.csv", "r")
# with open (r"C:\Users\ad-anestnorrislab\Desktop\pixel_conv.csv", "rb+") as f:
#     conversion_df = pd.read_csv(f, skiprows=4, names=['vidName', 'bucket_height', 'bucket_width', 'width_conv',
#                                                               'height_conv'])
# # print(conv_path)
conversion_df = pd.read_csv(conv_path, skiprows=4, names=['vidName', 'bucket_height', 'bucket_width', 'width_conv',
                                                          'height_conv'])

# individual velocity calculation for a single file
# this function will take the path of a file and run the function accordingly

def velocityCalc(path):
    # defining global variables to appease the demands of Python
    global velocity_df
    global conv_path
    global conversion_df

    # reading the DLC generated CSV file and generating a Pandas Dataframe with necessary info
    # this velocity function only focuses on snout velocity but can be adjusted to reflect other body parts of interest
    data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                   'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx',
                                                   'righteary',
                                                   'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                   'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                   'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                   'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                   'righthindpawlikelihood', 'tailbasex', 'tailbasey',
                                                   'taillikelihood'])

    # getting the animal name from DLC csv passed in
    animalName = []
    animalName[:] = ' '.join(path.split()[2:3])
    fullName = ' '.join(path.split()[:2]) + " " + ''.join(animalName[:2])
    short_name = fullName[52:]




    # getting animal name from pixel conversion CSV file
    conversion_df['animalFromVid'] = conversion_df.vidName.str[:17]
    # print(conversion_df['animalFromVid'])

    # variables that find conversion factors unique to each video
    # each video will have unique height and width conversion factors
    height_conv_factor = conversion_df.loc[conversion_df['animalFromVid'].str.lower() == short_name.lower()].height_conv.item()
    width_conv_factor = conversion_df.loc[conversion_df['animalFromVid'].str.lower() == short_name.lower()].width_conv.item()

    # print(conversion_df[conversion_df['animalFromVid'] == short_name].height_conv)
    data_df['TimeElapsed'] = data_df["frameNo"] / 30

    # divide by conversion factors to convert coordinates to cm
    data_df['X_cm'] = data_df['snoutX'].divide(width_conv_factor)
    data_df['Y_cm'] = data_df['snoutY'].divide(height_conv_factor)

    xy = data_df[['X_cm', 'Y_cm']]


    # rolling x and y coordinates to facilitate calculating the difference between coordinates
    b = np.roll(xy, -1, axis=0)[1:-1]
    a = xy[1:-1]

    # calculating change in xy coordinate
    dxy = np.linalg.norm(a - b, axis=1)

    # calculating change in time
    dt = (np.roll(data_df['TimeElapsed'], -1) - data_df['TimeElapsed'])[1:-1]

    # calculating the speed, change in displacement over time
    speeds = np.divide(dxy, dt)

    # generating speed dataframe
    speed_df = pd.DataFrame(data={'TimeElapsed': data_df['TimeElapsed'][1:-1], 'Speed': speeds,
                                  'Snout Likelihood': data_df['snoutLike'][1:-1]})

    # speed_normalized = (speed_df - speed_df.mean()) / speed_df.std()

    # calculates moving average over a span of approx. 3 seconds, can be changed
    speed_df['pandas_SMA_3'] = speed_df.iloc[:, 1].rolling(window=100).mean()

    # adding calculated velocity to global velocity DataFrame
    velocity_df['TimeElapsed'] = speed_df['TimeElapsed']
    velocity_df[fullName] = speed_df['pandas_SMA_3']

    # generating plots
    plt.plot(speed_df['TimeElapsed'], speed_df['pandas_SMA_3'], color='red', marker='o', markersize=0.1, linewidth=0.5,
             label='pandas_SMA_3')
    plt.xlabel('time (seconds)')
    plt.ylabel('velocity (cm/second)')
    animal = []
    animal[:] = ' '.join(path.split()[2:3])
    # plt.title('Snout Velocity vs. Time for: ' + ' '.join(path.split()[:2]) + " " + ''.join(animal[:3]))
    plt.title('Snout Velocity vs. Time for: ' + short_name)
    plt.show()


def accelerationCalc(path):
    # defining global variables to appease the demands of Python
    global velocity_df
    global acceleration_df
    global conv_path
    global conversion_df

    # reading the DLC generated CSV file and generating a Pandas Dataframe with necessary info
    # this velocity function only focuses on snout velocity but can be adjusted to reflect other body parts of interest
    data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                   'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx',
                                                   'righteary',
                                                   'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                   'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                   'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                   'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                   'righthindpawlikelihood', 'tailbasex', 'tailbasey',
                                                   'taillikelihood'])

    # getting the animal name from DLC csv passed in
    animalName = []
    animalName[:] = ' '.join(path.split()[2:3])
    fullName = ' '.join(path.split()[:2]) + " " + ''.join(animalName[:2])
    short_name = fullName[52:]

    # getting animal name from pixel conversion CSV file
    conversion_df['animalFromVid'] = conversion_df.vidName.str[:17]

    # variables that find conversion factors unique to each video
    # each video will have unique height and width conversion factors
    height_conv_factor = conversion_df.loc[conversion_df['animalFromVid'].str.lower() == short_name.lower()].height_conv.item()
    width_conv_factor = conversion_df.loc[conversion_df['animalFromVid'].str.lower() == short_name.lower()].width_conv.item()

    data_df['TimeElapsed'] = data_df["frameNo"] / 30

    # divide by conversion factors to convert coordinates to cm
    data_df['X_cm'] = data_df['snoutX'].divide(width_conv_factor)
    data_df['Y_cm'] = data_df['snoutY'].divide(height_conv_factor)
    xy = data_df[['X_cm', 'Y_cm']]

    # rolling x and y coordinates to facilitate calculating the difference between coordinates
    b = np.roll(xy, -1, axis=0)[1:-1]
    a = xy[1:-1]

    # calculating change in xy coordinates
    dxy = np.linalg.norm(a - b, axis=1)

    # calculating change in time
    dt = (np.roll(data_df['TimeElapsed'], -1) - data_df['TimeElapsed'])[1:-1]

    # calculating the speed, change in displacement over time
    speeds = np.divide(dxy, dt)

    # generating speed dataframe
    speed_df = pd.DataFrame(data={'TimeElapsed': data_df['TimeElapsed'][1:-1], 'Speed': speeds,
                                  'Snout Likelihood': data_df['snoutLike'][1:-1]})

    # speed_normalized = (speed_df - speed_df.mean()) / speed_df.std()

    # calculates moving average over a span of approx. 3 seconds, can be changed
    speed_df['pandas_SMA_3'] = speed_df.iloc[:, 1].rolling(window=100).mean()

    # adding calculated velocity to global velocity DataFrame
    velocity_df['TimeElapsed'] = speed_df['TimeElapsed']
    velocity_df[fullName] = speed_df['pandas_SMA_3']

    x_val_numpy = speed_df['TimeElapsed'].to_numpy()
    y_val_numpy = speed_df['pandas_SMA_3'].to_numpy()

    where_are_NaNs = np.isnan(y_val_numpy)
    y_val_numpy[where_are_NaNs] = 0
    # y_der = np.gradient(y_val_numpy, x_val_numpy, edge_order=3)
    # y_der_0 = np.nan_to_num(y_der)
    # y_der_cm = y_der

    y_der_cm = np.diff(y_val_numpy) / dt[:-1]
    where_are_NaNs2 = np.isnan(y_der_cm)
    y_der_cm[where_are_NaNs2] = 0
    # print(y_der_cm)

    speed_df['first der cm'] = y_der_cm
    # speed_df['first der cm'] = speed_df.iloc[:, 1].rolling(window=100).mean()

    acceleration_df['TimeElapsed'] = speed_df['TimeElapsed'][401:]
    # acceleration_df[short_name] = speed_df['first der cm']
    acceleration_df[short_name] = y_der_cm[400:]

    plt.plot(speed_df['TimeElapsed'][401:], y_der_cm[400:], color='green', marker='o', markersize=0.1,
             linewidth=0.5, label='pandas_SMA_3')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration(Pixels/second $^2$)')
    animal = []
    animal[:] = ' '.join(path.split()[2:3])
    plt.title('Snout Acceleration vs. Time for: ' + short_name)
    plt.show()


# if you pass a directory then parse through directory
def dirParse(directory):
    for file in os.listdir(directory):
        # ignore hidden files and look run velocity calculation on CSV files
        # could also use file.endswith if you only wanted to look for .csv files instead of ignoring hidden files
        # useful if you have other types of files in directory
        if not file.startswith('.') and os.path.isfile(os.path.join(directory, file)):
            # velocityCalc(os.path.join(directory, file))
            accelerationCalc(os.path.join(directory, file))
            # print(file)
#
# #
# dirParse("/Users/imehndiokho/PycharmProjects/DLCTools/drug_trials_saline")
# velocity_df.to_csv("/Users/imehndiokho/PycharmProjects/DLCTools/drug_trials_saline/velocity_saline.csv", index=False)
# acceleration_df.to_csv("/Users/imehndiokho/PycharmProjects/DLCTools/drug_trials_saline/acc_saline.csv", index=False)

velocityCalc(r"/Users/imehndiokho/PycharmProjects/DLCTools/drug_trials_saline/Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top DownDLC_resnet50_BigBinTopSep17shuffle1_240000filtered.csv")
# accelerationCalc(r"/Users/imehndiokho/PycharmProjects/DLCTools/csv_con/VGlut-cre C152 F2DLC_resnet50_EnclosedBehaviorMay27shuffle1_307000.csv")
# velocity_df['why'] = velocity_df['/Users/imehndiokho/PycharmProjects/DLCTools/csv_exp/VGlut-cre C147 F3']
# print(velocity_df)
#
# # # peaks, _ = find_peaks(-data_df['snoutY'], distance=100)
# peaks, _ = find_peaks(velocity_df.iloc[9000:18000]['why'], prominence=0.4, distance=10)
#
# #speed_df.iloc[:, 1]
# #
# velocity_df_peaks = pd.DataFrame(data=velocity_df.loc[velocity_df.TimeElapsed.isin(peaks)]['why'])
# # print(velocity_df_peaks)
# # parse_peaks = pd.DataFrame(data=velocity_df_peaks[velocity_df_peaks.why >= 30.5])
# #
# plt.plot(velocity_df.index/30, velocity_df['why'],color='green')
# plt.plot(velocity_df_peaks.index/30, velocity_df_peaks['why'],color='red', marker='o', label='snout max')
# plt.show()
# print(velocity_df_peaks.size)
