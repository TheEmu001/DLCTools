import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DONE: convert pixel coordinates to cm at the very beginning
#   every video must have its unique pixel conversion value
#   then append cm value for x and y pixel, use that to calculate distance and thereby velocity

# todo: format function in a way to pass it a path string/dir and generate velocities, append to pandas df, then
#  take pandas df and make it into csv containing velocity values for each animal frame by frame

# todo: generate an average of all animals
#   perhaps create a pandas df calculating all velocity frame by frame for each, convert df to csv
#   averages will need SEM as error bar
# TODO: recognition of multiple files in folder to each generate their own plot
#   this kne one not that urgent because well be generating averages

# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

#GLOBAL VAR, may not even need this??
velocity_df = pd.DataFrame(columns=['Time Elapsed'])


path_name = "Vglut-cre C137 F3-_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"
path_2 = "Vglut-cre C137 F4+_2DLC_resnet50_VGlutEnclosedBehaviorApr25shuffle1_151500.csv"

conv_path ="pixel conversion.csv"
conversion_df = pd.read_csv(conv_path, skiprows=4, names=['vidName', 'bucket_height', 'bucket_width', 'width_conv',
                                                          'height_conv'])

# individual calculation for a single file
def velocityCalc(path):
    global velocity_df
    global conv_path
    global conversion_df

    data_df = pd.read_csv(path, skiprows=3, names=['frameNo', 'snoutX', 'snoutY', 'snoutLike',
                                                   'LeftEarX', 'LeftEarY', 'LeftEarlikelihood', 'rightearx',
                                                   'righteary',
                                                   'rightearlikelihood', 'leftforepawx', 'leftforepawy',
                                                   'leftforewlikelihood', 'rightforepawx', 'rightforepawy',
                                                   'rightforepawlikelihood', 'lefthindpawx', 'lefthindpawy',
                                                   'lefthindpawlikelihood', 'righthindpawx', 'righthindpawy',
                                                   'righthindpawlikelihood', 'tailbasex', 'tailbasey',
                                                   'taillikelihood'])

    # getting animal name from csv path
    animalName = []
    animalName[:] = ' '.join(path.split()[2:3])
    fullName = ' '.join(path.split()[:2]) + " " + ''.join(animalName[:2])
    # print(fullName)

    # getting animal name from pixel conversion
    conversion_df['animalFromVid'] = conversion_df.vidName.str[:17]

    # find the height conversion, needs to be casted as an int since the conversion factor is unique to row
    # if not it will only work for one row...
    height_conv_factor = int(conversion_df.loc[conversion_df['animalFromVid'] == fullName].height_conv)
    width_conv_factor = int(conversion_df.loc[conversion_df['animalFromVid'] == fullName].width_conv)

    data_df['Time Elapsed'] = data_df["frameNo"] / 30

    # divide by factor to convert coordinates to cm
    data_df['X_cm'] = data_df['snoutX'].divide(width_conv_factor)
    data_df['Y_cm'] = data_df['snoutY'].divide(height_conv_factor)
    # xy = data_df[['snoutX', 'snoutY']]
    xy = data_df[['X_cm', 'Y_cm']]

    # print(xy)

    b = np.roll(xy, -1, axis=0)[1:-1]

    a = xy[1:-1]

    # change in xy
    dxy = np.linalg.norm(a - b, axis=1)

    # change in time
    dt = (np.roll(data_df['Time Elapsed'], -1) - data_df['Time Elapsed'])[1:-1]

    # calculating the speed, change in displacement over time
    speeds = np.divide(dxy, dt)

    speed_df = pd.DataFrame(data={'Time Elapsed': data_df['Time Elapsed'][1:-1], 'Speed': speeds,
                                  'Snout Likelihood': data_df['snoutLike'][1:-1]})

    speed_normalized = (speed_df - speed_df.mean()) / speed_df.std()

    speed_df['CMA'] = speed_df['Speed'][1:-1].expanding(min_periods=2).mean()

    speed_df['pandas_SMA_3'] = speed_df.iloc[:, 1].rolling(window=100).mean()
    # speed_df['pandas_SMA_3_cm'] = speed_df['pandas_SMA_3'] * (1/24.421)

    # plt.plot(speed_df['Time Elapsed'], speed_df['Speed'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='Raw Data')
    plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3'], color='red', marker='o', markersize=0.1, linewidth=0.5,
             label='pandas_SMA_3')
    # plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3_cm'],color='red', marker='o', markersize=0.1, linewidth=0.5, label='pandas_SMA_3')

    velocity_df['Time Elapsed'] = speed_df['Time Elapsed']
    velocity_df[fullName] = speed_df['pandas_SMA_3']

    plt.xlabel('time (seconds)')
    plt.ylabel('velocity (cm/second)')
    # plt.legend(loc=2)
    animal = []
    animal[:] = ' '.join(path.split()[2:3])
    plt.title('Snout Velocity vs. Time for: ' + ' '.join(path.split()[:2]) + " " + ''.join(animal[:3]))
    plt.show()


# if you pass a directory then parse through directory
def dirParse(directory):
    for file in os.listdir(directory):
        velocityCalc(file)


velocityCalc(path_name)
velocityCalc(path_2)
print(velocity_df)

