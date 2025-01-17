import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# prevent numpy exponential
# notation on print, default False
np.set_printoptions(suppress=True)

# global variables
# velocity DataFrame will store all velocity values for videos analyzed
velocity_df = pd.DataFrame(columns=['Time Elapsed'])

# conversion path *must* point to CSV and dataFrame is reading from CSV
# conversion path points to the location of the pixel conersion csv and will generate a Pandas Dataframe as a result
# every video will have a certain conversion value to convert x and y values from pixel to cm
# this accounts for video framing, an idea would be to add a tracking point in DLC and parse files accordingly
conv_path = "/Users/imehndiokho/Desktop/pixel conversion.csv"
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
                                                   'righthindpawlikelihood', 'tailbasex', 'tailbasey','taillikelihood'])

    # getting the animal name from DLC csv passed in
    animalName = []
    animalName[:] = ' '.join(path.split()[2:3])
    fullName = ' '.join(path.split()[:2]) + " " + ''.join(animalName[:2])

    # getting animal name from pixel conversion CSV file
    conversion_df['animalFromVid'] = conversion_df.vidName.str[:17]

    # variables that find conversion factors unique to each video
    # each video will have unique height and width conversion factors
    height_conv_factor = int(conversion_df.loc[conversion_df['animalFromVid'] == fullName].height_conv)
    width_conv_factor = int(conversion_df.loc[conversion_df['animalFromVid'] == fullName].width_conv)

    data_df['Time Elapsed'] = data_df["frameNo"] / 30

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
    dt = (np.roll(data_df['Time Elapsed'], -1) - data_df['Time Elapsed'])[1:-1]

    # calculating the speed, change in displacement over time
    speeds = np.divide(dxy, dt)

    # generating speed dataframe
    speed_df = pd.DataFrame(data={'Time Elapsed': data_df['Time Elapsed'][1:-1], 'Speed': speeds,
                                  'Snout Likelihood': data_df['snoutLike'][1:-1]})

    # speed_normalized = (speed_df - speed_df.mean()) / speed_df.std()

    # calculates moving average over a span of approx. 3 seconds, can be changed
    speed_df['pandas_SMA_3'] = speed_df.iloc[:, 1].rolling(window=100).mean()

    # adding calculated velocity to global velocity DataFrame
    velocity_df['Time Elapsed'] = speed_df['Time Elapsed']
    velocity_df[fullName] = speed_df['pandas_SMA_3']

    # generating plots
    plt.plot(speed_df['Time Elapsed'], speed_df['pandas_SMA_3'], color='red', marker='o', markersize=0.1, linewidth=0.5,
             label='pandas_SMA_3')
    plt.xlabel('time (seconds)')
    plt.ylabel('velocity (cm/second)')
    animal = []
    animal[:] = ' '.join(path.split()[2:3])
    plt.title('Snout Velocity vs. Time for: ' + ' '.join(path.split()[:2]) + " " + ''.join(animal[:3]))
    plt.show()


# if you pass a directory then parse through directory
def dirParse(directory):
    for file in os.listdir(directory):
        # ignore hidden files and look run velocity calculation on CSV files
        # could also use file.endswith if you only wanted to look for .csv files instead of ignoring hidden files
        # useful if you have other types of files in directory
        if not file.startswith('.') and os.path.isfile(os.path.join(directory, file)):
            velocityCalc(file)


dirParse("/Users/imehndiokho/Desktop/csv_files")
velocity_df.to_csv("/Users/imehndiokho/Desktop/velocity_all.csv", index=False)
