# Importing the toolbox (takes several seconds)
import warnings

import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import time_in_each_roi
from scipy import stats
from textwrap import wrap
from scipy import integrate

warnings.filterwarnings('ignore')

all_data = pd.DataFrame(columns=['init'])


def velocity(video, fps, no_seconds, DLCscorer, color):
    # DLCscorer='DLC_resnet50_BigBinTopSep17shuffle1_250000'

    dataname = str(Path(video).stem) + DLCscorer + 'filtered.h5'
    print(dataname)

    #loading output of DLC
    Dataframe = pd.read_hdf(os.path.join(dataname), errors='ignore')
    # Dataframe.reset_index(drop=True)

    #you can read out the header to get body part names!
    bodyparts=Dataframe.columns.get_level_values(1)

    bodyparts2plot=bodyparts

    # let's calculate velocity of the back
    # this can be changed to whatever body part
    bpt='back'
    vel = time_in_each_roi.calc_distance_between_points_in_a_vector_2d(np.vstack([Dataframe[DLCscorer][bpt]['x'].values.flatten(), Dataframe[DLCscorer][bpt]['y'].values.flatten()]).T)

    # frame rate of camera in those experiments
    # fps=30
    time=np.arange(len(vel))*1./fps
    #notice the units of vel are relative pixel distance [per time step]
    vel=vel

    # store in other variables:
    xsnout=Dataframe[DLCscorer][bpt]['x'].values
    ysnout=Dataframe[DLCscorer][bpt]['y'].values
    vsnout=vel

    a = pd.DataFrame(data={"xsnout": xsnout, "ysnout":ysnout})
    b = pd.DataFrame(data={"xsnout": xsnout[:-1], "ysnout":ysnout[:-1]})

    # calculate the difference from row under to row before
    # then calculate absolute value
    a['|diff X|'] = a['xsnout'].diff(-1)
    a['|diff X|'] = a['|diff X|'].abs()

    a['|diff Y|'] = a['ysnout'].diff(-1)
    a['|diff Y|'] = a['|diff Y|'].abs()

    # calculating the cummulative sum down the column
    a['sumX'] = a['|diff X|'].cumsum()
    a['sumY'] = a['|diff Y|'].cumsum()

    # squaring delta x and y values
    a['deltax^2'] = a['|diff X|'] ** 2
    a['deltay^2'] = a['|diff Y|'] ** 2

    # adding deltaX^2 + deltaY^2
    a['deltaSummed'] = a['deltax^2'] + a['deltay^2']

    # taking square root of deltaX^2 + deltaY^2
    a['eucDist'] = a['deltaSummed'] ** (1 / 2)
    a['eucDistSum'] = a['eucDist'].cumsum()

    all_data[video+" euc dist"] = a['eucDistSum']
    # plotting Velocity
    # plt.plot(time,vel)
    # plt.title("\n".join(wrap("Velocity(pixels/second) for "+video)))
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Speed in pixels per second')
    # plt.show()

    # Calculating and plotting Velocity Moving Average; currently set to 300 frame rolling avg or 10 seconds based on 30fps
    # change "no_seconds" to reflect the desired moving average
    # no_seconds = 10
    moving_average_duration_frames=fps*no_seconds

    velocity_pd = pd.Series(vel, time)
    velocity_moving_avg = velocity_pd.rolling(moving_average_duration_frames).mean()

    # moving_avg_vel_df = pd.DataFrame(velocity_moving_avg, columns=['Velocity Moving Average'])
    # print(moving_avg_vel_df)

    # velocity_df = pd.DataFrame(time, velocity_pd, columns=['Time', 'Instantaneous Velocity'])
    # print(vel*1./fps)
    inst_vel_df = pd.DataFrame(time, columns=['Time'])
    inst_vel_df['Instantaneous Velocity'] = vel
    # print(inst_vel_df)


    # removes outliers that are beyond 3 std
    outlier_inst_df= inst_vel_df[(np.abs(stats.zscore(inst_vel_df)) < 3).all(axis=1)]
    # determines rolling average for the extracted frames
    outlier_inst_df['Rolling'] = outlier_inst_df['Instantaneous Velocity'].rolling(moving_average_duration_frames).mean()


    outlier_inst_df['dist'] = outlier_inst_df['Time']*outlier_inst_df['Instantaneous Velocity']
    outlier_inst_df['sum'] = outlier_inst_df['dist'].cumsum()
    # plt.plot(outlier_inst_df['Time'], outlier_inst_df['sum'])
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Cummulative Distance Travelled')
    # plt.show()

    all_data[video+ " Time"] = outlier_inst_df['Time']
    all_data[video+ " Velocity"] = outlier_inst_df['Rolling']
    all_data[video+" Dist Travelled"] = outlier_inst_df['sum']
    all_data[video+" raw dist"] = (inst_vel_df["Instantaneous Velocity"]*inst_vel_df["Time"]).cumsum()
    all_data[video+" raw time"] = inst_vel_df["Time"]
if __name__ == '__main__':
    all_data.dropna(axis=1)
    dlc_250 = 'DLC_resnet50_BigBinTopSep17shuffle1_250000'
    dlc_240 = 'DLC_resnet50_BigBinTopSep17shuffle1_240000'

    """
    Female 5mgkg U50 at 60fps
    """
    velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    # velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    #
    plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down raw time'],
             all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down euc dist'], color='brown', label='u50F0')
    plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down raw time'],
             all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down euc dist'], color='red', label='u50F1')
    # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down Time'],
    #          all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down Dist Travelled'], color='brown')

    # only_U50_F = all_data.loc[:, ['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Dist Travelled',
    #                               'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down Dist Travelled',
    #                               'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down Dist Travelled']]
    # only_U50_F_other = all_data.loc[:, ['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down euc dist',
    #                               'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down euc dist',
    #                               'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down euc dist']]
    # # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average U50 Female Other"] = only_U50_F_other.mean(axis=1)
    #
    #
    # all_data["U50 Female SEM"] = stats.sem(only_U50_F, axis=1)
    # all_data.dropna(axis=1)
    #
    # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down raw time'], all_data["Average U50 Female Other"],
    #          label='U50 Female Dist Other', color='#c4b7ff')
    """
        Saline Females at 30fps
        """
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_250,
             color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_250,
             color=None)

    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down raw time'],
             all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down euc dist'], color='orange')
    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down raw time'],
             all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down euc dist'], color='orange')

    # create a dataframe with only Saline values to make it easier to average
    # .loc[all of the rows, [only relevant Saline columns]]
    only_saline_fem = all_data.loc[:, ['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down raw dist',
                                       'Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down raw dist']]
    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average Saline Female Dist"] = only_saline_fem.mean(axis=1)
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # all_data["SEM Saline Fem"] = stats.sem(only_saline_fem, axis=1)
    # sem_val = all_data["SEM Saline Fem"][:108000]
    # all_data.dropna(axis=1)
    # plt.plot(all_data.loc[:108000, ['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down raw time']],
    #          all_data.loc[:108000, ["Average Saline Female Dist"]],
    #          label='Saline Female Dist', color='#ef24b8')

    velocity(video='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down', fps=60, no_seconds=10,
             DLCscorer=dlc_250, color=None)
    velocity(video='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down', fps=60, no_seconds=10,
             DLCscorer=dlc_250, color=None)
    velocity(video='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down', fps=60, no_seconds=10,
             DLCscorer=dlc_250, color=None)

    plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down raw time'],
             all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down euc dist'], color='orange')
    plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down raw time'],
             all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down euc dist'], color='orange')
    plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down raw time'],
             all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down euc dist'], color='orange')

    # create a dataframe with only Saline values to make it easier to average
    # .loc[all of the rows, [only relevant Saline columns]]
    only_Nal_U50_F = all_data.loc[:, ['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down raw dist',
                                      'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down raw dist',
                                      'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down raw dist']]
    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    all_data["Average PreTreat Nal U50 Female Dist"] = only_Nal_U50_F.mean(axis=1)
    # level=[['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Dist Travelled',
    #                                   'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down Dist Travelled',
    #                                   'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down Dist Travelled']]
    all_data["Naltr Fem SEM"] = stats.sem(only_Nal_U50_F, axis=1)
    # then calculate the rolling mean over a specified period of time (based on df index)
    # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # all_data["Cumulative Sum Nal U50 Female"] = all_data["Average PreTreat Nal U50 Female Dist"].cumsum()
    all_data.dropna(axis=1)
    # plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down raw time'],
    #          all_data["Average PreTreat Nal U50 Female Dist"], label='Naltr PreTreat U50 Female Dist', color='#8096e6')
    # plt.fill_between(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Time'], all_data["Average PreTreat Nal U50 Female Dist"]-all_data["Naltr Fem SEM"],
    #                  all_data["Average PreTreat Nal U50 Female Dist"]+all_data["Naltr Fem SEM"], alpha=0.5, facecolor='#8096e6')
    plt.xlabel('time (seconds)')
    plt.ylabel('distance travelled (pixels)')
    plt.legend(loc=2)
    plt.title('Total Distance vs. Time [15 second window]')
    # # plt.axvspan(300, 600, alpha=0.25, color='blue')
    plt.show()
    # print(all_data)
