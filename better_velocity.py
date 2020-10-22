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
    # moving_avg_outlier_df = pd.DataFrame(velocity_moving_avg, columns=['Velocity Moving Average'])

    # plt.plot(outlier_inst_df['Time'][1800:],outlier_inst_df['Instantaneous Velocity'][1800:], color='green')
    # plt.title("\n".join(wrap("Velocity(pixels/second) for "+video)))
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Speed in pixels per second for')
    # plt.show()

    # moving_avg_vel_df.to_csv("/Users/imehndiokho/PycharmProjects/DLCTools/head_velocity_drug_trials/"+str(no_seconds)+"second_"+video+".csv", index=False)
    # inst_vel_df.to_csv("/Users/imehndiokho/PycharmProjects/DLCTools/drug_trials_saline/velocity_df_"+video+".csv", index=False)
    #
    # plt.plot(outlier_inst_df['Time'], outlier_inst_df['Rolling'], color='green')
    # plt.title("\n".join(wrap("Moving Average Velocity(pixels/second) ["+str(moving_average_duration_frames)+" frames] for "+video)))
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Speed in pixels per second')
    # plt.show()

    # rolling integration to find cummulative distance
    # forces = pd.read_csv(...)
    # wrk = np.trapz(forces, x=forces.index, axis=0)
    # work_done = pd.DataFrame(wrk[None, :], columns=forces.columns)

    outlier_inst_df['dist'] = outlier_inst_df['Time']*outlier_inst_df['Instantaneous Velocity']
    outlier_inst_df['sum'] = outlier_inst_df['dist'].cumsum()
    # plt.plot(outlier_inst_df['Time'], outlier_inst_df['sum'])
    # plt.xlabel('Time in seconds')
    # plt.ylabel('Cummulative Distance Travelled')
    # plt.show()

    # # # calculate the time elapsed per frame and append column
    # # data_df['Time Elapsed'] = time[:-2]
    #
    # # calculate the difference from row under to row before
    # # then calculate absolute value
    # data_df['|diff X|'] = data_df['xsnout'].diff(-1)
    # data_df['|diff X|'] = data_df['|diff X|'].abs()
    #
    # data_df['|diff Y|'] = data_df['ysnout'].diff(-1)
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
    # print(data_df)
    #
    # # what's being plotted
    # # plt.plot(time, data_df['sumX'],color='blue', marker='o', markersize=0.1, linewidth=0.1, label='xSum')
    # # plt.plot(time, data_df['sumY'],color='red', marker='o', markersize=0.1, linewidth=0.1, label='ySum')
    # plt.plot(time, data_df['eucDistSum'],color='green', marker='o', markersize=0.1, linewidth=0.1, label='distance')
    #
    # # plot formatting
    # plt.xlabel('time (seconds)')
    # plt.ylabel('distance travelled (pixels)')
    # plt.legend(loc=2)
    # # plt.title('total distance traveled vs. time: ' + path)
    # plt.title('Total Distance vs. Time for: ' + video)
    # # plt.axvspan(300, 600, alpha=0.25, color='blue')
    # plt.show()
    all_data[video+ " Time"] = outlier_inst_df['Time']
    all_data[video+ " Velocity"] = outlier_inst_df['Rolling']
    all_data[video+" Dist Travelled"] = outlier_inst_df['sum']
if __name__ == '__main__':

    # # PreTreat
    # # Note: when plotting these graphs, ran into the issue of the video with 60fps being graphed only half way, looks
    # # like pyplot will establish the x-axis based on the first plot, graph highest framerate first to make sure this
    # # isn't prblematic in the future
    # # AKA M1 graphed at 60fps looked like it was stopping prematurely bc graph was made first with a video of 30fps
    # velocity(video='Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down', fps=30, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down', fps=60, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top Down', fps=60, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top Down', fps=60, no_seconds=10, color=None)
    #
    #
    #
    # plt.plot(all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'], all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled'], label='M1', color='pink')
    # plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Time'], all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down '
    #                                                                                                      'Dist Travelled'], label='M2', color='purple')
    # plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top Down Time'], all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top '
    #                                                                                                      'Down Dist Travelled'], label='M3', color='red')
    # plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top Down Time'], all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top '
    #                                                                                                      'Down Dist Travelled'], label='M4', color='black')
    #
    # # create a dataframe with only Naltrexone values to make it easier to average
    # # .loc[all of the rows, [only relevant Naltrexone columns]]
    # only_saline = all_data.loc[:, ['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Dist Travelled', 'Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top Down Dist Travelled',
    #                                'Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top Down Dist Travelled']]
    # only_saline['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled'] = all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled'].iloc[::2].dropna().reset_index(drop=True).dropna()
    # all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'] = all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'].iloc[::2].dropna().reset_index(drop=True).dropna()
    # # df = dfA.reset_index().dropna()
    # print(only_saline['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled'])
    # # # work around to filter time periods to only be the relevant ones
    # # potato = np.isin(all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'],
    # #                  all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Time'])
    # # all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'] = \
    # #     all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'][potato]
    # #
    # # calculate mean for relevant Naltrexone columns, this calculates an average velocity for each point in time
    # all_data["Average Naltr Dist"] = only_saline.mean(axis=1)
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # # this part is a little weird since there is one video filmed at 60fps
    # # Maybe figure out away to get this to roll from the specified time?
    # all_data["Rolling Average Naltr"] = all_data["Average Naltr Dist"].rolling(300).mean()
    # plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Time'], all_data["Average Naltr Dist"], label='Average Naltr Dist', color='#b86c31')
    dlc_250 = 'DLC_resnet50_BigBinTopSep17shuffle1_250000'
    dlc_240 = 'DLC_resnet50_BigBinTopSep17shuffle1_240000'

    # 5mg/kg U50 Males - old data
    #
    # velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down', fps=60, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top Down', fps=60, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top Down', fps=60, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top Down', fps=60, no_seconds=10, color=None)
    #
    # # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top '
    # #                                                                                    'Down Dist Travelled'], label='M1', color='blue')
    # # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top '
    # #                                                                                    'Down Dist Travelled'], label='M2', color='blue')
    # # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top '
    # #                                                                                    'Down Dist Travelled'], label='M3', color='blue')
    # # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top '
    # #                                                                                    'Down Dist Travelled'], label='M4', color='blue')
    #
    # # create a dataframe with only 5mgkg U50 values to make it easier to average
    # # .loc[all of the rows, [only relevant u50 columns]]
    # only_5mgU50 = all_data.loc[:, ['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled','Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top Down Dist Travelled', 'Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top Down Dist Travelled', 'Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top Down Dist Travelled']]
    # # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average 5mgkg U50"] = only_5mgU50.mean(axis=1)
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 60fps, this is calculated over 600 frames to represent one second
    # all_data["Rolling Average 5mgkg U50"] = all_data["Average 5mgkg U50"].rolling(600).mean()
    # plt.plot(all_data["Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time"], all_data["Rolling Average 5mgkg U50"], label='Rolling Avg U50')


    # # 10mg/kg U50 Males
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M1_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240, color=None)
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240, color='purple')
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M3_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240, color='orange')
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M4_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240, color='green')
    #
    #
    # plt.plot(all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M1_Top Down Time'], all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M1_Top '
    #                                                                              'Down Dist Travelled'], label='M1', color='red')
    # plt.plot(all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M2_Top Down Time'], all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M2_Top '
    #                                                                              'Down Dist Travelled'], label='M2', color='purple')
    # plt.plot(all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M3_Top Down Time'], all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M3_Top '
    #                                                                              'Down Dist Travelled'], label='M3', color='orange')
    # plt.plot(all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M4_Top Down Time'], all_data['Paper_Redo_U50_Ai14_OPRK1_C1_M4_Top '
    #                                                                              'Down Dist Travelled'], label='M4', color='green')

    """
    U50 Males 5mgkg at 60fps
    """
    velocity(video='Paper_Redo_10_16_5mgkg_U50_Ai14_OPRK1_C1_M2_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M3_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250,
             color=None)
    velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M4_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250,
             color=None)

    only_u50_males_60fps = all_data.loc[:, ['Paper_Redo_10_16_5mgkg_U50_Ai14_OPRK1_C1_M2_Top Down Dist Travelled',
                                            'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M3_Top Down Dist Travelled',
                                            'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M4_Top Down Dist Travelled']]
    # # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    all_data["Average U50 Males Dist 60fps"] = only_u50_males_60fps.mean(axis=1)
    all_data["U50 Male SEM 60fps"] = stats.sem(only_u50_males_60fps, axis=1)
    all_data["Rolling Average U50 Males 60fps"] = all_data["Average U50 Males Dist 60fps"].rolling(900).mean()
    plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M4_Top Down Time'], all_data["Rolling Average U50 Males 60fps"],
             label='Rolling Average U50 Males 60fps', color='#700a8f')
    plt.fill_between(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C1_M4_Top Down Time'], all_data["Rolling Average U50 Males 60fps"]-all_data["U50 Male SEM 60fps"],
                     all_data["Rolling Average U50 Males 60fps"]+all_data["U50 Male SEM 60fps"], alpha=0.5, facecolor='#700a8f')
    """
    Female 5mgkg U50 at 60fps
    """
    # velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    # velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    # velocity(video='Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    # #
    # # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Time'],
    # #          all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Dist Travelled'], color='orange')
    # # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down Time'],
    # #          all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down Dist Travelled'], color='orange')
    # # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down Time'],
    # #          all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down Dist Travelled'], color='orange')
    #
    # # create a dataframe with only Saline values to make it easier to average
    # # .loc[all of the rows, [only relevant Saline columns]]
    # only_U50_F = all_data.loc[:, ['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Dist Travelled',
    #                               'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F1_Top Down Dist Travelled',
    #                               'Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F2_Top Down Dist Travelled']]
    # # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average U50 Female Dist"] = only_U50_F.mean(axis=1)
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # all_data["U50 Female SEM"] = stats.sem(only_U50_F, axis=1)
    # all_data["Rolling Average U50 Female"] = all_data["Average U50 Female Dist"].rolling(900).mean()
    # plt.plot(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Time'], all_data["Rolling Average U50 Female"],
    #          label='Rolling Average U50 Female', color='#c4b7ff')
    # plt.fill_between(all_data['Paper_Redo_10_17_5mgkg_U50_Ai14_OPRK1_C2_F0_Top Down Time'], all_data["Rolling Average U50 Female"]-all_data["U50 Female SEM"],
    #                  all_data["Rolling Average U50 Female"]+all_data["U50 Female SEM"], alpha=0.5, facecolor='#c4b7ff')

    """
    3mgkg Naltrexone PreTreat 5mgkg U50 Females at 60fps
    """
    # velocity(video='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    # velocity(video='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    # velocity(video='Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down', fps=60, no_seconds=10, DLCscorer=dlc_250, color=None)
    #
    # # plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Time'],
    # #          all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Dist Travelled'], color='orange')
    # # plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down Time'],
    # #          all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down Dist Travelled'], color='orange')
    # # plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down Time'],
    # #          all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down Dist Travelled'], color='orange')
    #
    # # create a dataframe with only Saline values to make it easier to average
    # # .loc[all of the rows, [only relevant Saline columns]]
    # only_Nal_U50_F = all_data.loc[:, ['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Dist Travelled',
    #                               'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down Dist Travelled',
    #                               'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down Dist Travelled']]
    # # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average PreTreat Nal U50 Female Dist"] = only_Nal_U50_F.mean(axis=1)
    # # level=[['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Dist Travelled',
    # #                                   'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F1_Top Down Dist Travelled',
    # #                                   'Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F2_Top Down Dist Travelled']]
    # all_data["Naltr Fem SEM"] = stats.sem(only_Nal_U50_F, axis=1)
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # all_data["Rolling Average Nal U50 Female"] = all_data["Average PreTreat Nal U50 Female Dist"].rolling(900).mean()
    # plt.plot(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Time'],
    #          all_data["Rolling Average Nal U50 Female"], label='Rolling Naltr Average U50 Female', color='#8096e6')
    # plt.fill_between(all_data['Paper_Redo_10_18_PreTreat_Nal_5mgkg_U50_Ai14_OPRK1_C1_F0_Top Down Time'], all_data["Rolling Average Nal U50 Female"]-all_data["Naltr Fem SEM"],
    #                  all_data["Rolling Average Nal U50 Female"]+all_data["Naltr Fem SEM"], alpha=0.5, facecolor='#8096e6')
    """Saline Males at 30fps
    """

    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240,
             color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240,
             color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240,
             color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_240,
             color=None)
    #
    # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down Time'],
    #          all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down Dist Travelled'], color='orange')
    # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down Time'],
    #          all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down Dist Travelled'], color='orange')
    # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Time'],
    #          all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Dist Travelled'], color='orange')
    # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Time'],
    #          all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Dist Travelled'], color='orange')

    # create a dataframe with only Saline values to make it easier to average
    # .loc[all of the rows, [only relevant Saline columns]]
    only_saline = all_data.loc[:, ['Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down Dist Travelled',
                                   'Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down Dist Travelled',
                                   'Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Dist Travelled',
                                   'Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Dist Travelled']]
    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    all_data["Average Saline Dist"] = only_saline.mean(axis=1)
    # then calculate the rolling mean over a specified period of time (based on df index)
    # since filmed in 30fps, this is calculated over 30 frames to represent one second
    all_data["Saline SEM Males"] = stats.sem(only_saline, axis=1)
    all_data["Rolling Average Saline"] = all_data["Average Saline Dist"].rolling(300).mean()
    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Time'], all_data["Rolling Average Saline"],
             label='Rolling Average Saline Males', color='#b2d88d')
    plt.fill_between(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Time'], all_data["Rolling Average Saline"]+all_data["Saline SEM Males"],
                     all_data["Rolling Average Saline"]-all_data["Saline SEM Males"], alpha=0.5, facecolor='#b2d88d')

    """
    Saline Females at 30fps
    """
    # velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_250,
    #          color=None)
    # velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down', fps=30, no_seconds=10, DLCscorer=dlc_250,
    #          color=None)
    #
    #
    # # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down Time'],
    # #          all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down Dist Travelled'], color='orange')
    # # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down Time'],
    # #          all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down Dist Travelled'], color='orange')
    #
    #
    # # create a dataframe with only Saline values to make it easier to average
    # # .loc[all of the rows, [only relevant Saline columns]]
    # only_saline_fem = all_data.loc[:, ['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down Dist Travelled',
    #                                'Paper_Redo_Saline_Ai14_OPRK1_C2_F2_Top Down Dist Travelled']]
    # # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    # all_data["Average Saline Female Dist"] = only_saline_fem.mean(axis=1)
    # # then calculate the rolling mean over a specified period of time (based on df index)
    # # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # all_data["SEM Saline Fem"] = stats.sem(only_saline_fem, axis=1)
    # all_data["Rolling Average Saline Female"] = all_data["Average Saline Female Dist"].rolling(450).mean()
    # plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down Time'], all_data["Rolling Average Saline Female"],
    #          label='Rolling Average Saline Female', color='#ef24b8')
    # plt.fill_between(all_data['Paper_Redo_Saline_Ai14_OPRK1_C2_F0_Top Down Time'], all_data["Rolling Average Saline Female"]-all_data["SEM Saline Fem"],
    #                  all_data["Rolling Average Saline Female"]+all_data["SEM Saline Fem"], alpha=0.5, facecolor='#ef24b8')

    plt.xlabel('time (seconds)')
    plt.ylabel('distance travelled (pixels)')
    plt.legend(loc=2)
    plt.title('Total Distance vs. Time [15 second window]')
    # # plt.axvspan(300, 600, alpha=0.25, color='blue')
    plt.show()
    # print(all_data)
