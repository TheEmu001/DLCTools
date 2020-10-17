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


def velocity(video, fps, no_seconds, color):
    DLCscorer='DLC_resnet50_BigBinTopSep17shuffle1_240000'

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

    # PreTreat
    # Note: when plotting these graphs, ran into the issue of the video with 60fps being graphed only half way, looks
    # like pyplot will establish the x-axis based on the first plot, graph highest framerate first to make sure this
    # isn't prblematic in the future
    # AKA M1 graphed at 60fps looked like it was stopping prematurely bc graph was made first with a video of 30fps
    velocity(video='Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down', fps=60, no_seconds=10, color=None)
    velocity(video='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down', fps=30, no_seconds=10, color=None)
    velocity(video='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top Down', fps=30, no_seconds=10, color=None)
    velocity(video='Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top Down', fps=30, no_seconds=10, color=None)

    plt.plot(all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'], all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled'], label='M1', color='pink')
    plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Time'], all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down '
                                                                                                         'Dist Travelled'], label='M2', color='purple')
    plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top Down Time'], all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top '
                                                                                                         'Down Dist Travelled'], label='M3', color='red')
    plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top Down Time'], all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top '
                                                                                                         'Down Dist Travelled'], label='M4', color='black')

    # create a dataframe with only Naltrexone values to make it easier to average
    # .loc[all of the rows, [only relevant Naltrexone columns]]
    only_saline = all_data.loc[:, ['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled','Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Dist Travelled', 'Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M3_Top Down Dist Travelled',
                                   'Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M4_Top Down Dist Travelled']]
    print(all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'])
    rando = 0
    for items in all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time']:
        if items not in all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Time'].values:
            # ind_val = all_data[all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'] == items].index
            # mask = all_data[all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'] == items]
            all_data[all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'].values == items].replace(to_replace=items, value=np.nan)
            rando=rando+1
            print("replaced "+str(rando)+" values")
            # all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'].drop(index=all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'].loc(items))
            # print(str(items)+" not in OG")
            # all_data.replace(items, np.nan)

    print(all_data['Paper_Redo_PreTreat5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'])

    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    all_data["Average Naltr Dist"] = only_saline.mean(axis=1)
    # then calculate the rolling mean over a specified period of time (based on df index)
    # since filmed in 30fps, this is calculated over 30 frames to represent one second
    # this part is a little weird since there is one video filmed at 60fps
    # Maybe figure out away to get this to roll from the specified time?
    all_data["Rolling Average Naltr"] = all_data["Average Naltr Dist"].rolling('1s').mean()
    plt.plot(all_data['Paper_Redo_5mg_kgU50PreTreatNaltrexone_Ai14_OPRK1_C1_M2_Top Down Time'], all_data["Average Naltr Dist"], label='Average Naltr Dist', color='#b86c31')

    #Saline
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down', fps=30, no_seconds=10, color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down', fps=30, no_seconds=10, color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down', fps=30, no_seconds=10, color=None)
    velocity(video='Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down', fps=30, no_seconds=10, color=None)

    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down Time'], all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down Dist Travelled'], color='orange')
    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down Time'], all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down Dist Travelled'], color='orange')
    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Time'], all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Dist Travelled'], color='orange')
    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Time'], all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Dist Travelled'], color='orange')

    # create a dataframe with only Saline values to make it easier to average
    # .loc[all of the rows, [only relevant Saline columns]]
    only_saline = all_data.loc[:, ['Paper_Redo_Saline_Ai14_OPRK1_C1_M1_trial2_Top Down Dist Travelled','Paper_Redo_Saline_Ai14_OPRK1_C1_M2_trial2_Top Down Dist Travelled', 'Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Dist Travelled',
                                   'Paper_Redo_Saline_Ai14_OPRK1_C1_M4_trial2_Top Down Dist Travelled']]
    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    all_data["Average Saline Dist"] = only_saline.mean(axis=1)
    # then calculate the rolling mean over a specified period of time (based on df index)
    # since filmed in 30fps, this is calculated over 30 frames to represent one second
    all_data["Rolling Average Saline"] = all_data["Average Saline Dist"].rolling(600).mean()
    plt.plot(all_data['Paper_Redo_Saline_Ai14_OPRK1_C1_M3_trial2_Top Down Time'], all_data["Rolling Average Saline"], label='Rolling Average Saline', color='#b2d88d')

    # 5mg/kg U50

    velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down', fps=60, no_seconds=10, color=None)
    velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top Down', fps=60, no_seconds=10, color=None)
    velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top Down', fps=60, no_seconds=10, color=None)
    velocity(video='Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top Down', fps=60, no_seconds=10, color=None)

    # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top '
    #                                                                                    'Down Dist Travelled'], label='M1', color='blue')
    # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top '
    #                                                                                    'Down Dist Travelled'], label='M2', color='blue')
    # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top '
    #                                                                                    'Down Dist Travelled'], label='M3', color='blue')
    # plt.plot(all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top Down Time'], all_data['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top '
    #                                                                                    'Down Dist Travelled'], label='M4', color='blue')

    # create a dataframe with only 5mgkg U50 values to make it easier to average
    # .loc[all of the rows, [only relevant u50 columns]]
    only_5mgU50 = all_data.loc[:, ['Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Dist Travelled','Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M2_Top Down Dist Travelled', 'Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M3_Top Down Dist Travelled', 'Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M4_Top Down Dist Travelled']]
    # calculate mean for relevant u50 columns, this calculates an average velocity for each point in time
    all_data["Average 5mgkg U50"] = only_5mgU50.mean(axis=1)
    # then calculate the rolling mean over a specified period of time (based on df index)
    # since filmed in 60fps, this is calculated over 600 frames to represent one second
    all_data["Rolling Average 5mgkg U50"] = all_data["Average 5mgkg U50"].rolling(600).mean()
    plt.plot(all_data["Paper_Redo_5mg_kgU50_Ai14_OPRK1_C1_M1_Top Down Time"], all_data["Rolling Average 5mgkg U50"], label='Rolling Avg U50')


    # 10mg/kg U50
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M1_Top Down', fps=30, no_seconds=10, color=None)
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M2_Top Down', fps=30, no_seconds=10, color='purple')
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M3_Top Down', fps=30, no_seconds=10, color='orange')
    # velocity(video='Paper_Redo_U50_Ai14_OPRK1_C1_M4_Top Down', fps=30, no_seconds=10, color='green')
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
    #

    plt.xlabel('time (seconds)')
    plt.ylabel('distance travelled (pixels)')
    plt.legend(loc=2)
    plt.title('Total Distance vs. Time')
    # # plt.axvspan(300, 600, alpha=0.25, color='blue')
    plt.show()
    # print(all_data)
