from doctest import ELLIPSIS_MARKER
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.impute import KNNImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

modelname = 'CQLSAC'
TARGETMODEL = 31196 #87842 #83811 #36422 #36049
RESAMPLING_RATE= 0.1
RESAMPLING= 3000
ACTION = ['ventilator'] #['ventilator', 'extubation']
HEMODYNAMICS = ['spo', 'sbp', 'hr', 'pip', 'apnea']
PIP = [20,25,30]
SPO = [97,95,92]
SBP = [20,25,30]
HR = [20,25,30]
APNEA = [2]


test_df = pd.read_csv(f'./results_test_{modelname}.csv')
snubh_df = pd.read_csv(f'./results_snubh_{modelname}.csv')
datex_df = pd.read_csv(f'./results_datex_{modelname}.csv')

def btstrap(a_data_frame, num_sample):
    btstr_data = pd.DataFrame(columns=a_data_frame.columns)
    for a_data in range(RESAMPLING):
        candidate = np.random.permutation(len(a_data_frame))[:num_sample]
        btstr_data = btstr_data.append(a_data_frame.iloc[candidate,:])
    return btstr_data

def ventcut(x):
    if  x < 10:
        return 0
    elif x < 30:
        return 20
    elif x < 50:
        return 40
    elif x < 70:
        return 60
    elif x < 90:
        return 80
    elif x < 110:
        return 100
    elif x < 130:
        return 120
    elif x < 150:
        return 140
    elif x < 170:
        return 160
    elif x < 190:
        return 180
    elif x < 210:
        return 200
    elif x < 230:
        return 220

    
    elif x < 250:
        return 240
    elif x < 270:
        return 260
    elif x < 290:
        return 280
    else:
        return 300


def ventsnubhcut(x):
    if  x < 10:
        return 0
    elif x < 30:
        return 20
    elif x < 50:
        return 40
    elif x < 70:
        return 60
    elif x < 90:
        return 80
    elif x < 110:
        return 100
    elif x < 130:
        return 120
    elif x < 150:
        return 140
    elif x < 170:
        return 160
    elif x < 190:
        return 180
    else:
        return 200

def extucut(x):
    if x < -105:
        return -100
    elif x < -95:
        return -90
    elif x < -85:
        return -80
    elif x < -65:
        return -70
    elif x < -55:
        return -60
    elif x < -45:
        return -50
    elif x < -35:
        return -40
    elif x < -25:
        return -30
    elif x < -15:
        return -20
    elif x < -5:
        return -10
    elif x < 5:
        return 0
    elif x < 15:
        return 10
    elif x < 25:
        return 20
    elif x < 35:
        return 30
    elif x < 45:
        return 40
    elif x < 55:
        return 50
    elif x < 65:
        return 60
    elif x < 75:
        return 70
    elif x < 85:
        return 80
    elif x < 95:
        return 90
    else:
        return 100

def snubhcut(x):
    if x <= -75:
        return -90
    elif x < -45:
        return -60
    elif x < -15:
        return -30
    elif x < 15:
        return 0
    elif x < 45:
        return 30
    elif x < 75:
        return 60
    elif x < 105:
        return 90
    elif x < 135:
        return 120
    elif x < 165:
        return 150
    elif x < 195:
        return 180
    elif x < 225:
        return 210
    else:
        return 240 


def diffplot(df, outcome):
    if len(df) > 3000:
        data = 'datex'
    elif len(df) > 1000:
        data = 'test'
    else:
        data = 'snubh'
    print(f'plotting {outcome} in {data}set')
    
    
    for action in ACTION:
        if data =="test":
            if action =='ventilator':
                df[action] = df['diff_vent'].apply(lambda x: ventcut(x))
           
            color = 'orange'

        elif data == "snubh":
            if action =='ventilator':
                df[action] = df['diff_vent'].apply(lambda x: ventsnubhcut(x))
            
            color = 'green'
            
        elif data == "datex":
            if action =='ventilator':
                df[action] = df['diff_vent'].apply(lambda x: ventcut(x))
            
            color = 'green'


        if outcome == 'sbp':
            for lim in SBP:
                out = f'{outcome}{lim}_0'
                plt.figure(figsize=(10, 10))
                # plt.plot(t, obs[:, HR] / 5 , label='HR / 5', color='pink')
                sns.lineplot(data=df, x=action, y=out, color='red', n_boot=RESAMPLING)
                plt.title(f'{outcome.upper()} > {lim}%')
                # plt.xlim([0, case_len])
                # plt.ylim([-2, 50])
                plt.ylabel('Lasting time under condition (sec)')
                plt.xlabel(f'Delayed ventilation time (sec)')
                plt.tight_layout()
                plt.savefig(f'./plot/{data}/{action}_{out}.tiff')
                # plt.savefig(f'{odir}/{case_len}.png'
                plt.close()
                
        if outcome == 'hr':
            for lim in HR:
                out = f'{outcome}{lim}_0'
                plt.figure(figsize=(10, 10))
                # plt.plot(t, obs[:, HR] / 5 , label='HR / 5', color='pink')
                sns.lineplot(data=df, x=action, y=out, color='green', n_boot=RESAMPLING)
                plt.title(f'{outcome.upper()} > {lim}%')
                # plt.xlim([0, case_len])
                # plt.ylim([-2, 50])
                plt.ylabel('Lasting time under condition (sec)')
                plt.xlabel(f'Discrepant ventilation time (sec)')
                plt.tight_layout()
                plt.savefig(f'./plot/{data}/{action}_{out}.tiff')
                # plt.savefig(f'{odir}/{case_len}.png'
                plt.close()
        if outcome == 'spo':
            for lim in SPO:
                out = f'{outcome}{lim}_0'
                plt.figure(figsize=(10, 10))
                # plt.plot(t, obs[:, HR] / 5 , label='HR / 5', color='pink')
                sns.lineplot(data=df, x=action, y=out, color='blue', n_boot=RESAMPLING)
                plt.title(f'{outcome.upper()}2 < {lim}%')
                # plt.xlim([0, case_len])
                # plt.ylim([-2, 50])
                plt.ylabel('Lasting time under condition (sec)')
                plt.xlabel(f'Discrepant ventilation time (sec)')
                plt.tight_layout()
                plt.savefig(f'./plot/{data}/{action}_{out}.tiff')
                # plt.savefig(f'{odir}/{case_len}.png'
                
        if outcome == 'pip':
            for lim in PIP:
                out = f'{outcome}{lim}_0'
                plt.figure(figsize=(10, 10))
                # plt.plot(t, obs[:, HR] / 5 , label='HR / 5', color='pink')
                sns.lineplot(data=df, x=action, y=out, color='purple', n_boot=RESAMPLING)
                plt.title(f'{outcome.upper()} > {lim}%')
                # plt.xlim([0, case_len])
                # plt.ylim([-2, 50])
                plt.ylabel('Lasting time under condition (sec)')
                plt.xlabel(f'Discrepant ventilation time (sec)')
                plt.tight_layout()
                plt.savefig(f'./plot/{data}/{action}_{out}.tiff')
                # plt.savefig(f'{odir}/{case_len}.png'
                plt.close()
                
        if outcome == 'apnea':
            for lim in APNEA:
                out = f'{outcome}{lim}_0'
                plt.figure(figsize=(10, 10))
                # plt.plot(t, obs[:, HR] / 5 , label='HR / 5', color='pink')
                sns.lineplot(data=df, x=action, y=out, color='orange', n_boot=RESAMPLING)
                plt.title(f'ETCO2 < {lim}mmHg ({outcome.upper()})')
                # plt.xlim([0, case_len])
                # plt.ylim([-2, 50])
                plt.ylabel('Lasting time under condition (sec)')
                plt.xlabel(f'Discrepant ventilation time (sec)')
                plt.tight_layout()
                plt.savefig(f'./plot/{data}/{action}_{out}.tiff')
                # plt.savefig(f'{odir}/{case_len}.png'
                plt.close()
                
        if outcome == 'composite':
            out = f'{outcome}_0'
            plt.figure(figsize=(10, 10))
            # plt.plot(t, obs[:, HR] / 5 , label='HR / 5', color='pink')
            sns.lineplot(data=df, x=action, y=out, color='gold', n_boot=RESAMPLING)
            plt.title(f'A composite of hemodynmaic instability')
            # plt.xlim([0, case_len])
            # plt.ylim([-2, 50])
            plt.ylabel('Duration under unstable condition (sec)')
            plt.xlabel(f'Discrepant ventilation time (sec)')
            #plt.xticks(np.arange(-180, 180, 13))
            plt.tight_layout()
            plt.savefig(f'./plot/{data}/{action}_{out}.tiff')
            # plt.savefig(f'{odir}/{case_len}.png'

for df in [datex_df, test_df, snubh_df]:
    for hemo in HEMODYNAMICS:
        diffplot(df, hemo)
    diffplot(df, 'composite')