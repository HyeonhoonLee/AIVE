import vitaldb
import pandas as pd
import os
import numpy as np
import random
import time


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.inspection import permutation_importance
import shap

import matplotlib.pyplot as plt
import seaborn as sns

# modellist: DQN, DDQN, SAC, BCQ, CQL
modelname = 'CQLSAC'

# ANEST_TYPE = 'primus'
MAX_CASES = 20000 
TARGETMODEL = 31196 

SUGGEST = 1  # continuous suggestion of RL model for (n) seconds

DEBUG = False
TOTALEVAL = True
REMOVESTATE = False


SEED = 42
# QUANTILE=4
N_ROUND = 500

N_ESTIMATORS = 5
N_WORKERS = 1

CLIP = 5
# MAX_ITER = 20
NMODEL = 1
EPOCHS = 1
UPDATE_FREQ = 1

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
set_seed(SEED)

CSVDIR = 'csv'

idir = '/home/ubuntu/hhl/research/kohi/kohi-project/data/220517/'

totaldir = './nptotal'
# traindir = './nptrain/'
validdir = './npvalid/'
testdir = './nptest/'
datexdir = './npdatex/'
snubhdir = './npsnubh/'

ototal = './resulttotal/'
# otrain = './resulttrain/'
ovalid = './resultvalid/'
otest = './resulttest/'
odatex = './resultdatex/'
osnubh = './resultsnubh/'

if not os.path.exists(ototal):
    os.mkdir(ototal)
if not os.path.exists(ovalid):
    os.mkdir(ovalid)
if not os.path.exists(otest):
    os.mkdir(otest)
if not os.path.exists(odatex):
    os.mkdir(odatex)
if not os.path.exists(osnubh):
    os.mkdir(osnubh)
    

# np.savez('withaopts.npz', c, s, qs, v, a, aopts)
df_finder = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_primus.csv'))
df_finder4 = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_snubhprimus.csv'))
df_finder5 = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_datex.csv'))
df_finder = pd.concat([df_finder, df_finder4, df_finder5], ignore_index=True)
print(df_finder.tail())

#df_clinical = pd.read_csv(os.path.join(idir, 'total_clinical_info_incl_lma_220803.csv'))
df_clinical = pd.read_csv(os.path.join(idir, 'total_clinical_info_incl_lma_220907.csv'))

df_clinical['fileid'] = df_clinical['fileid'].apply(lambda x: x.split('.')[0])
    

df_label = pd.read_csv(os.path.join(idir, 'result_15sec2mmHg_primus.csv'))

df_label4 = pd.read_csv(os.path.join(idir, 'result_15sec2mmHg_snubhprimus.csv'))
df_label5 = pd.read_csv(os.path.join(idir, 'result_15sec2mmHg_datex.csv'))
df_label = df_label.dropna()
#df_label['caseid'] = df_label['filename'].apply(lambda x: x.split('.')[0])
df_label['filename'] = df_label['filename'].apply(lambda x: x.split('.')[1])
#df_label4['caseid'] = df_label4['filename'].apply(lambda x: x.split('.')[0])
df_label4['filename'] = df_label4['filename'].apply(lambda x: x.split('.')[0])
df_label5['filename'] = df_label5['filename'].apply(lambda x: x.split('.')[0])

df_label = pd.concat([df_label, df_label4, df_label5], ignore_index=True)

#print(len(df_finder))
#print(len(df_label))

valid_files = [ f for f in os.listdir(validdir) if f.startswith(f"valid_{modelname}")]
print(f'{len(valid_files)} valid files for {modelname} are detected')

test_files = [ f for f in os.listdir(testdir) if f.startswith(f"test_{modelname}")]
print(f'{len(test_files)} test files for {modelname} are detected')

snubh_files = [ f for f in os.listdir(snubhdir) if f.startswith(f"snubh_{modelname}")]
print(f'{len(snubh_files)} snubh files for {modelname} are detected')

datex_files = [ f for f in os.listdir(datexdir) if f.startswith(f"datex_{modelname}")]
print(f'{len(datex_files)} datex files for {modelname} are detected')

def finder(caseid):
    return df_finder[df_finder['icase']==caseid]['filename'].to_numpy()[0]
            
def count_repeated_true(v):
    """
    [0 1 1 0 1 1 1 0] --> [0 1 2 0 1 2 3 0]
    """
    v = np.array(v).astype(int)
    reset_mask = (v == 0)
    valid_mask = ~reset_mask
    c = np.cumsum(valid_mask)
    # c = [0 1 2 2 3 4 5 5]
    d = np.diff(np.concatenate(([0.], c[reset_mask])))
    # d = [0 2 3]  # 앞에 있는 true의 갯수
    v[reset_mask] = -d
    return np.cumsum(v)

PPF20_CE = 0
RFTN20_CE = 1
EXP_SEVO = 2
PIP = 3
TV = 4
AWP = 5
CO2 = 6  # CO2 curve 로부터 구함
HR = 7
SPO2 = 8
SBP = 9
SB = 10
APNEA = 11 
VENT_STATE = 12
EXTU_STATE = 13

FEATURES = ['PPF', 'RTFN', 'SEVO', 'PIP','TV', 'AWP', 'ETCO2', 'HR', 'SPO2', 'SBP', 'SB', 'APNEA', 'INTU'] #'VENT_STATE',

print('processing valid files', end='...', flush=True)
for file in valid_files:
    if str(TARGETMODEL) in file:
        dat = np.load(os.path.join(validdir,file), allow_pickle=True)
        valid_s = dat['state']   # (1112292, 10)
        valid_a = dat['action']
        valid_c = dat['caseid']
        valid_aopts = dat['action_pred']

#valid_caseids = np.unique(valid_c)

print('processing test files', end='...', flush=True)
for file in test_files:
    if str(TARGETMODEL) in file:
        dat = np.load(os.path.join(testdir,file), allow_pickle=True)
        test_s = dat['state']   # (1112292, 10)
        test_a = dat['action']
        test_c = dat['caseid']
        test_aopts = dat['action_pred']

#test_caseids = np.unique(test_c)

print('processing snubh files', end='...', flush=True)
for file in snubh_files:
    if str(TARGETMODEL) in file:
        dat = np.load(os.path.join(snubhdir,file), allow_pickle=True)
        snubh_s = dat['state']   # (1112292, 10)
        snubh_a = dat['action']
        snubht_c = dat['caseid']
        snubh_aopts = dat['action_pred']

#snubh_caseids = np.unique(snubh_c)


def xgb_shap(valid_s, valid_a, test_s, test_a, task):
    num_round = N_ROUND 

    param = {
        "eta": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "tree_method": "gpu_hist",
        "eval_metric": "logloss"
    }

    # GPU accelerated training
    #dtrain = xgb.DMatrix(valid_s, label=valid_a, feature_names=FEATURES)
    dtrain = xgb.DMatrix(np.delete(valid_s, [VENT_STATE], 1), label=valid_a, feature_names=FEATURES)
    model = xgb.train(param, dtrain,num_round)

    model.set_param({"predictor": "gpu_predictor"})

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.delete(test_s, [VENT_STATE], 1))

    plt.figure(figsize=(10, 10))
    #shap.summary_plot(np.delete(shap_values, [VENT_STATE], 1), np.delete(test_s, [VENT_STATE], 1), feature_names=FEATURES[:VENT_STATE]+FEATURES[VENT_STATE+1:])  # , max_display=10
    shap.summary_plot(shap_values, np.delete(test_s, [VENT_STATE], 1), feature_names=FEATURES)
    
    if task.endswith('clin'):
        titlename = "Clinicians' policy"
    else:
        titlename = "AIVE's policy" 
    
    plt.title(f'{titlename}')
    plt.savefig(f'./plot/shap/shap_action0_{task}.tiff', bbox_inches="tight", pad_inches=1)
    plt.close('all')
    
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, np.delete(test_s, [VENT_STATE], 1), plot_type='bar', feature_names=FEATURES)
    #shap.summary_plot(np.delete(shap_values, [VENT_STATE], 1), np.delete(test_s, [VENT_STATE], 1), plot_type='bar', feature_names=FEATURES[:VENT_STATE]+FEATURES[VENT_STATE+1:])  # , max_display=10
    plt.title(f'{titlename}')
    plt.savefig(f'./plot/shap/shap_aciton0_bar_{task}.tiff', bbox_inches="tight", pad_inches=1)
    plt.close('all')

    print(f'{task} done')

print(f'xboost shap for clinician in testset')
#xgb_shap(valid_s, valid_a, test_s, test_a, 'test_clin')
xgb_shap(test_s, test_a, test_s, test_a, 'test_clin')
print(f'xboost shap for extuai in testset')
xgb_shap(test_s, test_aopts,  test_s, test_aopts, 'test_ai')
print(f'xboost shap for clinician in snubhset')
xgb_shap(snubh_s, snubh_a, snubh_s, snubh_a, 'snubh_clin')
print(f'xboost shap for extuai in snubhset')
xgb_shap(snubh_s, snubh_aopts, snubh_s, snubh_aopts, 'snubh_ai')
