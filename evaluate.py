import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing as mp
import vitaldb

MAX_CASES=20000
TARGETMODEL = 31196 

SUGGEST = 1  # continuous suggestion of RL model for (n) seconds

CSVDIR = 'csv'

modelname = 'CQLSAC'

UL = 1.2
LL  = 0.8

PTHRES = 0.95

idir = ''

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
#borame_files = [ f for f in os.listdir(boramedir) if f.startswith(f"borame_{modelname}")]
#print(f'{len(borame_files)} borame files for {modelname} are detected')
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

def makeresult(dir, file):
    
    dat = np.load(os.path.join(dir,file), allow_pickle=True)
    s = dat['state']   # (1112292, 10)
    a = dat['action']
    c = dat['caseid']
    aopts = dat['action_pred']
    paopts = dat['action_prob']
    #v = dat['qvalue'] # Estimated reward from RL
    #d = dat['qdata'] # Estimated reward from clinician (based on learned policy)
    r = dat['nreward'] # observable reward

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

    # DIFF = 1  ## Discrepancy time between actions of clinician and extuaiß
    # DIFF *= 6


    caseids = list(np.sort(np.unique(c)))
    len(caseids)  # 1775

    data = []

    exclude_data = []
    # empty = []
    # print(caseids)
    for caseid in caseids:
        # print(f'caseid = {caseid}')
        case_mask = (c==caseid)
        case_len = np.sum(case_mask)
        # print(case_len)
        # if case_len == 0:
        #     continue
        
        row = []
        s_case = s[case_mask]
        a_case = a[case_mask]
        #v_case = v[case_mask]
        #d_case = d[case_mask]
        aopts_case = aopts[case_mask]
        paopts_case = paopts[case_mask]
        r_case = r[case_mask]

        #SPO2
        spo92_0 = np.sum(s_case[:, SPO2] <92)
        spo95_0 = np.sum(s_case[:, SPO2] <95)
        spo97_0 = np.sum(s_case[:, SPO2] <97)
        
        #PIP
        pip20_0 = np.sum(s_case[:, PIP] > 1.2)
        pip25_0 = np.sum(s_case[:, PIP] > 1.25)
        pip30_0 = np.sum(s_case[:, PIP] > 1.30)
        
        #SBP
        sbp20_0 = np.sum(s_case[:, SBP] > 1.2)
        sbp25_0 = np.sum(s_case[:, SBP] > 1.25)
        sbp30_0 = np.sum(s_case[:, SBP] > 1.3)

        #HR
        hr20_0 = np.sum(s_case[:, HR] >1.2)  
        hr25_0 = np.sum(s_case[:, HR] >1.25) 
        hr30_0 = np.sum(s_case[:, HR] >1.3)  # or 120
        
        ##LOWER
        #PIP
        pip20_0_low = np.sum(s_case[:, PIP] < 0.8)
        pip25_0_low = np.sum(s_case[:, PIP] < 0.75)
        pip30_0_low = np.sum(s_case[:, PIP] < 0.7)

        
        #SBP
        sbp20_0_low = np.sum(s_case[:, SBP] < 0.8)
        sbp25_0_low = np.sum(s_case[:, SBP] < 0.75)
        sbp30_0_low = np.sum(s_case[:, SBP] < 0.7)

        
        #HR
        hr20_0_low = np.sum(s_case[:, HR] < 0.8)  
        hr25_0_low = np.sum(s_case[:, HR] < 0.75)  
        hr30_0_low = np.sum(s_case[:, HR] < 0.7)  # or 120
        
        #APNEA
        apnea2_0 = np.sum(np.where(s_case[:, APNEA] > 0, 1, 0))
        
        clin_vent = np.where(((a_case == 1)), 1, 0)
        rl_vent = np.where(((aopts_case == 1)), 1, 0)
        
        

        composite_0 = np.sum((s_case[:, SPO2] <95) | (s_case[:, SBP] > UL) | (s_case[:, HR] >UL))

        diff_vent = (clin_vent != rl_vent).sum()
    
        row = [caseid, case_len, clin_vent, rl_vent, 
            spo92_0, spo95_0, spo97_0,
            pip20_0, pip25_0, pip30_0, 
            sbp20_0, sbp25_0, sbp30_0,
            hr20_0, hr25_0, hr30_0, 
            apnea2_0, 
            pip20_0_low, pip25_0_low, pip30_0_low, 
            sbp20_0_low, sbp25_0_low, sbp30_0_low,
            hr20_0_low, hr25_0_low, hr30_0_low,
            composite_0, diff_vent,
            ]

        data.append(row)

    colnames = ['caseid', 'case_len', 'clin_vent', 'rl_vent', 
                'spo92_0', 'spo95_0', 'spo97_0',
                'pip20_0','pip25_0', 'pip30_0',
                'sbp20_0','sbp25_0', 'sbp30_0',
                'hr20_0', 'hr25_0',  'hr30_0',
                'apnea2_0', 
                'pip20_0_low', 'pip25_0_low', 'pip30_0_low',
                'sbp20_0_low', 'sbp25_0_low', 'sbp30_0_low',
                'hr20_0_low', 'hr25_0_low',  'hr30_0_low',
                'composite_0', 'diff_vent',
                ] #'peri_reintu', 
    data = pd.DataFrame(data, columns = colnames)

    # if dir == traindir:
    #     data.to_csv(os.path.join(otrain, f'./results_{file}.csv'), index = False)
    if dir == validdir:
        data.to_csv(os.path.join(ovalid, f'./results_{file}.csv'), index = False)
    elif dir == testdir:
        data.to_csv(os.path.join(otest, f'./results_{file}.csv'), index = False)
    #elif dir == boramedir:
    #    data.to_csv(os.path.join(oborame, f'./results_{file}.csv'), index = False)
    elif dir == snubhdir:
        data.to_csv(os.path.join(osnubh, f'./results_{file}.csv'), index = False)
    # print(f'Processing {file} is done')
    
    elif dir == datexdir:
        data.to_csv(os.path.join(odatex, f'./results_{file}.csv'), index = False)
    # print(f'Processing {file} is done')
    
    if len(exclude_data) > 0:
        print('Exclude data case id: ', exclude_data)

print('processing valid files', end='...', flush=True)
for file in valid_files:
    if str(TARGETMODEL) in file:
        makeresult(validdir, file)
        break

print('done')

print('processing test files', end='...', flush=True)
for file in test_files:
    if str(TARGETMODEL) in file:
        makeresult(testdir, file)
        break

print('done')


print('processing snubh files', end='...', flush=True)
for file in snubh_files:
    if str(TARGETMODEL) in file:
        makeresult(snubhdir, file)
        break
print('done')

print('processing datex files', end='...', flush=True)
for file in datex_files:
    if str(TARGETMODEL) in file:
        makeresult(datexdir, file)
        break
print('done')
