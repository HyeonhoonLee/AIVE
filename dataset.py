import vitaldb
import pandas as pd
import os
import numpy as np
import random
import glob

vital_path = ''

MAX_CASES = 20000 #9304는 pip까지 뽑는 케이스  아마도 9307? 9223?

PPF20_CE = 0
RFTN20_CE = 1
EXP_SEVO = 2
EXP_DES = 3
COMPLIANCE = 4
PIP= 5
PPLAT = 6
SET_PEEP = 7
SET_TV = 8
TV = 9
FIO2 = 10
AWP = 11 # AWP curve 로부터 구한 평균값 for each STATE 0,1,2
CO2 = 12 # CO2 curve 로부터 5초 이동평균값, Spontaneous CO2 on STATE 1,2
BIS = 13
EMG = 14
SEF = 15
HR = 16
SPO2 = 17
BT = 18
DBP = 19
MBP = 20
SBP = 21
SB = 22
APNEA = 23
BASE_HR = 24  #처음 10초간 HR값의 평균을 baseline
BASE_PIP = 25 #처음 10초간 PIP값의 평균을 baseline
BASE_TV = 26  #처음 10초간 TV값의 평균을 baseline
BASE_SBP = 27 #처음 10초간 SBP값의 평균을 baseline
VENT_STATE = 28
EXTU_STATE = 29 # 현재 상태



track_names = [ 'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE', 
                'Primus/EXP_SEVO', 'Primus/EXP_DES', 'Primus/COMPLIANCE', 
                'Primus/PIP_MBAR', 'Primus/PPLAT_MBAR', 'Primus/SET_INTER_PEEP', 'Primus/SET_TV_L', 'Primus/TV', 'Primus/FIO2',
                'Primus/AWP', 'Primus/CO2', 
                'BIS/BIS', 'BIS/EMG', 'BIS/SEF', 
                'Solar8000/HR', 'Solar8000/PLETH_SPO2',  'Solar8000/BT1', 'Solar8000/ART_DBP', 'Solar8000/ART_MBP', 'Solar8000/ART_SBP',  'Solar8000/NIBP_DBP', 'Solar8000/NIBP_MBP', 'Solar8000/NIBP_SBP',
                'Solar 8000M/HR', 'Solar 8000M/PLETH_SPO2', 'Solar 8000M/BT1', 'Solar 8000M/ART_DBP', 'Solar 8000M/ART_MBP', 'Solar 8000M/ART_SBP', 'Solar 8000M/NIBP_DBP', 'Solar 8000M/NIBP_MBP', 'Solar 8000M/NIBP_SBP' ] 

def get_reward(s):

    # Excessive value from baseline
    over_hr = np.divide(s[:, HR], s[:, BASE_HR], out=np.ones_like(s[:, HR]), where=s[:, BASE_HR]!=0)
    over_sbp = np.divide(s[:, SBP], s[:, BASE_SBP], out=np.ones_like(s[:, SBP]), where=s[:, BASE_SBP]!=0)
    over_pip = np.divide(s[:, PIP], s[:, BASE_PIP], out=np.ones_like(s[:, PIP]), where=s[:, BASE_PIP]!=0)

    reward = (
        ((s[:, SPO2] < 97) & (s[:, SPO2] > 0)) * (s[:, SPO2] - 97) * 2  +
        ((s[:, APNEA] > 6)) * (6 - s[:, APNEA]) * 1  +
        (over_hr > 1.2) * (1.2 - over_hr) * 20 +
        (over_sbp > 1.2) * (1.2 - over_sbp ) * 20 +
        (over_pip > 1.2) * (1.2 - over_pip ) * 10 
        #(-s[:, SB]) * (s[:, STATE] == 0) * 10
    ) 
    
    return reward

eligible_files = pd.read_csv('/home/ubuntu/hhl/research/kohi/kohi-project/data/220517/criteria/primus/eligible_files_primus.csv')['filename'].tolist()
vital_files = [f.split('/')[-1] for f in glob.glob(os.path.join(vital_path, '*.vital')) if str(f.split('/')[-1]) in eligible_files] # final well-labelled eligible files are listed here. (NOT above list)

cachepath = f'{MAX_CASES}cases_primus.npz'
cachepath_re = f'{MAX_CASES}cases_primus_re.npz'
if os.path.exists(cachepath):
    dat = np.load(cachepath)
    s = dat['s']
    a = dat['a']
    c = dat['c']
    print(f'Base "{cachepath}" file already made')
else:
    # 라벨 데이터를 읽어옴
    caseids = []
    filenames = []

    for file in vital_files:  
        filename = os.path.splitext(file)[0]
        filenames.append(filename)
        caseid = int((filename.split('_')[1] + filename.split('_')[2]))
        caseids.append(caseid)
    
    print(f'{len(caseids)} cases found')

    s = []
    a = []
    c = []
    d = []
    icase = 0  # 현재 로딩 중인 케이스 번호
    ncase = min(MAX_CASES, len(caseids))
    
    CSVDIR = 'csv'
    if not os.path.exists(CSVDIR):
        os.mkdir(CSVDIR)

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

    icase_result = []
    filename_result = []
    no_anyevents = []
    no_extu = []
    no_fio2 = []
    no_lb = []
    delay_fio2 = []
    no_sb = []
    #no_ventoff = []
    no_seq = []
    case_len = []
    no_prime = []
    #ventoffextu = []
    extulb = []
    
    # target = "E2_180620_143227"
    for filename, caseid in zip(filenames, caseids):  # 본 연구에 사용할 각 case에 대하여
        print('loading {} ({}/{})'.format(filename, icase, ncase), end='...', flush=True)
        # if target != filename:
        #     continue
        # 이벤트 타임을 로딩
        ipath = vital_path + f'/{filename}.vital' #wholecase_v2
        # print(ipath)
        vf = vitaldb.VitalFile(ipath, track_names=['EVENT'])
        # vals = vf.to_numpy(['Primus/AWP', 'Primus/SET_INTER_PEEP', 'Primus/FIO2'], 1 / 62.5)
        # print(vals)
        # FIO2 ~ SB
        # try:
        if  vf.find_track('EVENT') != None: ## EVENT가 아예 없는 경우를 배제
            trk = vf.find_track('EVENT')['recs']
            # print(trk)
            fio2 = None
            #ventoff = None
            extu = None
            lastbreath = None
            sb_pip = None
            sb_neg = None
            vent_off = []
            vent_on = []
            manual_off = []
            manual_on = []
            for rec in trk: # 하나씩 시간순으로 읽으면서
                if rec['val'] == 'FIO2':  # 여기서부터 깨우기 시작
                    fio2 = int(rec['dt'] - vf.dtstart)
                if rec['val'] == 'Extubation':
                    extu = int(rec['dt'] - vf.dtstart)
                if rec['val'] == 'SB_NEG':
                    sb_neg = int(rec['dt'] - vf.dtstart)
                if rec['val'] == 'SB_PIP':
                    sb_pip = int(rec['dt'] - vf.dtstart)
                if rec['val'] == 'Last_Breath':
                    lastbreath = int(rec['dt'] - vf.dtstart)
                if rec['val'] == 'Vent_OFF':
                    vent_off.append(int(rec['dt'] - vf.dtstart))
                if rec['val'] == 'Vent_ON':
                    vent_on.append(int(rec['dt'] - vf.dtstart))
                if rec['val'] == 'Manual_OFF':
                    manual_off.append(int(rec['dt'] - vf.dtstart))
                if rec['val'] == 'Manual_ON':
                    manual_on.append(int(rec['dt'] - vf.dtstart))

            
            if not extu:
                no_extu.append(filename)
                print('no extubation label')
                continue
            
            if not fio2:
                no_fio2.append(filename)
                print('no fio2 label')
                continue
            
            if not lastbreath:
                no_lb.append(filename)
                print('no lastbreath label')
                continue
            
            # SB assigned.
            sb = 0
            # spontaneous breathing
            if sb_neg and sb_pip:
                sb = min(sb_neg, sb_pip)
            elif sb_neg:
                sb = sb_neg
            elif sb_pip:
                sb = sb_pip
            
            if 0 < sb < fio2:
                fio2 = sb
                delay_fio2.append(filename)
                print('delayed fio2 raising')
                #continue   
            
            if not sb or (sb >= lastbreath):
                no_sb.append(filename)
                print('no spont breath with other resaons')
                continue   
        
            # lastbreath랑 extu가 완전 동시 이면, extu를 1/10초 전으로 assign
            if extu == lastbreath:
                extu = (lastbreath - 1)  
                extulb.append(filename)

            # 순서대로가 아니면 제거
            #if not (fio2 < ventoff < extu < lastbreath):
            if not (fio2 < extu < lastbreath):
                no_seq.append(filename)
                print('mis-ordered')
                continue
            
            # 1/10 초 간격 데이터 추출
            vf = vitaldb.VitalFile(ipath, track_names=track_names)
            vals = vf.to_numpy([
                'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE', 
                'Primus/EXP_SEVO', 'Primus/EXP_DES', 'Primus/COMPLIANCE', 
                'Primus/PIP_MBAR', 'Primus/PPLAT_MBAR', 'Primus/SET_INTER_PEEP', 'Primus/SET_TV_L', 'Primus/TV', 'Primus/FIO2',
                'Primus/AWP', 'Primus/CO2', 
                'BIS/BIS', 'BIS/EMG', 'BIS/SEF'
                ], 1 / 10)
            
            vals_HR = vf.to_numpy(['Solar 8000M/HR'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_HR))):
                vals_HR = vf.to_numpy(['Solar8000/HR'], 1 / 10)

            vals_SPO2 = vf.to_numpy(['Solar 8000M/PLETH_SPO2'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_SPO2))):
                vals_SPO2 = vf.to_numpy(['Solar8000/PLETH_SPO2'], 1 / 10)
                
            vals_BT = vf.to_numpy(['Solar 8000M/BT1'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_BT))):
                vals_BT = vf.to_numpy(['Solar8000/BT1'], 1 / 10)
                
            # NIBP
            vals_DBP = vf.to_numpy(['Solar 8000M/NIBP_DBP'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_DBP))):
                vals_DBP = vf.to_numpy(['Solar8000/NIBP_DBP'], 1 / 10)
            vals_MBP = vf.to_numpy(['Solar 8000M/NIBP_MBP'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_MBP))):
                vals_MBP = vf.to_numpy(['Solar8000/NIBP_MBP'], 1 / 10)
            vals_SBP = vf.to_numpy(['Solar 8000M/NIBP_SBP'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_SBP))):
                vals_SBP = vf.to_numpy(['Solar8000/NIBP_SBP'], 1 / 10)
                
            # A-line
            vals_DBPa = vf.to_numpy(['Solar 8000M/ART_DBP'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_DBPa))):
                vals_DBPa = vf.to_numpy(['Solar8000/ART_DBP'], 1 / 10)
            vals_MBPa = vf.to_numpy(['Solar 8000M/ART_MBP'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_MBPa))):
                vals_MBPa = vf.to_numpy(['Solar8000/ART_MBP'], 1 / 10)
            vals_SBPa = vf.to_numpy(['Solar 8000M/ART_SBP'], 1 / 10)
            if np.all(np.unique(np.isnan(vals_SBPa))):
                vals_SBPa = vf.to_numpy(['Solar8000/ART_SBP'], 1 / 10)

            ## if not using A-line, filled with NIBP   
            zeroindex_d = np.where(np.isnan(vals_DBPa))[0]
            np.put(vals_DBPa, zeroindex_d, vals_DBP[zeroindex_d])
            zeroindex_m = np.where(np.isnan(vals_MBPa))[0]
            np.put(vals_MBPa, zeroindex_m, vals_MBP[zeroindex_m])
            zeroindex_s = np.where(np.isnan(vals_SBPa))[0]
            np.put(vals_SBPa, zeroindex_s, vals_SBP[zeroindex_s])
            
            if not (np.nanmax(vals_HR) > 0) or not (np.nanmax(vals_SPO2) > 0) or not (np.nanmax(vals_SBPa) > 0):
                no_prime.append(filename)
                print('no primary outcome')
                continue
            
            vals = np.hstack((vals, vals_HR, vals_SPO2, vals_BT, vals_DBPa, vals_MBPa, vals_SBPa))

            
            # 시작 시점 이후의 데이터만 이용
            vals = vals[fio2 * 10:lastbreath * 10]
            vals = pd.DataFrame(vals).interpolate('linear', axis=0).fillna(method = 'ffill').fillna(method = 'bfill')
            vals = vals.fillna(0).values

            # 직전 1초 이내 최대값만 사용
            vals = vals[:((len(vals) // 10) * 10)]
            reshaped_vals = vals.transpose().reshape(vals.shape[1], -1, 10)
            vals = np.nanmax(reshaped_vals, axis=2).transpose()
            
            # Exclusion criteria: Case 길이
            if (len(vals) < 120) or (len(vals) > 1200):
                case_len.append(filename)
                print('excluded by case length')
                continue
            
            # Exclusion criteria: Primary outcome
            if not (np.nanmax(vals_HR) > 0) or not (np.nanmax(vals_SPO2) > 0) or not (np.nanmax(vals_SBPa) > 0):
                no_prime.append(filename)
                print('no primary outcome')
                continue
            
            # sb 여부를 state 변수로
            sbseq = np.zeros(len(vals))
            sbseq[sb - fio2] = 1
            sbseq = np.cumsum(sbseq)
            
            # CO2로부터 apnea를 생성
            apnea = count_repeated_true(vals[:,CO2] < 2)
            
            # 본 case의 state: track에 sb와 apnea를 추가
            case_s = np.hstack([vals, np.reshape(sbseq, (-1, 1)), np.reshape(apnea, (-1, 1))])

            # 처음 10초간 값의 평균을 baseline으로 잡음.
            base_hr = np.ones(len(vals)) * np.mean(vals[:10,HR])
            base_pip = np.ones(len(vals)) * np.mean(vals[:10,PIP])
            base_tv = np.ones(len(vals)) * np.mean(vals[:10,TV])
            base_sbp = np.ones(len(vals)) * np.mean(vals[:10,SBP])
            
            case_s = np.hstack([case_s, np.reshape(base_hr, (-1, 1)), np.reshape(base_pip, (-1, 1)), np.reshape(base_tv, (-1, 1)), np.reshape(base_sbp, (-1, 1))])
                        
            #make action space of initial vent on 
            init_venton = np.ones(len(vals)).astype(int)
            for i in range(len(vals)):
                if i+fio2 in vent_off:
                    init_venton[i:] = 0
                    break
                    
            #make action space of vent_status
            vent_status = np.ones(len(vals)).astype(int)
            for i in range(len(vals)):
                if i+fio2 in vent_off:
                    vent_status[i:] = 0
                if i+fio2 in vent_on:
                    vent_status[i:] = 1
            
            #make action space of manual_status
            manual_status = np.zeros(len(vals)).astype(int)
            for i in range(len(vals)):
                if i+fio2 in manual_off:
                    manual_status[i:] = 0
                if i+fio2 in manual_on:
                    manual_status[i:] = 1
            
            extubation = np.zeros(len(vals))
            extubation[extu - fio2] = 1
            #extubation = np.cumsum(extubation)

            case_a = np.vstack([init_venton, manual_status, extubation, vent_status]).T  
            
            case_d = np.zeros(len(vals))
            case_d[-1] = 1
            
            # 로딩된 vals 뒤에 caseid를 붙임
            s.extend(case_s)
            a.extend(case_a)
            c.extend([icase] * len(vals))
            d.extend(case_d)

            # 리뷰를 위해 출력
            df_case = pd.DataFrame(case_s, columns=['PPF', 'RFTN', 'SEVO', 'DES', 'COMPLIANCE', 'PIP', 'PPLAT', 'SET_PEEP', 'SET_TV', 'TV',
                                                    'FIO2', 'MEAN_AWP', 'CO2', 'BIS', 'EMG', 'SEF', 'HR', 'SPO2', 'BT', 'DBP', 
                                                    'MBP', 'SBP', 'SB', 'APNEA', 'BASE_HR', 'BASE_PIP', 'BASE_TV', 'BASE_SBP', #'MIN_AWP', 'MAX_AWP', 
                                                    ])
            df_case['INIT_VENTON'] = init_venton
            df_case['EXTU'] = extubation
            df_case['MANUAL'] = manual_status
            df_case['VENT_STATUS'] = vent_status
            
            df_case.to_csv(f'{CSVDIR}/{filename}.csv', index=False)

            icase_result.append(icase)
            filename_result.append(f'{filename}.vital')
            
            # 사용할 case
            print(f'{len(case_s)} samples loaded -> total {len(s)} samples')
            icase += 1
            if icase >= ncase:
                break
        else:
            no_anyevents.append(filename)
            print('No label at all')
            
    # save casenumber and vital filename match df 
    result_tuples = list(zip(icase_result, filename_result))
    df_finder = pd.DataFrame(result_tuples, columns=['icase', 'filename'])
    df_finder.to_csv(f'{MAX_CASES}cases_primus.csv', index=False)
    
    # save final inclusions with exclusions
    criteria = ['eligibles', 'vitalfiles(No track)','no_anyevents', 'no_extu', 'no_fio2', 'no_lb', 'no_sb',  'no_seq', 'case_len', 'no_prime', 'inclusion', 'delay_fio2', 'extulb', ] #'no_ventoff', 'ventoffextu', 
    files =  [eligible_files, vital_files, no_anyevents, no_extu, no_fio2, no_lb,no_sb,  no_seq, case_len, no_prime, filename_result, delay_fio2,  extulb] # no_ventoff, ventoffextu, 
    number =  [len(x) for x in files]
    #print(files)
    
    result=pd.DataFrame(zip(criteria, number))
    stat = pd.DataFrame({'eligibles': pd.Series(eligible_files), 'vitalfiles(No track)': pd.Series(vital_files), 
                        'no_anyevents': pd.Series(no_anyevents), 'no_extu': pd.Series(no_extu), 'no_fio2': pd.Series(no_fio2), 'no_lb': pd.Series(no_lb),
                        'no_sb': pd.Series(no_sb),  'no_seq': pd.Series(no_seq), 'case_len': pd.Series(case_len),  #'no_ventoff': pd.Series(no_ventoff),
                        'no_prime': pd.Series(no_prime), 'inclusion': pd.Series(filename_result), 'delay_fio2': pd.Series(delay_fio2),  'extulb': pd.Series(extulb)}) #'ventoffextu': pd.Series(ventoffextu),
    result.to_csv('inclusion_result_primus.csv', header= ['criteria', 'casenum'])
    stat.to_csv('inclusion_files_primus.csv')
    
    # save cache file
    s = np.array(s)
    a = np.array(a)
    c = np.array(c)
    d = np.array(d)
    
    np.savez(cachepath, s=s, a=a, c=c, d=d)

# cachepath_re = f'{MAX_CASES}cases_re.npz'
if os.path.exists(cachepath_re):  ## reward function 변경 시 이 부분만 바꾸면 됨.
    print(f'"{cachepath_re}" with reward file already made')
else:
    dat = np.load(cachepath)
    s = dat['s']
    a = dat['a']
    c = dat['c']
    d = dat['d']
    
    print(s.shape)
    print(a.shape)
    
    print(s[:,PPF20_CE].size)
    print((np.where(s[:,PPF20_CE]==0)[0]).size)
    print((np.where(s[:,RFTN20_CE]==0)[0]).size)
    print((np.where(s[:,EXP_SEVO]==0)[0]).size)

    print('Make vent action', end='...', flush=True)
    merged = np.add(a[:,1].astype(int), a[:,3].astype(int)) #Merge vent on/off and manual on/off
    a[:, 1] = np.where(merged == 2, 1, merged)
    
    a = a[:,1:3] # Remove init_venton and vent_status 
    print('done')
    
    df = pd.DataFrame(a, columns=['venton', 'extu'])
    print(df.groupby(['venton']).value_counts())
    
    print('generating delta of HR, PIP, TV, SBP', end='...', flush=True)
    s[:, HR] = np.divide(s[:, HR], s[:, BASE_HR], out=np.ones_like(s[:, HR]), where=s[:, BASE_HR]!=0)
    s[:, SBP] = np.divide(s[:, SBP], s[:, BASE_SBP], out=np.ones_like(s[:, SBP]), where=s[:, BASE_SBP]!=0)
    s[:, TV] = np.divide(s[:, TV], s[:, BASE_TV], out=np.ones_like(s[:, TV]), where=s[:, BASE_TV]!=0)
    s[:, PIP] = np.divide(s[:, PIP], s[:, BASE_PIP], out=np.ones_like(s[:, PIP]), where=s[:, BASE_PIP]!=0)
    print('done')

    
    print(f'averaging co2 and AWP', end='...', flush=True)
    n = 6
    def rolling_sum(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]
    def rolling_max(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)
    def rolling_min(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.min(rolling,axis=axis)
    co2_sum = np.append([0] * (n-1), rolling_sum(s[:,CO2], n))
    co2_max = np.append([0] * (n-1), rolling_max(s[:,CO2], n))
    co2_min = np.append([0] * (n-1), rolling_min(s[:,CO2], n))
    awp_max = np.append([0] * (n-1), rolling_max(s[:,AWP], n))
    awp_min = np.append([0] * (n-1), rolling_min(s[:,AWP], n))
    awp_sum = np.append([0] * (n-1), rolling_sum(s[:,AWP], n))
    s[:,CO2] = co2_sum / n #* (~(a[:,1] == 0)) #(~(s[:,STATE] == 0))
    s[:,AWP] = awp_sum / n
    valid_mask = np.append([False] * n, ~rolling_sum(np.diff(c), n).astype(bool))
    s = s[valid_mask]
    a = a[valid_mask]
    c = c[valid_mask]

    print('done')

    
    print('Remaining apnea time during vent off', end='...', flush=True)
    #s[:,APNEA] = s[:,APNEA] * (a[:,1] == 0)
    s[:,APNEA] = s[:,APNEA] * (a[:,0] == 0)
    print('done')
    
    
    print('making label encoding and condition variable', end='...', flush=True)
    #unique_a = np.unique(a, axis=1)
    s_vent = np.zeros(len(a))
    s_extu = np.zeros(len(a))
    a_cat = np.zeros(len(a))
    for i in range(len(a)):
        if (a[i] == [0,0]).all():
            a_cat[i] = 0
            s_vent[i] = 0
            s_extu[i] = 0
        elif (a[i] == [0,1]).all():
            #a_cat[i] = 1
            a_cat[i] = 0
            s_vent[i] = 0
            s_extu[i] = 1
        elif (a[i] == [1,0]).all():
            #a_cat[i] = 2
            a_cat[i] = 1
            s_vent[i] = 1
            s_extu[i] = 0
        elif (a[i] == [1,1]).all():
            #a_cat[i] = 3
            a_cat[i] = 1
            s_vent[i] = 1
            s_extu[i] = 1
    a = a_cat
    s = np.hstack([s, np.reshape(s_vent, (-1, 1)), np.reshape(s_extu, (-1, 1))])
    #print(a.shape)
    #print(a)
    print('done')
    
    print('generating s_next and a_next', end='...', flush=True)
    s_next = np.vstack([s[1:, :], s[0, :]])
    #a_next = np.vstack([a[1:, :], a[0, :]])
    a_next = np.concatenate([a[1:], [a[0]]])
    valid_mask = np.append(np.diff(c) == 0, False)
    s = s[valid_mask]
    s_next = s_next[valid_mask]
    a = a[valid_mask]
    a_next = a_next[valid_mask]
    c = c[valid_mask]
    d = np.append(np.diff(c) != 0, True)
    
    print('done')
    
    # reward를 구함
    r = get_reward(s)
    r_next = get_reward(s_next)


# 최종적으로 로딩 된 caseid
caseids = np.unique(c)
for i in range(0, max(caseids)+1):
    if i not in caseids:
        print(f'case #{i} is missing')

np.savez(cachepath_re, 
            state=s, action=a, reward=r, next_reward = r_next, next_state = s_next, next_action = a_next,
            caseid=c, done=d)

print('{} cases {} samples'.format(len(caseids), len(a)))
print(f'Arrays are saved on "{MAX_CASES}cases_primus_re.npz" w/ finder "{MAX_CASES}cases_primus.csv"')