from ensurepip import bootstrap
import vitaldb
import pandas as pd
import numpy as np
import sys
import os
import gc
import random
import time
import csv
from joblib import Parallel, delayed
from collections import deque

import tensorflow as tf
from tensorflow import keras

#학습 동시 진행시에는 아래 실행.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

modelname = 'CQLSAC'

# ANEST_TYPE = 'primus'
MAX_CASES = 20000 #9304는 pip까지 뽑는 케이스
SEED = 42

SUGGEST = 1
EPSILON = 1e-8
GAMMA = 0.95

NMODEL = 300
MAX_ITER = 10

RESAMPLING = 100
RESAMPLING_RATE = 0.2
CL = [0.05, 0.10]

BATCH_SIZE = 1024
INF_BATCH_SIZE = 8192
FNN_NODES = 64
LEARNING_RATE = 1e-4
REGULER_LAMDA = 0.005

GREEDY = 0.9

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(SEED)

seedlist = random.sample(range(1,100000), NMODEL) # 1부터 100000까지의 범위중에 500개를 중복없이 뽑겠다

print(f'random seeds are {seedlist[:5]}...{seedlist[-5:]} total {len(seedlist)}')

df_finder = pd.read_csv(os.path.join('./', f'{MAX_CASES}'+'cases_primus.csv'))
print(df_finder.head())


odir = 'output_'+modelname

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

def fqe_calcuate(data_path, BATCH_SIZE):
    data = np.load(data_path)
    
    if 'valid' in data_path:
        task = 'valid'
    elif 'test' in data_path:
        task = 'test'
    elif 'datex' in data_path:
        task = 'datex'
    else:
        task = 'snubh'
    c_valid = data['caseid']
    valid_aopts = data['action_pred']
    valid_aopts_prob = data['action_prob']
    a_valid = data['action']
    s_valid_= data[f'state_']
    ns_valid_=data[f'nstate_']
    nr_valid_ = data[f'nreward_']

    print(f'{task} FQE calculating #{model_idx}...')

    valid_caseids = np.unique(c_valid)
    
    # from keras.models import Model
    # from keras import Model
    # from keras.layers import Dense, Dropout, Input, concatenate
    # import tensorflow as tf
    tf.keras.backend.clear_session()
    
    valid_aopts_prob = np.take(valid_aopts_prob, valid_aopts)

    valid_s = []
    valid_a = []
    valid_na = []
    valid_aopts_ = []
    valid_naopts_= []
    valid_probs_ = []
    valid_c = []
    for case in valid_caseids:
        case_mask = (c_valid == case)
        #case_len = np.sum(case_mask)
        
        case_aopts = valid_aopts[case_mask]
        case_probs = valid_aopts_prob[case_mask]
        case_a = a_valid[case_mask]
        valid_s.append(s_valid_[case_mask])
        valid_a.append(case_a)
        valid_c.append(c_valid[case_mask])
        
        case_na = np.concatenate([case_a[1:],[0]])
        case_naopts = np.concatenate([case_aopts[1:],[0]])
        valid_na.append(case_na)
        valid_aopts_.append(case_aopts)
        valid_naopts_.append(case_naopts)
        valid_probs_.append(case_probs)

    valid_s = np.vstack(valid_s).astype(np.float32)
    valid_a = np.concatenate(valid_a).astype(np.int)
    valid_na = np.concatenate(valid_na).astype(np.int)
    valid_c = np.concatenate(valid_c).astype(np.int)
    valid_aopts_ = np.concatenate(valid_aopts_).astype(np.int)
    valid_naopts_ = np.concatenate(valid_naopts_)
    valid_probs_ = np.concatenate(valid_probs_)
    
    # 모델 설계
    def make_model():
        input_s = tf.keras.Input(batch_shape=(None, s_valid_.shape[-1]))
        input_a = tf.keras.Input(batch_shape=(None, 1), name='action')

        out = input_s
        out = tf.concat((out, input_a), axis=-1)
        out = tf.keras.layers.Dense(FNN_NODES, activation='elu', kernel_regularizer='l2')(out)
        out = tf.keras.layers.Dense(FNN_NODES, activation='elu', kernel_regularizer='l2')(out)

        out = tf.keras.layers.Dense(1, activation='softsign', name='Q')(out)
        
        model = tf.keras.Model(inputs=[input_s, input_a], outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse')
        return model 
    
    vemodel = make_model()
    #print(vemodel)
    
    for k in range(MAX_ITER):
        # 모든 action 을 넣어보고 q-value를 구함
        #print(valid_aopts_prob.shape)
        print(f'iteration for EXTUAI policy #{k+1}')
        qs_real = vemodel.predict([ns_valid_, valid_naopts_], batch_size=BATCH_SIZE, workers=4, use_multiprocessing=True).flatten()
        qs_dont = vemodel.predict([ns_valid_, (1 - valid_naopts_)], batch_size=BATCH_SIZE, workers=4, use_multiprocessing=True).flatten()
        qs_real *= valid_probs_
        qs_dont *= (1- valid_probs_)
        #print(qs_real.shape)
        qs = np.add(qs_real, qs_dont)
        #print(qs.shape)
        target_y = nr_valid_ + qs
        
        vemodel.fit([s_valid_, a_valid], target_y,
                    batch_size=BATCH_SIZE, 
                    epochs=1,
                    )
    vemodel.save(f"vemodel_{task}")
    del vemodel
    tf.keras.backend.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()
    
    if model_idx == 1:
        vcmodel = make_model()
        #print(vcmodel)
        for k in range(MAX_ITER):
            # 모든 action 을 넣어보고 q-value를 구함
            print(f"iteration for Clinicians' policy #{k+1}")
            qs_real = vcmodel.predict([ns_valid_, valid_na], batch_size=BATCH_SIZE, workers=4, use_multiprocessing=True).flatten()
            qs_dont = vcmodel.predict([ns_valid_, (1-valid_na)], batch_size=BATCH_SIZE, workers=4, use_multiprocessing=True).flatten()
            qs_real *= (np.ones(len(valid_na)) * GREEDY)
            qs_dont *= (np.ones(len(valid_na)) * (1-GREEDY))
            qs = np.add(qs_real, qs_dont)

            target_y = nr_valid_ + qs
            vcmodel.fit([s_valid_, a_valid], target_y,
                        batch_size=BATCH_SIZE, 
                        epochs=1,
                        )    
        vcmodel.save(f"vcmodel_{task}")
        del vcmodel
        tf.keras.backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()
    
    num_sample = 40
    if model_idx==1:
        print(f'{num_sample} episodes from {len(valid_caseids)} {task} episodes (cases) are resampled')
    
    
    label_targets = ['ve_error', 'vc_error']
  
    
    global savelist
    savelist = set()
    
    gc.collect()
    
    sampled_s_wor = []
    sampled_aopts_wor = []
    sampled_a_wor = []
    sampled_ap_wor = []
    sampled_s_wr = []
    sampled_aopts_wr = []
    sampled_a_wr = []
    sampled_ap_wr = []
    
     
    for i in range(RESAMPLING):
        candidate = np.random.permutation(len(valid_caseids))[:num_sample] #withoutreplacement
        #candidate = np.random.permutation(len(valid_a))[:num_sample] #withoutreplacement
        case_masks = np.where(np.isin(valid_c,candidate), 1, 0)

        sampled_s_wor.append(valid_s[case_masks])
        sampled_aopts_wor.append(valid_aopts_[case_masks])
        sampled_a_wor.append(valid_a[case_masks])
        sampled_ap_wor.append(valid_aopts_prob[case_masks])
        
        candidate = np.random.choice(np.arange(len(valid_caseids)), num_sample) #withreplacement
        
        case_masks = np.where(np.isin(valid_c,candidate), 1, 0)

        sampled_s_wr.append(valid_s[case_masks])
        sampled_aopts_wr.append(valid_aopts_[case_masks])
        sampled_a_wr.append(valid_a[case_masks])
        sampled_ap_wr.append(valid_aopts_prob[case_masks])
        
        if i % 99 == 0:
           print(f'{i}th bootstrapped', end=' ', flush=True)
        
    print('done')
    
    sampled_s_wor = np.vstack(sampled_s_wor)
    sampled_aopts_wor = np.concatenate(sampled_aopts_wor)
    sampled_a_wor = np.concatenate(sampled_a_wor)
    sampled_ap_wor = np.concatenate(sampled_ap_wor)
    sampled_s_wr = np.vstack(sampled_s_wr)
    sampled_aopts_wr = np.concatenate(sampled_aopts_wr)
    sampled_a_wr = np.concatenate(sampled_a_wr)
    sampled_ap_wr = np.concatenate(sampled_ap_wr)
    
    #print(sampled_s_wor.shape)
    
    # ExtuAI policy
    vemodel = tf.keras.models.load_model(f"vemodel_{task}")
    ve_wr_real = vemodel.predict([sampled_s_wr, sampled_aopts_wr], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=1).flatten()
    ve_wr_dont = vemodel.predict([sampled_s_wr, (1-sampled_aopts_wr)], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=1).flatten()
    ve_wr_real *= sampled_ap_wr
    ve_wr_dont *= (1- sampled_ap_wr)
    ve_wr = np.add(ve_wr_real, ve_wr_dont)
    
    ve_wor_real = vemodel.predict([sampled_s_wor, sampled_aopts_wor], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=1).flatten()
    ve_wor_dont = vemodel.predict([sampled_s_wor, (1-sampled_aopts_wor)], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=0).flatten()
    ve_wor_real *= sampled_ap_wor
    ve_wor_dont *= (1- sampled_ap_wor)
    ve_wor = np.add(ve_wor_real, ve_wor_dont)
    ve_err_dist = np.mean(np.split(ve_wr, RESAMPLING), axis=1) - np.mean(np.split(ve_wor, RESAMPLING), axis=1)
    
    # Clinician policy
    vcmodel = tf.keras.models.load_model(f"vcmodel_{task}")
    vc_wr_real = vcmodel.predict([sampled_s_wr, sampled_a_wr], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=0).flatten()
    vc_wr_dont = vcmodel.predict([sampled_s_wr, (1-sampled_a_wr)], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=0).flatten()
    vc_wr_real *= (np.ones(len(sampled_a_wr)) * GREEDY)
    vc_wr_dont *= (np.ones(len(sampled_a_wr)) * (1-GREEDY))
    vc_wr = np.add(vc_wr_real, vc_wr_dont)
    
    vc_wor_real = vcmodel.predict([sampled_s_wor, sampled_a_wor], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=0).flatten()
    vc_wor_dont = vcmodel.predict([sampled_s_wor, (1-sampled_a_wor)], batch_size=INF_BATCH_SIZE, workers=4, use_multiprocessing=True, verbose=0).flatten()
    vc_wor_real *= (np.ones(len(sampled_a_wor)) * GREEDY)
    vc_wor_dont *= (np.ones(len(sampled_a_wor)) * (1-GREEDY))
    vc_wor = np.add(vc_wor_real, vc_wor_dont)
    vc_err_dist =  np.mean(np.split(vc_wr, RESAMPLING), axis=1) - np.mean(np.split(vc_wor, RESAMPLING), axis=1)
    

    ve_errors = []
    vc_errors = []
    for cl in CL:
        ve_err_q1, ve_err_q2 = np.quantile(ve_err_dist, cl/2), np.quantile(ve_err_dist, (1-cl/2))
        vc_err_q1, vc_err_q2 = np.quantile(vc_err_dist, cl/2), np.quantile(vc_err_dist, (1-cl/2))
        ve_errors.append([ve_err_q1, ve_err_q2])
        vc_errors.append([vc_err_q1, vc_err_q2])
        
    vemodel = tf.keras.models.load_model(f"vemodel_{task}")
    ve = np.mean(vemodel.predict([s_valid_, valid_aopts_], batch_size=BATCH_SIZE, workers=4, use_multiprocessing=True))
    del vemodel
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    vcmodel = tf.keras.models.load_model(f"vcmodel_{task}")
    vc = np.mean(vcmodel.predict([s_valid_, a_valid], batch_size=BATCH_SIZE, workers=4, use_multiprocessing=True))
    del vcmodel
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    ve_cls = []
    vc_cls = []
    for (ve_err_q1, ve_err_q2), (vc_err_q1, vc_err_q2), cl in zip(ve_errors, vc_errors, CL):    
        ve_lower, ve_upper = ve - ve_err_q2, ve - ve_err_q1
        vc_lower, vc_upper = vc - vc_err_q2, vc - vc_err_q1
        ve_cls.append([ve, ve_lower, ve_upper])
        vc_cls.append([vc, vc_lower, vc_upper])
        
        print(f'{task} EXTUAI ({100-cl*100}%CI) : {ve} [{ve_lower}-{ve_upper}]')
        print(f'{task} CLIN ({100-cl*100}%CI) : {vc} [{vc_lower}-{vc_upper}]')
        
    return task, ve_cls, vc_cls
    #return (task), (ve, ve_lower, ve_upper), (vc, vc_lower, vc_upper)

model_idx = 0
starttime = time.process_time()

best_valid_extu_cl1=-1e+9
best_valid_clin_cl1=-1e+9
best_test_extu_cl1=-1e+9
best_test_clin_cl1=-1e+9
best_snubh_extu_cl1=-1e+9
best_snubh_clin_cl1=-1e+9
best_valid_extu_cl2=-1e+9
best_valid_clin_cl2=-1e+9
best_test_extu_cl2=-1e+9
best_test_clin_cl2=-1e+9
best_snubh_extu_cl2=-1e+9
best_snubh_clin_cl2=-1e+9
best_datex_extu_cl1=-1e+9
best_datex_clin_cl1=-1e+9
best_datex_extu_cl2=-1e+9
best_datex_clin_cl2=-1e+9

VALID_EXTU_CL1 = []
VALID_CLIN_CL1 = []
TEST_EXTU_CL1 = []
TEST_CLIN_CL1 = []
SNUBH_EXTU_CL1 = []
SNUBH_CLIN_CL1 = []
VALID_EXTU_CL2 = []
VALID_CLIN_CL2 = []
TEST_EXTU_CL2 = []
TEST_CLIN_CL2 = []
SNUBH_EXTU_CL2 = []
SNUBH_CLIN_CL2 = []

DATEX_EXTU_CL1 = []
DATEX_CLIN_CL1 = []
DATEX_EXTU_CL2 = []
DATEX_CLIN_CL2 = []


data_path = f'./fqe/FQE_{NMODEL}.npz'


if os.path.exists(data_path):
    data = np.load(data_path)
    seedlist = data['modellist']
    VALID_EXTU_CL1 = data['valid_extu_cl1']
    VALID_CLIN_CL1 = data['valid_clin_cl1']
    TEST_EXTU_CL1 = data['test_extu_cl1'] 
    TEST_CLIN_CL1 = data['test_clin_cl1']
    SNUBH_EXTU_CL1 = data['snubh_extu_cl1'] 
    SNUBH_CLIN_CL1 = data['snubh_clin_cl1']
    VALID_EXTU_CL2 = data['valid_extu_cl2']
    VALID_CLIN_CL2 = data['valid_clin_cl2']
    TEST_EXTU_CL2 = data['test_extu_cl2'] 
    TEST_CLIN_CL2 = data['test_clin_cl2']
    SNUBH_EXTU_CL2 = data['snubh_extu_cl2'] 
    SNUBH_CLIN_CL2 = data['snubh_clin_cl2']
    
else:
    #print(seedlist)
    for se in seedlist:
    #for se in [87842]:
        model_idx+=1
        print(f'model_#{model_idx}')
        validdata_path = './npvalid/valid_'+modelname+f'{se}'+'.npz'
        testdata_path = './nptest/test_'+modelname+f'{se}'+'.npz'
        snubhdata_path = './npsnubh/snubh_'+modelname+f'{se}'+'.npz'
        
        valid, ve_cls, vc_cls = fqe_calcuate(validdata_path, BATCH_SIZE)
        test, te_cls, tc_cls = fqe_calcuate(testdata_path, BATCH_SIZE)
        snubh, be_cls, bc_cls = fqe_calcuate(snubhdata_path, BATCH_SIZE)
    
        VALID_EXTU_CL1.append(ve_cls[0])
        VALID_CLIN_CL1.append(vc_cls[0])
        TEST_EXTU_CL1.append(te_cls[0])
        TEST_CLIN_CL1.append(tc_cls[0])
        SNUBH_EXTU_CL1.append(be_cls[0])
        SNUBH_CLIN_CL1.append(bc_cls[0])
        VALID_EXTU_CL2.append(ve_cls[1])
        VALID_CLIN_CL2.append(vc_cls[1])
        TEST_EXTU_CL2.append(te_cls[1])
        TEST_CLIN_CL2.append(tc_cls[1])
        SNUBH_EXTU_CL2.append(be_cls[1])
        SNUBH_CLIN_CL2.append(bc_cls[1])


        if best_valid_extu_cl1 < ve_cls[0][1]:
            best_valid_extu_cl1 = ve_cls[0][1]
        
        if best_test_extu_cl1 < te_cls[0][1]:
            best_test_extu_cl1 = te_cls[0][1]
            
        if best_snubh_extu_cl1 < be_cls[0][1]:
            best_snubh_extu_cl1 = be_cls[0][1]
            
            
        if best_test_clin_cl1 < tc_cls[0][2]:
            best_test_clin_cl1 = tc_cls[0][2]
            
        if best_snubh_clin_cl1 < bc_cls[0][2]:
            best_snubh_clin_cl1 = bc_cls[0][2]

        ####
        
        if best_valid_extu_cl2 < ve_cls[1][1]:
            best_valid_extu_cl2 = ve_cls[1][1]
        
        if best_test_extu_cl2 < te_cls[1][1]:
            best_test_extu_cl2 = te_cls[1][1]
            
        if best_snubh_extu_cl2 < be_cls[1][1]:
            best_snubh_extu_cl2 = be_cls[1][1]
        
        #if best_datex_extu_cl2 < de_cls[1][1]:
        #    best_datex_extu_cl2 = de_cls[1][1]
        
        if best_test_clin_cl2 < tc_cls[1][2]:
            best_test_clin_cl2 = tc_cls[1][2]
            
        if best_snubh_clin_cl2 < bc_cls[1][2]:
            best_snubh_clin_cl2 = bc_cls[1][2]
        
        #if best_datex_clin_cl2 < dc_cls[1][2]:
        #    best_datex_clin_cl2 = dc_cls[1][2]
        
            
        print('')
        print(f'best VALID EXTU ({100-CL[0]*100}% LB) : {best_valid_extu_cl1:.7f} | best TEST EXTU ({100-CL[0]*100}% LB) : {best_test_extu_cl1:.7f} | best TEST CLIN ({100-CL[0]*100}% UB) : {best_test_clin_cl1:.7f} until model #{model_idx}')
        print(f'best SNUBH EXTU ({100-CL[0]*100}% LB) : {best_snubh_extu_cl1:.7f} | best SNUBH CLIN ({100-CL[0]*100}% UB) : {best_snubh_clin_cl1:.7f} until model #{model_idx}')
        
        print(f'best VALID EXTU ({100-CL[1]*100}% LB) : {best_valid_extu_cl2:.7f} | best TEST EXTU ({100-CL[1]*100}% LB) : {best_test_extu_cl2:.7f} | best TEST CLIN ({100-CL[1]*100}% UB) : {best_test_clin_cl2:.7f} until model #{model_idx} ')
        print(f'best SNUBH EXTU ({100-CL[1]*100}% LB) : {best_snubh_extu_cl2:.7f} | best SNUBH CLIN ({100-CL[1]*100}% UB) : {best_snubh_clin_cl2:.7f} until model #{model_idx}')
         
     
    np.savez(f'./fqe/FQE_{NMODEL}.npz', 
            modellist=seedlist,
            valid_extu_cl1=VALID_EXTU_CL1,
            valid_clin_cl1=VALID_CLIN_CL1,
            test_extu_cl1=TEST_EXTU_CL1,
            test_clin_cl1=TEST_CLIN_CL1,
            snubh_extu_cl1=SNUBH_EXTU_CL1,
            snubh_clin_cl1=SNUBH_CLIN_CL1,
           
            ###
            valid_extu_cl2=VALID_EXTU_CL2,
            valid_clin_cl2=VALID_CLIN_CL2,
            test_extu_cl2=TEST_EXTU_CL2,
            test_clin_cl2=TEST_CLIN_CL2,
            snubh_extu_cl2=SNUBH_EXTU_CL2,
            snubh_clin_cl2=SNUBH_CLIN_CL2,
            )
    print('done!!')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Performance return    
for cl in CL:
    
    if cl ==CL[0]:
        VALID_EXTU, TEST_EXTU, TEST_CLIN, SNUBH_EXTU, SNUBH_CLIN, = VALID_EXTU_CL1, TEST_EXTU_CL1, TEST_CLIN_CL1, SNUBH_EXTU_CL1, SNUBH_CLIN_CL1
    else:
        VALID_EXTU, TEST_EXTU, TEST_CLIN, SNUBH_EXTU, SNUBH_CLIN, = VALID_EXTU_CL1, TEST_EXTU_CL1, TEST_CLIN_CL1, SNUBH_EXTU_CL1, SNUBH_CLIN_CL1
        
    valid_ai = []
    test_ai = []
    test_clin = []
    snubh_ai = []
    snubh_clin = []
    #datex_ai = []
    #datex_clin = []
    
    ME = 0
    LB = 1
    UB = 2
    num=0
    for se, ve, te, tc, be, bc in zip(seedlist, VALID_EXTU, TEST_EXTU, TEST_CLIN, SNUBH_EXTU, SNUBH_CLIN):
        vl= ve[LB]
        tl= te[LB]
        tu= tc[UB]
        bl= be[LB]
        bu= bc[UB]
      
        if num == 0:
            valid_ai.append(vl)
            test_ai.append(tl)
            test_clin.append(tu)
            snubh_ai.append(bl)
            snubh_clin.append(bu)

        else:
            if vl > valid_ai[-1]:
                valid_ai.append(vl)
                test_ai.append(tl)
                snubh_ai.append(tu)
  
            else:
                valid_ai.append(valid_ai[-1])
                test_ai.append(test_ai[-1])
                snubh_ai.append(snubh_ai[-1])
            
            if tu > test_clin[-1]:
                test_clin.append(tu)
            else:
                test_clin.append(test_clin[-1])
            if bu > snubh_clin[-1]:
                snubh_clin.append(bu)
            else:
                snubh_clin.append(snubh_clin[-1])

        num+=1

    plot_df = pd.DataFrame(zip(seedlist, valid_ai, test_ai, test_clin, snubh_ai, snubh_clin), columns=['Modelseed', 'Valid_AI', 'Test_AI', 'Test_Clin', 'SNUBH_AI', 'SNUBH_CLIN'])
    
    plot_df.to_csv(f'estimated_performance(FQE)_CL{cl}.csv')

    t = np.arange(0, len(seedlist))
    plt.figure(figsize=(10, 10))
    plt.plot(t, valid_ai, label=f'{100-cl*100}% LB for best ExtuAI policy (SNUH valid set)', color='red')
    plt.plot(t, test_ai, label=f'{100-cl*100}% LB for best ExtuAI policy (SNUH testset)', color='orange')
    plt.plot(t, test_clin, label=f"{100-cl*100}% UB for Clinicans' policy (SNUH testset)", color='green')
    plt.legend(loc="lower right")

    plt.title('Estimated performance return (FQE)')
    plt.xlabel('Number of models')
    plt.ylabel('Estimated policy value')
    plt.tight_layout()
    plt.savefig(f'./plot/estimated_performance(FQE)_CL{cl}_snuh.tiff')
    plt.close()
    
    
    t = np.arange(0, len(seedlist))
    plt.plot(t, snubh_ai, label=f'{100-cl*100}% LB for best ExtuAI policy (SNUBH set)', color='pink')
    plt.plot(t, snubh_clin, label=f"{100-cl*100}% UB for Clinicans' policy (SNUBH set)", color='blue')
    plt.legend(loc="lower right")

    plt.title('Estimated performance return (FQE)')
    plt.xlabel('Number of models')
    plt.ylabel('Estimated policy value')
    plt.tight_layout()
    plt.savefig(f'./plot/estimated_performance(FQE)_CL{cl}_snubh.tiff')
    plt.close()
    
    
    valid_ai = []
    test_ai = []
    test_clin = []
    snubh_ai = []
    snubh_clin = []
    datex_ai = []
    datex_clin = []
    
    ME = 0
    LB = 1
    UB = 2
    num=0
    for se, ve, te, tc, be, bc in zip(seedlist, VALID_EXTU, TEST_EXTU, TEST_CLIN, SNUBH_EXTU, SNUBH_CLIN):
        vl= ve[LB]
        tl= te[LB]
        tu= tc[UB]
        bl= be[LB]
        bu= bc[UB]
        #dl= de[LB]
        #du= dc[UB]
        if num == 0:
            valid_ai.append(vl)
            test_ai.append(tl)
            test_clin.append(tu)
            snubh_ai.append(bl)
            snubh_clin.append(bu)
         
        else:
            if vl > valid_ai[-1]:
                valid_ai.append(vl)    
            else:
                valid_ai.append(valid_ai[-1])

            if tl > test_ai[-1]:   
                test_ai.append(tl)
            else:
                test_ai.append(test_ai[-1])

            if bl > snubh_ai[-1]:
                snubh_ai.append(bl)
            else:
                snubh_ai.append(snubh_ai[-1])
           
            
            if tu > test_clin[-1]:
                test_clin.append(tu)
            else:
                test_clin.append(test_clin[-1])
                
            if bu > snubh_clin[-1]:
                snubh_clin.append(bu)
            else:
                snubh_clin.append(snubh_clin[-1])
                
            
        num+=1

    plot_df = pd.DataFrame(zip(seedlist, valid_ai, test_ai, test_clin, snubh_ai, snubh_clin), columns=['Modelseed', 'Valid_AI', 'Test_AI', 'Test_Clin', 'SNUBH_AI', 'SNUBH_CLIN'])
    
    plt.figure(figsize=(10,10))
    t = np.arange(0, len(seedlist))
    plt.plot(t, valid_ai, label=f"{100-cl*100}% LB for AIVE's policy (Validation set)", color='orange')
    plt.plot(t, test_ai, label=f"{100-cl*100}% LB for AIVE's policy (Internal test set)", color='red')
    plt.plot(t, snubh_ai, label=f"{100-cl*100}% LB for AIVE's policy (External test set)", color='pink', linestyle='dashed')
    plt.plot(t, test_clin, label=f"{100-cl*100}% UB for Clinicans' policy (Internal test set)", color='blue')
    plt.plot(t, snubh_clin, label=f"{100-cl*100}% UB for Clinicans' policy (External test set)", color='skyblue', linestyle='dashed')
    plt.legend(loc="lower right")

    #plt.title('Estimated performance return')
    #plt.xlim([0, len(seedlist)])
    #plt.xscale('log')
    plt.ylim([-0.6, 0.6])
    #plt.xlim([1,300])
    #plt.axis('scaled')
    plt.xscale("log")
    plt.xlabel('Number of models bootstrapped')
    plt.ylabel('Estimated policy value')
    plt.tight_layout()
    plt.savefig(f'./plot/best_estimated_performance(FQE)_CL{cl}_total.tiff')
    # plt.savefig(f'{odir}/{case_len}.png'
    plt.close()
    