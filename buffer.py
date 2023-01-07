import numpy as np
import pandas as pd
import random
import torch

import os
import time

from collections import deque, namedtuple
from sklearn.utils import shuffle


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

    
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device):

        self.device = device
        self.mode = None
        self.trainmemory = deque(maxlen=buffer_size)  
        self.validmemory  = deque(maxlen=buffer_size) 
        self.testmemory = deque(maxlen=buffer_size)
        self.snubhmemory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        #self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def load(self, source, seed):
        """Add a new experience to memory."""
        
        MAX_CASES = 20000 #9304는 pip까지 뽑는 케이스
        DEBUG = False
        DEBUGNUM=200000


        TESTSIZE = 0.15
        VALIDSIZE = 0.17651 # 전체 데이터셋에 대해서는 20%

        SEED = 42


        EPSILON = 1e-8

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

        print('loading data and concatenating', end='...', flush=True)
        # cachepath_re = f'{MAX_CASES}cases_{ANEST_TYPE}_re.npz'

        cachepath_re = f'{MAX_CASES}cases_primus_re.npz'
        data1 = np.load(os.path.join('../../',cachepath_re))
        state = data1['state']
        actions = data1['action']
        rewards = data1['reward']  #r_next
        next_rewards = data1['next_reward']
        next_state = data1['next_state']
        caseid = data1['caseid']
        terminals = data1['done']
        
        cachepath_re2 = f'{MAX_CASES}cases_datex_re.npz'
        data2 = np.load(os.path.join('../../',cachepath_re2))
        state2 = data2['state']
        actions2 = data2['action']
        rewards2 = data2['reward']  #r_next
        next_rewards2 = data2['next_reward']
        next_state2 = data2['next_state']
        caseid2 = data2['caseid']
        terminals2 = data2['done']
        

        cachepath_re4= f'{MAX_CASES}cases_snubhprimus_re.npz'
        data4 = np.load(os.path.join('../../',cachepath_re4))
        state4 = data4['state']
        actions4 = data4['action']
        rewards4 = data4['reward']  #r_next
        next_rewards4 = data4['next_reward']
        next_state4 = data4['next_state']
        caseid4 = data4['caseid']
        terminals4 = data4['done']


        EXCLUDE = [EXP_DES, COMPLIANCE, PPLAT, SET_PEEP, SET_TV, FIO2, BIS, EMG, SEF, BT, DBP, MBP, BASE_HR, BASE_PIP , BASE_TV, BASE_SBP]
        
        
        #SNUH-Primus
        #states = np.concatenate([state1, state2])
        states = np.delete(state, EXCLUDE, axis=1)
        # states = np.hstack([states[:,:3], states[:,4:10], states[:,-1].reshape(-1, 1)])  #baseline값이 들어있는 column은 제외하기 위함.

        #next_states = np.concatenate([next_state1, next_state2])
        next_states  = np.delete(next_state, EXCLUDE, axis=1)
        # next_states = np.hstack([next_states[:,:3], next_states[:,4:10], next_states[:,-1].reshape(-1, 1)])  #baseline값이 들어있는 column은 제외하기 위함.

        #SNUH-Datex
        states2 = np.delete(state2, EXCLUDE, axis=1)
        next_states2  = np.delete(next_state2, EXCLUDE, axis=1)
        
        #SNUBH
        states4 = np.delete(state4, EXCLUDE, axis=1)
        next_states4  = np.delete(next_state4, EXCLUDE, axis=1)

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


        print(f'Standard scaling w/o SB, STATE', end='...', flush=True) # APNEA, 
        #Standard scaling
        s_mean, s_std = np.mean(states, axis=0), np.std(states, axis=0)
        r_mean, r_std = np.mean(rewards), np.std(rewards)

        #SNUH-Primus
        states_ = (states - s_mean) / (s_std + EPSILON)
        next_states_ = (next_states - s_mean) / (s_std + EPSILON)

        #SNUH-Datex
        states2_ = (states2 - s_mean) / (s_std + EPSILON)
        next_states2_ = (next_states2 - s_mean) / (s_std + EPSILON)
        
        #SNUBH
        states4_ = (states4 - s_mean) / (s_std + EPSILON)
        next_states4_ = (next_states4 - s_mean) / (s_std + EPSILON)

        print(f' mean{np.mean(rewards)}, min{np.min(rewards)}, max{np.max(rewards)}, median{np.median(rewards)}, 1/4Q{np.quantile(rewards, 0.25)}, 3/4Q{np.quantile(rewards, 0.75)}')

        #SNUH-Primus
        rewards = np.clip(rewards, -20,  0) # -EPSILON
        next_rewards = np.clip(next_rewards, -20,  0) # -EPSILON

        rewards_ = (rewards - r_mean) / (r_std + EPSILON)
        next_rewards_ = (next_rewards - r_mean) / (r_std + EPSILON)
        
        #SNUH-Datex
        rewards2 = np.clip(rewards2, -20,  0) # -EPSILON
        next_rewards2 = np.clip(next_rewards2, -20,  0) # -EPSILON

        rewards2_ = (rewards2 - r_mean) / (r_std + EPSILON)
        next_rewards2_ = (next_rewards2 - r_mean) / (r_std + EPSILON)

        #SNUBH
        rewards4 = np.clip(rewards4, -20, 0)
        next_rewards4 = np.clip(next_rewards4, -20, 0)

        rewards4_ = (rewards4 - r_mean) / (r_std + EPSILON)
        next_rewards4_ = (next_rewards4 - r_mean) / (r_std + EPSILON)

        
        states_[:, SB], next_states_[:, SB] = states[:, SB], next_states[:, SB]
        states_[:, VENT_STATE], next_states_[:, VENT_STATE] = states[:, VENT_STATE], next_states[:, VENT_STATE]
        states_[:, EXTU_STATE], next_states_[:, EXTU_STATE] = states[:, EXTU_STATE], next_states[:, EXTU_STATE]


        states2_[:, SB], next_states2_[:, SB] = states2[:, SB], next_states2[:, SB]
        states2_[:, VENT_STATE], next_states2_[:, VENT_STATE] = states2[:, VENT_STATE], next_states2[:, VENT_STATE]
        states2_[:, EXTU_STATE], next_states2_[:, EXTU_STATE] = states2[:, EXTU_STATE], next_states2[:, EXTU_STATE]
        
        
        states4_[:, SB], next_states4_[:, SB]  = states4[:, SB], next_states4[:, SB]
        states4_[:, VENT_STATE], next_states4_[:, VENT_STATE] = states4[:, VENT_STATE], next_states4[:, VENT_STATE]
        states4_[:, EXTU_STATE], next_states4_[:, EXTU_STATE] = states4[:, EXTU_STATE], next_states4[:, EXTU_STATE]

        print('...done')
        '''
        print(f'log scaling on APNEA', end='...', flush=True)
        states_[:, APNEA] = np.log(states[:, APNEA], where=(states[:, APNEA]!=0))
        print('...done')
        '''

        if DEBUG:
            states = states[:DEBUGNUM, ]
            states_ = states_[:DEBUGNUM,:]
            next_states = next_states[:DEBUGNUM]
            next_states_ = next_states_[:DEBUGNUM, :]
            actions = actions[:DEBUGNUM]
            rewards_ = rewards_[:DEBUGNUM]
            rewards = rewards[:DEBUGNUM]
            next_rewards = next_rewards[:DEBUGNUM]
            next_rewards_ = next_rewards_[:DEBUGNUM]
            caseid = caseid[:DEBUGNUM]
            terminals = terminals[:DEBUGNUM]
            MAX_ITER = 1
            NMODEL = 1
            

        caseids = np.unique(caseid)
            
        print(f'{len(caseids)} cases are loaded')

        '''For one hot encoding
        num = np.unique(actions, axis=0)
        num = num.shape[0]
        actions_hot = np.eye(num)[actions]
        '''
        print(f'total state space: {states_.shape}')
        print(f'total action space: {actions.shape} * {len(np.unique(actions))}')
        print(f'total reward space: {rewards_.shape}')
        # print(terminals.shape)
        #print(rewards4.shape)
        caseids = shuffle(caseids, random_state=SEED)

        n_test = round((len(caseids) * TESTSIZE))
        n_train = len(caseids) - n_test 

        trainvalidcase = caseids[:n_train] 
        testcase = caseids[n_train:]

        if source == 'snuh':
            trainvalidcase = shuffle(trainvalidcase, random_state=seed)

            n_valid = round(len(trainvalidcase) * VALIDSIZE)
            n_train = len(trainvalidcase) - n_valid
            traincase = trainvalidcase[:n_train]
            validcase = trainvalidcase[n_train:]

            print(f'total; #{len(caseids)}, train #{len(traincase)},  valid #{len(validcase)},  test #{len(testcase)}')
            
            train_mask = np.isin(caseid, traincase)
            s_train = states_[train_mask]
            ns_train = next_states_[train_mask]
            a_train = actions[train_mask]
            r_train = rewards_[train_mask]
            nr_train = next_rewards_[train_mask]
            c_train = caseid[train_mask]
            d_train = terminals[train_mask]
    
            self.trainmemory = (s_train, a_train[..., None], nr_train[..., None], ns_train, d_train[..., None], c_train[..., None])
            
            #vatecase = np.concatenate([validcase, testcase])
    
            valid_mask = np.isin(caseid, validcase)
            s_valid = states[valid_mask]
            ns_valid = next_states[valid_mask]
            a_valid = actions[valid_mask]
            r_valid = rewards[valid_mask]
            nr_valid = next_rewards[valid_mask]
            c_valid = caseid[valid_mask]
            d_valid = terminals[valid_mask]
            
            s_valid_ = states_[valid_mask]
            ns_valid_= next_states_[valid_mask]
            nr_valid_ = next_rewards_[valid_mask]
            
            self.validmemory = (s_valid_, a_valid[..., None], nr_valid[..., None], ns_valid_, d_valid[..., None], c_valid[..., None])
            self.validoriginal = (s_valid, a_valid, nr_valid, ns_valid, d_valid, c_valid)
            
            #Test case
            test_mask = np.isin(caseid, testcase)
            s_test = states[test_mask]
            ns_test = next_states[test_mask]
            a_test = actions[test_mask]
            r_test = rewards[test_mask]
            nr_test = next_rewards[test_mask]
            c_test = caseid[test_mask]
            d_test = terminals[test_mask]
            
            s_test_ = states_[test_mask]
            ns_test_= next_states_[test_mask]
            nr_test_ = next_rewards_[test_mask]
            
            self.testmemory = (s_test_, a_test[..., None], nr_test[..., None], ns_test_, d_test[..., None], c_test[..., None])
            self.testoriginal = (s_test, a_test, nr_test, ns_test, d_test, c_test)
            
        if source == 'snubh':
            self.snubhmemory = (states4_, actions4[..., None], next_rewards4_[..., None], next_states4_, terminals4[..., None], caseid4[..., None])
            self.snubhoriginal = (states4, actions4, next_rewards4, next_states4, terminals4, caseid4)
            print(f'snubh state space: {states4_.shape}')
            print(f'snubh action space: {actions4.shape} * {len(np.unique(actions))}')
            print(f'snubh reward space: {rewards4_.shape}')
        
        if source == 'datex':
            self.datexmemory = (states2_, actions2[..., None], next_rewards2_[..., None], next_states2_, terminals2[..., None], caseid2[..., None])
            self.datexoriginal = (states2, actions2, next_rewards2, next_states2, terminals2, caseid2)


    def get_data(self, mode='valid', original=True):
        
        if mode=='train':
            if original:
                s, a, nr, ns, d, c = self.trainoriginal
                return s, a, nr, ns, d, c 
            else:
                s, a, nr, ns, d, c  = self.trainmemory
                return s, a.squeeze(), nr.squeeze(), ns, d.squeeze(), c.squeeze()
        
        
        if mode=='valid':
            if original:
                s, a, nr, ns, d, c = self.validoriginal
                return s, a, nr, ns, d, c 
            else:
                s, a, nr, ns, d, c  = self.validmemory
                return s, a.squeeze(), nr.squeeze(), ns, d.squeeze(), c.squeeze()
            
        if mode=='test':
            if original:
                s, a, nr, ns, d, c  = self.testoriginal
                return s, a, nr, ns, d, c 
            else:
                s, a, nr, ns, d, c  = self.testmemory
                return s, a.squeeze(), nr.squeeze(), ns, d.squeeze(), c.squeeze()
        
        if mode=='snubh':
            if original:
                s, a, nr, ns, d, c  = self.snubhoriginal
                return s, a, nr, ns, d, c 
            else:
                s, a, nr, ns, d, c  = self.snubhmemory
                return s, a.squeeze(), nr.squeeze(), ns, d.squeeze(), c.squeeze()
        
        if mode=='datex':
            if original:
                s, a, nr, ns, d, c  = self.datexoriginal
                return s, a, nr, ns, d, c 
            else:
                s, a, nr, ns, d, c  = self.datexmemory
                return s, a.squeeze(), nr.squeeze(), ns, d.squeeze(), c.squeeze()
    
    def __len__(self):
        """Return the current size of internal memory."""
        if self.mode == 'train':
            return len(self.trainmemory[0])
        if self.mode == 'valid':
            return len(self.validmemory[0])
        if self.mode == 'test':
            return len(self.testmemory[0])
        if self.mode == 'snubh':
            return len(self.snubhmemory[0])
        if self.mode == 'datex':
            return len(self.datexmemory[0])
