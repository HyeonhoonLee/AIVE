import numpy as np
import pandas as pd
import random
import torch

import os
import time

from collections import deque, namedtuple
from sklearn.utils import shuffle


class Buffer:

    def __init__(self, buffer_size, batch_size, device):

        self.device = device
        self.mode = None
        self.trainmemory = deque(maxlen=buffer_size)  
        self.validmemory  = deque(maxlen=buffer_size) 
        self.testmemory = deque(maxlen=buffer_size)
        self.snubhmemory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def load(self, source, seed):   
           
        TESTSIZE = 0.15
        VALIDSIZE = 0.17651 
        SEED = 42
        EPSILON = 1e-8
 
        PPF20_CE = 0
        RFTN20_CE = 1
        EXP_SEVO = 2
        PIP = 3
        TV = 4
        AWP = 5
        CO2 = 6 
        HR = 7
        SPO2 = 8
        SBP = 9
        SB = 10
        APNEA = 11 
        VENT_STATE = 12
        EXTU_STATE = 13

        print('loading data and concatenating', end='...', flush=True)

        data1 = np.load('dataset.npz')
        state = data1['state']
        actions = data1['action']
        rewards = data1['reward'] 
        next_rewards = data1['next_reward']
        next_state = data1['next_state']
        caseid = data1['caseid']
        terminals = data1['done']

        data4 = np.load('dataset4.npz')
        state4 = data4['state']
        actions4 = data4['action']
        rewards4 = data4['reward']
        next_rewards4 = data4['next_reward']
        next_state4 = data4['next_state']
        caseid4 = data4['caseid']
        terminals4 = data4['done']      

        s_mean, s_std = np.mean(states, axis=0), np.std(states, axis=0)
        r_mean, r_std = np.mean(rewards), np.std(rewards)

        states_ = (states - s_mean) / (s_std + EPSILON)
        next_states_ = (next_states - s_mean) / (s_std + EPSILON)

        states4_ = (states4 - s_mean) / (s_std + EPSILON)
        next_states4_ = (next_states4 - s_mean) / (s_std + EPSILON)

        print(f' mean{np.mean(rewards)}, min{np.min(rewards)}, max{np.max(rewards)}, median{np.median(rewards)}, 1/4Q{np.quantile(rewards, 0.25)}, 3/4Q{np.quantile(rewards, 0.75)}')

        rewards = np.clip(rewards, -20,  0) 
        next_rewards = np.clip(next_rewards, -20,  0) 

        rewards_ = (rewards - r_mean) / (r_std + EPSILON)
        next_rewards_ = (next_rewards - r_mean) / (r_std + EPSILON)


        rewards4 = np.clip(rewards4, -20, 0)
        next_rewards4 = np.clip(next_rewards4, -20, 0)

        rewards4_ = (rewards4 - r_mean) / (r_std + EPSILON)
        next_rewards4_ = (next_rewards4 - r_mean) / (r_std + EPSILON)
        
        states_[:, SB], next_states_[:, SB] = states[:, SB], next_states[:, SB]
        states_[:, VENT_STATE], next_states_[:, VENT_STATE] = states[:, VENT_STATE], next_states[:, VENT_STATE]
        states_[:, EXTU_STATE], next_states_[:, EXTU_STATE] = states[:, EXTU_STATE], next_states[:, EXTU_STATE]
        
        states4_[:, SB], next_states4_[:, SB]  = states4[:, SB], next_states4[:, SB]
        states4_[:, VENT_STATE], next_states4_[:, VENT_STATE] = states4[:, VENT_STATE], next_states4[:, VENT_STATE]
        states4_[:, EXTU_STATE], next_states4_[:, EXTU_STATE] = states4[:, EXTU_STATE], next_states4[:, EXTU_STATE]

        print('...done')

        caseids = np.unique(caseid)
            
        print(f'{len(caseids)} cases are loaded')

        print(f'total state space: {states_.shape}')
        print(f'total action space: {actions.shape} * {len(np.unique(actions))}')
        print(f'total reward space: {rewards_.shape}')

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
        
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.trainmemory[0])
        if self.mode == 'valid':
            return len(self.validmemory[0])
        if self.mode == 'test':
            return len(self.testmemory[0])
        if self.mode == 'snubh':
            return len(self.snubhmemory[0])

