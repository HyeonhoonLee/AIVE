import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import Buffer
import glob
from utils import save, collect_random
import random
from agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset, sampler
import matplotlib.pyplot as plt
import os
import pandas as pd

modelname = 'CQLSAC'

NMODEL = 500

MAX_CASES = 20000 

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

df_finder = pd.read_csv('cases.csv')
print(df_finder.head())

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-discrete", help="Run name, default: CQL-SAC")
    parser.add_argument("--epochs", type=int, default=50, help="Number of iteration, default: 50")
    parser.add_argument("--buffer_size", type=int, default=100_000_000, help="Maximal training dataset size, default: 100_000_000")
    parser.add_argument("--seed", type=int, default=42, help="Seed, default: 42")
    parser.add_argument("--save_every", type=int, default=1, help="Saves the network every x epochs, default: 1")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size, default: 8192")
    
    args = parser.parse_args()
    return args

def prep_dataloader(dataset, batch_size=256, seed=42, weight=True):
    tensors = {}
    tuples= ["states", "actions", "rewards", "next_states", "terminals", "caseids"]
    for k, v in list(zip(tuples, dataset)):
        if  (k != "terminals") or (k != "caseids") or (k != "actions"):
            tensors[k] = torch.from_numpy(v).float()
        else:
            tensors[k] = torch.from_numpy(v).long()
    
    if weight:
        target = dataset[1]
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t.astype(int)] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        weightedsampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle=False
    else:
        shuffle=True
        weightedsampler = None
        
    tensordata = TensorDataset(tensors["states"],
                               tensors["actions"][:, None],
                               tensors["rewards"][:, None],
                               tensors["next_states"],
                               tensors["terminals"][:, None],
                               tensors["caseids"][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=shuffle, sampler=weightedsampler, num_workers=8, pin_memory=True)
    
    return dataloader

def train(config, se, seednum):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with wandb.init(project="AIVE", name=config.run_name, config=config, entity="hyeonhoonlee"):

        buffer = Buffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        buffer.load('snuh', se)
        
        (states, actions, rewards, next_states, dones, caseids) = dataset = buffer.get_data(mode='train', original=False)
        
        dataloader = prep_dataloader(dataset, batch_size=config.batch_size, weight=True)
        
        agent = CQLSAC(state_size = states.shape[1],
                        action_size = len(np.unique(actions)),
                        device = device,
                        gamma = 0.95,
                        hidden_size = 64, 
                        learning_rate = 5e-4,
                        with_lagrange = False,
                        target_action_gap = 0.0
                        )
        
        wandb.watch(agent, log="gradients", log_freq=10)

        epochs = 0
        batches=0
        bestloss=-1e+8
        patience=0
        earlystop=5
        for i in range(1, config.epochs+1):
            
            for batch_idx, experience in enumerate(dataloader):
                states, actions, rewards, next_states, dones, caseids = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(states, actions, rewards, next_states, dones)
                batches += 1

            epochs += 1
            
            print("Epochs: {}/{} | Policy Loss: {}".format(i, config.epochs, policy_loss))
            

            wandb.log({"Reward": rewards,
                    "Epochs": epochs,
                    "Policy Loss": policy_loss,
                    "Alpha Loss": alpha_loss,
                    "Lagrange Alpha Loss": lagrange_alpha_loss,
                    "CQL1 Loss": cql1_loss,
                    "CQL2 Loss": cql2_loss,
                    "Bellmann error 1": bellmann_error1,
                    "Bellmann error 2": bellmann_error2,
                    "Alpha": current_alpha,
                    "Lagrange Alpha": lagrange_alpha,
                    "Batches": batches,
                    "Buffer size": buffer.__len__()})
            
            if i % config.save_every == 0:
                save(config, save_name="CQL-SAC-discrete", model=agent.actor_local, wandb=wandb, ep=0)
                
            if policy_loss > bestloss:
                bestloss = policy_loss
                patience=0    
            else:
                patience +=1
                
            if patience > earlystop:
                print('Early stopped')
                break
                
    
    for task in ['valid', 'test']:
        (states, actions, rewards, next_states, dones, caseids) = buffer.get_data(mode=f'{task}', original=False)
        
        (s_test, a_test, nr_test, ns_test, d_test, c_test) = buffer.get_data(mode=f'{task}', original=True)

        a_test = a_test.astype(int)
        c_test = c_test.astype(int)
        
        test_aopts, test_aopts_prob, _ = agent.get_action_prob(states)
        
        test_vclin, test_vmodel = agent.get_value(states, actions, test_aopts)
        
        data_path = f'{task}_'+modelname+f'{se}'+'.npz'
        np.savez(data_path, 
                caseid=c_test,
                state =s_test,
                nstate=ns_test,
                done=d_test,
                qvalue=test_vmodel,
                qdata =test_vclin,
                action_pred=test_aopts,
                action_prob=test_aopts_prob,
                action=a_test,
                nreward=nr_test,
                state_=states,
                nstate_=next_states,
                nreward_=rewards)
        
    
    buffer.load('snubh', se)
    for task in ['snubh']:
        (states, actions, rewards, next_states, dones, caseids) = buffer.get_data(mode=f'{task}', original=False)
        
        (s_test, a_test, nr_test, ns_test, d_test, c_test) = buffer.get_data(mode=f'{task}', original=True)
        a_test = a_test.astype(int)
        
        test_aopts, test_aopts_prob, _ = agent.get_action_prob(states)
        
        test_vclin, test_vmodel = agent.get_value(states, actions, test_aopts)
        
        data_path = f'{task}_'+modelname+f'{se}'+'.npz'
        np.savez(data_path, 
                caseid=c_test,
                state =s_test,
                nstate=ns_test,
                done=d_test,
                qvalue=test_vmodel,
                qdata =test_vclin,
                action_pred=test_aopts,
                action_prob=test_aopts_prob,
                action=a_test,
                #reward=r_test,
                nreward=nr_test,
                state_=states,
                nstate_=next_states,
                nreward_=rewards)
 
    print(f'Seednum {seednum} is done')
    
if __name__ == "__main__":
    config = get_config()
    random.seed(config.seed)
    seedlist = random.sample(range(1,100000), NMODEL) 
    print(f'random seeds are {seedlist[:5]}...{seedlist[-5:]} total {len(seedlist)}')
    seednum=0
    for se in seedlist[113:]:
        train(config, se, seednum)
        seednum+=1