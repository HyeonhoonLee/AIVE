#import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
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

MAX_CASES = 20000 #9304는 pip까지 뽑는 케이스

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

df_finder = pd.read_csv(os.path.join('../../', f'{MAX_CASES}'+'cases_primus.csv'))
#df_finder2 = pd.read_csv(os.path.join(f'{MAX
# _CASES}'+'cases_datex.csv'))
#df_finder3 = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_borame.csv'))
df_finder4 = pd.read_csv(os.path.join('../../', f'{MAX_CASES}'+'cases_snubhprimus.csv'))
#df_finder5 = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_snubhdatex.csv'))
#df_finder = pd.concat([df_finder, df_finder2, df_finder3, df_finder4, df_finder5], ignore_index=True)
print(df_finder.head())

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-discrete", help="Run name, default: CQL-SAC")
    #parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--epochs", type=int, default=50, help="Number of iteration, default: 50")
    parser.add_argument("--buffer_size", type=int, default=100_000_000, help="Maximal training dataset size, default: 100_000_000")
    parser.add_argument("--seed", type=int, default=42, help="Seed, default: 42")
    #parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
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
    # env = gym.make(config.env)
    
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #steps = 0
    #average10 = deque(maxlen=10)
    #total_steps = 0
    
    with wandb.init(project="extuai", name=config.run_name, config=config, entity="hyeonhoonlee"):

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
        #buffer.load('snuh', config.seed)
        buffer.load('snuh', se)
        
        (states, actions, rewards, next_states, dones, caseids) = dataset = buffer.get_data(mode='train', original=False)
        
        dataloader = prep_dataloader(dataset, batch_size=config.batch_size, weight=True)
        
        agent = CQLSAC(state_size = states.shape[1],
                        action_size = len(np.unique(actions)),
                        device = device,
                        gamma = 0.95, #0.99,
                        #tau = 1e-2,
                        hidden_size = 64, #256,
                        learning_rate = 5e-4,
                        #temp = 1.0, #1.0,
                        with_lagrange = False,
                        #cql_weight = 1.0,
                        target_action_gap = 0.0 #1.0
                        
                        )
        
        wandb.watch(agent, log="gradients", log_freq=10)
        #if config.log_video:
        #    env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)
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
            
            #print("Iters: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, iter_steps,))
            print("Epochs: {}/{} | Policy Loss: {}".format(i, config.epochs, policy_loss))
            

            wandb.log({"Reward": rewards,
                    #"Average10": np.mean(average10),
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
                    #"Episode": i,
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
        #print(np.bincount(a_test.astype(int)))
        a_test = a_test.astype(int)
        c_test = c_test.astype(int)
        
        test_aopts, test_aopts_prob, _ = agent.get_action_prob(states)
        
        test_vclin, test_vmodel = agent.get_value(states, actions, test_aopts)
        
        data_path = f'../../np{task}/{task}_'+modelname+f'{se}'+'.npz'
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

        if (task =='test') and (seednum % 10 == 0):

            test_aopts_prob = np.floor(test_aopts_prob*10) 
            test_caseids = np.unique(caseids)

            for case in test_caseids[:30]:
                case_mask = (c_test == case)
                case_len = np.sum(case_mask)

                if case_len == 0:
                    continue
                
                min_v = min(np.concatenate([test_vmodel, test_vclin]))
                max_v = max(np.concatenate([test_vmodel, test_vclin]))
                # min max scaled for visualization
                value_rl = (test_vmodel[case_mask]-min_v) / (max_v -  min_v) 
                value_clin = (test_vclin[case_mask]-min_v) / (max_v -  min_v)
                

                t = np.arange(0, case_len)
                plt.figure(figsize=(30, 5))
                plt.plot(t, s_test[case_mask, HR] * 10 + 5, label='dHR*10+5', color='pink')
                plt.plot(t, s_test[case_mask, SPO2] / 5 , label='SpO2 / 5', color='blue')
                plt.plot(t, s_test[case_mask, SBP] * 10 + 20, label='dSBP*10+20', linestyle='dashdot', color='brown')
                plt.plot(t, s_test[case_mask, PIP] * 10 + 15, label='dPIP*10+15', color='brown')
                plt.plot(t, s_test[case_mask, CO2], label='CO2', color='yellow')
                plt.plot(t, s_test[case_mask, AWP], label='AWP', color='gray')
                plt.plot(t, s_test[case_mask, TV] * 10 + 10, label='dTV*10+10', color='olive')
                
                plt.plot(t, s_test[case_mask, EXP_SEVO] * 10, label='Sevo * 10', color='orange')
                plt.plot(t, s_test[case_mask, PPF20_CE] * 10, label='Propofol * 10', color='green')
                plt.plot(t, s_test[case_mask, RFTN20_CE] * 10, label='Remifentanil * 10', linestyle='dashed', color='green')
                plt.plot(t, s_test[case_mask, SB] * 10, label='SB', color='purple')
                #plt.plot(t, s_test[case_mask, VENT_STATUS]* 10, label='Mechnial Vent.', color='gray')

                plt.plot(t, -nr_test[case_mask] , label='Penalty', color='red')
                
                #plt.fill_between(t, 0, 50, where=s_test[case_mask, VENT_STATUS]==1, label='Mechnial Vent.', alpha=0.1, color = 'gray')
                #print(a_test[case_mask, 0])
                #[0,0,1], [0,0,0], [0,1,0], [0,1,1], [1,0,0], [1,0,1]
                # print(a_test[case_mask])
                # print(test_aopts[case_mask])
                #plt.plot(t, a_test[case_mask]*10, label='clin', color='black')
                
                #or (a_test[case_mask]==3) or (a_test[case_mask]==5
                
                #plt.fill_between(t, -3.5, -1, where=(a_test[case_mask]==1), label='extu_clin', alpha=1.0, color = 'red')
                #plt.fill_between(t, -3.5, -1, where=(a_test[case_mask]==3), alpha=1.0, color = 'red')
            # plt.fill_between(t, -3.5, -1, where=(a_test[case_mask]==5), alpha=1.0, color = 'red')
                plt.fill_between(t, -3.5, -1, where=(a_test[case_mask]==1), label='venton_clin', alpha=1.0, color = 'blue')
                #plt.fill_between(t, -6, -3.5, where=(a_test[case_mask]==3), alpha=1.0, color = 'blue')
            # plt.fill_between(t, -8.5, -6, where=(a_test[case_mask]==4), label='venton_clin', alpha=1.0, color = 'green')
            # plt.fill_between(t, -8.5, -6, where=(a_test[case_mask]==5), alpha=1.0, color = 'green')

                for i in range(0,11,1):
                    if i ==10:
                        #label1, label2, = 'extu_rl', 'venton_rl'
                        label1 = 'venton_rl'
                    else:
                        #label1, label2, = None, None
                        label1 = None
                    plt.fill_between(t, -7, -4, where=(test_aopts[case_mask]==1)&(test_aopts_prob[case_mask, 1]==i), label=label1, alpha=i/10, color = 'blue')
                    #plt.fill_between(t, -11, -8.5, where=(test_aopts[case_mask]==3)&(test_aopts_prob[case_mask, 3]==i), alpha=i/10, color = 'red')
                    #plt.fill_between(t, -11, -8.5, where=(test_aopts[case_mask]==5)&(test_aopts_prob[case_mask, 5]==i), alpha=i/10, color = 'red')
                    #plt.fill_between(t, -13.5, -11, where=(test_aopts[case_mask]==2)&(test_aopts_prob[case_mask, 2]==i), label=label2, alpha=i/10, color = 'blue')
                    #plt.fill_between(t, -13.5, -11, where=(test_aopts[case_mask]==3)&(test_aopts_prob[case_mask, 3]==i), alpha=i/10, color = 'blue')
                # plt.fill_between(t, -16, -13.5, where=(test_aopts[case_mask]==4)&(test_aopts_prob[case_mask, 4]==i), label=label3, alpha=i/10, color = 'green')
                # plt.fill_between(t, -16, -13.5, where=(test_aopts[case_mask]==5)&(test_aopts_prob[case_mask, 5]==i), alpha=i/10, color = 'green')
                #plt.plot(t, a_test[case_mask] * 10, label='State', color='black')
                #plt.plot(t, test_aopts[case_mask] * 10, label='Action (RL)', linestyle='dashed', color='black')
                
                plt.plot(t, value_rl * 50, label='Value (RL)', linestyle='dotted', color='red')
                plt.plot(t, value_clin * 50, label='Value (clinician)', linestyle='dotted', color='blue')
                #print(f'case#{case}=> max_r: {round(max(-nr_test[case_mask]), 2)}, mean_r: {round(np.mean(-nr_test[case_mask]), 2)}, max_q: {round(max(test_vmodel[case_mask]), 2)}, mean_q: {round(np.mean(test_vmodel[case_mask]), 2)}')

                plt.legend(loc="upper left")

                # target = df_finder.iloc[case, 1]
                target = df_finder[df_finder['icase']==case]['filename'].to_numpy()[0]
                plt.title('PoVA_'+modelname+f' in caseid #{case} #{target} ')
                plt.tight_layout()
                plt.xlim([0, case_len])
                plt.ylim([-7, 50])
                plt.savefig(f'../../output_{modelname}/{target}.tiff')
                print(target)
                # plt.savefig(f'{odir}/{case_len}.png'
                plt.close()
        
    
    buffer.load('snubh', se)
    for task in ['snubh']:
        (states, actions, rewards, next_states, dones, caseids) = buffer.get_data(mode=f'{task}', original=False)
        
        (s_test, a_test, nr_test, ns_test, d_test, c_test) = buffer.get_data(mode=f'{task}', original=True)
        #print(np.bincount(a_test.astype(int)))
        a_test = a_test.astype(int)
        
        test_aopts, test_aopts_prob, _ = agent.get_action_prob(states)
        
        test_vclin, test_vmodel = agent.get_value(states, actions, test_aopts)
        
        data_path = f'../../np{task}/{task}_'+modelname+f'{se}'+'.npz'
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
    
    '''   
    buffer.load('datex', se)
    for task in ['datex']:
        (states, actions, rewards, next_states, dones, caseids) = buffer.get_data(mode=f'{task}', original=False)
        
        (s_test, a_test, nr_test, ns_test, d_test, c_test) = buffer.get_data(mode=f'{task}', original=True)
        #print(np.bincount(a_test.astype(int)))
        a_test = a_test.astype(int)
        
        test_aopts, test_aopts_prob, _ = agent.get_action_prob(states)
        
        test_vclin, test_vmodel = agent.get_value(states, actions, test_aopts)
        
        data_path = f'../../np{task}/{task}_'+modelname+f'{se}'+'.npz'
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
    '''    
    print(f'Seednum {seednum} is done')
    
if __name__ == "__main__":
    config = get_config()
    random.seed(config.seed)
    seedlist = random.sample(range(1,100000), NMODEL) # 1부터 100000까지의 범위중에 500개를 중복없이 뽑겠다. 
    print(f'random seeds are {seedlist[:5]}...{seedlist[-5:]} total {len(seedlist)}')
    #print(seedlist[200:230])
    seednum=0
    for se in seedlist[113:]:
    #for se in [42, 43]: 
        train(config, se, seednum)
        seednum+=1