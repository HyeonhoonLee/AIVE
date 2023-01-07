import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats
from scipy.stats import chisquare, fisher_exact

from statannotations.Annotator import Annotator

plt.rcParams.update({'figure.max_open_warning': 0})

modelname = 'CQLSAC'
TARGETMODEL = 31196 #87842 #83811 #20731(가장최신) #36422 #36049
DIFF = [10, 30, 60]
# DIFF = [15, 60, 90, 300, 60]
HEMODYNAMICS = ['pip', 'spo', 'sbp', 'hr', 'apnea']#'hr'  '


PIP = [20,25,30]
SPO = [97,95,92]
SBP = [20,25,30]
HR = [20,25,30]
APNEA = [2]

ACTION = ['vent']#['vent', 'extu']
STATES = ['0', '1'] #['0','1', '01']
# EVAL = ['Good', "Mistake", "Bad", "Very Bad", "Horrible"]
# EVAL = ['Optimal', "Suboptimal", "Discrepent"]
EVAL = ['Agreed', 'Discrepant']
ototal = './resulttotal/'
# otrain = './resulttrain/'
ovalid = './resultvalid/'
otest = './resulttest/'
odatex = './resultdatex/'
osnubh = './resultsnubh/'

MAX_CASES=20000
# MAX_CASES_ADD=50
# np.savez('withaopts.npz', c, s, qs, v, a, aopts)
df_finder = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_primus.csv'))

df_finder4 = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_snubhprimus.csv'))
df_finder5 = pd.read_csv(os.path.join(f'{MAX_CASES}'+'cases_datex.csv'))
df_finder = pd.concat([df_finder, df_finder4, df_finder5], ignore_index=True)

test_results = [ f for f in os.listdir(otest) if f.startswith(f"results_test_{modelname}") and (str(TARGETMODEL) in f)]
print(f'{len(test_results)} test results for {modelname} are detected')
snubh_results = [ f for f in os.listdir(osnubh) if f.startswith(f"results_snubh_{modelname}") and (str(TARGETMODEL) in f)]
print(f'{len(snubh_results)} snubh files for {modelname} are detected')
datex_results = [ f for f in os.listdir(odatex) if f.startswith(f"results_datex_{modelname}") and (str(TARGETMODEL) in f)]
print(f'{len(datex_results)} datex files for {modelname} are detected')

def discrepency(x):
    diff = np.abs(x)
    if diff < 6:
        return 0
    elif 6<= diff <=12:
        return 1
    elif 12< diff:
        return 2
    # elif 20< diff:
    #     return 3
    
def discrepency_vent(x):
    diff = np.abs(x)
    if diff <= 100:
        return 0
    elif 100< diff <=500:
        return 1
    elif 500< diff:
        return 2
    
def combine_sp(x, y):
    if (x ==1) & (y==1):
        return "Bad"
    if ((x ==1) & (y==0)) | ((x ==0) & (y==1)) :
        return "One"
    if (x ==0) & (y==0):
        return "Good"

def combine_pip(x, y):
    if ((x ==1) & (y==0)) :
        return "Vent"
    if (x ==0) & (y==0):
        return "Good"
    

def combine_all(x, y):
    if (x ==0) & (y==0):
        return "Happy"
    else:
        return "Bad"
    
def weighted_eval(x, y):
    if x + y == 0:
        return EVAL[0]
    else:
        return EVAL[1]

def weighted_eval_one(x):
    if x == 0:
        return EVAL[0]
    else:
        return EVAL[1]

    
EVAL = EVAL[:2]
#1 control ticks, labels, legends
def pretty(axs, lim_idx, st_idx, color):
    if st_idx > 0:
        axs[lim_idx, st_idx].set(yticklabels=[])  
        axs[lim_idx, st_idx].set(ylabel=None)
        axs[lim_idx, st_idx].set_yticks([])
    if (st_idx ==1):
        if lim_idx == 0:
            patches = [matplotlib.patches.Patch(color=color[i], label=t) for i,t in enumerate(t.get_text() for t in axs[lim_idx, st_idx].get_xticklabels())]
            axs[lim_idx, st_idx].legend(handles=patches, bbox_to_anchor=(1.02,1), fontsize=8) #, loc="upper left"
        # else:
        #     patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(t.get_text() for t in axs[lim_idx, st_idx].get_xticklabels())]
        #     axs[lim_idx, st_idx].legend(handles=patches, bbox_to_anchor=(1.04,1), fontsize=8)   
    if lim_idx == 2:
        axs[lim_idx, st_idx].set(xlabel=f'STATE{STATES[st_idx]}')
    else:
        axs[lim_idx, st_idx].set(xlabel=None)
    axs[lim_idx, st_idx].set(xticklabels=[])
    axs[lim_idx, st_idx].set_xticks([])

#2 save figure    
def savefig(modelname, hemo, fig, dataset, the, action):
    fig.suptitle(f'{hemo.upper()} change by {action} with {modelname} on {dataset}set',fontweight ="bold")
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.80, top=0.9, wspace=0.10, hspace=0.20)
    # plt.tight_layout()
    plt.savefig(f'./plot/{hemo}/stat_{action}{the}_{hemo}_{modelname}_on_{dataset}set.tiff')

def makeresult(results):
    
    #if results[0].startswith("results_valid"):
    #    dataset = "valid"
    if results[0].startswith("results_test"):
        dataset = "test"
    if results[0].startswith("results_datex"):
        dataset = "datex"
    elif results[0].startswith("results_snubh"):
        dataset = "snubh"
        
        
    dfs = []
    
    for result in results:
    
        if dataset == "valid":
            data = pd.read_csv(os.path.join(ovalid, result))
            
        elif dataset == "test":
            data = pd.read_csv(os.path.join(otest, result))
        elif dataset == "datex":
            data = pd.read_csv(os.path.join(odatex, result))
        elif dataset == "snubh":
            data = pd.read_csv(os.path.join(osnubh, result))
        # print(f'before: {len(data)}')
        # data = data[data.case_len > 173]
        # print(f'after: {len(data)}')
        
            
       # data["large_extu"] = data.diff_extu.apply(lambda x : discrepency(x))
        data["large_vent"] = data.diff_vent.apply(lambda x : discrepency_vent(x))
        
        for hemo in HEMODYNAMICS:
            for the in ["", '_low']:
                if hemo == 'pip':
                    for lim in PIP:
                     
                        data[f"discrepency_{hemo}{lim}{the}_vent"] = data.apply(lambda x: weighted_eval_one(x.large_vent), axis = 1 )
                if hemo == 'spo':
                    the = ""
                    for lim in SPO:

                        data[f"discrepency_{hemo}{lim}{the}_vent"] = data.apply(lambda x: weighted_eval_one(x.large_vent), axis = 1 )
                if hemo == 'sbp':
                    for lim in SBP:

                        data[f"discrepency_{hemo}{lim}{the}_vent"] = data.apply(lambda x: weighted_eval_one(x.large_vent), axis = 1 )
                if hemo == 'hr':
                    for lim in HR:    
    
                        data[f"discrepency_{hemo}{lim}{the}_vent"] = data.apply(lambda x: weighted_eval_one(x.large_vent), axis = 1 )
                
                if hemo == 'apnea':
                    for lim in APNEA:    

                        data[f"discrepency_{hemo}{lim}_vent"] = data.apply(lambda x: weighted_eval_one(x.large_vent), axis = 1 )

        #for composite outcome

        data[f"discrepency_composite_vent"] = data.apply(lambda x: weighted_eval_one(x.large_vent), axis = 1 )
    
        dfs.append(data)
        
    data = pd.concat(dfs, ignore_index=True) # 모든 모델들을 합침
    data.to_csv(f'./results_{dataset}_{modelname}.csv', index = False)
    #data.iloc[:,-50:].to_csv(f'./dis_results_{dataset}_{modelname}.csv', index = False)
    print(data["discrepency_spo97_vent"].value_counts())
    
    def finder(caseid):
        return df_finder[df_finder['icase']==caseid]['filename'].to_numpy()[0]
    
    #action_dist = data[["caseid","case_len", "clin_extu", "clin_vent", "rl_extu", "rl_vent", 'diff_extu', 'diff_vent']]
    action_dist = data[["caseid","case_len", "clin_vent", "rl_vent", 'diff_vent']]
    action_dist['filename'] = action_dist['caseid'].apply(lambda x : finder(x))
    action_dist.to_csv('action_dist.csv')
    print(action_dist.describe())
    
    
    BOUND = 200
    #action_dist['diff_extu'] = action_dist['diff_extu'].apply(lambda x : np.clip(x, -BOUND, BOUND))
    #action_dist['diff_vent'] = action_dist['diff_vent'].apply(lambda x : np.clip(x, -BOUND, BOUND))
    
    #sns.jointplot(data=action_dist, x="diff_extu", y="diff_vent",xlim=[-200,200], ylim=[0,1000], alpha=0.5, color='red', s=5)
    if dataset == 'test':
        nbin = 40
    else:
        nbin = 30
    plt.figure(figsize=(20,20))
    sns.displot(data=action_dist, x="diff_vent", alpha=0.5, color='red', kde=True, bins=nbin)
    if dataset == 'test':
        plt.xlim([0, 400])
    else:
        plt.xlim([0, 400])
    plt.xlabel('Discrepant ventilation time (sec)')
    plt.ylabel('Case number')
    if dataset == 'test':
        plt.title(f'Case distribution on internal test set')
    else:
        plt.title(f'Case distribution on external test set')
    plt.tight_layout()
    plt.savefig(f'./plot/discrepency_dist_{dataset}_{modelname}.tiff')
    plt.close()
    print('histogram made')
    
    sns.lineplot(data=data, x='diff_vent', y='spo97_0', color='red', n_boot=2000)
    plt.savefig(f'./plot/spo.tiff')
    #3 make plot
    for action in ACTION:
        for hemo in HEMODYNAMICS:
            for the in ["", '_low']:
                plt.figure(figsize=(200, 200))
                fig, axs = plt.subplots(3, len(STATES))
                color = ['royalblue', 'darkred']
                for st_idx, st in enumerate(STATES):
                    st = '0' # For my own convenience... 
                    if hemo == 'pip':
                        for lim_idx, lim in enumerate(PIP):
                            ax = sns.barplot(x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, ax= axs[lim_idx, st_idx], order=EVAL, palette=color)
                            if lim == 20:
                                axs[lim_idx, st_idx].set(ylim=(0, 120))
                            elif lim == 30:
                                axs[lim_idx, st_idx].set(ylim=(0, 120))    
                            else:
                                axs[lim_idx, st_idx].set(ylim=(0, 120))
                            
                            if the =="":
                                axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}>{lim}%")
                            else:
                                axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}<{lim}%")
                            pretty(axs, lim_idx, st_idx, color)
                            annotator = Annotator(ax, [(EVAL[0], EVAL[1])], x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, order=EVAL, palette=color)
                            annotator.configure(test='t-test_welch', text_format='star', loc='outside')
                            annotator.apply_and_annotate()
                            
                    if hemo == 'spo':
                        the = ""
                        for lim_idx, lim in enumerate(SPO):
                            ax = sns.barplot(x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, ax= axs[lim_idx, st_idx], order=EVAL, palette=color)
                            if lim == 97:
                                axs[lim_idx, st_idx].set(ylim=(0,500))
                            else:
                                axs[lim_idx, st_idx].set(ylim=(0,500))
                            axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}<{lim}%")
                            pretty(axs, lim_idx, st_idx, color)
                            annotator = Annotator(ax, [(EVAL[0], EVAL[1])], x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, order=EVAL)
                            annotator.configure(test='t-test_welch', text_format='star', loc='outside')
                            annotator.apply_and_annotate()
                            
                    if hemo == 'sbp':
                        for lim_idx, lim in enumerate(SBP):
                            ax = sns.barplot(x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}', data=data, ax= axs[lim_idx, st_idx], order=EVAL, palette=color)
                            if lim == 10:
                                axs[lim_idx, st_idx].set(ylim=(0, 180))
                            else:
                                axs[lim_idx, st_idx].set(ylim=(0, 180))
                            axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}>{lim}")
                            pretty(axs, lim_idx, st_idx, color)
                            annotator = Annotator(ax, [(EVAL[0], EVAL[1])], x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, order=EVAL, palette=color)
                            annotator.configure(test='t-test_welch', text_format='star', loc='outside')
                            annotator.apply_and_annotate()
                            
                    if hemo == 'hr':
                        for lim_idx, lim in enumerate(HR):
                            ax = sns.barplot(x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, ax= axs[lim_idx, st_idx], order=EVAL, palette=color)
                            if lim == 100:
                                axs[lim_idx, st_idx].set(ylim=(0, 180))
                            else:
                                axs[lim_idx, st_idx].set(ylim=(0, 180))
                            if the =="":
                                axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}>{lim}%")
                            else:
                                axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}<{lim}%")
                            pretty(axs, lim_idx, st_idx, color)
                            annotator = Annotator(ax, [(EVAL[0], EVAL[1])], x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, order=EVAL, palette=color)
                            annotator.configure(test='t-test_welch', text_format='star', loc='outside')
                            annotator.apply_and_annotate()
                    
                    if hemo == 'apnea':
                        if the != '_low':
                            for lim_idx, lim in enumerate(APNEA):
                                sns.barplot(x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}', data=data, ax= axs[lim_idx, st_idx], order = EVAL, palette=color)
                                if lim == 10:
                                    axs[lim_idx, st_idx].set(ylim=(0, 200))
                                else:
                                    axs[lim_idx, st_idx].set(ylim=(0, 200))
                                axs[lim_idx, st_idx].set_ylabel(f"{hemo.upper()}<{lim}")
                                pretty(axs, lim_idx, st_idx, color)
                                annotator = Annotator(ax, [(EVAL[0], EVAL[1])], x=f"discrepency_{hemo}{lim}_{action}", y=f'{hemo}{lim}_{st}{the}', data=data, order=EVAL, palette=color)
                                annotator.configure(test='t-test_welch', text_format='star', loc='outside')
                                annotator.apply_and_annotate()

                savefig(modelname, hemo, fig, dataset, the, action)
                plt.close()
        
    print(f'Hemodynamic figures on {dataset}set made')
    
    

    # print(f'mean: {np.mean(data.case_len)}')
    # print(f'SD: {np.std(data.case_len)}')
    # print(f'MIN: {np.min(data.case_len)}')
    # print(f'MAX: {np.max(data.case_len)}')


    # # Make dummy row for visualize 0 value of 'Good' Column
    # dummy = [0 for n in range(len(data.columns))]
    # # col_idx = []
    # for idx, col in enumerate(data.columns):
    #     if col =='discrepency_reintu':
    #         dummy[idx] = 'Good'
    #     elif col == 'discrepency_mortality':
    #         dummy[idx] = 'Good'
    
def makereturn(total_results, valid_results, test_results,  snubh_results, datex_results): 
#def makereturn(total_results, valid_results):
    plt.figure(figsize=(10,5))
    
    dfs=[]
    for idx, result in enumerate(total_results):
        num = idx+1
        model_num = len(total_results)
        dat = pd.read_csv(os.path.join(ototal, result))
        data = pd.concat([dat]*(model_num-idx), ignore_index=True)
        data['model_num'] = [i for i in range(num, model_num+1)] * len(dat)
        data['dataset'] = ["Observed clinicians' policy"] * len(data)
        data['return'] = data['observable_r']
        dfs.append(data[['model_num','return', 'dataset']])
        # total_df['model_num'].append([idx]* len(data))
        # pd.concat([total_df['estimated_rl'], data['estimated_rl']])
    total_df = pd.concat(dfs, ignore_index=True)
    # observable_r의 그림을 일관되게 만들어 줌
    total_df['return'] =  np.ones(len(total_df['return'])) * np.mean(total_df['return'])

    dfs=[]
    for idx, result in enumerate(total_results):
        num = idx+1
        dat = pd.read_csv(os.path.join(ototal, result))
        data = pd.concat([dat]*(model_num-idx), ignore_index=True)
        data['model_num'] = [i for i in range(num, model_num+1)] * len(dat)
        data['dataset'] = ["Learned clinicians' policy"] * len(data)
        data['return'] = data['estimated_clinic']
        dfs.append(data[['model_num','return', 'dataset']])
    df2 = pd.concat(dfs, ignore_index=True)

    print('total results are processed')

    dfs=[]
    for idx, result in enumerate(valid_results):
        num = idx+1
        dat = pd.read_csv(os.path.join(ovalid, result))
        data = pd.concat([dat]*(model_num-idx), ignore_index=True)
        data['model_num'] = [i for i in range(num, model_num+1)] * len(dat)
        data['dataset'] = ["ExtuAI policy on validation set"] * len(data)
        data['return'] = data['estimated_rl']
        dfs.append(data[['model_num','return', 'dataset']])
    df3 = pd.concat(dfs, ignore_index=True)

    print('valid results are processed')

    dfs=[]
    for idx, result in enumerate(test_results):
        num = idx+1
        dat = pd.read_csv(os.path.join(otest, result))
        data = pd.concat([dat]*(model_num-idx), ignore_index=True)
        data['model_num'] = [i for i in range(num, model_num+1)] * len(dat)
        data['dataset'] = ["ExtuAI policy on test set"] * len(data)
        data['return'] = data['estimated_rl']
        dfs.append(data[['model_num','return', 'dataset']])
    df4 = pd.concat(dfs, ignore_index=True)

    print('test results are processed')
    
    dfs=[]
    for idx, result in enumerate(borame_results):
        num = idx+1
        dat = pd.read_csv(os.path.join(oborame, result))
        data = pd.concat([dat]*(model_num-idx), ignore_index=True)
        data['model_num'] = [i for i in range(num, model_num+1)] * len(dat)
        data['dataset'] = ["ExtuAI policy on bormae set"] * len(data)
        data['return'] = data['estimated_rl']
        dfs.append(data[['model_num','return', 'dataset']])
    df5 = pd.concat(dfs, ignore_index=True)

    print('borame results are processed')
    
    dfs=[]
    for idx, result in enumerate(snubh_results):
        num = idx+1
        dat = pd.read_csv(os.path.join(osnubh, result))
        data = pd.concat([dat]*(model_num-idx), ignore_index=True)
        data['model_num'] = [i for i in range(num, model_num+1)] * len(dat)
        data['dataset'] = ["ExtuAI policy on snubh set"] * len(data)
        data['return'] = data['estimated_rl']
        dfs.append(data[['model_num','return', 'dataset']])
    df6 = pd.concat(dfs, ignore_index=True)

    print('snubh results are processed')

    total_df = pd.concat([total_df, df2, df3, df4, df5, df6], ignore_index=True)
    #total_df = pd.concat([total_df, df2, df3], ignore_index=True)
    total_df.to_csv('performance_return.csv')
    
    r_max = max(total_df['return'])
    r_min = min(total_df['return'])
    
    # def normalize_return(x, r_max, r_min):
    #     return (x - r_min) / (r_max - r_min)
    
    # total_df['return'] = total_df['return'].apply(lambda x : normalize_return(x, r_max, r_min))

    sns.lineplot(data = total_df, x = 'model_num', y = 'return', hue='dataset')
    plt.legend(bbox_to_anchor=(1.00,1), fontsize=8)
    plt.xlabel('Number of models',fontweight ="bold")
    plt.ylabel('Estimated performance return',fontweight ="bold")
    plt.tight_layout()
    plt.savefig(f'./plot/peformance_return_{modelname}.tiff')
    plt.close()

    plt.figure(figsize=(10,5))
    bar_df = total_df[total_df['model_num']==model_num]
    sns.barplot(data = bar_df, x = 'model_num', y = 'return', hue='dataset')
    plt.legend(bbox_to_anchor=(1.00,1), fontsize=8)
    plt.xlabel('Number of models',fontweight ="bold")
    plt.ylabel('Estimated performance return',fontweight ="bold")
    plt.tight_layout()
    plt.savefig(f'./plot/(bar)peformance_return_{modelname}.tiff')
    plt.close()
    
#makeresult(valid_results)

makeresult(test_results)
makeresult(snubh_results)
#makeresult(datex_results)

#makereturn(total_results, valid_results, test_results, borame_results, snubh_results)
#makereturn(total_results, valid_results, test_results, snubh_results)
#makereturn(total_results, valid_results)
