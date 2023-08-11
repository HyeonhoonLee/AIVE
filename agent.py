import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import numpy as np
import math
import copy

from torch.distributions import Categorical 

class CQLSAC(nn.Module):
    
    def __init__(self,
                        state_size,
                        action_size,
                        device,
                        gamma: float=0.99,
                        tau: float=1e-2,
                        hidden_size: int=256,
                        learning_rate: float=5e-4,
                        with_lagrange: bool=False,
                        target_action_gap: float=0.0
                ):

        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = gamma 
        self.tau = tau  
        hidden_size = hidden_size  
        learning_rate = learning_rate 
        self.clip_grad_param = 1

        self.target_entropy = -action_size

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
        self.with_lagrange = with_lagrange
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate) 
        
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
        self.softmax = nn.Softmax(dim=-1)

    
    def get_action(self, state, eval=False):
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()
    
    def get_action_prob(self, state, eval=False):
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            action, action_prob, log_action_prob = self.actor_local.get_action(state)
        return action.numpy(), action_prob.detach().cpu().numpy(), log_action_prob.detach().cpu().numpy()
    
    def get_value(self, state, action, aopt):
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            q1 = self.critic1(state)
            q2 = self.critic2(state)
            
            q1_data = q1.gather(1, torch.Tensor(action[:, None]).long().to(self.device)).squeeze()
            q2_data = q2.gather(1, torch.Tensor(action[:, None]).long().to(self.device)).squeeze()
            
            q1_model = q1.gather(1, torch.Tensor(aopt[:, None]).long().to(self.device)).squeeze()
            q2_model = q2.gather(1, torch.Tensor(aopt[:, None]).long().to(self.device)).squeeze()
            
            q_data, q_model = torch.min(torch.stack([q1_data,q2_data],dim=1).detach().cpu(),axis=1)[0], torch.min(torch.stack([q1_model,q2_model],dim=1).detach().cpu(),axis=1)[0]
        
        return q_data, q_model

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)
            
        q1 = self.critic1(states)   
        q2 = self.critic2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)

        return actor_loss, log_action_pi
    
    def learn(self, states, actions, rewards, next_states, dones, d=1):
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha, bc=False)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        
        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)
        
        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

    def soft_update(self, local_model , target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
