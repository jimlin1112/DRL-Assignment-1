# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(16, 64)
        self.affine2 = nn.Linear(64, 32)
        self.affine3 = nn.Linear(32, 6)    # 輸出層
        self.dropout = nn.Dropout(p=0.05)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.saved_log_probs = []
        self.rewards = []
        self.ff = 0

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.affine2(x)
        x = F.relu(x)
        x = self.dropout(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=0)




    def update(self, gamma=0.99):
        reward2 = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            reward2.append(R)
        self.rewards = reward2[::-1]

        # rewards = torch.tensor(self.rewards).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, self.rewards):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.saved_log_probs[:]
        del self.rewards[:]

policy_model = Policy()
policy_model.load_state_dict(torch.load("policy_model.pth"))
policy_model.eval()


eps = np.finfo(np.float32).eps.item()

def get_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    probs = policy_model(obs)
    # if policy_model.ff == 1000:
    #     policy_model.ff += 1
    #     print(probs)
    # elif policy_model.ff < 1000:
    #     policy_model.ff += 1

    m = Categorical(probs)
    action = m.sample()
    policy_model.saved_log_probs.append(m.log_prob(action))
    return action.item()
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
