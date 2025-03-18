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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# print(device)


class ActorCritic(nn.Module):
    def __init__(self,  gamma=0.9, lr=1e-4):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 6)
        
        self.critic = nn.Linear(128, 1)

        self.log_probs = []
        self.rewards = []
        self.values = []
        self.gamma = gamma
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value




    def update(self):
        gamma = self.gamma
        reward2 = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            reward2.append(R)
        rewards = reward2[::-1]

        rewards = torch.tensor(self.rewards).to(device)
        # rewards = ((rewards) / (rewards.std() + eps)).to(device)

        values = torch.cat(self.values).squeeze()
        advantages = rewards - values
        actor_loss = - (torch.stack(self.log_probs) * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, rewards)
        loss = (actor_loss + critic_loss).to(device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]

policy_model = ActorCritic().to(device)
# policy_model.load_state_dict(torch.load("policy_model.pth"))

policy_model.load_state_dict(torch.load("policy_model.pth", map_location=torch.device('cpu')))
policy_model.eval()



eps = np.finfo(np.float32).eps.item()

def get_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    probs, value = policy_model(obs)
    # if policy_model.ff == 1000:
    #     policy_model.ff += 1
    #     print(probs)
    # elif policy_model.ff < 1000:
    #     policy_model.ff += 1

    m = Categorical(probs)
    action = m.sample()
    policy_model.log_probs.append(m.log_prob(action))
    policy_model.values.append(value)
    return action.item()
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.


    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
