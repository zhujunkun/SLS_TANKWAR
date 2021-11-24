import parl
import random
from copy import deepcopy
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from parl.utils import LinearDecayScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(parl.Agent):
    def __init__(self, alg, epsilon=None, update_steps=2):
        super().__init__(alg)
        self.update_steps = update_steps
        self.cur_steps = 0
        self.scl = LinearDecayScheduler(epsilon, 1000000)

    def sample(self, obs):
        self.epsilon = self.scl.step(1)
        if random.random() < self.epsilon:
            move, shot = random.randint(0, 3), random.randint(0, 1)
            return [move, shot], self.epsilon
        else:
            return self.predict(obs), self.epsilon

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        move, shot = self.alg.predict(obs)
        return [move, shot]

    def learn(self, obs, act, reward, next_obs, done):
        if self.cur_steps % self.update_steps == 0:
            self.alg.sync_target()
        self.cur_steps += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.clip(reward, -1, 1)
        reward = np.expand_dims(reward, axis=-1)
        done = np.expand_dims(done, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float, device=device)
        act = torch.tensor(act, dtype=torch.int, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.float, device=device)

        loss = self.alg.learn(obs, act, reward, next_obs, done)
        return loss
    
class Agent_enemy(parl.Agent):
    def __init__(self, alg, epsilon=None, update_steps=2):
        super().__init__(alg)
        self.update_steps = update_steps
        self.cur_steps = 0
        self.scl = LinearDecayScheduler(epsilon, 1000000)

    def sample(self, obs):
        self.epsilon = self.scl.step(1)
        if random.random() < self.epsilon:
            move, shot = random.randint(0, 3), random.randint(0, 1)
            return [move, shot], self.epsilon
        else:
            return self.predict(obs), self.epsilon

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        move, shot = self.alg.predict(obs)
        return [move, shot]

    def learn(self, obs, act, reward, next_obs, done):
        if self.cur_steps % self.update_steps == 0:
            self.alg.sync_target()
        self.cur_steps += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.clip(reward, -1, 1)
        reward = np.expand_dims(reward, axis=-1)
        done = np.expand_dims(done, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float, device=device)
        act = torch.tensor(act, dtype=torch.int, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.float, device=device)

        loss = self.alg.learn(obs, act, reward, next_obs, done)
        return loss

class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        self.model = model.to(device)
        self.target_model = deepcopy(self.model).to(device)

        self.gamma = gamma
        self.loss_func = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, obs):
        move_Q, shot_Q = self.model(obs)
        move_Q = move_Q.cpu().detach().numpy().squeeze()
        shot_Q = shot_Q.cpu().detach().numpy().squeeze()
        move, shot = np.argmax(move_Q), np.argmax(shot_Q)
        return [move, shot]

    def learn(self, obs, act, reward, next_obs, done):
        act = act.type(torch.int64)
        move_act, shot_act = act[:, 0, :], act[:, 1, :]

        move_Q, shot_Q = self.model(obs)
        pred_move, pred_shot = move_Q.gather(1, move_act), shot_Q.gather(1, shot_act)
        with torch.no_grad():
            target_move_Q, target_shot_Q = self.target_model(next_obs)
            target_move_Q = target_move_Q.max(1, keepdim=True)[0]
            target_shot_Q = target_shot_Q.max(1, keepdim=True)[0]

            target_move_Q = reward + (1 - done) * self.gamma * target_move_Q
            target_shot_Q = reward + (1 - done) * self.gamma * target_shot_Q

        loss = self.loss_func(target_move_Q, pred_move) + self.loss_func(target_shot_Q, pred_shot)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)


class DQNModel(parl.Model):
    def __init__(self, obs_dim, move_dim, shot_dim):
        super(DQNModel, self).__init__()
        self.l1 = nn.Linear(obs_dim, 500)
        self.l2 = nn.Linear(500, 200)
        # move
        self.l3 = nn.Linear(200, move_dim)
        # shot
        self.l4 = nn.Linear(200, shot_dim)

    def forward(self, obs):
        out = F.relu(self.l1(obs))
        out = F.relu(self.l2(out))
        move_Q = self.l3(out)
        shot_Q = self.l4(out)
        return move_Q, shot_Q
    
class DQNModel_enemy(parl.Model):
    def __init__(self, obs_dim, move_dim, shot_dim):
        super(DQNModel_enemy, self).__init__()
        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 200)
        nn.Dropout(p=0.2)
        # move
        self.l3 = nn.Linear(200, move_dim)
        # shot
        self.l4 = nn.Linear(200, shot_dim)

    def forward(self, obs):
        out = F.relu(self.l1(obs))
        out = F.relu(self.l2(out))
        move_Q = self.l3(out)
        shot_Q = self.l4(out)
        return move_Q, shot_Q

class ReplayBuffer(object):
    def __init__(self, max_size, ):
        self.replay_buffer = deque(maxlen=max_size)
        self.size = 0
        self.max_size = max_size

    def store(self, experience):
        # (obs, act, reward, next_obs, done)
        self.replay_buffer.append(experience)
        if self.size < self.max_size:
            self.size += 1

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        obs_batch, act_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in batch:
            s, a, r, s_, done = experience
            obs_batch.append(s)
            act_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_)
            done_batch.append(done)
        return np.array(obs_batch), np.array(act_batch), np.array(reward_batch), np.array(next_obs_batch), np.array(done_batch)
