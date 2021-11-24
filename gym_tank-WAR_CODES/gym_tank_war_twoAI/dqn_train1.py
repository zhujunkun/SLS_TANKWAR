import gym
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from statistics import median, mean
from collections import Counter
from tank_env import TankEnv
import torch.nn.functional as F
import argparse
device = torch.device("cpu")

env = TankEnv('Tank Game')
    


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
class DQNAgent:

    def __init__(self,model):
        self.training_data = []
        self.model=model.to(device)
    def initial_experience(self, score_requirements=10, n_games=50):
        scores = []
        accepted_scores = []
        for _ in range(n_games):
            observation = env.reset()
            score = 0
            game_memory = []
            for i in range(500):
                action = [random.randrange(0, 4),random.randrange(0, 2)]
                game_memory.append([observation, action])
                observation, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break
            if score > 0:
                accepted_scores.append(score)
                self.training_data.extend(game_memory)
            scores.append(score)
        training_data_save = np.array(self.training_data)
        np.save('training_data.npy', training_data_save)




    def train_model(self, batch_size=64, n_epoch=5, lr=1e-3, verbose=False):
        X = torch.tensor([i[0] for i in self.training_data], dtype=torch.float)
        y = torch.tensor([i[1] for i in self.training_data], dtype=torch.long)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        #self.model((X.shape[1],4,2))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for e in range(n_epoch):
            running_loss = 0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                output1,output2 = self.model(inputs)

                output=[]
                output3=[]
                label1=[]
                label2=[]
                for i in range(len(output1)):
                    
                    
                    output.append(output1[i].detach().numpy().tolist())
                    output3.append(output2[i].detach().numpy().tolist())

                    label1.append(labels[i][0].detach().numpy().tolist())
                    label2.append(labels[i][1].detach().numpy().tolist())

                output = torch.from_numpy(np.array(output))
                output3 = torch.from_numpy(np.array(output3))
                label1=torch.from_numpy(np.array(label1))
                label2=torch.from_numpy(np.array(label2))
                #label1=label1.requires_grad_()
                #label2=label2.requires_grad_()

                loss = criterion(output, label1.long())+ criterion(output3, label2.long())
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                if verbose:
                    print(f"Training loss: {running_loss / len(dataloader)}")
        return self.model

class DQNModel(torch.nn.Module):
    def __init__(self, obs_dim=315, move_dim=4, shot_dim=2):
        super(DQNModel, self).__init__()
        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 200)
        self.l3 = nn.Linear(200, 100)
        # move
        self.l4 = nn.Linear(100, move_dim)
        # shot
        self.l5 = nn.Linear(100, shot_dim)

    def forward(self, obs):
        out = F.relu(self.l1(obs))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        move_Q = self.l4(out)
        shot_Q = self.l5(out)
        return move_Q, shot_Q

if __name__ == "__main__":
    total_avg_reward = 0
    for run in tqdm(range(30)):
        model=DQNModel(315,4,2)
        cartpole_agent = DQNAgent(model)
        cartpole_agent.initial_experience(n_games=1500)
        model = cartpole_agent.train_model(n_epoch=10, lr=1e-3)
        scores = []
        choices = []
        for episode in range(10):
            score = 0
            game_memory = []
            prev_obs = []
            env.reset()
            done = False
            while not done:
                if len(prev_obs) == 0:
                    action = [random.randrange(0, 4),random.randrange(0, 2)]
                else:
                    with torch.no_grad():
                        action = [np.argmax(model(torch.Tensor(prev_obs).view(1, len(prev_obs)))[0].numpy()),np.argmax(model(torch.Tensor(prev_obs).view(1, len(prev_obs)))[1].numpy())]

                choices.append(action)

                new_observation, reward, done, info = env.step(action)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score += reward
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        total_avg_reward += avg_score
    print(f"Average reward over 30 runs over 100 episodes: {total_avg_reward / 30}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='TankEnv', help='game')
    parser.add_argument('--total_steps', type=int, default=int(1e6), help='maximum training steps')
    parser.add_argument('--test_every_steps', type=int, default=1000, help='the step interval for evaluating')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')

    args = parser.parse_args()
    main()
    env.close()
