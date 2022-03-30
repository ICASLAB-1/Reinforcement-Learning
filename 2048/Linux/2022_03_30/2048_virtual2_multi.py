import gym_2048
import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import ray
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from torchsummary import summary

start_time = time.time()
ray.init()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device} device')


main_env = gym.make('2048-v0')
virtual1_env = gym.make('2048-v0')
virtual2_env = gym.make('2048-v0')

env_list = [virtual1_env, virtual2_env]
gamma = 0.9
batch_size = 512
max_memory = batch_size*8
memory = []
dense = 128

layer_count = 14 # 2 ~ 2048 까지의 타일 종류의 수
table = {2**i:i for i in range(layer_count)}
print(table)


def preprocess(obs):
    x = np.zeros([layer_count, 4, 4]) # 4X4 의 observation space, 12개의 타일 종류
    for k in range(12):
        matching = pow(2, k)
        for i in range(4):
            for j in range(4):
                if k == 0:
                    if obs[i,j] == 0:
                        x[k,i,j] = 1
                else:
                    if obs[i,j] == matching:
                        x[k,i,j] = 1
    return x


class Multi_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels=layer_count, out_channels=dense, kernel_size=(2, 1))
        self.conv_b = nn.Conv2d(in_channels=layer_count, out_channels=dense, kernel_size=(1, 2))
        self.conv_aa = nn.Conv2d(in_channels=dense, out_channels=dense, kernel_size=(2, 1))
        self.conv_ab = nn.Conv2d(in_channels=dense, out_channels=dense, kernel_size=(1, 2))
        self.conv_ba = nn.Conv2d(in_channels=dense, out_channels=dense, kernel_size=(2, 1))
        self.conv_bb = nn.Conv2d(in_channels=dense, out_channels=dense, kernel_size=(1, 2))
        self.fc1 = nn.Linear(7424, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        a = F.relu(self.conv_a(x))
        b = F.relu(self.conv_b(x))
        aa = F.relu(self.conv_aa(a))
        ab = F.relu(self.conv_ab(a))
        ba = F.relu(self.conv_ba(b))
        bb = F.relu(self.conv_bb(b))

        trans = [t.view(t.size(0), -1) for t in [a, b, aa, ab, ba, bb]]
        flat = [torch.flatten(t, start_dim=1) for t in trans]
        # flat = [torch.flatten(t) for t in [a, b, aa, ab, ba, bb]]
        concat = torch.cat(flat, dim=1)
        act = F.relu(self.fc1(concat))
        act = self.fc2(act)

        return act


def append_sample(memory, state, action, reward, next_state, done):
    memory.append([state, action, reward, next_state, done])


def train_model(model, target_model, optim, cri):
    np.random.shuffle(memory)
    len = max_memory // batch_size
    for k in range(len):
        mini_batch = memory[k*batch_size:(k+1)*batch_size]

        states = np.zeros((batch_size, layer_count, 4, 4))
        next_states = np.zeros((batch_size, layer_count, 4, 4))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        target = model(states).detach().numpy()
        next_target = target_model(next_states).detach().numpy()

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + gamma * np.amax(next_target[i])

        Y = torch.tensor(target, dtype=torch.float32)
        dataset = TensorDataset(states, Y)
        train_loader = DataLoader(dataset=dataset, batch_size=512)

        cost = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            y_pred = model(x)
            loss = cri(y_pred, y)
            loss.backward()
            optim.step()
            cost += loss

        print("epoch : {}, loss : {:.6f}".format(k, cost))


def softmax(logits):
    logits = logits.detach().numpy()
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits


@ray.remote
def going_game(env, model, epsilon):
    prev_obs = env.reset()
    prev_max = np.max(prev_obs)
    not_move_list = np.array([1, 1, 1, 1])
    local_memory = []
    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            x = preprocess(prev_obs)
            x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            logits = model(x)
            prob = softmax(logits)
            prob = prob * not_move_list
            action = np.argmax(prob)

        obs, reward, done, _ = env.step(action)

        if reward == 0 and np.array_equal(obs, prev_obs):
            not_move_list[action] = 0
            continue
        else:
            not_move_list = np.array([1, 1, 1, 1])

        now_max = np.max(obs)
        if prev_max < now_max:
            prev_max = now_max
            reward = math.log(now_max, 2) * 0.1
        else:
            reward = 0

        reward += np.count_nonzero(prev_obs) - np.count_nonzero(obs) + 1

        append_sample(local_memory, preprocess(prev_obs), action, reward, preprocess(obs), done)

        prev_obs = obs

        if done:
            break

    # return [preprocess(prev_obs), action, reward, preprocess(obs), done]
    return local_memory


model = Multi_Network()
torch.save(model.state_dict(), 'target2.pth')
target_model = Multi_Network()
target_model.load_state_dict(torch.load('target2.pth'))

model.to(device)
target_model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()
# summary(model, (14, 4, 4))

max_episodes = 10001
epsilon = 0.9
epsilon_min = 0.0001

scores = []
steps = []
max_tile = []
iteration = 0

train_count = 0

for i in range(max_episodes):
    if i % 100 == 0 and i != 0:
        print('scores mean: ', np.mean(scores[-100:]), 'step mean: ', np.mean(steps[-100:]), 'iteration: ', iteration,
              'epsilon: ', epsilon)

    score = 0
    step = 0

    main_prev_obs = main_env.reset()
    main_prev_max = np.max(main_prev_obs)
    not_move_list = np.array([1, 1, 1, 1])
    futures = [going_game.remote(env, model, epsilon) for env in env_list]
    results = ray.get(futures)
    for result in results:
        for set in result:
            memory.append(set)

    while True:
        if random.random() < epsilon:
            main_action = main_env.action_space.sample()
        else:
            main_x = preprocess(main_prev_obs)
            main_x = np.expand_dims(main_x, axis=0)
            main_x = torch.tensor(main_x, dtype=torch.float32)
            main_x = main_x.to(device)
            main_logits = model(main_x)
            main_prob = softmax(main_logits)
            main_prob = main_prob * not_move_list
            main_action = np.argmax(main_prob)

        main_obs, main_reward, main_done, _ = main_env.step(main_action)

        score += main_reward
        step += 1

        if main_reward == 0 and np.array_equal(main_obs, main_prev_obs):
            not_move_list[main_action] = 0
            continue
        else:
            not_move_list = np.array([1, 1, 1, 1])

        main_now_max = np.max(main_obs)
        if main_prev_max < main_now_max:
            main_prev_max = main_now_max
            main_reward = math.log(main_now_max, 2) * 0.1
        else:
            main_reward = 0

        main_reward += np.count_nonzero(main_prev_obs) - np.count_nonzero(main_obs) + 1

        append_sample(memory, preprocess(main_prev_obs), main_action, main_reward, preprocess(main_obs), main_done)

        if len(memory) >= max_memory:
            train_model(model, target_model, optimizer, criterion)
            memory = []

            train_count += 1
            if train_count % 4 == 0:
                torch.save(model.state_dict(), 'target2.pth')
                target_model.load_state_dict(torch.load('target2.pth'))

        main_prev_obs = main_obs

        if epsilon > epsilon_min and iteration % 10 == 0:
            epsilon = epsilon / 1.005

        iteration += 1

        if main_done:
            break

    scores.append(score)
    steps.append(score)
    max_tile.append(np.max(main_obs))

    print(i, 'score: ', score, 'step: ', step, 'max tile: ', np.max(main_obs), 'memory len: ', len(memory), 'epsilon: ', epsilon)

torch.save(model.state_dict(), '2048_virtual2_original_normal_ray.pth')
df1 = pd.DataFrame(scores)
df2 = pd.DataFrame(steps)
df3 = pd.DataFrame(max_tile)
df1.to_csv("virtual2_scores_ray.csv")
df2.to_csv("virtual2_steps_ray.csv")
df3.to_csv("virtual2_max_tile_ray.csv")

print("running time : ", time.time() - start_time)
N = 100
rolling_mean = [np.mean(scores[x:x+N]) for x in range(len(scores)-N+1)]

plt.figure(figsize=(12,4))
plt.subplot(1, 3, 1)
plt.scatter(range(len(scores)), scores, marker='.')
plt.subplot(1, 3, 2)
plt.plot(rolling_mean)
plt.subplot(1, 3, 3)
plt.scatter(range(len(max_tile)), max_tile, marker='.')
plt.show()
