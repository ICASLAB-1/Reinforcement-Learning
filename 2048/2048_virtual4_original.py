import gym_2048
import gym
import numpy as np
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt

main_env = gym.make('2048-v0')
virtual1_env = gym.make('2048-v0')
virtual2_env = gym.make('2048-v0')
virtual3_env = gym.make('2048-v0')
virtual4_env = gym.make('2048-v0')

gamma = 0.9
batch_size = 512
max_memory = batch_size*8
memory = []

layer_count = 14 # 2 ~ 2048 까지의 타일 종류의 수
table = {2**i:i for i in range(layer_count)}
print(table)


def preprocess(obs):
    x = np.zeros([4, 4, layer_count]) # 4X4 의 observation space, 12개의 타일 종류
    for i in range(4):
        for j in range(4):
            if obs[i,j] > 0:
                v = min(obs[i,j], 2**(layer_count-1))
                x[i,j,table[v]] = 1
            else:
                x[i,j,0] = 1
    return x


def build_model():
    dense1 = 128
    dense2 = 128

    x = tf.keras.Input(shape=(4, 4, layer_count))

    conv_a = tf.keras.layers.Conv2D(dense1, kernel_size=(2, 1), activation='relu')(x)
    conv_b = tf.keras.layers.Conv2D(dense1, kernel_size=(1, 2), activation='relu')(x)

    conv_aa = tf.keras.layers.Conv2D(dense2, kernel_size=(2, 1), activation='relu')(conv_a)
    conv_ab = tf.keras.layers.Conv2D(dense2, kernel_size=(1, 2), activation='relu')(conv_a)
    conv_ba = tf.keras.layers.Conv2D(dense2, kernel_size=(2, 1), activation='relu')(conv_b)
    conv_bb = tf.keras.layers.Conv2D(dense2, kernel_size=(1, 2), activation='relu')(conv_b)

    flat = [tf.keras.layers.Flatten()(a) for a in [conv_a, conv_b, conv_aa, conv_ab, conv_ba, conv_bb]]

    concat = tf.keras.layers.Concatenate()(flat)
    dense1 = tf.keras.layers.Dense(256, activation='relu')(concat)
    out = tf.keras.layers.Dense(4, activation='linear')(dense1)

    model = tf.keras.Model(inputs=x, outputs=out)
    model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.0005), loss='mse')
    return model


def append_sample(memory, state, action, reward, next_state, done):
    memory.append([state, action, reward, next_state, done])


def train_model():
    np.random.shuffle(memory)

    len = max_memory // batch_size
    for k in range(len):
        mini_batch = memory[k*batch_size:(k+1)*batch_size]

        states = np.zeros((batch_size, 4, 4, layer_count))
        next_states = np.zeros((batch_size, 4, 4, layer_count))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = model.predict(states)
        next_target = target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + gamma * np.amax(next_target[i])

        model.fit(states, target, batch_size=batch_size, epochs=1)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits

model = build_model()
target_model = build_model()

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
    main_prev_obs = main_env.reset()
    virtual1_prev_obs = virtual1_env.reset()
    virtual2_prev_obs = virtual2_env.reset()
    virtual3_prev_obs = virtual3_env.reset()
    virtual4_prev_obs = virtual4_env.reset()

    score = 0
    step = 0

    not_move_list = np.array([1, 1, 1, 1])

    main_prev_max = np.max(main_prev_obs)
    virtual1_prev_max = np.max(virtual1_prev_obs)
    virtual2_prev_max = np.max(virtual2_prev_obs)
    virtual3_prev_max = np.max(virtual3_prev_obs)
    virtual4_prev_max = np.max(virtual4_prev_obs)

    while True:
        if random.random() < epsilon:
            virtual1_action = virtual1_env.action_space.sample()
        else:
            virtual1_x = preprocess(virtual1_prev_obs)
            virtual1_logits = model.predict(np.expand_dims(virtual1_x, axis=0))[0]
            virtual1_prob = softmax(virtual1_logits)
            virtual1_prob = virtual1_prob * not_move_list
            virtual1_action = np.argmax(virtual1_prob)

        virtual1_obs, virtual1_reward, virtual1_done, _ = virtual1_env.step(virtual1_action)

        if virtual1_reward == 0 and np.array_equal(virtual1_obs, virtual1_prev_obs):
            not_move_list[virtual1_action] = 0
            continue
        else:
            not_move_list = np.array([1, 1, 1, 1])

        virtual1_now_max = np.max(virtual1_obs)
        if virtual1_prev_max < virtual1_now_max:
            virtual1_prev_max = virtual1_now_max
            virtual1_reward = math.log(virtual1_now_max, 2) * 0.1
        else:
            virtual1_reward = 0

        virtual1_reward += np.count_nonzero(virtual1_prev_obs) - np.count_nonzero(virtual1_obs) + 1

        append_sample(memory, preprocess(virtual1_prev_obs), virtual1_action, virtual1_reward, preprocess(virtual1_obs), virtual1_done)

        virtual1_prev_obs = virtual1_obs

        if virtual1_done:
            break

    while True:
        if random.random() < epsilon:
            virtual2_action = virtual2_env.action_space.sample()
        else:
            virtual2_x = preprocess(virtual2_prev_obs)
            virtual2_logits = model.predict(np.expand_dims(virtual2_x, axis=0))[0]
            virtual2_prob = softmax(virtual2_logits)
            virtual2_prob = virtual2_prob * not_move_list
            virtual2_action = np.argmax(virtual2_prob)

        virtual2_obs, virtual2_reward, virtual2_done, _ = virtual2_env.step(virtual2_action)

        if virtual2_reward == 0 and np.array_equal(virtual2_obs, virtual2_prev_obs):
            not_move_list[virtual2_action] = 0
            continue
        else:
            not_move_list = np.array([1, 1, 1, 1])

        virtual2_now_max = np.max(virtual2_obs)
        if virtual2_prev_max < virtual2_now_max:
            virtual2_prev_max = virtual2_now_max
            virtual2_reward = math.log(virtual2_now_max, 2) * 0.1
        else:
            virtual2_reward = 0

        virtual2_reward += np.count_nonzero(virtual2_prev_obs) - np.count_nonzero(virtual2_obs) + 1

        append_sample(memory, preprocess(virtual2_prev_obs), virtual2_action, virtual2_reward, preprocess(virtual2_obs), virtual2_done)

        virtual2_prev_obs = virtual2_obs

        if virtual2_done:
            break

    while True:
        if random.random() < epsilon:
            virtual3_action = virtual3_env.action_space.sample()
        else:
            virtual3_x = preprocess(virtual3_prev_obs)
            virtual3_logits = model.predict(np.expand_dims(virtual3_x, axis=0))[0]
            virtual3_prob = softmax(virtual3_logits)
            virtual3_prob = virtual3_prob * not_move_list
            virtual3_action = np.argmax(virtual3_prob)

        virtual3_obs, virtual3_reward, virtual3_done, _ = virtual3_env.step(virtual3_action)

        if virtual3_reward == 0 and np.array_equal(virtual3_obs, virtual3_prev_obs):
            not_move_list[virtual3_action] = 0
            continue
        else:
            not_move_list = np.array([1, 1, 1, 1])

        virtual3_now_max = np.max(virtual3_obs)
        if virtual3_prev_max < virtual3_now_max:
            virtual3_prev_max = virtual3_now_max
            virtual3_reward = math.log(virtual3_now_max, 2) * 0.1
        else:
            virtual3_reward = 0

        virtual3_reward += np.count_nonzero(virtual3_prev_obs) - np.count_nonzero(virtual3_obs) + 1

        append_sample(memory, preprocess(virtual3_prev_obs), virtual3_action, virtual3_reward, preprocess(virtual3_obs), virtual3_done)

        virtual3_prev_obs = virtual3_obs

        if virtual3_done:
            break

    while True:
        if random.random() < epsilon:
            virtual4_action = virtual4_env.action_space.sample()
        else:
            virtual4_x = preprocess(virtual4_prev_obs)
            virtual4_logits = model.predict(np.expand_dims(virtual4_x, axis=0))[0]
            virtual4_prob = softmax(virtual4_logits)
            virtual4_prob = virtual4_prob * not_move_list
            virtual4_action = np.argmax(virtual4_prob)

        virtual4_obs, virtual4_reward, virtual4_done, _ = virtual4_env.step(virtual4_action)

        if virtual4_reward == 0 and np.array_equal(virtual4_obs, virtual4_prev_obs):
            not_move_list[virtual4_action] = 0
            continue
        else:
            not_move_list = np.array([1, 1, 1, 1])

        virtual4_now_max = np.max(virtual4_obs)
        if virtual4_prev_max < virtual4_now_max:
            virtual4_prev_max = virtual4_now_max
            virtual4_reward = math.log(virtual4_now_max, 2) * 0.1
        else:
            virtual4_reward = 0

        virtual4_reward += np.count_nonzero(virtual4_prev_obs) - np.count_nonzero(virtual4_obs) + 1

        append_sample(memory, preprocess(virtual4_prev_obs), virtual4_action, virtual4_reward, preprocess(virtual4_obs), virtual4_done)

        virtual4_prev_obs = virtual4_obs

        if virtual4_done:
            break

    while True:
        if random.random() < epsilon:
            main_action = main_env.action_space.sample()
        else:
            main_x = preprocess(main_prev_obs)
            main_logits = model.predict(np.expand_dims(main_x, axis=0))[0]
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
            train_model()
            memory = []

            train_count += 1
            if train_count % 4 == 0:
                target_model.set_weights(model.get_weights())

        main_prev_obs = main_obs

        if epsilon > epsilon_min and iteration % 2500 == 0:
            epsilon = epsilon / 1.005

        if main_done:
            break

    scores.append(score)
    steps.append(score)
    max_tile.append(np.max(main_obs))


    if np.mean(scores[-100:]) > 30000:
        model.save('2048_virtual_original_best.h5')
        N = 100
        rolling_mean = [np.mean(scores[x:x + N]) for x in range(len(scores) - N + 1)]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.scatter(range(len(scores)), scores, marker='.')
        plt.subplot(1, 3, 2)
        plt.plot(rolling_mean)
        plt.subplot(1, 3, 3)
        plt.scatter(range(len(max_tile)), max_tile, marker='.')
        plt.show()

    print(i, 'score: ', score, 'step: ', step, 'max tile: ', np.max(main_obs), 'memory len: ', len(memory))

model.save('2048_virtual_original_normal.h5')

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
