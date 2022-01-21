import gym_2048
import gym
import numpy as np
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool


def preprocess(obs):
    layer_count = 14  # 2 ~ 2048 까지의 타일 종류의 수
    table = {2 ** i: i for i in range(layer_count)}

    x = np.zeros([4, 4, layer_count]) # 4X4 의 observation space, 12개의 타일 종류
    for i in range(4):
        for j in range(4):
            if obs[i,j] > 0:
                v = min(obs[i,j], 2**(layer_count-1))
                x[i,j,table[v]] = 1
            else:
                x[i,j,0] = 1
    return x


def episode_start(env, model, epsilon, not_move_list, memory):
    prev_obs = env.reset()
    prev_max = np.max(prev_obs)
    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            x = preprocess(prev_obs)
            logits = model.predict(np.expand_dims(x, axis=0))[0]
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
        memory.append([preprocess(prev_obs), action, reward, preprocess(obs), done])
        prev_obs = obs

        if done:
            break


def build_model(layer_count):
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


def train_model(layer_count):
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


if __name__ == '__main__':
    main_env = gym.make('2048-v0')
    virtual1_env = gym.make('2048-v0')
    virtual2_env = gym.make('2048-v0')
    virtual3_env = gym.make('2048-v0')
    virtual4_env = gym.make('2048-v0')

    gamma = 0.9
    batch_size = 512
    max_memory = batch_size * 8
    memory = []

    model = build_model(14)
    target_model = build_model(14)

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

        not_move_list = np.array([1, 1, 1, 1])

        p = Pool(4)
        ret = p.starmap(episode_start, [(virtual1_env, model, epsilon, not_move_list, memory),
                                        (virtual2_env, model, epsilon, not_move_list, memory),
                                        (virtual3_env, model, epsilon, not_move_list, memory),
                                        (virtual4_env, model, epsilon, not_move_list, memory)])
        print(ret)

        p.close()
        p.join()

        main_prev_obs = main_env.reset()
        main_prev_max = np.max(main_prev_obs)
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

            memory.append([preprocess(main_prev_obs), main_action, main_reward, preprocess(main_obs), main_done])

            if len(memory) >= max_memory:
                train_model(14)
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
            model.save('2048_virtual_multiprocess_original_best.h5')
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

    model.save(model, '2048_virtual_multiprocess_original_normal.h5')

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

