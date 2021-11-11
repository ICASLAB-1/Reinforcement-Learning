import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym_2048
from tkinter import *
import time

table = {2**i:i for i in range(14)}


SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", \
                         32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", \
                         512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}

CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2", \
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2", \
                   512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}

FONT = ("Verdana", 40, "bold")


class GameGrid(Frame):
    def __init__(self, matrix):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')

        self.grid_cells = []
        self.init_grid()
        self.update_grid_cells(matrix)

        self.wait_visibility()
        self.matrix = matrix
    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4,
                          height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def update_grid_cells(self, matrix):
        self.matrix = matrix
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number],
                                                    fg=CELL_COLOR_DICT[new_number])

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits


def preprocess(obs):
    x = np.zeros([4, 4, 14]) # 4X4 의 observation space, 12개의 타일 종류
    for i in range(4):
        for j in range(4):
            if obs[i,j] > 0:
                v = min(obs[i,j], 2**(14-1))
                x[i,j,table[v]] = 1
            else:
                x[i,j,0] = 1
    return x


def display_mode(model_name):
    env = gym.make('2048-v0')
    obs_list = []
    model = tf.keras.models.load_model(model_name)
    for p in range(5):
        for i in range(100):
            prev_obs = env.reset()
            score = 0
            step = 0
            not_move_list = np.array([1, 1, 1, 1])
            root = Tk()
            gamegrid = GameGrid(prev_obs)
            time.sleep(5)
            while True:
                obs_list.append(prev_obs)
                gamegrid.update_grid_cells(prev_obs)
                root.update()
                time.sleep(0.1)
                x = preprocess(prev_obs)
                logits = model.predict(np.expand_dims(x, axis=0))[0]
                prob = softmax(logits)
                prob = prob * not_move_list
                action = np.argmax(prob)
                obs, reward, done, _ = env.step(action)

                score += reward
                step += 1

                prev_obs = obs

                if done:
                    break
            time.sleep(30)
            root.destroy()
            root.mainloop()


def no_display_mode(model_name):
    env = gym.make('2048-v0')
    scores = []
    max_tiles = []
    model = tf.keras.models.load_model(model_name)

    for p in range(5):
        for i in range(100):
            prev_obs = env.reset()
            score = 0
            step = 0
            not_move_list = np.array([1, 1, 1, 1])
            while True:
                x = preprocess(prev_obs)
                logits = model.predict(np.expand_dims(x, axis=0))[0]
                prob = softmax(logits)
                prob = prob * not_move_list
                action = np.argmax(prob)
                obs, reward, done, _ = env.step(action)

                score += reward
                step += 1

                prev_obs = obs

                if done:
                    break

            scores.append(score)
            max_tiles.append(np.max(obs))
            print(i, 'score: ', score, 'step: ', step, 'max tile: ', np.max(obs))

    max_list = [2 ** i for i in range(7, 13)]
    max_index = [0, 0, 0, 0, 0, 0]

    for j in range(len(max_tiles)):
        for k in range(6):
            if max_tiles[j] == max_list[k]:
                max_index[k] += 1

    l = np.arange(6)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(scores)), scores, marker='.')
    plt.subplot(1, 2, 2)
    plt.bar(l, max_index)
    plt.xticks(l, max_list)
    plt.show()


def main():
    model_name = '2048_virtual_original_normal.h5'
    display_mode(model_name)


if __name__=="__main__":
    main()



