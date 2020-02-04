import gym
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
import random
from tqdm import tqdm
from collections import deque
from time import sleep
import os

FILENAME = "model-cartpole.h5"


class CartPole():

    def __init__(self, alpha=0.01, gamma=0.95, memory=2000, episodes=5000, num_sample=32, load_from_file=False):
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.exploration_rate = 1
        self.exploration_decay = 0.995
        self.exploration_min = 0.001
        self.num_sample = num_sample
        self.memory = deque(maxlen=2000)
        self.env = gym.make("CartPole-v0").env
        self.brain = self.__create_brain(load_from_file)

    def __create_brain(self, load_from_file):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        if load_from_file and os.path.isfile(FILENAME):
            model.load_weights(FILENAME)
            self.exploration_rate = self.exploration_min
        return model

    def save_brain(self):
        self.brain.save(FILENAME)

    def pick_action(self, state, rand=True):
        if np.random.rand() <= self.exploration_rate:
            return self.env.action_space.sample()
        return np.argmax(self.brain.predict(state)[0])

    def __learn(self):
        if len(self.memory) < self.num_sample:
            return

        minibatch = random.sample(self.memory, self.num_sample)

        for current_state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.max(self.brain.predict(next_state)[0])

            prediction = self.brain.predict(current_state)
            prediction[0][action] = target
            self.brain.fit(current_state, prediction, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def fit(self, save_brain=True):
        pbar = tqdm(range(self.episodes))

        try:
            score_prec = 0
            for _ in pbar:

                current_state = self.env.reset()
                current_state = np.reshape(current_state, (1, 4))
                done = False

                score = 0

                while not done:
                    self.env.render()
                    action = self.pick_action(current_state)
                    next_state, reward, done, _ = self.env.step(action)

                    next_state = np.reshape(next_state, (1, 4))

                    self.memory.append(
                        (current_state, action, reward, next_state, done))

                    current_state = next_state
                    score += 1

                    pbar.set_description(f"PS={score_prec} | AS={score}")

                score_prec = score
                self.__learn()

            if save_brain:
                self.save_brain()

        finally:
            self.save_brain()

        pbar.close()

    def visualize(self):

        state = np.reshape(self.env.reset(), (1, 4))

        while True:
            self.env.render()
            action = self.pick_action(state, rand=False)
            state, _, done, _ = self.env.step(action)
            state = np.reshape(state, (1, 4))
            # sleep(0.1)

            if done:
                state = np.reshape(self.env.reset(), (1, 4))


if __name__ == "__main__":
    c = CartPole(load_from_file=True)
    c.fit()
    c.visualize()
