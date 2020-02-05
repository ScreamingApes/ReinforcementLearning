import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
import os
import numpy as np
import random
import math
import matplotlib.pyplot as plt

FILENAME = "mountain-weight.h5"


class Agent():

    def __init__(self, action_dim, state_dim, brain_layers=[64, 24], gamma=0.99, learning_rate=0.01, num_samples=32, load_from_file=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.exploration_rate = 0.5
        self.exploration_min = 0.1
        self.exploration_decay = 0.995
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        self.num_samples = num_samples
        self.brain = self.__init_brain(brain_layers, load_from_file)

    def __init_brain(self, brain_layers, load_from_file):
        model = Sequential()
        model.add(
            Dense(brain_layers[0], input_dim=self.state_dim, activation='relu'))
        for n in brain_layers[1:]:
            model.add(Dense(n, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
        if load_from_file and os.path.isfile(FILENAME):
            model.load_weights(FILENAME)
            self.exploration_rate = self.exploration_min
            print(f"[X] Loaded brain in {FILENAME}")
        return model

    def save_brain(self):
        self.brain.save(FILENAME)
        print(f"[X] Saved brain in {FILENAME}")

    def pick_action(self, state, rand=True):
        if rand and random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(0, self.action_dim)
        return np.argmax(self.brain.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.num_samples:
            return

        minibatches = random.sample(self.memory, self.num_samples)

        for current_state, action, reward, next_state, done in minibatches:
            target = reward if done else reward + self.gamma * \
                np.max(self.brain.predict(next_state)[0])

            target_q = self.brain.predict(current_state)
            target_q[0][action] = target

            self.brain.fit(current_state, target_q, verbose=0, epochs=1)

        if self.exploration_rate >= self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class MountainCar():

    def __init__(self, episodes=10000, load_from_file=False):
        self.env = gym.make("MountainCar-v0").env
        self.episodes = episodes
        self.max_iterations = 1000
        self.space_dim = self.env.observation_space.shape[0]
        self.agent = Agent(self.env.action_space.n,
                           self.space_dim, load_from_file=load_from_file)
        self.max_reward = 10
        self.evaluation_data = []

    def action_to_string(self, action):
        return ('LEFT', 'NO ACTION', 'RIGHT')[action]

    def fit(self, visualize=False, evaluation=[]):
        pbar = tqdm(range(self.episodes))

        try:

            score_prec = 0
            completed = 0

            for e in pbar:

                current_state = self.env.reset()
                current_state = np.reshape(current_state, (1, self.space_dim))
                done = False
                score = 0

                max_pos = self.env.observation_space.low[0]

                for i in range(self.max_iterations):

                    if visualize and e % 500 == 0:
                        self.env.render()

                    action = self.agent.pick_action(current_state)
                    next_state, reward, done, _ = self.env.step(action)

                    next_state = np.reshape(next_state, (1, self.space_dim))

                    if next_state[0][0] > max_pos:
                        max_pos = next_state[0][0]
                        reward = 1

                    if done:
                        reward = 1000

                    # if visualize and e % 500 == 0:
                    #    pbar.write(self.action_to_string(action))

                    self.agent.remember(current_state, action,
                                        reward, next_state, done)

                    current_state = next_state
                    score += reward

                    if done and i != self.max_iterations - 1:
                        completed += 1
                        break

                    pbar.set_description("C=%d | PS=%d | AS=%d | ER=%.2f" % (
                        completed, score_prec, score, self.agent.exploration_rate))

                score_prec = score
                self.agent.learn()

                if e in evaluation:
                    self.evaluate(e+1)

        finally:
            self.agent.save_brain()

    def evaluate(self, training_episode, episodes=100):

        pbar = tqdm(range(episodes))
        pbar.set_description(f"Evaluating after {training_episode} ep")

        finished = 0
        for _ in pbar:

            state = self.env.reset().reshape((1, self.space_dim))

            for i in range(self.max_iterations):

                action = self.agent.pick_action(state)
                state, _, done, _ = self.env.step(action)

                state = state.reshape((1, self.space_dim))

                if done and i != self.max_iterations - 1:
                    finished += 1
                    break

        self.evaluation_data.append(finished / episodes)

    def visualize(self):

        state = self.env.reset().reshape((1, self.space_dim))
        frames = 0
        finished = 0
        total = 0

        try:
            while True:
                self.env.render()

                action = self.agent.pick_action(state, rand=False)
                state, _, done, _ = self.env.step(action)

                print(self.action_to_string(action))

                state = state.reshape((1, self.space_dim))
                frames += 1

                if done:
                    if frames < 200:
                        finished += 1
                    frames = 0
                    total += 1
                    state = self.env.reset().reshape((1, self.space_dim))

                if frames == 200:
                    frames = 0
                    state = self.env.reset().reshape((1, self.space_dim))
        finally:
            print(
                f"Total: {total} | finished: {finished} | {finished/total*100}%")


def plot(eps, data):

    fig, ax = plt.subplots()
    ax.plot(eps, data)

    ax.set(xlabel='episodes', ylabel='success')
    ax.grid()

    fig.savefig("plot.png")
    plt.show()


if __name__ == "__main__":
    episodes_training = 10000
    step_eval = 500
    evaluation_steps = [0] + [i * step_eval -
                              1 for i in range(1, int(episodes_training / step_eval) + 1)]

    sim = MountainCar(episodes=episodes_training, load_from_file=False)
    sim.fit(visualize=False, evaluation=evaluation_steps)

    plot(evaluation_steps, sim.evaluation_data)
