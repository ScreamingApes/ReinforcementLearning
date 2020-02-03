import gym
import numpy as np
from math import floor
from tqdm import tqdm
from random import uniform
from time import sleep
from sys import argv

outfile = "cartpole.npy"


class CartPole():

    def __init__(self, buckets=(10, 10, 100, 100,), episodes=10_000, epsilon=0.1, gamma=0.9, alpha=0.2):
        self.env = gym.make("CartPole-v0").env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.buckets = buckets
        self.episodes = episodes

        self.q_table = np.zeros(buckets + (self.env.action_space.n,))
        self.env.observation_space.low = np.array(
            self.env.observation_space.low, dtype=np.float64)
        self.env.observation_space.high = np.array(
            self.env.observation_space.high, dtype=np.float64)

    def discretize(self, obs):
        step = [(self.env.observation_space.high[i]-self.env.observation_space.low[i]
                 ) / self.buckets[i] for i, _ in enumerate(self.env.observation_space.high)]
        disc_obs = [int(floor((obs[i] - self.env.observation_space.low[i]) / step[i]))
                    for i, _ in enumerate(obs)]
        return tuple([min(max(0, do), bn-1) for do, bn in zip(disc_obs, self.buckets)])

    def pick_action(self, rand=True):
        if rand and uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_curr = self.discretize(self.env.state)
            return np.argmax(self.q_table[state_curr])

    def update_q_table(self, old_state, action, new_state, reward):
        self.q_table[old_state][action] += self.alpha * (reward + self.gamma * np.max(
            self.q_table[new_state] - self.q_table[old_state][action]))

    def fit(self):

        for _ in tqdm(range(self.episodes)):
            state_curr = self.discretize(self.env.reset())
            done = False

            while not done:
                action = self.pick_action()
                obs, reward, done, _ = self.env.step(action)
                state_new = self.discretize(obs)

                self.update_q_table(state_curr, action, state_new, reward)

                state_curr = state_new

    def visualize(self):
        self.env.reset()

        while True:
            action = self.pick_action(rand=False)
            self.env.step(action)

            self.env.render()
            sleep(.1)

        # self.env.close()

    def save_q_table(self):
        np.save(outfile, self.q_table)
        print("Saved Q table in", outfile)

    def load_q_table(self):
        self.q_table = np.load(outfile)
        print("Loaded Q table from", outfile)


if __name__ == "__main__":
    c = CartPole()

    if len(argv) > 1 and argv[1] == "load":
        c.load_q_table()
        c.visualize()
    elif len(argv) > 1 and argv[1] == "save":
        c.fit()
        c.save_q_table()
    else:
        print("Usage: %s [save|load]" % argv[0])
