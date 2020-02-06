import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
import random

FILENAME = "pong-model.h5"


class Agent_Pong():

    def __init__(self, action_dim, state_dim, gamma=0.99, learning_rate=0.005, num_samples=32, load_from_file=False):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.exploration_rate = 1
        self.exploration_min = 0.1
        self.exploration_decay = 0.9995
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=1_000_000)
        self.num_samples = num_samples
        self.brain = self.init_brain()

    def init_brain(self):
        model = Sequential()
        model.add(Conv2D(16, 8, strides=(4, 4), activation='relu',
                         input_shape=self.state_dim))
        model.add(Conv2D(32, 4, strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='linear'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.learning_rate))
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


class Pong():

    def __init__(self, episodes=100_000):
        self.env = gym.make("Pong-v0").env
        self.episodes = episodes
        self.agent = Agent_Pong(self.env.action_space.n, (160, 160, 4))
        self.last_four_frames = deque(maxlen=4)

    # converts an rgb image to gray scale image
    def __rgb2gray(self, rgb_img):
        return np.dot(rgb_img, [0.2989, 0.5870, 0.1140])

    def __reduce_size(self, img):
        # remove score bar and bottom bar
        return img[34:194]

    def __img2net(self, img):
        return self.__reduce_size(self.__rgb2gray(img))

    # phi function in Mnih et al paper
    def get_state(self):
        return np.array(self.last_four_frames)

    def __add_frame(self, f):
        self.last_four_frames.append(self.__img2net(f))

    def fit(self, visualize=False):

        for e in tqdm(range(self.episodes)):

            # take the first four frames and convert them to gray scale and reduce size
            self.__add_frame(self.env.reset())
            for _ in range(3):
                action = self.env.action_space.sample()
                state, _, _, _ = self.env.step(action)
                self.__add_frame(state)

            done = False

            while not done:

                if visualize and e % 500 == 0:
                    self.env.render()

                current_state = self.get_state()
                action = self.agent.pick_action(current_state)
                next_state, reward, done, _ = self.env.step(action)

                self.__add_frame(next_state)
                next_state = self.get_state()

                self.agent.remember(current_state, action,
                                    reward, next_state, done)

                self.agent.learn()


if __name__ == "__main__":
    sim = Pong()
    sim.fit(visualize=True)
