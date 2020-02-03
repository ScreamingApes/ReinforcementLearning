import gym
from IPython.display import clear_output
from time import sleep
from tqdm import tqdm
import random
import numpy as np

EPISODES = 100_000
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.6

# tutorial: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

def print_frames(frames, s=0.1):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i+1}/{len(frames)}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(s)


def random_actions():

    env = gym.make("Taxi-v3").env

    env.reset()

    epochs = 0
    penalties, reward = 0, 0

    frames = []

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        if reward == -10:
            penalties += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })

        epochs += 1

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))

    print_frames(frames)


def pick_action(env, q_table, rand=True):
    if rand and random.uniform(0, 1) < EPSILON:
        # explore
        return env.action_space.sample()
    else:
        # exploit
        return np.argmax(q_table[env.s])


def q_learning():
    env = gym.make("Taxi-v3").env
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # training agent
    for _ in tqdm(range(EPISODES)):
        state = env.reset()
        done = False

        while not done:
            action = pick_action(env, q_table)

            next_state, reward, done, _ = env.step(action)

            old_qvalue = q_table[state, action]
            next_max = np.max(q_table[next_state])

            q_table[state, action] = (1-ALPHA)*old_qvalue + ALPHA*(reward + GAMMA * next_max)

            state = next_state
        
    print("Training finished")
    return q_table

def q_evaluate(q_table, episodes=100):
    env = gym.make("Taxi-v3").env
    total_steps, total_penalties = 0, 0

    for _ in tqdm(range(episodes)):
        env.reset()
        step, reward, penalties = 0, 0, 0

        done = False

        while not done:
            action = pick_action(env, q_table, False)
            _, reward, done, _ = env.step(action)

            if reward == -10:
                penalties += 1

            step += 1

        total_steps += step
        total_penalties += penalties

    print(f"Results after {episodes} episodes")
    print(f"Average steps per episode: {total_steps / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")

def q_visualize(q_table):
    env = gym.make("Taxi-v3").env
    state = env.reset()

    done = False

    frames = []

    while not done:
        action = pick_action(env, q_table, False)
        state, reward, done, _ = env.step(action)

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })
    
    print_frames(frames, 0.5)

if __name__ == "__main__":
    # random_actions()
    q_table = q_learning()
    q_evaluate(q_table)
    input()
    q_visualize(q_table)
