# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# Using Policy Iteration to Solve FrozenLake8x8 Problem from OpenAI Gym

import numpy as np
import gym
from gym import wrappers

def run_episode(env, policy, gamma=1.0, render=False):

    obs = env.reset()
    total_reward = 0
    step_idx = 0

    while True:
        if render:
            env.render()

        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx) * reward
        step_idx += 1

        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [
        run_episode(env, policy, gamma, False) for _ in range(n)
    ]

    return np.mean(scores)

def extract_policy(env, v, gamma = 1.0):
    env_nS = env.nrow * env.ncol        # possible states
    env_nA = 4                          # possible actions -> UP, DOWN, LEFT, RIGHT

    policy = np.zeros((env.nrow, env.ncol))           # develop a sample policy

    for s in range(env_nS):
        q_sa = np.zeros(env_nA)
        for a in range(env_nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])

