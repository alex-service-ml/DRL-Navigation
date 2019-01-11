# Code from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
# Original credit to jaromiru for his implementation of this code!
# Copyright (c) 2018 Jaromír Janisch

import numpy as np
import random
from collections import namedtuple

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:   # stored as ( s, a, r, s_, done ) in SumTree
    e = 0.01  # prevent division by 0
    a = 0.6

    def __init__(self, capacity, batch_size, n=1e5, b=0.4):
        self.max_priority = 1.      # Set initial max priority
        self.batch_size = batch_size
        self.n = n
        self.b = b
        self.tree = SumTree(capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, state, action, reward, next_state, done):
        p = self._getPriority(self.max_priority)  # Adjusted to follow the original PER paper
        self.tree.add(p, self.experience(state, action, reward, next_state, done))

    def sample(self):
        indices, experiences = [], []
        segment = self.tree.total() / self.batch_size  # Divide priority into equal segments

        states = []  # TODO: write as a tensor directly?
        actions = []
        rewards = []
        next_states = []
        dones = []
        weights = []
        for i in range(self.batch_size):  # Select an experience from each segment
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)

            (idx, p, data) = self.tree.get(s)

            states.append(data.state)
            actions.append(data.action)
            rewards.append(data.reward)
            next_states.append(data.next_state)
            dones.append(data.done)
            weights.append(p)
            # TODO: Rest of the experience values

            indices.append(idx)
            experiences.append((*data, p)) # TODO: match behavior of ReplayBuffer
            # batch.append( (idx, data) )

        weights = (self.n * np.vstack(weights)) ** -self.b
        weights /= np.max(weights)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)

        return indices, (states, actions, rewards, next_states, dones, weights)  # batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
