import numpy as np
import random
from collections import namedtuple, deque

from model import BananaNet, VisualBananaNet

import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BananAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agent_type=BananaNet, memory=None, checkpoint_filename=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        print('Agent type:', agent_type)
        self.qnetwork_local = agent_type(state_size, action_size).to(device)
        if checkpoint_filename:
            print('Loading checkpoint', checkpoint_filename)
            checkpoint = torch.load(checkpoint_filename)
            self.qnetwork_local.load_state_dict(checkpoint)
        self.qnetwork_target = agent_type(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = memory
        if memory is None:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.double_q_learn(experiences, GAMMA)

    def act(self, state, eps=0., evaluate=False):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if evaluate:
            self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def double_q_learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

                Params
                ======
                    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
                    gamma (float): discount factor
                """
        states, actions, rewards, next_states, dones = experiences
        # print('NEXT STATES:', next_states.shape)
        local_max_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        outputs = self.qnetwork_target(next_states).detach()
        target_max = outputs.gather(1, local_max_action)
        targets = rewards + gamma * target_max * (1 - dones)

        # forward pass local network
        output = self.qnetwork_local(states).gather(1, actions)

        # calculate loss & gradient for local network
        self.optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(output, targets)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # TODO: probably have to detach target_max?
        target_max = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        targets = rewards + gamma * target_max * (1 - dones)

        # forward pass local network
        output = self.qnetwork_local(states).gather(1, actions)

        # calculate loss & gradient for local network
        self.optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(output, targets)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print('ADDING EXPERIENCE TO BUFFER', state.shape, next_state.shape)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class VisualBananAgent(BananAgent):
    def __init__(self, state_size, action_size,  memory=None, checkpoint_filename=None):
        super().__init__(state_size,
                         action_size,
                         agent_type=VisualBananaNet,
                         memory=memory,
                         checkpoint_filename=checkpoint_filename)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.double_q_learn(experiences, GAMMA)