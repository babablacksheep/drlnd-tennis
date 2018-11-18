import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 1e-3          # learning rate of the actor
LR_CRITIC = 1e-3         # learning rate of the critic
WEIGHT_DECAY = 1e-5      # L2 weight decay
UPDATE_EVERY = 100       # how often to update the network
LEARN_TIMES = 50         # how many times to learn each avtive step
NOISE_REDUCE = 0.9997    # reduce factor of noise in action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, maddpg_agent, seed=10):
        self.maddpg_agent = maddpg_agent
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def transpose_to_tensor(self, input_list):
        make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
        out_tensor = torch.stack(list(map(make_tensor, input_list)), dim=0)
        return out_tensor
        
    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object.
        Params
        ======
            obs_all_agents (list, array): list of observations for each agent
        """

        actions = np.array([agent.act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)])
        return actions

    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.noise.reset()

    def denoise(self, noise_reduce):
        """Reduce the noise in action"""
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.noise.scale = ddpg_agent.noise.scale * noise_reduce

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def step(self, obs_tuple, actions_tuple, rewards_tuple, next_obs_tuple, dones_tuple):
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Params
        ======
            obs_tuple, actions_tuple, rewards_tuple, next_obs_tuple, dones_tuple: s, a, r, s', done tuples from environment
        """
        # Save 
        transition = obs_tuple, actions_tuple, rewards_tuple, next_obs_tuple, dones_tuple
        self.memory.push(transition)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Learn, if enough samples are available in memory every UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and (not self.t_step):
            self.denoise(NOISE_REDUCE)
            for _ in range(LEARN_TIMES):
                experiences = self.memory.sample()         
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        obs_tuple, actions_tuple, rewards_tuple, next_obs_tuple, dones_tuplet = experiences
        state = [ob.ravel() for ob in obs_tuple]
        action = [ac.ravel() for ac in actions_tuple]
        next_state = [next_ob.ravel() for next_ob in next_obs_tuple]
        target_actions = []
        transpose_obs_tuple = []

        for i,agent in enumerate(self.maddpg_agent):
            next_obs = self.transpose_to_tensor([next_ob[i] for next_ob in next_obs_tuple])
            next_obs = next_obs.to(device)
            target_actions.append(agent.actor_target(next_obs))
            transpose_obs_tuple.append([ob[i] for ob in obs_tuple])
        target_actions = torch.cat(target_actions, dim=1)

        for i in range(len(self.maddpg_agent)):
            agent = self.maddpg_agent[i]
            reward = [re[i] for re in rewards_tuple]
            
            done = [do[i] for do in dones_tuplet]
            experiences = [transpose_obs_tuple, state, action, reward, next_state, done, target_actions]
            self.single_learn(agent, experiences, gamma, i)


    def single_learn(self, agent, experiences, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples for an agent.
           critic loss = batch mean of (y- Q(s,a) from target network)^2
           y = reward of this timestep + discount * Q(st+1,at+1) from target network

        Params
        ======
            agent (object): the agent to learn in this turn
            experiences (Tuple): tuple of (st, s, a, ri, s', done, a') tuples 
            gamma (float): discount factor
        """
        state, action, reward, next_state, done = map(self.transpose_to_tensor, experiences[1:-1])
        transpose_obs_tuple = experiences[0]
        target_actions = experiences[-1]
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_state = next_state.to(device)
        reward = reward.to(device)
        done = done.to(device)
        agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            Q_targets_next = agent.critic_target(next_state, target_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = reward.view(-1, 1) + gamma * Q_targets_next * (1 - done.view(-1, 1))
        # Compute Q expected for current states (y_i)
        action = action.to(device)
        state = state.to(device)
        Q_expected = agent.critic_local(state, action)

        # Compute critic loss
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(Q_expected, Q_targets.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # detach the other agents to save computation
        # saves some time for computing derivative
        actions_pred = [ self.maddpg_agent[i].actor_local(self.transpose_to_tensor(ob).to(device)) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(self.transpose_to_tensor(ob).to(device)).detach()
                   for i, ob in enumerate(transpose_obs_tuple) ]
        actions_pred = torch.cat(actions_pred, dim=1)
        agent.actor_optimizer.zero_grad()
        actor_loss = -agent.critic_local(state, actions_pred).mean()
        # Minimize the loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),1)
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)



class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed, 
                 lr_actor=LR_ACTOR, lr_critic=LR_CRITIC,weight_decay= WEIGHT_DECAY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of entire state
            action_size (int): dimension of each action
            random_seed (int): random seed
            n_agent (int): number of agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*2, action_size*2, random_seed).to(device)
        self.critic_target = Critic(state_size*2, action_size*2, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Add noise process to agent
        self.noise = OUNoise(action_size, random_seed**2)

    def act(self, obs, add_noise=True):
        """Returns actions for given state as per current policy."""

        obs = torch.from_numpy(np.expand_dims(obs,0)).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(obs).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.squeeze(np.clip(action, -1, 1),axis=0)

    def target_act(self, obs):
        """get target network actions from the agent in the MADDPG object """
        obs = torch.from_numpy(np.expand_dims(obs,0)).float().to(device)
        action = self.actor_target(obs)
        return action


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, scale=1.0):
        """Initialize parameters and noise process."""
        self.scale = scale
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.np_seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


class ReplayBuffer:
    def __init__(self, size, batch_size, seed):
        self.size = size
        self.batch_size = batch_size
        self.deque = deque(maxlen=self.size)
        self.seed = random.seed(seed)

    def transpose_list(self, mylist):
        return list(map(list, zip(*mylist)))

    def push(self, transition):
        """push into the buffer"""
        input_to_buffer = self.transpose_list(transition)
        self.deque.append(transition)

    def sample(self):
        """sample from the buffer"""
        samples = random.sample(self.deque, self.batch_size)
        return self.transpose_list(samples)

    def __len__(self):
        return len(self.deque)
