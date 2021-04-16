import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Implementation of D2RL: Deep Dense Architectures in Reinforcement Learning (D2RL)
# Paper: https://arxiv.org/abs/2010.09163

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain("relu")
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
		super(Actor, self).__init__()

		in_dim = hidden_dim + state_dim
		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(in_dim, hidden_dim)
		self.l3 = nn.Linear(in_dim, hidden_dim)
		self.l4 = nn.Linear(in_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, action_dim)
		
		self.max_action = max_action
		self.apply(weight_init)

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = torch.cat([a, state], dim=1)
		a = F.relu(self.l2(a))
		a = torch.cat([a, state], dim=1)
		a = F.relu(self.l3(a))
		a = torch.cat([a, state], dim=1)
		a = F.relu(self.l4(a))
		return self.max_action * torch.tanh(self.l5(a))


# Returns Q-value for given state/action pairs
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Critic, self).__init__()

		in_dim = hidden_dim + state_dim + action_dim
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(in_dim, hidden_dim)
		self.l3 = nn.Linear(in_dim, hidden_dim)
		self.l4 = nn.Linear(in_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l6 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l7 = nn.Linear(in_dim, hidden_dim)
		self.l8 = nn.Linear(in_dim, hidden_dim)
		self.l9 = nn.Linear(in_dim, hidden_dim)
		self.l10 = nn.Linear(hidden_dim, 1)
		self.apply(weight_init)

	def forward(self, state, action):
		sa = torch.cat([state, action], dim=1)

		q1 = F.relu(self.l1(sa))
		q1 = torch.cat([q1, sa], dim=1)
		q1 = F.relu(self.l2(q1))
		q1 = torch.cat([q1, sa], dim=1)
		q1 = F.relu(self.l3(q1))
		q1 = torch.cat([q1, sa], dim=1)
		q1 = F.relu(self.l4(q1))
		q1 = self.l5(q1)

		q2 = F.relu(self.l6(sa))
		q2 = torch.cat([q2, sa], dim=1)
		q2 = F.relu(self.l7(q2))
		q2 = torch.cat([q2, sa], dim=1)
		q2 = F.relu(self.l8(q2))
		q2 = torch.cat([q2, sa], dim=1)
		q2 = F.relu(self.l9(q2))
		q2 = self.l10(q2)

		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], dim=1)

		q1 = F.relu(self.l1(sa))
		q1 = torch.cat([q1, sa], dim=1)
		q1 = F.relu(self.l2(q1))
		q1 = torch.cat([q1, sa], dim=1)
		q1 = F.relu(self.l3(q1))
		q1 = torch.cat([q1, sa], dim=1)
		q1 = F.relu(self.l4(q1))
		q1 = self.l5(q1)

		return q1


class TD3_D2RL(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hidden_dim=256,
	):
		self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		action = self.actor(state)
		return action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample batches from replay buffer  
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(min=-self.noise_clip, max=self.noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(min=-self.max_action, max=self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# save the model
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	# load the model
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		