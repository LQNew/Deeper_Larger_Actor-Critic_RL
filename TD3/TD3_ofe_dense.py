import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from OFENet.ofenet import DensenetBlock

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Implementation of Online Feature Extractor Network (OFENet)
# Paper: https://arxiv.org/abs/2003.01629
# Reimplementation of Training Larger Networks for Deep Reinforcement Learning, add `Densenet` structure in Actor and Critic network.
# Paper: https://arxiv.org/abs/2102.07920

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
	def __init__(
		self, 
		state_features_dim, action_dim, 
		max_action, 
		num_layers=2, activation="swish", hidden_dim=2048
	):
		super(Actor, self).__init__()
		
		hidden_layers = []
		for i in range(num_layers):
			hidden_layers.append(state_features_dim + i*hidden_dim)
		output_features_dim = hidden_layers[-1] + hidden_dim

		self.hidden_blocks = nn.ModuleList([])
		self.hidden_blocks.extend([
			DensenetBlock(
				input_dim=hidden_layers[i], output_dim=hidden_dim, \
				activation=activation, layernorm=False
			) for i in range(num_layers)
		])

		self.fc = nn.Linear(output_features_dim, action_dim)
		self.max_action = max_action
		self.apply(weight_init)

	def forward(self, state_features):
		features = state_features
		for hidden_block in self.hidden_blocks:
			features = hidden_block(features)
		return self.max_action * torch.tanh(self.fc(features))


# Returns Q-value for given state/action pairs
class Critic(nn.Module):
	def __init__(
		self, 
		state_action_features_dim, 
		num_layers=2, activation="swish", hidden_dim=2048
	):
		super(Critic, self).__init__()

		hidden_layers = []

		for i in range(num_layers):
			hidden_layers.append(state_action_features_dim + i*hidden_dim)
		output_features_dim = hidden_layers[-1] + hidden_dim

		# Q1 architecture
		self.hidden_blocks_1 = nn.ModuleList([])
		self.hidden_blocks_1.extend([
			DensenetBlock(
				input_dim=hidden_layers[i], output_dim=hidden_dim, \
				activation=activation, layernorm=False
			) for i in range(num_layers)
		])
		self.fc1 = nn.Linear(output_features_dim, 1)

		# Q2 architecture
		self.hidden_blocks_2 = nn.ModuleList([])
		self.hidden_blocks_2.extend([
			DensenetBlock(
				input_dim=hidden_layers[i], output_dim=hidden_dim, \
				activation=activation, layernorm=False
			) for i in range(num_layers)
		])
		self.fc2 = nn.Linear(output_features_dim, 1)
		self.apply(weight_init)

	def forward(self, hidden_state_action_features):
		q1_features = hidden_state_action_features
		for hidden_block_1 in self.hidden_blocks_1:
			q1_features = hidden_block_1(q1_features)
		q1 = self.fc1(q1_features)

		q2_features = hidden_state_action_features
		for hidden_block_2 in self.hidden_blocks_2:
			q2_features = hidden_block_2(q2_features)
		q2 = self.fc2(q2_features)
		return q1, q2


	def Q1(self, hidden_state_action_features):
		q1_features = hidden_state_action_features
		for hidden_block_1 in self.hidden_blocks_1:
			q1_features = hidden_block_1(q1_features)
		q1 = self.fc1(q1_features)
		return q1


class TD3_OFE_DENSE(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		feature_extractor,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hidden_dim=2048,
		activation="swish",
	):
		self._extractor = feature_extractor
		self.device = device

		self.actor = Actor(
			state_features_dim=self._extractor.state_features_dim, action_dim=action_dim, max_action=max_action, \
			activation=activation, hidden_dim=hidden_dim).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(
			state_action_features_dim=self._extractor.state_action_features_dim, \
			activation=activation, hidden_dim=hidden_dim).to(self.device)
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
		state = torch.as_tensor(state.reshape(1, -1), device=self.device, dtype=torch.float32)
		state_features = self._extractor.features_of_states(state)
		action = self.actor(state_features)
		return action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample batches from replay buffer  
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		with torch.no_grad():
			_next_state_features = self._extractor.features_of_states(next_state)  # compute Z(o_{t+1})
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(min=-self.noise_clip, max=self.noise_clip)
			_next_action = (self.actor_target(_next_state_features) + noise).clamp(min=-self.max_action, max=self.max_action)
			sa_hidden_target_state = self._extractor.features_of_states_actions(next_state, _next_action)  # compute Z(o_{t+1}, a_{t+1})

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(sa_hidden_target_state)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		sa_hidden_state = self._extractor.features_of_states_actions(state, action)
		current_Q1, current_Q2 = self.critic(sa_hidden_state)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			_state_features = self._extractor.features_of_states(state)  # Z(o_t)
			policy_action = self.actor(_state_features)
			sa_policy_hidden_state = self._extractor.features_of_states_actions(state, policy_action) 
			actor_loss = -self.critic.Q1(sa_policy_hidden_state).mean()
			
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