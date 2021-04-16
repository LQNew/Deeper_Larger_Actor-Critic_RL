import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Soft Actor-Critic (SAC)
# Paper: https://arxiv.org/abs/1801.01290
# Implementation of Online Feature Extractor Network (OFENet)
# Paper: https://arxiv.org/abs/2003.01629

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

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_features_dim, action_dim, max_action, hidden_dim=256):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_features_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)
		self.max_action = max_action
		self.apply(weight_init)

	def forward(self, state_features, deterministic=False, with_logprob=True):
		a = F.relu(self.l1(state_features))
		a = F.relu(self.l2(a))
		mu_a = self.mean(a)
		log_std_a = self.log_std(a)
		log_std_a = torch.clamp(log_std_a, LOG_STD_MIN, LOG_STD_MAX)
		std_a = torch.exp(log_std_a)
		# Only used for evaluating policy at test time.
		if deterministic:
			z = mu_a
		else:
			noise = torch.randn_like(mu_a)  # sampled from guassian distribution
			z = mu_a + noise * std_a  # reparameterization trick
		action = torch.tanh(z) 

		if with_logprob and not deterministic:
			logp_pi = gaussian_logprob(noise, log_std_a).sum(axis=-1)
			logp_pi = logp_pi - (1.0 - action**2).clamp(min=1e-6).log().sum(axis=-1)
		else:
			logp_pi = None
		return self.max_action * action, logp_pi


# Returns Q-value for given state/action pairs
class Critic(nn.Module):
	def __init__(self, state_action_features_dim, hidden_dim=256):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_action_features_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_action_features_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)
		self.apply(weight_init)

	def forward(self, hidden_state_action):
		sa = hidden_state_action

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


class SAC_OFE(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		feature_extractor,
		discount=0.99,
		tau=0.005,
		alpha=0.2,
		hidden_dim=256,
	):
		self._extractor = feature_extractor
		self.device = device

		self.actor = Actor(self._extractor.state_features_dim, action_dim, max_action, hidden_dim).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(self._extractor.state_action_features_dim, hidden_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.alpha = alpha

	def select_action(self, state, deterministic=False):
		state = torch.as_tensor(state.reshape(1, -1), device=self.device, dtype=torch.float32)
		state_features = self._extractor.features_of_states(state)
		action, _ = self.actor(state_features, deterministic, False)
		return action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		# Sample batches from replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		with torch.no_grad():
			_next_state_features = self._extractor.features_of_states(next_state)  # compute Z(o_{t+1})
			# Select action according to policy 
			_next_action, logp_pi_next_action = self.actor(_next_state_features)
			logp_pi_next_action = torch.unsqueeze(logp_pi_next_action, 1)
			sa_hidden_target_state = self._extractor.features_of_states_actions(next_state, _next_action)  # compute Z(o_{t+1}, a_{t+1})
			
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(sa_hidden_target_state)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * (target_Q - self.alpha * logp_pi_next_action)

		# Get current Q estimates
		sa_hidden_state = self._extractor.features_of_states_actions(state, action)
		current_Q1, current_Q2 = self.critic(sa_hidden_state)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		_state_features = self._extractor.features_of_states(state)  # Z(o_t)
		policy_action, logp_pi_action = self.actor(_state_features)
		logp_pi_action = torch.unsqueeze(logp_pi_action, 1)
		sa_policy_hidden_state = self._extractor.features_of_states_actions(state, policy_action) 

		Q1_pi, Q2_pi = self.critic(sa_policy_hidden_state)
		Q_pi = torch.min(Q1_pi, Q2_pi)
		actor_loss = (self.alpha * logp_pi_action - Q_pi).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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