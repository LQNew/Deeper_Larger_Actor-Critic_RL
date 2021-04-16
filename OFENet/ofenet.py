import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

def swish(features, beta=1.0):
	return features * F.sigmoid(beta * features)


def get_actionvation_fn(activation: str) -> Callable:
	if activation == "relu":
		return F.relu
	elif activation == "swish":
		return swish
	else:
		raise RuntimeError("activation fn {} not supported".format(activation))


class DensenetBlock(nn.Module):
	def __init__(
		self, 
		input_dim, output_dim=40,
		activation="swish", 
		layernorm=False,
	):
		super(DensenetBlock, self).__init__()

		self.activation_fn = get_actionvation_fn(activation)
		self.fc = nn.Linear(input_dim, output_dim)

		self.layernorm = layernorm
		if self.layernorm:
			self.layer_normalizer = nn.LayerNorm(output_dim)

	def forward(self, inputs):
		identity_map = inputs
		features = self.fc(inputs)

		if self.layernorm:
			features = self.layer_normalizer(features)

		features = self.activation_fn(features)
		features = torch.cat([features, identity_map], dim=1)
		return features


class OFENet(nn.Module):
	def __init__(
		self, 
		state_dim, action_dim, output_dim, 
		hidden_dim=40,
		num_layers=6,
		activation="swish",
		layernorm=True,
	):
		super(OFENet, self).__init__()

		state_layers = []
		action_layers = []

		for i in range(num_layers):
			state_layers.append(state_dim + i*hidden_dim)
		self.state_features_dim = state_layers[-1] + hidden_dim

		for i in range(num_layers):
			action_layers.append(self.state_features_dim + action_dim + i*hidden_dim)
		self.state_action_features_dim = action_layers[-1] + hidden_dim

		self.state_blocks = nn.ModuleList([])
		self.state_blocks.extend([
			DensenetBlock(
				input_dim=state_layers[i], output_dim=hidden_dim, \
				activation=activation, layernorm=layernorm
			) for i in range(num_layers)
		])

		self.action_blocks = nn.ModuleList([])
		self.action_blocks.extend([
			DensenetBlock(
				input_dim=action_layers[i], output_dim=hidden_dim, \
				activation=activation, layernorm=layernorm
			) for i in range(num_layers)
		])
		self.out_layer = nn.Linear(self.state_action_features_dim, output_dim)
		self.output_dim = output_dim
	
	def forward(self, states, actions):
		features = states
		for state_block in self.state_blocks:
			features = state_block(features)
		features = torch.cat([features, actions], dim=1)
		for action_block in self.action_blocks:
			features = action_block(features)
		next_states = self.out_layer(features)
		return next_states  # fc[Z(o_t, a_t)] --> add one fc layer for predicting next state.
	
	def features_of_states(self, states):
		features = states
		for state_block in self.state_blocks:
			features = state_block(features)
		return features  # Z(o_t)

	def features_of_states_actions(self, states, actions):
		state_features = self.features_of_states(states)
		features = torch.cat([state_features, actions], dim=1)
		for action_block in self.action_blocks:
			features = action_block(features)
		return features  # Z(o_t, a_t)


class Aux_Encoder(object):
	def __init__(
		self,
		device,
		state_dim, action_dim, output_dim, 
		hidden_dim=40,
		num_layers=6,
		activation="swish",
		layernorm=True,
	):	
		super(Aux_Encoder, self).__init__()
		self.extractor = OFENet(
			state_dim, action_dim, output_dim, 
			hidden_dim, num_layers,
			activation, layernorm
		).to(device)
		self.extractor_optimizer = torch.optim.Adam(self.extractor.parameters(), lr=3e-4)
	
	def train(self, states, actions, next_states):
		predicted_states = self.extractor(states, actions)
		target_dim = self.extractor.output_dim
		target_states = next_states[:, :target_dim]
		feature_loss = F.mse_loss(target_states, predicted_states)

		# Optimize the extractor 
		self.extractor_optimizer.zero_grad()
		feature_loss.backward()
		self.extractor_optimizer.step()

		# save the model
	def save(self, filename):
		torch.save(self.extractor.state_dict(), filename + "_extractor")
		torch.save(self.extractor_optimizer.state_dict(), filename + "_extractor_optimizer")

	# load the model
	def load(self, filename):
		self.extractor.load_state_dict(torch.load(filename + "_extractor"))
		self.extractor_optimizer.load_state_dict(torch.load(filename + "_extractor_optimizer"))
