"""For running D2RL coupled with DDPG / TD3 / SAC."""
import random
import torch
import gym
import argparse
import os
import time
import numpy as np
# ------------------------------
from DDPG import DDPG_d2rl
# -------------------------------
from TD3 import TD3_d2rl
# -------------------------------
from SAC import SAC_d2rl
# -------------------------------
from utils import replay_buffer

from spinupUtils.logx import EpochLogger
from spinupUtils.run_utils import setup_logger_kwargs

def test_agent(policy, eval_env, seed, logger, eval_episodes=10):
	for _ in range(eval_episodes):
		state, done, ep_ret, ep_len = eval_env.reset(), False, 0, 0
		while not done:
			if args.policy.startswith("SAC"):
				action = policy.select_action(np.array(state), deterministic=True)
			else:
				action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			ep_ret += reward
			ep_len += 1
		logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="SAC")                   # Policy name
	parser.add_argument("--env", default="HalfCheetah-v2")           # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=3e6, type=int)    # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                  # Discount factor
	parser.add_argument("--tau", default=0.005)                      # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
	parser.add_argument("--alpha", default=0.2, type=float)          # For sac entropy
	parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--exp_name", type=str)       				 # Name for algorithms
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print(f"---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"---------------------------------------")

	# Make envs
	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	eval_env.seed(args.seed)  # eval env for evaluating the agent
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	# ----------------------------------------------
	if args.policy == "DDPG_d2rl":
		# if the formal argument defined in function `DDPG()` are regular params, can pass `**-styled` actual argument.
		policy = DDPG_d2rl.DDPG_D2RL(**kwargs)
	# ---------------------------------------------------
	elif args.policy == "TD3_d2rl":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3_d2rl.TD3_D2RL(**kwargs)
	# ----------------------------------------------
	elif args.policy == "SAC_d2rl":
		kwargs["alpha"] = args.alpha
		policy = SAC_d2rl.SAC_D2RL(**kwargs)
	else:
		raise ValueError(f"Invalid Policy: {args.policy}!")
	
	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		if not os.path.exists(f"./models/{policy_file}"):
			assert f"The loading model path of `../models/{policy_file}` does not exist! "
		policy.load(f"./models/{policy_file}")
	
	# Setup loggers
	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=False)
	logger = EpochLogger(**logger_kwargs)

	_replay_buffer = replay_buffer.ReplayBuffer(state_dim, action_dim)
	
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	start_time = time.time()

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < int(args.start_timesteps):
			action = env.action_space.sample()
		else:
			if args.policy.startswith("SAC"):
				action = policy.select_action(np.array(state))
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)

		# If env stops when reaching max-timesteps, then `done_bool = False`, else `done_bool = True`
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0  

		# Store data in replay buffer
		_replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= int(args.start_timesteps):
			policy.train(_replay_buffer, args.batch_size)

		if done: 
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			logger.store(EpRet=episode_reward, EpLen=episode_timesteps)
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		if (t + 1) % args.eval_freq == 0:
			test_agent(policy, eval_env, args.seed, logger)
			if args.save_model: 
				policy.save(f"./models/{file_name}")
			logger.log_tabular("EpRet", with_min_and_max=True)
			logger.log_tabular("TestEpRet", with_min_and_max=True)
			logger.log_tabular("EpLen", average_only=True)
			logger.log_tabular("TestEpLen", average_only=True)
			logger.log_tabular("TotalEnvInteracts", t+1)
			logger.log_tabular("Time", time.time()-start_time)
			logger.dump_tabular()
