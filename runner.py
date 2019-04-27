import numpy as np
from collections import deque


#--------------------------
# Compute the discounted returns
#--------------------------
def discounted_returns(mb_rewards, mb_last_idx, gamma=1):
	mb_returns = np.zeros_like(mb_rewards)
	
	for i in range(mb_rewards.shape[0]):
		running_add = 0
		
		for t in reversed(range(0, int(mb_last_idx[i, 1] + 1))):
			running_add = gamma * running_add + mb_rewards[i, t]
			mb_returns[i, t] = running_add

	return mb_returns


class Runner():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(
		self, 
		sess, 
		sent_ph,
		given_sent_ph, 
		fake_sent_ph, 
		last_idx_ph,
		given_last_idx_ph,
		fake_last_idx_ph, 
		fake_action_ph,
		z_ph, 
		sos_id, 
		eos_id, 
		gamma=1,
		lamb=0.1
	):
		self.sess = sess
		self.sent_ph = sent_ph
		self.given_sent_ph = given_sent_ph
		self.fake_sent_ph = fake_sent_ph
		self.last_idx_ph = last_idx_ph
		self.given_last_idx_ph = given_last_idx_ph
		self.fake_last_idx_ph = fake_last_idx_ph
		self.fake_action_ph = fake_action_ph
		self.z_ph = z_ph
		self.sos_id = sos_id
		self.eos_id = eos_id 
		self.gamma = gamma
		self.lamb = lamb

		#Recoder
		self.reward_buf = deque(maxlen=128)
		self.sent_reward_buf = deque(maxlen=128)


	#--------------------------
	# Sample
	#--------------------------
	def sample(self, model, z, mb_size=64, max_len=32):
		#1. Initialization
		mb_gen_sent = np.zeros([mb_size, 1], np.int32)
		mb_last_idx = np.zeros([mb_size, 2], np.int32)

		for i in range(mb_size):
			mb_gen_sent[i, 0] = self.sos_id
			mb_last_idx[i, 0] = i
			mb_last_idx[i, 1] = 0

		#2. For each timestep
		for i in range(max_len):
			mb_action = self.sess.run(model.action, feed_dict={
				self.given_sent_ph     : mb_gen_sent,
				self.given_last_idx_ph : mb_last_idx,
				self.z_ph              : z
			})

			for j in range(mb_size):
				mb_last_idx[j, 1] = i + 1

			mb_gen_sent = np.insert(mb_gen_sent, i + 1, mb_action, axis=1)

		#3. Compute last_idx
		mb_gen_sent = mb_gen_sent[:, 1:]

		for i in range(mb_size):
			for j in range(max_len):
				if mb_gen_sent[i, j] == self.eos_id or j == max_len - 1:
					mb_last_idx[i, 1] = j
					break

		return mb_gen_sent, mb_last_idx


	#--------------------------
	# Sample actions given states
	#--------------------------
	def sample_action(self, model, sent, last_idx, z):
		action = self.sess.run(model.action, feed_dict={
			self.given_sent_ph     : sent,
			self.given_last_idx_ph : last_idx,
			self.z_ph              : z
		})

		return action


	#--------------------------
	# Run for an episode
	#--------------------------
	def run(self, model, z, mb_size=64, max_len=32):
		#1. Initialization
		#--------------------------
		#mb_gen_sent:     (mb_size, 1)
		#mb_last_idx:     (mb_size, 2)
		#mb_obs:          (max_len, mb_size, max_len)
		#mb_actions:      (max_len, mb_size)
		#mb_values:       (max_len, mb_size)
		#mb_rewards:      (max_len, mb_size)
		#mb_sent_rewards: (max_len, mb_size)
		mb_gen_sent = np.zeros([mb_size, 1], np.int32)
		mb_last_idx = np.zeros([mb_size, 2], np.int32)
		mb_obs      = np.zeros([max_len, mb_size, max_len], np.int32)
		mb_actions  = np.zeros([max_len, mb_size], np.int32)
		mb_values   = np.zeros([max_len, mb_size])
		mb_rewards  = np.zeros([max_len, mb_size])
		mb_sent_rewards = np.zeros([max_len, mb_size])

		for i in range(mb_size):
			mb_gen_sent[i, 0] = self.sos_id
			mb_last_idx[i, 0] = i
			mb_last_idx[i, 1] = 0

		#2. For each timestep
		#--------------------------
		for i in range(max_len):
			#mb_gen_sent: (mb_size, cur_seq_len)
			mb_obs[i, :, :i+1] = mb_gen_sent[:, :]

			#mb_action: (mb_size)
			#mb_value:  (mb_size)
			mb_action, mb_value = self.sess.run([model.action, model.value], feed_dict={
				self.given_sent_ph     : mb_gen_sent,
				self.given_last_idx_ph : mb_last_idx,
				self.z_ph              : z
			})

			#mb_sa_reward:   (mb_size, 1)
			mb_sa_reward = self.sess.run(model.sa_prob_fake, feed_dict={
				self.fake_sent_ph     : mb_gen_sent,  
				self.fake_last_idx_ph : mb_last_idx,
				self.fake_action_ph   : mb_action
			})

			mb_gen_sent = np.insert(mb_gen_sent, i + 1, mb_action, axis=1)

			#mb_sent_reward: (mb_size, 1)
			mb_sent_reward = self.sess.run(model.prob_fake, feed_dict={
				self.fake_sent_ph     : mb_gen_sent[:, 1:],  
				self.fake_last_idx_ph : mb_last_idx
			})

			for j in range(mb_size):
				mb_last_idx[j, 1] = i + 1

			mb_actions[i, :] = mb_action[:]
			mb_values[i, :]  = mb_value[:]
			mb_rewards[i, :] = self.lamb * mb_sa_reward[:, 0]
			mb_sent_rewards[i, :] = mb_sent_reward[:, 0]


		#3. Transpose
		#--------------------------
		#mb_obs:          (mb_size, max_len, max_len)
		#mb_actions:      (mb_size, max_len)
		#mb_values:       (mb_size, max_len)
		#mb_rewards:      (mb_size, max_len)
		#mb_sent_rewards: (mb_size, max_len)
		mb_obs     = np.transpose(mb_obs, (1, 0, 2))
		mb_actions = np.transpose(mb_actions, (1, 0))
		mb_values  = np.transpose(mb_values, (1, 0))
		mb_rewards = np.transpose(mb_rewards, (1, 0))
		mb_sent_rewards = np.transpose(mb_sent_rewards, (1, 0))


		#4. Compute discounted returns & last_idx for mb_action
		#--------------------------
		avg_sent_reward = 0.0

		for i in range(mb_size):
			for j in range(max_len):
				if mb_actions[i, j] == self.eos_id:
					mb_last_idx[i, 1] = j
					mb_rewards[i, j] += mb_sent_rewards[i, j]
					avg_sent_reward += mb_sent_rewards[i, j]
					break
				elif j == max_len - 1:
					mb_last_idx[i, 1] = j
					break

		self.record(mb_rewards, mb_last_idx, avg_sent_reward / mb_size)

		#mb_returns: (mb_size, max_len)
		mb_returns = discounted_returns(mb_rewards, mb_last_idx, gamma=self.gamma)
 
		return mb_obs, mb_actions, mb_values, mb_returns, mb_last_idx


	#--------------------------
	# Record average reward
	#--------------------------
	def record(self, mb_rewards, mb_last_idx, avg_sent_reward):
		mb_size = mb_rewards.shape[0]
		total_reward = 0.0

		for i in range(mb_size):
			for j in range(mb_last_idx[i, 1] + 1):
				total_reward += mb_rewards[i, j]

		self.reward_buf.append(total_reward / mb_size)
		self.sent_reward_buf.append(avg_sent_reward)


	#--------------------------
	# Get performance
	#--------------------------
	def get_performance(self):
		if len(self.reward_buf) == 0:
			return 0.0, 0.0, 0.0
		else:
			return self.reward_buf[-1], np.mean(self.reward_buf), np.mean(self.sent_reward_buf)