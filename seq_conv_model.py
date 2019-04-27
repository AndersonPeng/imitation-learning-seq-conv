import ops
import distribs
import tensorflow as tf
import numpy as np


#--------------------------
# Residual block
#--------------------------
def res_block(inp, dilation, k_w, causal, name):
	h = ops.conv1d(inp, 128, dilation=dilation, k_w=k_w, causal=causal, name=name)
	h = tf.nn.relu(h)

	return h + inp


#Sequential convolution generator model
class SeqConvModel():
	#--------------------------
	# Constructor
	#--------------------------
	def __init__(
		self,
		sent,
		given_sent,
		fake_sent,
		last_idx,
		given_last_idx,
		fake_last_idx,
		real_action,
		fake_action,
		z,
		dilations,
		vocab_dim,
		embed_dim,
		k_w=2,
		alpha=1.0,
		name="seq_conv_gail"
	):
		with tf.variable_scope(name):
			#1. Embedding
			#-----------------------------
			embed_sent = ops.embed(sent, vocab_dim, embed_dim, name="embed")
			given_embed_sent = ops.embed(given_sent, vocab_dim, embed_dim, name="embed", reuse=True)
			fake_embed_sent = ops.embed(fake_sent, vocab_dim, embed_dim, name="embed", reuse=True)
			real_action_embed = ops.embed(real_action, vocab_dim, embed_dim, name="embed", reuse=True)
			fake_action_embed = ops.embed(fake_action, vocab_dim, embed_dim, name="embed", reuse=True)


			#2. Generator
			#-----------------------------
			gen_state_logits, gen_state_logits_last = self.state_encoder(given_embed_sent, given_last_idx, z, dilations, k_w)
			logits, logits_last, greedy_pred, prob = self.generator(gen_state_logits, given_last_idx, vocab_dim)

			self.logits = logits
			self.logits_last =  logits_last
			self.greedy_pred = greedy_pred
			self.prob = prob
			self.distrib = distribs.CategoricalDistrib(logits_last, alpha)
			self.action = self.distrib.sample()
			

			#3. Value network
			#-----------------------------
			value = self.value_net(gen_state_logits_last)
			
			self.value = value[:, 0]


			#4. State-action discriminator
			#-----------------------------
			real_state_logits, real_state_logits_last = self.state_encoder(embed_sent, last_idx, z, dilations, k_w, reuse=True)
			fake_state_logits, fake_state_logits_last = self.state_encoder(fake_embed_sent, fake_last_idx, z, dilations, k_w, reuse=True)

			sa_logits_real, sa_prob_real = self.state_action_discriminator(real_state_logits_last, real_action_embed)
			sa_logits_fake, sa_prob_fake = self.state_action_discriminator(fake_state_logits_last, fake_action_embed, reuse=True)

			self.sa_logits_real = sa_logits_real
			self.sa_logits_fake = sa_logits_fake
			self.sa_prob_real = sa_prob_real
			self.sa_prob_fake = sa_prob_fake


			#5. Discriminator
			#-----------------------------
			logits_real, prob_real = self.discriminator(real_state_logits_last)
			logits_fake, prob_fake = self.discriminator(fake_state_logits_last, reuse=True)

			self.logits_real = logits_real
			self.logits_fake = logits_fake
			self.prob_real = prob_real
			self.prob_fake = prob_fake

		#embed weight
		with tf.variable_scope(name, reuse=True):
			self.embed_W = tf.get_variable("embed/embed_W")


	#--------------------------
	# State encoder
	#--------------------------
	def state_encoder(self, embed_sent, last_idx, z, dilations, k_w=2, reuse=False):
		with tf.variable_scope("state_enc", reuse=reuse):
			#1. Dilated causal convolution
			#-----------------------------
			#embed_sent: (mb_size, seq_len, embed_dim)
			h = embed_sent

			#h: (mb_size, seq_len, 128)
			for i, dilation in enumerate(dilations):
				h = res_block(h, dilation, k_w, causal=True, name="res_causal_conv{}_{}".format(i, dilation))

			#2. Concat z
			#-----------------------------
			#z_expanded: (mb_size, 1, z_dim)
			#z_seq:      (mb_size, seq_len, z_dim)
			#concat_z_h: (mb_size, seq_len, z_dim+128)
			#z_expanded = tf.expand_dims(z, [1])
			#z_seq = tf.tile(z_expanded, [1, tf.shape(h)[1], 1])
			#concat_z_h = tf.concat([z_seq, h], 2)

			#3. Output logits & logits_last
			#-----------------------------
			#logits:      (mb_size, seq_len, 128)
			#logits_last: (mb_size, 128)
			logits = h
			logits_last = tf.gather_nd(h, last_idx)

			return logits, logits_last


	#--------------------------
	# Generator
	#--------------------------
	def generator(self, state_logits, last_idx, vocab_dim, reuse=False):
		with tf.variable_scope("gen", reuse=reuse):
			#h:      (mb_size, seq_len, 128)
			h = ops.conv1d(state_logits, 128, dilation=1, k_w=1, name="conv1")
			h = tf.nn.relu(h)

			#logits:      (mb_size, seq_len, vocab_dim)
			#logits_last: (mb_size, vocab_dim)
			#greedy_pred: (mb_size, seq_len)
			#prob:        (mb_size, seq_len, vocab_dim)
			logits = ops.conv1d(h, vocab_dim, dilation=1, k_w=1, name="conv_logits")
			logits_last = tf.gather_nd(logits, last_idx)
			greedy_pred = tf.argmax(logits, axis=-1)
			prob = tf.nn.softmax(logits, axis=-1)

			return logits, logits_last, greedy_pred, prob


	#--------------------------
	# Value network
	#--------------------------
	def value_net(self, state_logits_last, reuse=False):
		with tf.variable_scope("value_net", reuse=reuse):
			#h: (mb_size, 128)
			h = ops.fc(state_logits_last, 128, name="fc1")
			h = tf.nn.relu(h)
			
			#value: (mb_size, 1)
			value = ops.fc(h, 1, name="fc_out")

			return value


	#--------------------------
	# State-action discriminator
	#--------------------------
	def state_action_discriminator(self, state_logits_last, action_embed, reuse=False):
		with tf.variable_scope("sa_dis", reuse=reuse):
			#h: (mb_size, 128)
			h = ops.fc(tf.concat([state_logits_last, action_embed], 1), 128, name="fc1")
			h = tf.nn.relu(h)

			#logits: (mb_size, 1)
			#prob:   (mb_size, 1)
			logits = ops.fc(h, 1, name="fc_out")
			prob = tf.nn.sigmoid(logits)

			return logits, prob


	#--------------------------
	# Discriminator
	#--------------------------
	def discriminator(self, state_logits_last, reuse=False):
		with tf.variable_scope("dis", reuse=reuse):
			#h: (mb_size, 128)
			h = ops.fc(state_logits_last, 128, name="fc1")
			h = tf.nn.relu(h)

			#logits: (mb_size, 1)
			#prob:   (mb_size, 1)
			logits = ops.fc(h, 1, name="fc_out")
			prob = tf.nn.sigmoid(logits)

			return logits, prob