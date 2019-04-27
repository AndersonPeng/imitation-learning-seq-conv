import ops
import tensorflow as tf
import numpy as np


class Word2vecModel():
	#----------------------------
	# Constructor
	#----------------------------
	def __init__(self, x_ph, vocab_dim, embed_dim, name="word2vec"):
		#Word2vec model
		with tf.variable_scope(name):
			embed_x = ops.embed(x_ph, vocab_dim, embed_dim, name="embed")

			softmax_W = tf.get_variable(
				"softmax_W",
				[vocab_dim, embed_dim],
				initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1.0 / embed_dim))
			)
			softmax_b = tf.get_variable(
				"softmax_b",
				[vocab_dim],
				initializer=tf.constant_initializer(0.0)
			)

		#Compute similarity
		with tf.variable_scope(name, reuse=True):
			embed_W = tf.get_variable("embed/embed_W")

		norm = tf.sqrt(tf.reduce_sum(tf.square(embed_W), 1, keepdims=True))
		normalized_embed_W = embed_W / norm
		valid_embed = tf.nn.embedding_lookup(normalized_embed_W, x_ph)
		similarity = tf.matmul(valid_embed, tf.transpose(normalized_embed_W))

		self.x_ph = x_ph
		self.embed_x = embed_x
		self.embed_W = embed_W
		self.normalized_embed_W = normalized_embed_W
		self.softmax_W = softmax_W
		self.softmax_b = softmax_b
		self.similarity = similarity


	#----------------------------
	# Find top k nearest words
	#----------------------------
	def find_nearest(self, sess, x_mb, k=8):
		mb_size = x_mb.shape[0]

		sim = sess.run(self.similarity, feed_dict={
			self.x_ph: x_mb
		})

		nearest = np.zeros([mb_size, k], np.int32)
		for i in range(mb_size):
			nearest[i, :] = (-sim[i, :]).argsort()[:k]

		return nearest