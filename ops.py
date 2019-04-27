import tensorflow as tf
import numpy as np


#--------------------------
# Fully connected
# inp: (mb_size, in_dim)
#--------------------------
def fc(inp, out_dim, name="fc", bias_init=0.0):
	inp_shape = inp.get_shape().as_list()
	stddev = tf.sqrt(3.0 / (inp_shape[-1] + out_dim))

	with tf.variable_scope(name):
		W = tf.get_variable(
			"W",
			[inp_shape[-1], out_dim],
			initializer=tf.random_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			"b",
			[out_dim],
			initializer=tf.constant_initializer(bias_init)
		)

		#return: (mb_size, out_dim)
		return tf.matmul(inp, W) + b


#----------------------------
# Convolution 1D
# with dilated & causal
# inp: (mb_size, seq_len, in_c_dim)
#----------------------------
def conv1d(inp, out_c_dim, dilation=1, k_w=1, causal=False, bias_init=0.0, name="conv1d"):
	inp_shape = inp.get_shape().as_list()
	stddev = tf.sqrt(3.0 / (k_w*inp_shape[-1] + out_c_dim))

	with tf.variable_scope(name):
		W = tf.get_variable(
			"W",
			[1, k_w, inp_shape[-1], out_c_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			"b",
			[out_c_dim],
			initializer=tf.constant_initializer(bias_init)
		)

		#Causal conv 1D
		if causal:
			#padded:   (mb_size, (k_w-1)*dilation + seq_len, in_c_dim)
			#expanded: (mb_size, 1, (k_w-1)*dilation + seq_len, in_c_dim)
			#out:      (mb_size, 1, seq_len, in_c_dim)
			padded = tf.pad(inp, [[0, 0], [(k_w-1) * dilation, 0], [0, 0]])
			expanded = tf.expand_dims(padded, dim=1)
			out = tf.nn.atrous_conv2d(expanded, W, rate=dilation, padding="VALID") + b

		#Normal conv 1D
		else:
			#expanded: (mb_size, 1, seq_len, in_c_dim)
			#out:      (mb_size, 1, seq_len, in_c_dim)
			expanded = tf.expand_dims(inp, dim=1)
			out = tf.nn.atrous_conv2d(expanded, W, rate=dilation, padding="SAME") + b

		#out: (mb_size, seq_len, in_c_dim)
		return tf.squeeze(out, [1])


#--------------------------
# Convolution 2D
# inp: (mb_size, hight, width, in_c_dim)
#--------------------------
def conv2d(inp, out_c_dim, k_w=5, k_h=5, stride_x=2, stride_y=2, padding="SAME", bias_init=0.0, name="conv2d"):
	inp_shape = inp.get_shape().as_list()
	stddev = tf.sqrt(3.0 / (k_h*k_w*inp_shape[-1] + out_c_dim))

	with tf.variable_scope(name):
		W = tf.get_variable(
			"W",
			[k_h, k_w, inp_shape[-1], out_c_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			"b",
			[out_c_dim],
			initializer=tf.constant_initializer(bias_init)
		)

		conv = tf.nn.conv2d(
			inp,
			W,
			strides=[1, stride_y, stride_x, 1],	#[mb_size, y, x, c_dim]
			padding=padding
		)

		#return: (mb_size, hight', width', out_c_dim)
		return tf.nn.bias_add(conv, b)


#--------------------------
# Deconvolution 2D
# inp: (mb_size, hight, width, in_c_dim)
#--------------------------
def deconv2d(inp, out_shape, k_w, k_h, stride_x=1, stride_y=1, padding="SAME", name="deconv2d", bias_init=0.0):
	inp_shape = inp.get_shape().as_list()
	stddev = tf.sqrt(3.0 / (k_h*k_w*out_shape[-1] + inp_shape[-1]))

	with tf.variable_scope(name):
		W = tf.get_variable(
			"W",
			[k_h, k_w, out_shape[-1], inp_shape[-1]],
			initializer=tf.truncated_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			"b",
			[out_shape[-1]],
			initializer=tf.constant_initializer(bias_init)
		)

		deconv = tf.nn.conv2d_transpose(
			inp, 
			W, 
			output_shape=out_shape, 
			strides=[1, stride_y, stride_x, 1], #[mb_size, y, x, c_dim]
			padding=padding
		)

		#return: (mb_size, hight', width', out_c_dim)
		return tf.nn.bias_add(deconv, b)


#--------------------------
# Embedding
# inp: (s1, ..., s_n)
#--------------------------
def embed(inp, in_dim, embed_dim, name="embedding", reuse=False):
	stddev = tf.sqrt(3.0 / (in_dim + embed_dim))

	with tf.variable_scope(name, reuse=reuse):
		embed_W = tf.get_variable(
			"embed_W",
			[in_dim, embed_dim],
			tf.float32,
			initializer=tf.random_normal_initializer(stddev=stddev)
		)

		#return: (s1, ..., s_n, embed_dim)
		return tf.nn.embedding_lookup(embed_W, inp)


#---------------------------
# Single layer LSTM cell
#---------------------------
def single_layer_lstmCell(s_dim):
	return tf.nn.rnn_cell.LSTMCell(
		s_dim,
		initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)
	)


#---------------------------
# Multi layer LSTM cell
#---------------------------
def multi_layer_lstmCell(s_dim, n_layer):
	return tf.nn.rnn_cell.MultiRNNCell([single_layer_lstmCell(s_dim) for _ in range(n_layer)])


#---------------------------
# self-defined LSTM cell
#---------------------------
def lstmCell(in_dim, s_dim, name="lstm", bias_init=0.0):
	stddev = tf.sqrt(3.0 / (in_dim + s_dim))

	with tf.variable_scope(name):
		Wi = tf.get_variable("Wi", [in_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		Ui = tf.get_variable("Ui", [s_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		bi = tf.get_variable("bi", [s_dim], initializer=tf.constant_initializer(bias_init))

		Wf = tf.get_variable("Wf", [in_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		Uf = tf.get_variable("Uf", [s_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		bf = tf.get_variable("bf", [s_dim], initializer=tf.constant_initializer(bias_init))

		Wo = tf.get_variable("Wo", [in_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		Uo = tf.get_variable("Uo", [s_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		bo = tf.get_variable("bo", [s_dim], initializer=tf.constant_initializer(bias_init))

		Wc = tf.get_variable("Wc", [in_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		Uc = tf.get_variable("Uc", [s_dim, s_dim], initializer=tf.random_normal_initializer(stddev=stddev))
		bc = tf.get_variable("bc", [s_dim], initializer=tf.constant_initializer(bias_init))

	def unit(x, state):
		s_prev, h_prev = tf.unstack(state)

		#Input Gate
		i = tf.sigmoid(
			tf.matmul(x, Wi) + tf.matmul(h_prev, Ui) + bi
		)

		#Forget Gate
		f = tf.sigmoid(
			tf.matmul(x, Wf) + tf.matmul(h_prev, Uf) + bf
		)

		#Output Gate
		o = tf.sigmoid(
			tf.matmul(x, Wo) + tf.matmul(h_prev, Uo) + bo
		)

		#New Memory Cell
		s = tf.sigmoid(
			tf.matmul(x, Wc) + tf.matmul(h_prev, Uc) + bc
		)

		#Final Memory Cell
		s = f*s_prev + i*s

		#Current state
		h = o * tf.nn.tanh(s)

		return tf.stack([s, h])

	return unit


#--------------------------
# LSTM dynamic RNN
# inp:     (mb_size, max_seq_len, in_dim)
# seq_len: (mb_size)
#--------------------------
def dynamic_lstm(inp, s_dim, n_layer=1, init_state=None, mb_size=None, seq_len=None, name="lstm"):
	#init_state:  (mb_size, s_dim)
	#final_state: (mb_size, s_dim)
	#rnn_outputs: (mb_size, max_seq_len, s_dim)
	lstm_cell = multi_layer_lstmCell(s_dim, n_layer)

	#Given initial state
	if init_state is not None:
		rnn_outputs, final_state = tf.nn.dynamic_rnn(
			lstm_cell, 
			inp, 
			sequence_length=seq_len, 
			initial_state=init_state,
			scope=name
		)
		return rnn_outputs, final_state

	#Given initial state placeholder (default is zero state)
	elif mb_size is not None:
		init_state_ph = lstm_cell.zero_state(mb_size, tf.float32)
		rnn_outputs, final_state = tf.nn.dynamic_rnn(
			lstm_cell, 
			inp, 
			sequence_length=seq_len, 
			initial_state=init_state,
			scope=name
		)
		return rnn_outputs, final_state, init_state_ph

	#Initial state is zero state
	else:
		rnn_outputs, final_state = tf.nn.dynamic_rnn(
			lstm_cell, 
			inp, 
			sequence_length=seq_len, 
			dtype=tf.float32,
			scope=name 
		)
		return rnn_outputs, final_state


#--------------------------
# LSTM static RNN
# inp:         (mb_size, max_seq_len, in_dim)
# init_states: [(mb_size, s_dim), (mb_size, s_dim)] * n_layer
#--------------------------
def static_lstm(inp, init_states, n_layer=1, bias_init=0.0, name="static_lstm"):
	in_dim = inp.get_shape().as_list()[-1]
	s_dim = init_states[0][0].get_shape().as_list()[-1]

	#Create lstm cells
	with tf.variable_scope(name):
		lstm_cells = [lstmCell(in_dim, s_dim, name="lstm0")] \
					+ [lstmCell(s_dim, s_dim, name="lstm"+str(i)) for i in range(1, n_layer)]
		states = [tf.stack(init_states[i]) for i in range(n_layer)]

	#Create static rnn graph
	#------------------------
	#rnn_inputs:  [(mb_size, in_dim)] * max_seq_len
	#rnn_outputs: [(mb_size, s_dim)] * max_seq_len
	rnn_inputs = tf.unstack(inp, axis=1)
	rnn_outputs = []

	for rnn_inp in rnn_inputs:
		h = rnn_inp
		next_states = []
		for i in range(n_layer):
			next_state = lstm_cells[i](h, states[i])
			s, h = tf.unstack(next_state)
			next_states.append(next_state)

		states = next_states
		rnn_outputs.append(h)

	#rnn_outputs:  (mb_size, max_seq_len, s_dim)
	#final_states: [(2, mb_size, s_dim)] * n_layer
	rnn_outputs = tf.transpose(tf.stack(rnn_outputs), perm=[1, 0, 2])
	final_states = states

	return rnn_outputs, final_states


#--------------------------
# LSTM tensor array RNN
# inp:        (mb_size, max_seq_len, in_dim)
# init_state: [(mb_size, s_dim), (mb_size, s_dim)] * n_layer
#--------------------------
def tensor_array_lstm(inp, init_states, n_layer=1, bias_init=0.0, name="tensor_array_lstm"):
	inp_shape = inp.get_shape().as_list()
	max_seq_len = inp_shape[1]
	in_dim = inp_shape[-1]
	s_dim = init_states[0][0].get_shape().as_list()[-1]

	#Create lstm cells
	with tf.variable_scope(name):
		lstm_cells = [lstmCell(in_dim, s_dim, name="lstm0")] \
					+ [lstmCell(s_dim, s_dim, name="lstm"+str(i)) for i in range(1, n_layer)]
		states = [tf.stack(init_states[i]) for i in range(n_layer)]

	#Create tensor array
	in_arr = tf.TensorArray(
		dtype=tf.float32,
		size=max_seq_len,
		dynamic_size=False,
		infer_shape=True
	)
	out_arr = tf.TensorArray(
		dtype=tf.float32,
		size=max_seq_len,
		dynamic_size=False,
		infer_shape=True
	)
	out_state_arr = tf.TensorArray(
		dtype=tf.float32,
		size=max_seq_len,
		dynamic_size=False,
		infer_shape=True
	)

	for i in range(max_seq_len):
		in_arr = in_arr.write(i, inp[:, i, :])

	#Define while loop body
	def recur(i, states_prev, in_arr, out_arr, out_state_arr):
		h = in_arr.read(i)
		states_cur = []
		
		for j in range(n_layer):
			state_cur = lstm_cells[j](h, states_prev[j])
			s, h = tf.unstack(state_cur)
			states_cur.append(state_cur)
		
		out_arr = out_arr.write(i, h)
		out_state_arr = out_state_arr.write(i, tf.stack(states_cur))

		return i+1, states_cur, in_arr, out_arr, out_state_arr

	#Construct while loop
	_, _, in_arr, out_arr, out_state_arr = tf.while_loop(
		cond=lambda i, _1, _2, _3, _4: i < max_seq_len,
		body=recur,
		loop_vars=(
			tf.constant(0, dtype=tf.int32),
			states,
			in_arr,
			out_arr,
			out_state_arr
		)
	)

	#rnn_outputs:  (mb_size, max_seq_len, s_dim)
	#final_states: (n_layer, 2, mb_size, s_dim)
	rnn_outputs = tf.transpose(out_arr.stack(), perm=[1, 0, 2])
	final_state = out_state_arr.stack()[-1, :, :]

	return rnn_outputs, final_state


#--------------------------
# LSTM tensor array decoder
# start_embed: (mb_size, embed_dim)
# init_state:  (mb_size, s_dim)
#--------------------------
def tensor_array_lstm_decoder(
	start_embed, 
	init_states, 
	embed_W, 
	out_dim, 
	max_seq_len, 
	n_layer=1, 
	greedy=False, 
	bias_init=0.0,
	name="lstm_decoder"
):
	s_dim = init_states[0][0].get_shape().as_list()[-1]
	embed_dim = start_embed.get_shape().as_list()[-1]
	stddev = tf.sqrt(3.0 / (s_dim + out_dim))

	#Create projection weights & lstm cells
	with tf.variable_scope(name):
		W = tf.get_variable(
			"W",
			[s_dim, out_dim],
			initializer=tf.random_normal_initializer(stddev=stddev)
		)
		b = tf.get_variable(
			"b",
			[out_dim],
			initializer=tf.constant_initializer(bias_init)
		)

		lstm_cells = [lstmCell(embed_dim, s_dim, name="lstm0")] \
					+ [lstmCell(s_dim, s_dim, name="lstm"+str(i)) for i in range(1, n_layer)]
		states = [tf.stack(init_states[i]) for i in range(n_layer)]

	#Create tensor array
	in_arr = tf.TensorArray(
		dtype=tf.float32,
		size=max_seq_len+1,
		dynamic_size=False,
		infer_shape=True
	)
	out_arr = tf.TensorArray(
		dtype=tf.float32,
		size=max_seq_len,
		dynamic_size=False,
		infer_shape=True
	)
	out_logits_arr = tf.TensorArray(
		dtype=tf.float32,
		size=max_seq_len,
		dynamic_size=False,
		infer_shape=True
	)
	in_arr = in_arr.write(0, start_embed)

	#Define while loop body
	def recur(i, states_prev, in_arr, out_arr, out_logits_arr):
		h = in_arr.read(i)
		states_cur = []

		for j in range(n_layer):
			state_cur = lstm_cells[j](h, states_prev[j])
			s, h = tf.unstack(state_cur)
			states_cur.append(state_cur)

		logits = tf.matmul(h, W) + b

		if greedy:
			out_sample = sample(logits)
		else:
			out_sample = tf.multinomial(logits, 1)[:, 0]

		out_embed = tf.nn.embedding_lookup(embed_W, out_sample)

		in_arr = in_arr.write(i+1, out_embed)
		out_arr = out_arr.write(i, h)
		out_logits_arr = out_logits_arr.write(i, logits)

		return i+1, states_cur, in_arr, out_arr, out_logits_arr

	#Construct while loop
	_, _, in_arr, out_arr, out_logits_arr = tf.while_loop(
		cond=lambda i, _1, _2, _3, _4: i < max_seq_len,
		body=recur,
		loop_vars=(
			tf.constant(0, dtype=tf.int32),
			states,
			in_arr,
			out_arr,
			out_logits_arr
		)
	)

	#rnn_outputs: (mb_size, max_seq_len, s_dim)
	#logits:      (mb_size, max_seq_len, out_dim)
	#pred:        (mb_size, max_seq_len)
	rnn_outputs = tf.transpose(out_arr.stack(), perm=[1, 0, 2])
	logits = tf.transpose(out_logits_arr.stack(), perm=[1, 0, 2])
	pred = sample(logits)

	return rnn_outputs, logits, pred


#--------------------------
# Layer normalization
#--------------------------
def layer_norm(inp, eps=1e-8, trainable=True, name="layer_norm"):
	with tf.variable_scope(name):
		shape = inp.get_shape()

		beta = tf.get_variable(
			"beta",
			[shape[-1]],
			initializer=tf.constant_initializer(0.0),
			trainable=trainable
		)
		gamma = tf.get_variable(
			"gamma",
			[shape[-1]],
			initializer=tf.constant_initializer(1.0),
			trainable=trainable
		)

		mean, var = tf.nn.moments(inp, axes=[len(shape) - 1], keep_dims=True)
		inp = (inp - mean) / tf.sqrt(var + eps)

		return gamma * inp + beta


#--------------------------
# Sample from the Cetegorical
# distrib. (greedy)
#--------------------------
def sample(logits, axis=-1):
	noise = tf.random_uniform(tf.shape(logits))
	return tf.argmax(logits - tf.log(-tf.log(noise)), axis)


#-----------------------------
# Sample from top k prob
#-----------------------------
def sample_top(prob, top_k=3):
	idx = np.argsort(prob)[::-1]
	idx = idx[:top_k]

	p = prob[idx]
	p = p / np.sum(p)

	return np.random.choice(idx, p=p)