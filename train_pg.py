import ops
from seq_conv_model import SeqConvModel
from word2vec_model import Word2vecModel
from runner import Runner
import tensorflow as tf
import numpy as np
import pickle as pkl
import os


#--------------------------
# Exponential decay
#--------------------------
def exp_decay(lr, global_step, decay_step, decay_rate):
	return lr * (decay_rate**(global_step / decay_step))


#--------------------------
# Pad a batch of sentences
#--------------------------
def pad_sentence_batch(sentence_batch, pad_id, sos_id=None, eos_id=None):
	max_len = max([len(s) for s in sentence_batch])
	output_batch = np.zeros([len(sentence_batch), max_len+1], np.int32)

	#x_0, x_1, ..., x_T, <EOS>, <PAD>, ...
	if eos_id is not None:
		for i, s in enumerate(sentence_batch):
			output_batch[i, :] = np.array(s + [eos_id] + [pad_id]*(max_len - len(s)))[:]
	
	#<SOS>, x_0, x_1, ..., x_T, <PAD>, ...
	elif sos_id is not None:
		for i, s in enumerate(sentence_batch):
			output_batch[i, :] = np.array([sos_id] + s + [pad_id]*(max_len - len(s)))[:]
	
	#x_0, x_1, ..., x_T, <PAD>, ...
	else:
		for i, s in enumerate(sentence_batch):
			output_batch[i, :] = np.array(s + [pad_id]*(max_len - len(s)))[:]

	return output_batch


#--------------------------
# Generate state-action batch
#--------------------------
def gen_state_action_batch(mb_sent, mb_last_idx, mb_size=64):
	sample_sent     = np.zeros([mb_size, mb_sent.shape[-1]], np.int32)
	sample_last_idx = np.zeros([mb_size, 2], np.int32)
	sample_actions  = np.zeros([mb_size], np.int32)

	#sent: <SOS>, x_0, x_1, ..., x_T, <EOS>, <PAD>, <PAD>, ...
	for i in range(mb_size):
		j = np.random.randint(0, mb_size)
		k = np.random.randint(0, mb_last_idx[j, 1]+1)

		sample_last_idx[i, 0] = i
		sample_last_idx[i, 1] = k
		sample_sent[i, :]     = mb_sent[j, :]
		sample_actions[i]     = mb_sent[j, k+1]
		
	return sample_sent, sample_last_idx, sample_actions


#--------------------------
# Sample state-action batch
#--------------------------
def sample_state_action_batch(mb_obs, mb_actions, mb_values, mb_returns, mb_last_idx, mb_size=64):
	mb_sample_obs      = np.zeros([mb_size, mb_obs.shape[-1]], np.int32)
	mb_sample_actions  = np.zeros([mb_size], np.int32)
	mb_sample_values   = np.zeros([mb_size])
	mb_sample_returns  = np.zeros([mb_size])
	mb_sample_last_idx = np.zeros([mb_size, 2], np.int32)

	for i in range(mb_size):
		j = np.random.randint(0, mb_size)
		k = np.random.randint(0, mb_last_idx[j, 1]+1)

		mb_sample_obs[i, :]      = mb_obs[j, k, :]
		mb_sample_actions[i]     = mb_actions[j, k]
		mb_sample_values[i]      = mb_values[j, k]
		mb_sample_returns[i]     = mb_returns[j, k]
		mb_sample_last_idx[i, 0] = i
		mb_sample_last_idx[i, 1] = k

	return mb_sample_obs, mb_sample_actions, mb_sample_values, mb_sample_returns, mb_sample_last_idx


#Parameters
#---------------------------
max_len = 32
mb_size = 128
dilations = [1, 2, 4, 8, 16]
z_dim = 64
embed_dim = 128
gamma = 0.95
lamb = 0.05
ent_weight = 0.01
value_weight = 0.5
max_grad_norm = 0.5
lr = 1e-4
lr_decay = 0.99
n_iter = 10000
disp_step = 10
save_step = 100
save_dir = "./save"
embed_save_dir = "./save_embed"


#Dataset
#---------------------------
token_data, w2i, i2w = pkl.load(open("./dataset/src_dataset.pkl", "rb"))
vocab_dim = len(w2i)


#Placeholder
#---------------------------
#sent_ph:           (mb_size, seq_len)
#given_sent_ph:     (mb_size, seq_len)
#fake_sent_ph:      (mb_size, seq_len)
#last_idx_ph:       (mb_size, 2)
#given_last_idx_ph: (mb_size, 2)
#fake_last_idx_ph:  (mb_size, 2)
#z_ph:              (mb_size, z_dim)
#action_ph:         (mb_size)
#return_ph:         (mb_size)
#adv_ph:            (mb_size)
sent_ph           = tf.placeholder(tf.int32, [None, None], name="sentence")
given_sent_ph     = tf.placeholder(tf.int32, [None, None], name="given_sentence")
fake_sent_ph      = tf.placeholder(tf.int32, [None, None], name="fake_sentence")
last_idx_ph       = tf.placeholder(tf.int32, [None, 2], name="last_idx")
given_last_idx_ph = tf.placeholder(tf.int32, [None, 2], name="given_last_idx_ph")
fake_last_idx_ph  = tf.placeholder(tf.int32, [None, 2], name="fake_last_idx")
z_ph              = tf.placeholder(tf.float32, [None, z_dim], name="z")
action_ph         = tf.placeholder(tf.int32, [None], name="action")
fake_action_ph    = tf.placeholder(tf.int32, [None], name="fake_action")
return_ph         = tf.placeholder(tf.float32, [None], name="return")
adv_ph            = tf.placeholder(tf.float32, [None], name="advantage")
lr_ph             = tf.placeholder(tf.float32, [])
x_ph              = tf.placeholder(tf.int32, [None])


#Model & runner
#---------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

word2vec_model = Word2vecModel(x_ph, vocab_dim, embed_dim)

model = SeqConvModel(
	sent_ph,
	given_sent_ph,
	fake_sent_ph,
	last_idx_ph,
	given_last_idx_ph,
	fake_last_idx_ph,
	action_ph,
	fake_action_ph,
	z_ph,
	dilations,
	vocab_dim,
	embed_dim
)

runner = Runner(
	sess, 
	sent_ph,
	given_sent_ph, 
	fake_sent_ph, 
	last_idx_ph,
	given_last_idx_ph,
	fake_last_idx_ph,
	fake_action_ph, 
	z_ph, 
	w2i["<sos>"], 
	w2i["<eos>"],
	gamma=gamma,
	lamb=lamb
)

word2vec_embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="word2vec/embed")
embed_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq_conv_gail/embed")
state_enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq_conv_gail/state_enc")
gen_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq_conv_gail/gen")
value_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq_conv_gail/value_net")
dis_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq_conv_gail/dis")
sa_dis_vars    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq_conv_gail/sa_dis")
t_vars = embed_vars+state_enc_vars+gen_vars+value_net_vars+dis_vars+sa_dis_vars
tf.contrib.slim.model_analyzer.analyze_vars(t_vars, print_info=True)


#Loss & opt
#---------------------------
#G_loss
neg_log_prob = model.distrib.neg_logp(action_ph)
pg_loss = tf.reduce_mean(adv_ph * neg_log_prob)
ent = tf.reduce_mean(model.distrib.entropy())
value_loss = tf.reduce_mean(tf.squared_difference(model.value, return_ph) / 2.0)
G_loss = pg_loss + ent_weight*ent + value_weight*value_loss

#D_loss
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits=model.logits_real,
	labels=tf.ones_like(model.prob_real)
))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits=model.logits_fake,
	labels=tf.zeros_like(model.prob_fake)
))
D_loss = D_loss_real + D_loss_fake

#D_sa_loss
D_sa_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits=model.sa_logits_real,
	labels=tf.ones_like(model.sa_prob_real)
))
D_sa_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits=model.sa_logits_fake,
	labels=tf.zeros_like(model.sa_prob_fake)
))
D_sa_loss = D_sa_loss_real + D_sa_loss_fake

#Opt
G_grads = tf.gradients(G_loss, embed_vars+state_enc_vars+gen_vars+value_net_vars)
G_grads, actor_grad_norm = tf.clip_by_global_norm(G_grads, max_grad_norm)
G_grads = list(zip(G_grads, embed_vars+state_enc_vars+gen_vars+value_net_vars))
G_opt = tf.train.AdamOptimizer(lr_ph).apply_gradients(G_grads)

D_opt = tf.train.AdamOptimizer(lr_ph).minimize(D_loss, var_list=embed_vars+state_enc_vars+dis_vars)
D_sa_opt = tf.train.AdamOptimizer(lr_ph).minimize(D_sa_loss, var_list=embed_vars+state_enc_vars+sa_dis_vars)


#Load the model
#---------------------------
sess.run(tf.global_variables_initializer())

if os.path.exists(embed_save_dir):
	saver = tf.train.Saver(var_list=word2vec_embed_vars, max_to_keep=2)
	ckpt = tf.train.get_checkpoint_state(embed_save_dir)
	if ckpt:
		print("Loading the word2vec weight ... ", end="")
		saver.restore(sess, ckpt.model_checkpoint_path)
		sess.run(tf.assign(model.embed_W, word2vec_model.embed_W))
		print("Done.")

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

saver = tf.train.Saver(var_list=t_vars, max_to_keep=2)
ckpt = tf.train.get_checkpoint_state(save_dir)
if ckpt:
	print("Loading the model ... ", end="")
	global_step = int(ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1])
	saver.restore(sess, ckpt.model_checkpoint_path)
	print("Done.")
else:
	global_step = 0


#Training
#---------------------------
n_data = len(token_data)
mb_z = np.random.randn(mb_size, z_dim)
mb_last_idx = np.zeros([mb_size, 2], np.int32)

for i in range(mb_size):
	mb_last_idx[i, 0] = i

for it in range(global_step, n_iter):
	#1. Train D (Adversarial)
	rand_idx = np.random.randint(0, n_data - mb_size)
	mb_token = token_data[rand_idx : rand_idx+mb_size]

	for j in range(mb_size):
		mb_last_idx[j, 1] = len(mb_token[j])

	mb_sent = pad_sentence_batch(mb_token, w2i["<pad>"], eos_id=w2i["<eos>"])
	mb_fake_sent, mb_fake_last_idx = runner.sample(model, mb_z, mb_size=mb_size, max_len=mb_sent.shape[-1])

	prob_fake_cur, D_loss_cur, _ = sess.run([model.prob_fake, D_loss, D_opt], feed_dict={
		sent_ph          : mb_sent,
		fake_sent_ph     : mb_fake_sent,
		last_idx_ph      : mb_last_idx,
		fake_last_idx_ph : mb_fake_last_idx,
		lr_ph            : exp_decay(lr, it, 10000, lr_decay)
	})

	#2. Train D_sa (Adversarial)
	mb_sent = np.insert(mb_sent, 0, np.array([w2i["<sos>"]]*mb_size, np.int32), axis=1)
	mb_fake_sent = np.insert(mb_fake_sent, 0, np.array([w2i["<sos>"]]*mb_size, np.int32), axis=1)

	sample_sent, sample_last_idx, sample_actions = gen_state_action_batch(
		mb_sent, 
 		mb_last_idx,
		mb_size=mb_size
	)
	sample_fake_sent, sample_fake_last_idx, sample_fake_actions = gen_state_action_batch(
		mb_fake_sent, 
 		mb_fake_last_idx,
		mb_size=mb_size//2
	)
	sample_fake_actions2 = runner.sample_action(model, mb_sent[:mb_size//2], mb_last_idx[:mb_size//2], mb_z)
	
	sample_fake_sent = np.concatenate([sample_fake_sent, mb_sent[:mb_size//2]], axis=0)
	sample_fake_last_idx = np.concatenate([sample_fake_last_idx, mb_last_idx[:mb_size//2]], axis=0)
	sample_fake_actions = np.concatenate([sample_fake_actions, sample_fake_actions2], axis=0)

	for j in range(mb_size):
		sample_fake_last_idx[j, 0] = j

	sa_prob_fake_cur, D_sa_loss_cur, _ = sess.run([model.sa_prob_fake, D_sa_loss, D_sa_opt], feed_dict={
		sent_ph          : sample_sent,
		fake_sent_ph     : sample_fake_sent,
		last_idx_ph      : sample_last_idx,
		fake_last_idx_ph : sample_fake_last_idx,
		action_ph        : sample_actions,
		fake_action_ph   : sample_fake_actions,
		lr_ph            : exp_decay(lr, it, 10000, lr_decay)
	})

	#3. Train G (Policy Gradient)
	mb_obs, mb_actions, mb_values, mb_returns, mb_fake_last_idx = runner.run(
		model, 
		mb_z, 
		mb_size=mb_size,
		max_len=max_len
	)
	sample_fake_sent, sample_fake_actions, sample_values, sample_returns, sample_fake_last_idx = sample_state_action_batch(
		mb_obs, 
		mb_actions, 
		mb_values,
		mb_returns, 
		mb_fake_last_idx, 
		mb_size=mb_size
	)
	sample_advs = sample_returns - sample_values
	sample_advs = (sample_advs - sample_advs.mean()) / (sample_advs.std() + 1e-8)

	G_loss_cur, ent_cur, _ = sess.run([G_loss, ent, G_opt], feed_dict={
		given_sent_ph     : sample_fake_sent,
		given_last_idx_ph : sample_fake_last_idx,
		action_ph         : sample_fake_actions,
		return_ph         : sample_returns,
		adv_ph            : sample_advs,
		lr_ph            : exp_decay(lr, it, 10000, lr_decay)
	})

	#4. Show the result
	if it % disp_step == 0:
		mb_gen_sent, mb_gen_last_idx = runner.sample(model, mb_z, mb_size=mb_size, max_len=24)
		avg_reward, running_reward, running_sent_reward = runner.get_performance()

		print("[{:5d} / {:5d}]".format(it, n_iter))
		print("-----------------------------------")
		print("D_loss              = {:.6f}".format(D_loss_cur))
		print("D_sa_loss           = {:.6f}".format(D_sa_loss_cur))
		print("G_loss              = {:.6f}".format(G_loss_cur))
		print("entropy             = {:.6f}".format(ent_cur))
		print("prob_fake           = {:.6f}".format(prob_fake_cur.mean()))
		print("sa_prob_fake        = {:.6f}".format(sa_prob_fake_cur.mean()))
		print("avg_reward          = {:.6f}".format(avg_reward))
		print("running_reward      = {:.6f}".format(running_reward))
		print("running_sent_reward = {:.6f}".format(running_sent_reward))
		print()

		for j in range(5):
			print(" ".join([i2w[t] for t in mb_gen_sent[j, :mb_gen_last_idx[j, 1]+1]]))
		print()

	#Save
	if it % save_step == 0:
		print("Saving the model ... ", end="")
		saver.save(sess, save_dir + "/model.ckpt", global_step=it)
		print("Done.")
		print()