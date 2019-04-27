import ops
from seq_conv_model import SeqConvModel
from word2vec_model import Word2vecModel
from runner import Runner
import tensorflow as tf
import numpy as np
import pickle as pkl
import os


#----------------------------
# Pad a batch of sentences
#----------------------------
def pad_sentence_batch(sentence_batch, pad_id, sos_id=None, eos_id=None):
	max_len = max([len(s) for s in sentence_batch])
	output_batch = []

	#x_0, x_1, ..., x_T, <EOS>, <PAD>, ...
	if eos_id is not None:
		for s in sentence_batch:
			output_batch.append(s + [eos_id] + [pad_id]*(max_len - len(s)))
	
	#<SOS>, x_0, x_1, ..., x_T, <PAD>, ...
	elif sos_id is not None:
		for s in sentence_batch:
			output_batch.append([sos_id] + s + [pad_id]*(max_len - len(s)))
	
	#x_0, x_1, ..., x_T, <PAD>, ...
	else:
		for s in sentence_batch:
			output_batch.append(s + [pad_id]*(max_len - len(s)))

	return output_batch


#Parameters
#---------------------------
n_epoch = 4
mb_size = 128
dilations = [1, 2, 4, 8, 16]
z_dim = 64
embed_dim = 128
lr = 1e-4
disp_step = 200
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
	w2i["<eos>"]
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
seq_len = last_idx_ph[:, 1] + 1

loss = tf.reduce_mean(tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=model.logits,
	labels=sent_ph
), tf.sequence_mask(seq_len, tf.reduce_max(seq_len), dtype=tf.float32)))

opt = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(loss, var_list=embed_vars+state_enc_vars+gen_vars)


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
n_mb = n_data // mb_size
start_epoch = global_step // n_mb
mb_last_idx = np.zeros([mb_size, 2], np.int32)
mb_given_last_idx = np.zeros([mb_size, 2], np.int32)

for i in range(mb_size):
	mb_last_idx[i, 0] = i
	mb_given_last_idx[i, 0] = i

for i_epoch in range(start_epoch, n_epoch):
	#Save
	print("Saving the model ... ", end="")
	saver.save(sess, save_dir + "/model.ckpt", global_step=global_step)
	print("Done.")
	print()

	for idx in range(n_mb):
		#1. Generate a batch
		rand_idx = np.random.randint(0, n_data - mb_size)
		mb_token = token_data[rand_idx : rand_idx+mb_size]

		#sent:       x_0, x_1, ..., x_T, <EOS>
		#given_sent: <SOS>, x_0, x_1, ..., x_T
		for j in range(mb_size):
			mb_last_idx[j, 1] = len(mb_token[j])
			mb_given_last_idx[j, 1] = len(mb_token[j])

		mb_sent = np.array(pad_sentence_batch(mb_token, w2i["<pad>"], eos_id=w2i["<eos>"]))
		mb_given_sent = np.array(pad_sentence_batch(mb_token, w2i["<pad>"], sos_id=w2i["<sos>"]))
		mb_z = np.random.randn(mb_size, z_dim)

		#2. Train
		loss_cur, _ = sess.run([loss, opt], feed_dict={
			sent_ph           : mb_sent,
			given_sent_ph     : mb_given_sent,
			last_idx_ph       : mb_last_idx,
			given_last_idx_ph : mb_given_last_idx,
			z_ph              : mb_z
		})

		#3. Show the result
		if global_step % disp_step == 0:
			mb_gen_sent, mb_gen_last_idx = runner.sample(model, mb_z, mb_size=mb_size, max_len=24)

			print("----------------------------------------")
			print("[{:4d} / {:4d}] [{:5d} / {:5d}] loss = {:.4f}".format(
				i_epoch, start_epoch+n_epoch, idx, n_mb, loss_cur
			))

			for j in range(5):
				print(" ".join([i2w[t] for t in mb_gen_sent[j, :mb_gen_last_idx[j, 1]+1]]))
			print()

		global_step += 1