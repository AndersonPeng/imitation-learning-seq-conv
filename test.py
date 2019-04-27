import ops
from seq_conv_model import SeqConvModel
from runner import Runner
import tensorflow as tf
import numpy as np
import pickle as pkl
import os


#Parameters
#---------------------------
dilations = [1, 2, 4, 8, 16]
z_dim = 64
embed_dim = 128
alphas = [1e-8, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
disp_step = 100
save_dir = "./save"
result_dir = "./result"


#Dataset
#---------------------------
token_data, w2i, i2w = pkl.load(open("./dataset/src_dataset.pkl", "rb"))
vocab_dim = len(w2i)


for alpha in alphas:
	print("alpha = {:.2f}".format(alpha))
	print("----------------------------")
	tf.reset_default_graph()

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


	#Model & runner
	#---------------------------
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

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
		embed_dim,
		alpha=alpha
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


	#Load the model
	#---------------------------
	sess.run(tf.global_variables_initializer())

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)
	ckpt = tf.train.get_checkpoint_state(save_dir)
	if ckpt:
		print("Loading the model ... ", end="")
		global_step = int(ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1])
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Done.")
	else:
		global_step = 0


	#Testing
	#---------------------------
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)

	n_show = 20
	fp = open(os.path.join(result_dir, "sample_" + str(alpha) + ".txt"), "w")

	for i in range(8):
		mb_gen_sent, mb_gen_last_idx = runner.sample(model, np.random.randn(128, z_dim), mb_size=128, max_len=32)

		for j in range(128):
			fp.write(" ".join([i2w[t] for t in mb_gen_sent[j, :mb_gen_last_idx[j, 1]+1]]) + "\n")

	fp.close()

	mb_gen_sent, mb_gen_last_idx = runner.sample(model, np.random.randn(n_show, z_dim), mb_size=n_show, max_len=32)

	for j in range(n_show):
		print(" ".join([i2w[t] for t in mb_gen_sent[j, :mb_gen_last_idx[j, 1]+1]]))
	print()