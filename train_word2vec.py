from word2vec_model import Word2vecModel
import numpy as np
import tensorflow as tf
import pickle as pkl
import os


#Parameters
#------------------------
n_iter = 300000
mb_size = 128
embed_dim = 128
n_sampled = 128
lr = 1e-4
disp_step = 5000
save_step = 50000
save_dir = "./save_embed"


#Dataset
#------------------------
token_data, w2i, i2w = pkl.load(open("./dataset/src_dataset.pkl", "rb"))
x_data, y_data = pkl.load(open("./dataset/src_skipgram.pkl", "rb"))

x_data = np.array(x_data)
y_data = np.array(y_data)
vocab_dim = len(w2i)

del token_data


#Placeholders
#------------------------
x_ph = tf.placeholder(tf.int32, [None])
y_ph = tf.placeholder(tf.int32, [None])


#Model
#------------------------
model = Word2vecModel(x_ph, vocab_dim, embed_dim)
tf.contrib.slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)


#Loss & opt
#------------------------
loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
	weights=model.softmax_W,
	biases=model.softmax_b,
	inputs=model.embed_x,
	labels=tf.expand_dims(y_ph, axis=1),
	num_sampled=n_sampled,
	num_classes=vocab_dim
))
opt = tf.train.AdamOptimizer(lr).minimize(loss)


#Load the model
#---------------------------
sess = tf.Session()
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


#Training
#------------------------
for it in range(global_step, n_iter):
	#Generate a batch
	rand_idx = np.random.randint(0, len(x_data)-mb_size)
	x_mb = x_data[rand_idx : rand_idx+mb_size]
	y_mb = y_data[rand_idx : rand_idx+mb_size]

	#Train
	loss_cur, _ = sess.run([loss, opt], feed_dict={
		x_ph: x_mb,
		y_ph: y_mb
	})

	#Show the result
	if it % disp_step == 0:
		nearest = model.find_nearest(sess, x_mb[:6])

		print("[{:5d} / {:5d}] loss = {:.6f}".format(it, n_iter, loss_cur))
		print("--------------------------")

		for i in range(6):
			log = "[{}]".format(i2w[x_mb[i]])

			for j in range(8):
				log = "{} {}".format(log, i2w[nearest[i, j]])

			print(log)
		print()

	#Save
	if it % save_step == 0:
		print("Saving the model ... ", end="")
		saver.save(sess, save_dir + "/model.ckpt", global_step=it)
		print("Done.")
		print()