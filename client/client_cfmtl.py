import numpy as np
import nest_asyncio
# np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from communication import COMM
import sys
import os
from time import *
import random
import data_pre as data
from sklearn.decomposition import PCA
import math
from tqdm import tqdm

from model_alex_full import model
from keras.utils import to_categorical


np.random.seed(0)
tf.set_random_seed(0)
config = tf.ConfigProto()

current_user_id = int(sys.argv[1])
NUM_OF_TOTAL_USERS = 7

LOW_SCALE = 10
HIGH_SCALE = 50
train_N_set = np.random.randint(low=LOW_SCALE, high=HIGH_SCALE, size=NUM_OF_TOTAL_USERS)
NUM_TRAIN_EXAMPLES_PER_USER = train_N_set[current_user_id]
# NUM_TRAIN_EXAMPLES_PER_USER = 10
NUM_TEST_EXAMPLES_PER_USER = 50

NUM_OF_CLASS = 3
D_OF_DATA = 900
_BATCH_SIZE = 5

# local_rho = 2*(1e-3)
alpha = 1*(1e-3)
beta = 5*(1e-4)
rho = 2*(1e-3)

local_iter = 20
oueter_iter = 50
initial_learning_rate = 1e-2
step_rate = 100
decay = 0.999

MODEL_INDICATE = 0

loss_record = np.zeros(2000)

# build the model and initialize the first training
W_DIM = 271203#751303 for 4 layers, 271203 for 2 layers
W = np.zeros(W_DIM)
F = np.zeros((NUM_OF_TOTAL_USERS, NUM_OF_TOTAL_USERS))
Omega = np.zeros((NUM_OF_TOTAL_USERS, W_DIM))
U = np.zeros((NUM_OF_TOTAL_USERS, W_DIM))

tf.reset_default_graph()

_SAVE_PATH = './save_model'

x, y, weights, logits, y_pred_cls, global_step, is_training = model() 

W_flat = tf.keras.backend.flatten(weights[0])
for model_i in range(1,len(weights)):
	temp = tf.keras.backend.flatten(weights[model_i])
	W_flat = tf.concat([W_flat, temp], 0)
# print(W_flat.shape)

# W_DIM = W_flat.shape
# USER_ID =  tf.placeholder(tf.float32, name="USER_ID")
F_i_tensor =  tf.placeholder(tf.float32, shape=[NUM_OF_TOTAL_USERS], name="F_i_tensor")
Omega_tensor =  tf.placeholder(tf.float32, shape=[NUM_OF_TOTAL_USERS, W_DIM], name="Omega_tensor")
U_tensor =  tf.placeholder(tf.float32, shape=[NUM_OF_TOTAL_USERS, W_DIM], name="U_tensor")

# LOSS
finalloss = 0.0
regular_1 = 0.0
regular_2 = 0.0

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
# regular_1 = alpha * tf.reduce_sum(tf.square(W_flat))
coe_1 = alpha + 0.5 * rho * tf.reduce_sum(tf.square(F_i_tensor))
regular_1 = coe_1 * tf.reduce_sum(tf.square(W_flat))
for j in range(NUM_OF_TOTAL_USERS):
      regular_2 += F_i_tensor[j] * tf.tensordot(U_tensor[j] + rho * Omega_tensor[j], W_flat, 1)

finalloss = loss + regular_1 + regular_2

#OPTIMIZER
local_learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, step_rate, decay, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=local_learning_rate).minimize(finalloss, var_list=None, global_step=global_step)
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
training_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# SAVER
with tf.name_scope('summaries'): 
  tf.summary.scalar('Training loss', finalloss)
merged = tf.summary.merge_all()

saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
saver_model = tf.train.Saver()

sess.run(tf.global_variables_initializer())

def shuffle_training_data(train_data, train_labels):
    seed = 0
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)
    return train_data, train_labels

def local_run(user_x_train, user_y_train, experiment_iter):
  
	# for outer_i in range(10):
	for epoch_i in range(local_iter):

	  # training process train(i)
		batch_size = int(math.ceil(len(user_x_train) / _BATCH_SIZE))
		train_x1, train_y1 = shuffle_training_data(user_x_train, user_y_train)
		print("Epoch : " + str(epoch_i+1))

		for s in tqdm(range(batch_size)):

			batch_xs = train_x1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
			batch_ys = train_y1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
 
			summary, i_global, _, model_weight, train_learning_rate, batch_loss, batch_acc = sess.run(
			        [merged, global_step, optimizer, weights, local_learning_rate, finalloss, training_accuracy],
			        feed_dict={x: batch_xs, y: batch_ys})
			train_writer.add_summary(summary, i_global)

		print("the training accuracy for epoch " + str(epoch_i+1)+ " loss: "+ str(batch_loss) + " accuracy: " + str(batch_acc*100))
		print(train_learning_rate)
		print(i_global)

	# saver.save(sess, save_path=_SAVE_PATH, global_step=i_global)

	return w_flat, batch_loss

def local_test(user_x_test, user_y_test, experiment_iter):

  # testing process
  i = 0
  test_data_len = user_x_test.shape[0]
  predicted_class = np.zeros(shape=test_data_len, dtype=np.int)
  while i < test_data_len:
      j = min(i + _BATCH_SIZE, test_data_len)
      test_batch_xs = user_x_test[i:j, :]
      test_batch_ys = user_y_test[i:j, :]
      predicted_class[i:j] = sess.run(
          y_pred_cls,
          feed_dict={x: test_batch_xs, y: test_batch_ys, is_training:False}
      )
      i = j

  correct = (np.argmax(user_y_test, axis=1) == predicted_class)
  test_acc = correct.mean()#*100
  correct_numbers = correct.sum()

  return test_acc

#load data and pre-process
x_coll, y_coll, dimension = data.load_data(current_user_id)
(x_train,x_test,y_train,y_test) = data.generate_data(NUM_TRAIN_EXAMPLES_PER_USER, NUM_TEST_EXAMPLES_PER_USER, x_coll, y_coll)
print(data.count_analysis(y_train))
print(data.count_analysis(y_test))
y_train = to_categorical(y_train, NUM_OF_CLASS)
y_test = to_categorical(y_test, NUM_OF_CLASS)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#prepare the communication module
server_addr = "localhost"
server_port = 9999
comm = COMM(server_addr,server_port,current_user_id)

comm.send2server('hello',-1)
print(comm.recvfserver())


outer_i = 0
sig_stop = 0

while True:

	for epoch_i in range(local_iter):

	  # training process train(i)
		batch_size = int(math.ceil(len(x_train) / _BATCH_SIZE))
		train_x1, train_y1 = shuffle_training_data(x_train, y_train)
		print("Epoch : " + str(outer_i*local_iter + epoch_i+1))

		for s in tqdm(range(batch_size)):

			batch_xs = train_x1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
			batch_ys = train_y1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
 
			summary, i_global, _, model_weight, batch_loss, batch_acc = sess.run(
			        [merged, global_step, optimizer, weights, finalloss, training_accuracy],
			        feed_dict={x: batch_xs, y: batch_ys,
			        F_i_tensor: F[current_user_id], Omega_tensor:Omega, U_tensor:U})
			train_writer.add_summary(summary, i_global)

		print("the training accuracy for epoch " + str(outer_i*local_iter + epoch_i+1)+ " loss: "+ str(batch_loss) + " accuracy: " + str(batch_acc*100))
		loss_record[outer_i*local_iter + epoch_i + 1] = batch_loss

	#get the weights and send to server
	w_flat = np.array([])
	for i in range(len(model_weight)):
		temp = model_weight[i].reshape(-1)
		w_flat = np.append(w_flat,temp)

	local_loss = batch_loss

	comm.send2server(w_flat,0)

	comm.send2server(local_loss,1)

	Omega, U, F, sig_stop = comm.recvOUF()

	print(F)
	# print(np.max(Omega))

	outer_i += 1

	if sig_stop == 1 or outer_i == oueter_iter:
		np.savetxt("F.txt", F)
		break

local_accuracy = local_test(x_test, y_test, MODEL_INDICATE)
print("clusterfl node %d: "%(current_user_id), local_accuracy)
loss_record[0] = local_accuracy

comm.disconnect(1)
sess.close()
np.savetxt("./loss_record/%d_local_loss.txt"%(current_user_id), loss_record)
