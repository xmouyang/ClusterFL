import socketserver
import pickle, struct
import sys
from threading import Lock, Thread
import threading
import numpy as np
from sklearn.decomposition import PCA
# from server_model_alex_full import model
from keras.utils import to_categorical
#import data_pre as data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from server_model_alex_full import model
# np.set_printoptions(threshold=np.inf)

NUM_OF_TOTAL_USERS = 7
NUM_OF_WAIT = 7
W_DIM = 271203 #751303 for 4 layers, 271203 for 2 layers
inner_iteration = 3
T_thresh = 10

iteration_count = 0
regular = 1e5
alpha = 1*(1e-3)
beta = 5*(1e-4)
rho = 2*(1e-3)

W = np.zeros((NUM_OF_TOTAL_USERS,W_DIM))
F = np.zeros((NUM_OF_TOTAL_USERS,NUM_OF_TOTAL_USERS))
KL_distance = np.zeros((NUM_OF_TOTAL_USERS,NUM_OF_TOTAL_USERS))
Omega = np.zeros((NUM_OF_TOTAL_USERS, W_DIM))
U = np.zeros((NUM_OF_TOTAL_USERS,W_DIM))
Loss = np.zeros(NUM_OF_TOTAL_USERS)
Loss[Loss<np.inf] = 1e5
Loss_cache = np.zeros(NUM_OF_TOTAL_USERS)
conver_indicator = 1e5
loss_record = np.zeros(1000)
normalized_dloss = np.zeros((NUM_OF_TOTAL_USERS,T_thresh))
update_flag = np.ones(NUM_OF_TOTAL_USERS)

NUM_OF_CLASS = 3
softmax_temperature = 6.0

server_x_test = np.loadtxt("server_x_test.txt")
server_y_test = np.loadtxt("server_y_test.txt")



def judge_stragglers():
	temp_dloss = np.zeros(NUM_OF_TOTAL_USERS)
	if iteration_count > 0:
		dloss_iter = (iteration_count-1)%T_thresh

		for i in range(NUM_OF_TOTAL_USERS):
			if(update_flag[i]==1):
				temp_dloss[i] = abs(Loss[i]-Loss_cache[i])
			else:
				temp_dloss[i] = 0

		for i in range(NUM_OF_TOTAL_USERS):
			cluster_id = int(i/3)
			sum_cluster = 0
			for j in range(int(NUM_OF_TOTAL_USERS/3)):
				sum_cluster += temp_dloss[cluster_id*3+j]
			if sum_cluster < 1e-6:
				normalized_dloss[i,dloss_iter] = 0
			else:
				normalized_dloss[i,dloss_iter] = temp_dloss[i]/sum_cluster
		gamma_q = np.mean(normalized_dloss,axis=1)

		if np.any(gamma_q > 1e-6):
			for i in range(NUM_OF_TOTAL_USERS):
				cluster_id = int(i/3)
				other_gamma_cluster = 0
				for j in range(int(NUM_OF_TOTAL_USERS/3)):
					other_gamma_cluster += gamma_q[cluster_id*3+j]
				other_gamma_cluster -= gamma_q[i]
				if (gamma_q[i] / other_gamma_cluster) > 10:
					update_flag[i] = 0
	return 0

def model_output_test(W_i):

	# how to feed the model with weight W_i
	node_weight = W_i

	# print("right here 0")

	node_logits = model(server_x_test, node_weight)

	return node_logits

def server_KL_distance():

	output_user = []

	for user_id in range(NUM_OF_TOTAL_USERS):
		output_user.append(model_output_test(W[user_id]))

	# print("right here 1")

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for main_user_id in range(NUM_OF_TOTAL_USERS):

		for user_id in range(NUM_OF_TOTAL_USERS):

			main_output = output_user[main_user_id]/softmax_temperature
			other_output = output_user[user_id]/softmax_temperature
			KL_tensor = tf.reduce_mean(tf.nn.softmax(main_output, 1) * (tf.nn.log_softmax(main_output, 1) - tf.nn.log_softmax(other_output, 1) ))

			KL_distance[main_user_id, user_id] = sess.run(KL_tensor)

			# print("right here 2")
			print(KL_distance[main_user_id, user_id])

def server_update_F():

	server_KL_distance()
	print(KL_distance)

	# print("right here 3")

	# centered Kl matrix
	centered_matrix = np.zeros((NUM_OF_TOTAL_USERS, NUM_OF_TOTAL_USERS))
	mean_w = np.mean(KL_distance, axis=0)
	for user_id in range(NUM_OF_TOTAL_USERS):
		centered_matrix[user_id] = KL_distance[user_id] - mean_w

	global F
	F = np.zeros((NUM_OF_TOTAL_USERS, NUM_OF_TOTAL_USERS))
	
	pca = PCA(n_components= NUM_OF_TOTAL_USERS - 1)
	
	compoments = pca.fit_transform(centered_matrix) # use KL_distance
	# compoments = pca.fit_transform(W)
	
	#(m, m-1) * (m-1,m)
	P = np.dot(compoments, compoments.T)
	for i in range(NUM_OF_TOTAL_USERS):
	    for j in range(NUM_OF_TOTAL_USERS):
	        if P[i,j] < 0:
	            P[i,j] = 0
	print("P:",P)

	# temp_relation = np.zeros((self.m, self.m))
	# for i in range(self.m):
	#     for j in range(self.m):
	#         temp_relation[i,j] = P[i,j] / sqrt(P[i,i] * P[j,j])
	        # if temp_relation[i,j] < 0.2:
	        #     P[i,j] = 0
	        
	count_positive = np.zeros(NUM_OF_TOTAL_USERS)
	for j in range(NUM_OF_TOTAL_USERS):
	    sum_column = np.sum(P[:,j])
	    for i in range(NUM_OF_TOTAL_USERS):
	        if P[i,j] >= 1e-10:
	            count_positive[j] += 1
	        P[i,j] = P[i,j] / sum_column

	F = P
	# for j in range(NUM_OF_TOTAL_USERS):
	#     F[:,j] /= np.sqrt(count_positive[j])

	print("F:",F)

def server_update():
	
	# Solve for Omega_t+1
	for j in range(NUM_OF_TOTAL_USERS):
		Omega[j] = (1.0 / (rho - 2 * beta)) * (rho * np.dot(F[:,j],W) - U[j])
	# Solve for U_t+1
	for j in range(NUM_OF_TOTAL_USERS):
		U[j] = U[j] + rho * (Omega[j] - np.dot(F[:,j],W))

	global iteration_count
	iteration_count+=1
	if iteration_count%inner_iteration == 0:
		server_update_F()

	global regular
	omega = np.dot(F.T,W)
	regular = alpha*np.trace(np.dot(W,W.T)) - beta * np.trace(np.dot(omega,omega.T))

	global conver_indicator
	conver_indicator = 0
	print("Loss: ",Loss)
	print("Loss_cache: ",Loss_cache)

	for i in range(NUM_OF_TOTAL_USERS):
		if update_flag[i]==1:
			conver_indicator += np.abs(Loss[i]-Loss_cache[i])

	loss_record[iteration_count] = np.sum(Loss)

	# judge_stragglers()
	print("conver_indicator: ",conver_indicator)
	
def reinitialize():
	print(Loss_cache,Loss)
	global iteration_count
	print("The iteration number: ", iteration_count)
	W[W<np.inf] = 0
	F[F<np.inf] = 0
	Omega[Omega<np.inf] = 0
	U[U<np.inf] = 0
	Loss[Loss<np.inf] = 1e5
	Loss_cache[Loss_cache<np.inf] = 0
	loss_record[loss_record<np.inf] = 0
	update_flag[update_flag<np.inf] = 1
	normalized_dloss[normalized_dloss<np.inf] = 0
	iteration_count = 0
	global NUM_OF_WAIT
	NUM_OF_WAIT = 7

	global regular
	regular = 1e5
	
	global conver_indicator
	conver_indicator = 1e5
	barrier_update()

	#for i in range(300):
		#print(loss_record[i])

barrier_start = threading.Barrier(NUM_OF_WAIT,action = None, timeout = None)
barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)

def barrier_update():
	global NUM_OF_WAIT
	print("update the barriers to NUM_OF_WAIT: ",NUM_OF_WAIT)
	global barrier_W
	barrier_W = threading.Barrier(NUM_OF_WAIT,action = server_update, timeout = None)
	global barrier_end
	barrier_end = threading.Barrier(NUM_OF_WAIT, action = reinitialize, timeout = None)

class MyTCPHandler(socketserver.BaseRequestHandler):
	def handle(self):
		while True:
			try:
				#receive the size of content
				header = self.request.recv(4)
				size = struct.unpack('i', header)

				#receive the id of client
				u_id = self.request.recv(4)
				user_id = struct.unpack('i',u_id)

				# receive the type of message, defination in communication.py
				mess_type = self.request.recv(4)
				mess_type = struct.unpack('i',mess_type)[0]

				#print("This is the {}th node with message type {}".format(user_id[0],mess_type))

				#receive the body of message
				recv_data = b""
				
				while sys.getsizeof(recv_data)<size[0]:
					recv_data += self.request.recv(size[0]-sys.getsizeof(recv_data))
				
				#if hello message, barrier until all clients arrive and send a message to start
				if mess_type == -1:
					try:
						barrier_start.wait(120)
					except Exception as e:
						print("start wait timeout...")

					start_message = 'start'
					start_data = pickle.dumps(start_message, protocol = 0)
					size = sys.getsizeof(start_data)
					header = struct.pack("i",size)
					self.request.sendall(header)
					self.request.sendall(start_data)


				#if W message, update Omega and U or F
				elif mess_type == 0:
					weights = pickle.loads(recv_data)
					W[user_id] = weights

					try:
						barrier_W.wait(120)
					except Exception as e:
						print("wait W timeout...")

					Omega_data = pickle.dumps(Omega, protocol = 0)
					Omega_size = sys.getsizeof(Omega_data)
					Omega_header = struct.pack("i",Omega_size)
					#print("The Omega matrix is like: \n",Omega)
					self.request.sendall(Omega_header)
					self.request.sendall(Omega_data)
					# print("send Omega to client {} with the size of {}".format(user_id[0],size))


					U_data = pickle.dumps(U,protocol = 0)
					U_size = sys.getsizeof(U_data)
					U_header = struct.pack("i",U_size)
					#print("The U matrix is like: \n",U)
					self.request.sendall(U_header)
					self.request.sendall(U_data)
					# print("send U to client {} with the size of {}".format(user_id[0],size))
					

					F_data = pickle.dumps(F, protocol = 0)
					F_size = sys.getsizeof(F_data)
					F_header = struct.pack("i",F_size)
					self.request.sendall(F_header)
					self.request.sendall(F_data)
					# print("send F to client {} with the shape of {}...".format(user_id[0],F.shape))
					
					# print(update_flag)
					global conver_indicator
					# print("conver_indicator: ", conver_indicator)
					#if convergence, stop all the clients
					if update_flag[user_id]==0:
						sig_stop = struct.pack("i",2)
						global NUM_OF_WAIT
						NUM_OF_WAIT-=1
						barrier_update()
						self.finish()

					elif(np.abs(conver_indicator)<1e-2):
						sig_stop = struct.pack("i",1)
					else:
						sig_stop = struct.pack("i",0)
					self.request.sendall(sig_stop)


				# if Loss message, record the loss
				elif mess_type == 1:

					loss = pickle.loads(recv_data)
					Loss_cache[user_id] = Loss[user_id]
					Loss[user_id] = (loss + regular)/NUM_OF_TOTAL_USERS

				elif mess_type == 9:
					break

				elif mess_type == 10:
					try:
						barrier_end.wait(5)
					except Exception as e:
						print("finish timeout...")
					break


			except Exception as e:
				print('err',e)
				break



if __name__ == "__main__":
	HOST, PORT = "0.0.0.0", 9999 
	server = socketserver.ThreadingTCPServer((HOST,PORT),MyTCPHandler)
	server.serve_forever(poll_interval = 0.5)
