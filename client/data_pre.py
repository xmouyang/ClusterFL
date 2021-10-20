from PIL import Image
import cv2 as cv
import numpy as np
np.set_printoptions(threshold = np.inf)
import matplotlib.pyplot as plt
from skimage import data_dir,io,color
from sklearn.model_selection import train_test_split
import random
random.seed(0)

cluster_set = ['hsh_', 'mmw_']
class_set = ['_walk','_up','_down']
label = [0,1,2]

NUM_OF_CLASS = 3
DIMENSION_OF_FEATURE = 900


def load_data(user_id):

	# dataset append and split

	if user_id < 4:
		cluster_id = 0
		intra_user_id = user_id + 1 # 0,1,2,3 to 1,2,3,4
	else:
		cluster_id = 1
		intra_user_id = user_id - 3 # 4,5,6 to 1,2,3

	# x append
	cluster_des = str(cluster_set[cluster_id])

	coll_class = []
	coll_label = []

	for class_id in range(NUM_OF_CLASS):
	
		read_path = '/Users/ouyangxiaomin/Desktop/fed_datasets/imu_data_7/' + \
		cluster_des + str(intra_user_id) + str(class_set[class_id]) + '_nor' + '.txt'

		temp_original_data = np.loadtxt(read_path,delimiter=',')
		temp_coll = temp_original_data.reshape(-1, DIMENSION_OF_FEATURE)
		count_img = temp_coll.shape[0]
		temp_label = class_id * np.ones(count_img)

		# print(temp_original_data.shape)
		# print(temp_coll.shape)

		coll_class.extend(temp_coll)
		coll_label.extend(temp_label)

	coll_class = np.array(coll_class)
	coll_label = np.array(coll_label)

	print(coll_class.shape)
	print(coll_label.shape)

	return coll_class, coll_label, DIMENSION_OF_FEATURE


def generate_data(num_of_train_per_node, num_of_test_per_node, x_coll, y_coll):
	

	node_x_train = np.zeros((num_of_train_per_node, DIMENSION_OF_FEATURE))
	node_y_train = np.zeros(num_of_train_per_node)
	node_x_test = np.zeros((num_of_test_per_node, DIMENSION_OF_FEATURE))
	node_y_test = np.zeros(num_of_test_per_node)

	test_percent = 0.4

	x_train,x_test,y_train,y_test = \
	train_test_split(x_coll,y_coll,test_size = test_percent,random_state = 0)

	num_of_all_train_data = y_train.shape[0]
	num_of_all_test_data = y_test.shape[0]
	train_index = random.sample(range(0,num_of_all_train_data), num_of_train_per_node)
	test_index = random.sample(range(0,num_of_all_test_data), num_of_test_per_node)

	# train sample in one node
	for train_id in range(num_of_train_per_node):

		node_x_train[train_id, :] = np.array(x_train[train_index[train_id]]).flatten().astype('float32')
		node_y_train[train_id] = y_train[train_index[train_id]]

	# test sample of one class in one node
	for test_id in range(num_of_test_per_node):

		node_x_test[test_id, :] = np.array(x_test[test_index[test_id]]).flatten().astype('float32')
		node_y_test[test_id] = y_test[test_index[test_id]]

	return node_x_train,node_x_test,node_y_train,node_y_test


def count_analysis(y):

	count_class = np.zeros(NUM_OF_CLASS)

	for class_id in range(NUM_OF_CLASS):
		count_class[class_id] = np.sum( y == class_id )

	return count_class


# NUM_OF_USERS = 9
# NUM_TRAIN_EXAMPLES_PER_USER = (50 * np.ones(NUM_OF_USERS)).astype(int)
# NUM_TEST_EXAMPLES_PER_USER = (40 * np.ones(NUM_OF_USERS)).astype(int)

# [x_coll, y_coll, D] = load_data()

# node_train_index= node_define(NUM_TRAIN_EXAMPLES_PER_USER, NUM_OF_USERS)
# node_test_index= node_define(NUM_TEST_EXAMPLES_PER_USER, NUM_OF_USERS)

# (x_train, y_train, x_test, y_test) = \
# generate_data(NUM_TRAIN_EXAMPLES_PER_USER, NUM_TEST_EXAMPLES_PER_USER, NUM_OF_USERS, node_train_index, node_test_index, x_coll, y_coll)

# count_train_class = count_analysis(y_train, node_train_index, NUM_OF_USERS)
# count_test_class = count_analysis(y_test, node_test_index, NUM_OF_USERS)
# print(count_train_class)
# print(count_test_class)


# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y_test)





