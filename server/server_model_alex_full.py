# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.random.set_seed(0)
tf.set_random_seed(0)
import numpy as np
# from tensorflow.contrib.layers import flatten
# from tensorflow.python.tools import inspect_checkpoint
# import tensorflow.contrib as tf_contrib

def maxpool(x, ksize, strides, padding = "SAME"):
    """max-pooling layer"""
    return tf.nn.max_pool(x, 
                          ksize = [1, ksize, ksize, 1], 
                          strides = [1, strides, strides, 1], 
                          padding = padding, 
                          name='maxpooling')

def dropout(x, rate, is_training):
    """drop out layer"""
    return tf.layers.dropout(x, rate, name='dropout', training = is_training)

def lrn(x, depth_r=2, alpha=0.0001, beta=0.75, bias=1.0):
    """local response normalization"""
    return tf.nn.local_response_normalization(x, 
                                              depth_radius = depth_r, 
                                              alpha = alpha, 
                                              beta = beta, 
                                              bias = bias, 
                                              name='lrn')

def conv(x, ksize, strides, output_size, name, activation_func=tf.nn.relu, padding = "SAME", bias=0.0):
    """conv layer"""
    with tf.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w = tf.Variable(tf.random_normal([ksize, ksize, input_size, output_size], 
                        dtype=tf.float32, 
                        stddev=0.01), 
                        name='weights')
        b = tf.Variable(tf.constant(value=bias, 
                        dtype=tf.float32, 
                        shape=[output_size]), 
                        name='bias')
        
        conv = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)
        
        if activation_func:
            conv = activation_func(conv)
            
        return conv

def batch_norm(x, name, is_training):
    with tf.variable_scope(name):
        return tf_contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, updates_collections=None)

def fc(x, w_value, b_value, activation_func=tf.nn.relu):
    """fully connected layer"""

    # input_size = x.get_shape().as_list()[-1]
    w = tf.Variable(w_value, 
                    name='weights')
    b = tf.Variable(b_value, 
                    name='bias')

    out = tf.nn.xw_plus_b(x, w, b)
    if activation_func:
        return activation_func(out)
    
    else:
        return out

def model(x_image, weight_vector):

    _IMAGE_SIZE = 900
    _NUM_CLASSES = 3

    # layer 6
    with tf.name_scope('fc1layer'):
        w_value = tf.reshape(weight_vector[0:270000],[900,300])
        b_value = weight_vector[270000:270300]
        fc1 = fc(x=x_image, w_value=w_value, b_value=b_value)
        # print(fc.w)
        #print('fc1:{}'.format(fc1.get_shape().as_list()))
        # fc1 = dropout(fc1, keepPro, is_training)
        # fc1 = batch_norm(fc1, 'bn6', is_training)

    # layer 8 - output
    with tf.name_scope('fc2layer'):
        w_value = tf.reshape(weight_vector[270300:271200],[300,_NUM_CLASSES])
        b_value = weight_vector[271200:271203]
        logits = fc(x=fc1, w_value=w_value, b_value=b_value, activation_func=None)

    with tf.variable_scope('softmax'):
        softmax = tf.nn.softmax(logits=logits)


    y_pred_cls = tf.argmax(softmax, axis=1)

    return logits


