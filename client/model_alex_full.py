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

def fc(x, output_size, name, activation_func=tf.nn.relu):
    """fully connected layer"""
    with tf.variable_scope(name):
        input_size = x.get_shape().as_list()[-1]
        w = tf.Variable(tf.random_normal([input_size, output_size], 
                        dtype=tf.float32, 
                        stddev=0.01), 
                        name='weights')
        b = tf.Variable(tf.constant(value=0.0, 
                        dtype = tf.float32, 
                        shape=[output_size]), 
                        name='bias')
    
        out = tf.nn.xw_plus_b(x, w, b)
        if activation_func:
            return activation_func(out)
        
        else:
            return out

def batch_norm(x, name, is_training):
    with tf.variable_scope(name):
        return tf_contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, updates_collections=None)

def model():
    _IMAGE_SIZE = 900
    _NUM_CLASSES = 3

    with tf.variable_scope('teacher_alex'):
        # keepPro = 0.5
        #temperature = 1.0
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE ], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

        is_training = tf.placeholder_with_default(True, shape=())

        x_image = x

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

        # layer 6
        with tf.name_scope('fc1layer'):
            fc1 = fc(x=x_image, output_size=300, name='fc1')

        # layer 8 - output
        with tf.name_scope('fc2layer'):
            logits = fc(x=fc1, output_size=_NUM_CLASSES, activation_func=None, name='fc2')

        with tf.variable_scope('softmax'):
            softmax = tf.nn.softmax(logits=logits)

    weights = []
    weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc1layer/fc1/weights:0'))
    weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc1layer/fc1/bias:0'))
    weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc2layer/fc2/weights:0'))
    weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc2layer/fc2/bias:0'))
    # weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc3layer/fc3/weights:0'))
    # weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc3layer/fc3/bias:0'))
    # weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc4layer/fc4/weights:0'))
    # weights.append(tf.get_default_graph().get_tensor_by_name('teacher_alex/fc4layer/fc4/bias:0'))

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, weights, logits, y_pred_cls, global_step, is_training


