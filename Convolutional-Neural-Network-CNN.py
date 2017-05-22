import tensofrlow as tf
import numpy as np


def one_hot_encode(x):
    """
    This function one hot encodes a list of sample labels and return a one-hot encoded vector for each label.
    """
    import numpy as np
    arr = np.zeros([len(x),10])
    for i in range(len(x)):
        arr[(i,x[i])]=1
    return arr


def neural_net_image_input(image_shape):
    shape=[None,image_shape[0],image_shape[1],image_shape[2]]
    return tf.placeholder(dtype=tf.float32, shape=(shape), name="x")


def neural_net_label_input(n_classes):
    return tf.placeholder(dtype=tf.float32, name="y", shape=(None,n_classes))



def neural_net_keep_prob_input():
    return tf.placeholder(dtype=tf.float32, name="keep_prob")
    
    
    
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    This function applies convolution layer then max pooling layer to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    """
 
    
    weight = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[-1],
                                           conv_num_outputs], stddev=0.1))
    
    bias = tf.Variable(tf.zeros(conv_num_outputs, dtype=tf.float32))
    conv_layer = tf.nn.conv2d(x_tensor, weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    nonlinear_activation_layer = tf.nn.relu(conv_layer)
    max_pooling_layer = tf.nn.max_pool(nonlinear_activation_layer, ksize=[1, pool_ksize[0], pool_ksize[1], 1], 
                                strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return max_pooling_layer
    
    
    
def fully_conn(x_tensor, num_outputs):
    num_features = x_tensor.shape[1].value
    weights = tf.Variable(tf.random_normal([num_features, num_outputs], stddev=0.1))
    biases = tf.Variable(tf.zeros([num_outputs]))
    fully_connected = tf.add(tf.matmul(x_tensor, weights), biases)
    fully_connected = tf.nn.relu(fully_connected)
    return fully_connected
    
    
    
 def output(x_tensor, num_outputs):
    num_features = x_tensor.shape[1].value
    weights = tf.Variable(tf.random_normal([num_features, num_outputs], stddev=0.1))
    biases = tf.Variable(tf.zeros([num_outputs]))
    output_layer = tf.add(tf.matmul(x_tensor, weights), biases)
    return output_layer 
    
    
 def conv_net(x, keep_prob):
    """
    This function creates a Convolutional Neural Network model
    """
    conv2d_maxpool_layer_1 = conv2d_maxpool(x, 64, (8,8), (4,4), (4,4),(2,2))
    conv2d_maxpool_layer_2 = conv2d_maxpool(conv2d_maxpool_layer_1, 64, (8,8), (4,4), (4,4),(2,2))
    conv2d_maxpool_layer_3 = conv2d_maxpool(conv2d_maxpool_layer_2, 64, (8,8), (4,4), (4,4),(2,2))   

    flatten_layer = flatten(conv2d_maxpool_layer_3)
    flatten_layer = tf.nn.dropout(flatten_layer, keep_prob)

    fully_conn_layer = fully_conn(flatten_layer, 256)
    fully_conn_layer = tf.nn.dropout(fully_conn_layer, keep_prob)

    return output(fully_conn_layer, 10)
