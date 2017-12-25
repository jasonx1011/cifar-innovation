import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

PRINT_INFO_ON = False


def print_info(input_tensor, output_tensor, name):
    if PRINT_INFO_ON:
        print("==================================================")
        print("{}:".format(name))
        print("input x_tensor = {}".format(input_tensor))
        print("result = {}".format(output_tensor))
        print("==================================================")
    return


def input_image(image_shape, name):
    with tf.name_scope(name):
        result = tf.placeholder(tf.float32, shape=[None, *image_shape], name=name)
        tf.summary.histogram(name, result)
    print_info("Placeholder", result, name)
    return result


def input_label(n_classes, name):
    with tf.name_scope(name):
        result = tf.placeholder(tf.float32, shape=[None, n_classes], name=name)
        tf.summary.histogram(name, result)
    print_info("Placeholder", result, name)
    return result


def cust(x_tensor, name, func_str):
    # implement w2*sin(w0*x + b0) + w3*cos(w1*x + b1)
    dim = x_tensor.get_shape().as_list()
    print(dim)

    with tf.name_scope(name):
        weight_0 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        weight_1 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        weight_2 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        weight_3 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        weight_4 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        weight_mult = tf.Variable(tf.truncated_normal([1], stddev=0.05))
        bias_0 = tf.Variable(tf.zeros(dim[1:]))
        bias_1 = tf.Variable(tf.zeros(dim[1:]))

        if func_str == "iden_iden":
            p1_output = tf.identity(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.identity(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sin_cos":
            p1_output = tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.cos(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sin_iden":
            p1_output = tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.identity(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sin_tanh":
            p1_output = tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.tanh(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sin_relu":
            p1_output = tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.nn.relu(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "relu_relu":
            p1_output = tf.nn.relu(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.nn.relu(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sin_tan":
            p1_output = tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.tan(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sintan_sintan":
            p1_output = tf.sin(tf.tan(tf.add(tf.multiply(weight_0, x_tensor), bias_0)))
            p2_output = tf.sin(tf.tan(tf.add(tf.multiply(weight_1, x_tensor), bias_1)))
        elif func_str == "sintan_tansin":
            p1_output = tf.sin(tf.tan(tf.add(tf.multiply(weight_0, x_tensor), bias_0)))
            p2_output = tf.tan(tf.sin(tf.add(tf.multiply(weight_1, x_tensor), bias_1)))
        elif func_str == "sin_sin":
            p1_output = tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0))
            p2_output = tf.sin(tf.add(tf.multiply(weight_1, x_tensor), bias_1))
        elif func_str == "sinsin_sinsin":
            p1_output = tf.sin(tf.sin(tf.add(tf.multiply(weight_0, x_tensor), bias_0)))
            p2_output = tf.sin(tf.sin(tf.add(tf.multiply(weight_1, x_tensor), bias_1)))
        elif func_str == "sintan_tansin_iden":
            p1_output = tf.sin(tf.tan(tf.add(tf.multiply(weight_0, x_tensor), bias_0)))
            p2_output = tf.tan(tf.sin(tf.add(tf.multiply(weight_1, x_tensor), bias_1)))
            p3_output = tf.identity(x_tensor)
        else:
            raise SystemExit("func parameter in cust layer is invalid!")

        if func_str == "sintan_tansin_iden":
            w_p1 = tf.multiply(weight_2, p1_output)
            w_p2 = tf.multiply(weight_3, p2_output)
            w_p3 = tf.multiply(weight_4, p3_output)
            result = tf.add(tf.add(w_p1, w_p2), w_p3)
        else:
            w_p1 = tf.multiply(weight_2, p1_output)
            w_p2 = tf.multiply(weight_3, p2_output)
            result = tf.add(w_p1, w_p2)

        tf.summary.histogram(name, result)

    print_info(x_tensor, result, name)
    return result


def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name, option):
    """
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    """
    b, h, w, c = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], c,
                                              conv_num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    with tf.name_scope(name):
        conv_layer = tf.nn.conv2d(x_tensor, weight,
                                  strides=[1, conv_strides[0], conv_strides[1], 1], padding="SAME")
        conv_layer = tf.nn.bias_add(conv_layer, bias)
        if option == 0:
            result = cust(conv_layer, "cust")
        elif option == 1:
            result = tf.identity(conv_layer)
        elif option == 2:
            result = tf.sin(conv_layer)
        elif option == 3:
            result = tf.nn.relu(conv_layer)
        elif option == 4:
            result = tf.nn.relu6(conv_layer)
        elif option == 5:
            result = tf.cos(conv_layer)
        else:
            raise SystemExit("option out of range!")
        tf.summary.histogram(name, result)
    print_info(x_tensor, result, name)
    return result


def maxpool(x_tensor, pool_ksize, pool_strides, name):
    """
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    """
    with tf.name_scope(name):
        result = tf.nn.max_pool(x_tensor,
                                       ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                                       strides=[1, pool_strides[0], pool_strides[1], 1],
                                       padding='SAME', name=name)
        tf.summary.histogram(name, result)
    print_info(x_tensor, result, name)
    return result


def flatten(x_tensor, name):
    """
    Flatten input layer
    """
    # without the 1st param, which is Batch Size
    shape = x_tensor.get_shape().as_list()
    flatten_dim = np.prod(shape[1:])
    with tf.name_scope(name):
        result = tf.reshape(x_tensor, [-1, flatten_dim], name=name)
        tf.summary.histogram(name, result)
    print_info(x_tensor, result, name)
    return result


def fully_connect(x_tensor, num_outputs, name):
    """
    Apply a fully connection layer
    """
    # shape_list = x_tensor.get_shape().as_list()
    with tf.name_scope(name):
        result = tf.layers.dense(inputs=x_tensor,
                                 units=num_outputs,
                                 activation=None,
                                 # activation=tf.nn.relu,
                                 # activation=tf.nn.elu,
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 name=name)
        tf.summary.histogram(name, result)
    print_info(x_tensor, result, name)
    return result


def output(x_tensor, num_outputs, name):
    """
    Apply a output layer with linear output activation function
    """
    # shape_list = x_tensor.get_shape().as_list()
    # linear output activation function: activation = None
    with tf.name_scope(name):
        result = tf.layers.dense(inputs=x_tensor,
                                 units=num_outputs,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(),
                                 name=name)
        tf.summary.histogram(name, result)
    print_info(x_tensor, result, name)
    return result


def conv_net(x, n_classes, name, option):

    with tf.name_scope(name):
        conv_layer = conv2d(x, 128, (3, 3), (2, 2), "conv2d_layer_0", option)
        conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_0")

        conv_layer = conv2d(conv_layer, 256, (3, 3), (2, 2), "conv2d_layer_1", option)
        conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_1")

        flatten_layer = flatten(conv_layer, "flatten_layer_0")

        fully_layer = fully_connect(flatten_layer, 256, "fully_connect_layer_0")
        fully_layer = fully_connect(fully_layer, 128, "fully_connect_layer_1")

        output_layer = output(fully_layer, n_classes, "output_layer")

    return output_layer


def conv_cust_net(x, n_classes, name, option, func_str):

    with tf.name_scope(name):
        conv_layer = conv2d(x, 128, (3, 3), (2, 2), "conv2d_layer_0", option)
        conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_0")

        conv_layer = conv2d(conv_layer, 256, (3, 3), (2, 2), "conv2d_layer_1", option)
        conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_1")

        flatten_layer = flatten(conv_layer, "flatten_layer_0")

        # fully_layer = fully_connect(flatten_layer, 256, "fully_connect_layer_0")
        fully_layer = fully_connect(flatten_layer, 128, "fully_connect_layer_0")
        fully_layer = cust(fully_layer, "cust_fully_layer_0", func_str=func_str)
        # fully_layer = fully_connect(fully_layer, 128, "fully_connect_layer_1")

        output_layer = output(fully_layer, n_classes, "output_layer")

    return output_layer


def build(lr, act_option, net_option, func_str):

    image_shape = (32, 32, 3)
    n_classes = 10

    with tf.name_scope("inputs"):
        # Inputs
        x = input_image(image_shape, "input_image")
        y = input_label(n_classes, "input_label")

    # Model
    if net_option == 0:
        logits = conv_net(x, n_classes, "conv_net", act_option)
    elif net_option == 1:
        logits = conv_cust_net(x, n_classes, "conv_cust_net", act_option, func_str)
    else:
        raise SystemExit("net_option out of range!")

    with tf.name_scope("logits"):
        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

    with tf.name_scope("cost"):
        # Loss and Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("cost", cost)

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    with tf.name_scope("accuracy_scope"):
        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar("accuracy", accuracy)

    return x, y, logits, cost, optimizer, correct_pred, accuracy

