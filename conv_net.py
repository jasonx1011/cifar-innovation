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


def func(weight, x_tensor, bias, func_str):
    f_in = tf.add(tf.multiply(weight, x_tensor), bias)
    if func_str == "iden":
        res = tf.identity(f_in)
    elif func_str == "sin":
        res = tf.sin(f_in)
    elif func_str == "cos":
        res = tf.cos(f_in)
    elif func_str == "tan":
        res = tf.tan(f_in)
    elif func_str == "relu":
        res = tf.nn.relu(f_in)
    else:
        raise SystemExit("invalid func_str in def func!")
    return res


def cust(x_tensor, name, func_str, exp_opt):
    # implement w2*sin(w0*x + b0) + w3*cos(w1*x + b1)
    dim = x_tensor.get_shape().as_list()
    flatten_dim = 1
    for i in dim[1:]:
        flatten_dim = flatten_dim * i
    # flatten_dim_mult_2 = flatten_dim * 2
    # print("flatten_dim = {}".format(flatten_dim))
    # print("flatten_dim_mult_2 = {}".format(flatten_dim_mult_2))

    # print("cust_layer input dim = {}".format(dim))

    exp_opt = "constant"
    # exp_opt = "two_weights"
    # exp_opt = "dense"

    with tf.name_scope(name):
        if exp_opt == "constant":
            flag_const_w = True
        elif exp_opt == "two_weights" or exp_opt == "dense":
            flag_const_w = False
        else:
            raise SystemExit("invalid exp_opt!")

        if flag_const_w:
            weight_0 = tf.constant(1.0, shape=dim[1:])
            weight_1 = tf.constant(1.0, shape=dim[1:])
            bias_0 = tf.constant(0.0, shape=dim[1:])
            bias_1 = tf.constant(0.0, shape=dim[1:])
        else:
            weight_0 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
            weight_1 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
            bias_0 = tf.Variable(tf.zeros(dim[1:]))
            bias_1 = tf.Variable(tf.zeros(dim[1:]))

        weight_2 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        weight_3 = tf.Variable(tf.truncated_normal(dim[1:], stddev=0.05))
        bias_2 = tf.Variable(tf.zeros(dim[1:]))

        if exp_opt == "dense":
            str_list = func_str.split('_')
            if len(str_list) != 2:
                raise SystemExit("func_str does not contain exact two parameters!")
            else:
                print("str_list: {}".format(str_list))

                act_opt = []
                for item in str_list:
                    if item == "iden":
                        act_opt.append(None)
                    elif item == "sin":
                        act_opt.append(tf.sin)
                    elif item == "cos":
                        act_opt.append(tf.cos)
                    elif item == "tan":
                        act_opt.append(tf.tan)
                    elif item == "relu":
                        act_opt.append(tf.nn.relu)
                    else:
                        raise SystemExit("invalid func_str for exp_opt == 'dense'!")

                with tf.name_scope(name):
                    p1_output = tf.layers.dense(inputs=x_tensor,
                                                units=flatten_dim,
                                                activation=act_opt[0],
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                name=name + "p1")

                    p2_output = tf.layers.dense(inputs=x_tensor,
                                                units=flatten_dim,
                                                activation=act_opt[1],
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                name=name + "p2")
        else: # exp_opt == "constant" or "two_weights"
            str_list = func_str.split('_')
            if len(str_list) == 2:
                print("str_list: {}".format(str_list))
                p1_output = func(weight_0, x_tensor, bias_0, str_list[0])
                p2_output = func(weight_1, x_tensor, bias_1, str_list[1])
            else:
                raise SystemExit("func_str does not contain exact two parameters!")

        w_p1 = tf.multiply(weight_2, p1_output)
        w_p2 = tf.multiply(weight_3, p2_output)
        result = tf.add(tf.add(w_p1, w_p2), bias_2)

        tf.summary.histogram(name, result)

    print_info(x_tensor, result, name)
    return result


def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name, act_option):
    """
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param name: scope name
    :param act_option: activation function
    """
    b, h, w, c = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], c,
                                              conv_num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    with tf.name_scope(name):
        conv_layer = tf.nn.conv2d(x_tensor, weight,
                                  strides=[1, conv_strides[0], conv_strides[1], 1], padding="SAME")
        conv_layer = tf.nn.bias_add(conv_layer, bias)
        if act_option == "iden":
            result = tf.identity(conv_layer)
        elif act_option == "sin":
            result = tf.sin(conv_layer)
        elif act_option == "cos":
            result = tf.cos(conv_layer)
        elif act_option == "tan":
            result = tf.tan(conv_layer)
        elif act_option == "relu":
            result = tf.nn.relu(conv_layer)
        elif act_option == "relu6":
            result = tf.nn.relu6(conv_layer)
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


# def conv_net(x, n_classes, name, act_option):
#
#     with tf.name_scope(name):
#         conv_layer = conv2d(x, 128, (3, 3), (2, 2), "conv2d_layer_0", act_option)
#         conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_0")
#
#         conv_layer = conv2d(conv_layer, 256, (3, 3), (2, 2), "conv2d_layer_1", act_option)
#         conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_1")
#
#         flatten_layer = flatten(conv_layer, "flatten_layer_0")
#
#         fully_layer = fully_connect(flatten_layer, 256, "fully_connect_layer_0")
#         fully_layer = fully_connect(fully_layer, 128, "fully_connect_layer_1")
#
#         output_layer = output(fully_layer, n_classes, "output_layer")
#
#     return output_layer


def conv_cust_net(x, n_classes, name, pre_net_option, pre_net_act, cust_func_str, exp_opt):

    with tf.name_scope(name):
        if pre_net_option == "conv2d_maxpool":
            conv_layer = conv2d(x, 128, (3, 3), (2, 2), "conv2d_layer_0", pre_net_act)
            conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_0")

            conv_layer = conv2d(conv_layer, 256, (3, 3), (2, 2), "conv2d_layer_1", pre_net_act)
            conv_layer = maxpool(conv_layer, (3, 3), (2, 2), "maxpool_layer_1")

            flatten_layer = flatten(conv_layer, "flatten_layer_0")

        elif pre_net_option == "flatten":
            flatten_layer = flatten(x, "flatten_layer_0")
        else:
            raise SystemExit("invalid net_option!")

        # fully_layer = fully_connect(flatten_layer, 256, "fully_connect_layer_0")
        fully_layer = fully_connect(flatten_layer, 128, "fully_connect_layer_0")
        fully_layer = cust(fully_layer, "cust_fully_layer_0", func_str=cust_func_str, exp_opt=exp_opt)
        # fully_layer = fully_connect(fully_layer, 128, "fully_connect_layer_1")

        output_layer = output(fully_layer, n_classes, "output_layer")

    return output_layer


def build(lr, pre_net_option, pre_net_act, cust_func_str, exp_opt):

    image_shape = (32, 32, 3)
    n_classes = 10

    with tf.name_scope("inputs"):
        # Inputs
        x = input_image(image_shape, "input_image")
        y = input_label(n_classes, "input_label")

    # Model
    logits = conv_cust_net(x, n_classes, "conv_cust_net", pre_net_option, pre_net_act, cust_func_str, exp_opt)

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

