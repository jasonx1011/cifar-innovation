from datetime import datetime
import matplotlib.pyplot as plt
import os
import pickle
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
import timeit

from preprocess import preprocess_all

import conv_net
import file_manager

# downloaded and unzipped from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
CIFAR10_DIR = "./cifar-10-batches-py/"

# preprocess output dir
PRE_DATA_DIR = "./pre_data/"

# CIFAR10 label names
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# tensorboard log dir
TF_LOGDIR = "./tf_logs/"

# tensorboard log dir
SAVE_DIR = "./save/"

# tensorboard log dir
RESTORE_RUN_SAVE_EDIR = "./restore_run_save/"

def train_network(session, optimizer, x, y, feature_batch, label_batch):
    session.run([optimizer], feed_dict={x: feature_batch,
                                        y: label_batch})
    return None


def print_stats(session, merged, x, y, feature_batch, label_batch, valid_features, valid_labels, cost, accuracy):
    loss, summary_train = session.run([cost, merged], feed_dict={x: feature_batch,
                                                                 y: label_batch})

    valid_acc, summary_valid = session.run([accuracy, merged], feed_dict={x: valid_features,
                                                                          y: valid_labels})

    print('batch training loss = {:>5.3f}, valid_acc = {:>1.3f}'.format(loss, valid_acc))

    return summary_train, summary_valid


def next_batch(x, y, batch_size, re_shuffle):
    if len(x) != len(y):
        raise SystemError("In def next_batch(x, y), len(x) != len(y)")

    idx = 0
    while idx < len(x):
        batch_x = x[idx: idx + batch_size]
        batch_y = y[idx: idx + batch_size]

        if re_shuffle:
            batch_x, batch_y = shuffle(batch_x, batch_y)

        yield batch_x, batch_y
        idx += batch_size


def make_hparam_string(lr, epochs, batch_size, act_option, net_option, func_str):
    log_timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    hparam_str = "lr_{:.0E},{},{},act_op_{},net_op_{},{},{}".format(lr, epochs, batch_size,
                                                                 act_option, net_option,
                                                                 func_str, log_timestr)
    # hparam_str = "lr_{:.0E},{},{},h_layers".format(lr, epochs, batch_size)
    # idx = 0
    # while idx < len(hidden_layers):
    #     hparam_str = hparam_str + "_" + str(hidden_layers[idx])
    #     idx += 1
    return hparam_str


def run_graph(train_set, valid_set, lr, epochs, batch_size, turn_on_tb, act_option, net_option, func_str, restore_model):
    # unpack training set and validation set

    train_features, train_labels = train_set
    valid_features, valid_labels = valid_set

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # build up nets
    x, y, logits, cost, optimizer, correct_pred, accuracy = conv_net.build(lr, act_option, net_option, func_str)

    hparam_str = make_hparam_string(lr, epochs, batch_size, act_option, net_option, func_str)
    print("=====================")
    print("{}".format(hparam_str))
    print("=====================")

    merged = tf.summary.merge_all()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # for i in [x, y, logits, cost, optimizer, correct_pred, accuracy]:
    #     print("===")
    #     print(i)
    #     print("===")

    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        if restore_model:
            # Restore variables from disk.
            saver.restore(sess, os.path.join(SAVE_DIR) + "model.ckpt")
            print("Model restored.")
        else:
            sess.run(tf.global_variables_initializer())

        if turn_on_tb:
            train_writer = tf.summary.FileWriter(TF_LOGDIR + hparam_str + "/train/", sess.graph)
            valid_writer = tf.summary.FileWriter(TF_LOGDIR + hparam_str + "/valid/")

        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in next_batch(train_features, train_labels, batch_size, re_shuffle=False):
                train_network(sess, optimizer, x, y, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            summary_train, summary_valid = print_stats(sess, merged, x, y, batch_features, batch_labels,
                                                       valid_features, valid_labels, cost, accuracy)

            if turn_on_tb:
                train_writer.add_summary(summary_train, epoch)
                valid_writer.add_summary(summary_valid, epoch)

        # Save the variables to disk.
        if restore_model:
            save_path = saver.save(sess, os.path.join(RESTORE_RUN_SAVE_EDIR) + "model.ckpt")
        else:
            save_path = saver.save(sess, os.path.join(SAVE_DIR) + "model.ckpt")
        print("Model saved in file: {}".format(save_path))

    return


def main():
    # preprocess_raw_data = True
    preprocess_raw_data = False
    turn_on_tb = True

    if preprocess_raw_data:
        file_manager.backup_files([PRE_DATA_DIR])
        preprocess_all(CIFAR10_DIR, PRE_DATA_DIR)

    if turn_on_tb:
        file_manager.backup_files([TF_LOGDIR])

    # it's a tuple of (*_features, *_labels)
    train_set = pickle.load(open(PRE_DATA_DIR + 'train_batch_1.p', mode='rb'))
    valid_set = pickle.load(open(PRE_DATA_DIR + 'validation.p', mode='rb'))
    test_set = pickle.load(open(PRE_DATA_DIR + 'test.p', mode='rb'))

    # for i in [train_set, valid_set, test_set]:
    #     print("*_features, *_labels")
    #     print(type(i[0]), type(i[1]))
    #     print(i[0].shape, i[1])
    rand_idx = randint(0, len(train_set[0] - 1))
    plt.imshow(train_set[0][rand_idx])
    print(LABEL_NAMES)
    print(train_set[1][rand_idx])
    plt.show()

    restore_model = True
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.isdir(RESTORE_RUN_SAVE_EDIR):
        os.makedirs(RESTORE_RUN_SAVE_EDIR)

    # Hyper parameters default values
    # AdamOptimizer default initial lr = 0.001 = 1e-3
    # lr = 4E-3
    lr = 3E-3
    # epochs = 100
    # epochs = 20
    epochs = 5
    batch_size = 512
    # batch_size = 32
    # hidden_layers = [16, 32]

    # for act_option in [1, 2, 3, 4]:
    for act_option in [1]:
        for net_option in [1]:
            if net_option == 1:
                # for func_str in ["sin_x_tan", "sin_cos", "identity"]:
                # for func_str in ["sin_iden", "sin_relu", "sin_sin", "iden_iden"]:
                # for func_str in ["iden_iden", "sin_tanh"]:
                # for func_str in ["iden_iden"]:
                # for func_str in ["sintan_tansin"]:
                for func_str in ["sin_cos"]:
                # for func_str in ["sintan_tansin", "iden_iden", "sin_sin"]:
                    start_time = timeit.default_timer()
                    run_graph(train_set, valid_set, lr, epochs, batch_size,
                              turn_on_tb, act_option, net_option, func_str, restore_model)
                    end_time = timeit.default_timer()
                    print("=====================")
                    print("Run time = {:.2f} mins".format((end_time - start_time) / 60))
                    print("=====================")


if __name__ == "__main__":
    main()
