import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import timeit
from sklearn.utils import shuffle

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

def train_network(session, optimizer, x, y, feature_batch, label_batch):
    session.run([optimizer], feed_dict={x: feature_batch,
                                        y: label_batch})
    return None


def print_stats(session, x, y, feature_batch, label_batch, valid_features, valid_labels, cost, accuracy):
    loss = session.run(cost, feed_dict={x: feature_batch,
                                        y: label_batch})

    valid_acc = session.run(accuracy, feed_dict={x: valid_features,
                                                 y: valid_labels})

    print('batch training loss = {:>5.3f}, valid_acc = {:>1.3f}'.format(loss, valid_acc))
    return None


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


def run_graph(train_set, valid_set):
    # unpack training set and validation set

    train_features, train_labels = train_set
    valid_features, valid_labels = valid_set

    # build up nets
    x, y, logits, cost, optimizer, correct_pred, accuracy = conv_net.build()

    # hyper parameters
    epochs = 5
    batch_size = 256

    for i in [x, y, logits, cost, optimizer, correct_pred, accuracy]:
        print("===")
        print(i)
        print("===")

    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in next_batch(train_features, train_labels, batch_size, re_shuffle=False):
                train_network(sess, optimizer, x, y, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, x, y, batch_features, batch_labels, valid_features, valid_labels, cost, accuracy)

    return


def main():
    # preprocess_raw_data = True
    preprocess_raw_data = False

    if preprocess_raw_data:
        file_manager.backup_files([PRE_DATA_DIR])
        preprocess_all(CIFAR10_DIR, PRE_DATA_DIR)

    # it's a tuple of (*_features, *_labels)
    train_set = pickle.load(open(PRE_DATA_DIR + 'train_batch_1.p', mode='rb'))
    valid_set = pickle.load(open(PRE_DATA_DIR + 'validation.p', mode='rb'))
    test_set = pickle.load(open(PRE_DATA_DIR + 'test.p', mode='rb'))
    for i in [train_set, valid_set, test_set]:
        print("*_features, *_labels")
        print(type(i[0]), type(i[1]))
        print(i[0].shape, i[1])
    plt.imshow(train_set[0][11])
    print(LABEL_NAMES)
    print(train_set[1][11])
    plt.show()

    start_time = timeit.default_timer()

    run_graph(train_set, valid_set)

    end_time = timeit.default_timer()
    print("Run time = {:.2f} mins".format((end_time - start_time) / 60))


if __name__ == "__main__":
    main()