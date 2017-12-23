import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def _label_name():
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return label_names


def _normalize(x):
    result = x / float(255)
    return result


def _one_hot_encode(x):
    result = np.eye(10)[x]
    return result


def _preprocess_and_save(features, labels, filename):
    features = _normalize(features)
    labels = _one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def _load_data_set(cifar10_dir, batch_id):
    with open(cifar10_dir + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    print(batch.keys())
    # label_names = batch['label_names']
    # print(label_names)

    return features, labels


def preprocess_all(cifar10_dir, pre_data_dir):

    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = _load_data_set(cifar10_dir, batch_i)
        validation_count = int(len(features) * 0.1)

        # Prprocess and save a batch of training data
        _preprocess_and_save(
            features[:-validation_count],
            labels[:-validation_count],
            pre_data_dir + 'train_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        np.array(valid_features),
        np.array(valid_labels),
        pre_data_dir + 'validation.p')

    with open(cifar10_dir + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the test data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all test data
    _preprocess_and_save(
        np.array(test_features),
        np.array(test_labels),
        pre_data_dir + 'test.p')