import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from preprocess import preprocess_all, _label_name

import file_manager

# downloaded and unzipped from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
CIFAR10_DIR = "./cifar-10-batches-py/"

# preprocess output dir
PRE_DATA_DIR = "./pre_data/"

LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    # preprocess_raw_data = True
    preprocess_raw_data = False

    if preprocess_raw_data:
        file_manager.backup_files([PRE_DATA_DIR])
        preprocess_all(CIFAR10_DIR, PRE_DATA_DIR)

    train_features, train_labels = pickle.load(open(PRE_DATA_DIR + 'train_batch_1.p', mode='rb'))
    valid_features, valid_labels = pickle.load(open(PRE_DATA_DIR + 'validation.p', mode='rb'))
    test_features, test_labels = pickle.load(open(PRE_DATA_DIR + 'test.p', mode='rb'))
    for i in [train_features, train_labels, valid_features, valid_labels, test_features, test_labels]:
        print(type(i))
        print(i.shape)
    plt.imshow(train_features[11])
    print(LABEL_NAMES)
    print(train_labels[11])
    plt.show()


if __name__ == "__main__":
    main()