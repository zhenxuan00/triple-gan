'''
Create semi-supervised datasets for different models
'''
import numpy as np

def create_ssl_data(x, y, n_classes, n_labelled, seed):
    # 'x': data matrix, nxk
    # 'y': label vector, n
    # 'n_classes': number of classes
    # 'n_labelled': number of labelled data
    # 'seed': random seed

    # check input
    if n_labelled%n_classes != 0: 
        print n_labelled
        print n_classes
        raise("n_labelled (wished number of labelled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = n_labelled/n_classes

    rng = np.random.RandomState(seed)
    index = rng.permutation(x.shape[0])
    x = x[index]
    y = y[index]

    # select first several data per class
    data_labelled = [0]*n_classes
    index_labelled = []
    index_unlabelled = []
    for i in xrange(x.shape[0]):
        if data_labelled[y[i]] < n_labels_per_class:
            data_labelled[y[i]] += 1
            index_labelled.append(i)
        else:
            index_unlabelled.append(i)
    
    x_labelled = x[index_labelled]
    y_labelled = y[index_labelled]
    x_unlabelled = x[index_unlabelled]
    y_unlabelled = y[index_unlabelled]
    return x_labelled, y_labelled, x_unlabelled, y_unlabelled


def create_ssl_data_subset(x, y, n_classes, n_labelled, n_labelled_per_time, seed):
    assert n_labelled%n_labelled_per_time==0
    times = n_labelled/n_labelled_per_time
    x_labelled, y_labelled, x_unlabelled, y_unlabelled = create_ssl_data(x, y, n_classes, n_labelled_per_time, seed)
    while (times > 1):
        x_labelled_new, y_labelled_new, x_unlabelled, y_unlabelled = create_ssl_data(x_unlabelled, y_unlabelled, n_classes, n_labelled_per_time, seed)
        x_labelled = np.vstack((x_labelled, x_labelled_new))
        y_labelled = np.hstack((y_labelled, y_labelled_new))
        times -= 1
    return x_labelled, y_labelled, x_unlabelled, y_unlabelled