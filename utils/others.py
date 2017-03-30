import numpy as np
import theano.tensor as T
import theano, lasagne


def get_pad(pad):
    if pad not in ['same', 'valid', 'full']:
        pad = tuple(map(int, pad.split('-')))
    return pad

def get_pad_list(pad_list):
    re_list = []
    for p in pad_list:
        re_list.append(get_pad(p))
    return re_list

# nonlinearities
def get_nonlin(nonlin):
    if nonlin == 'rectify':
        return lasagne.nonlinearities.rectify
    elif nonlin == 'leaky_rectify':
        return lasagne.nonlinearities.LeakyRectify(0.1)
    elif nonlin == 'tanh':
        return lasagne.nonlinearities.tanh
    elif nonlin == 'sigmoid':
        return lasagne.nonlinearities.sigmoid
    elif nonlin == 'maxout':
        return 'maxout'
    elif nonlin == 'none':
        return lasagne.nonlinearities.identity
    else:
        raise ValueError('invalid non-linearity \'' + nonlin + '\'')
        
def get_nonlin_list(nonlin_list):
    re_list = []
    for n in nonlin_list:
        re_list.append(get_nonlin(n))
    return re_list

def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)

def array2file_2D(array,logfile):
    assert len(array.shape) == 2, array.shape
    with open(logfile,'a') as f:
       for i in xrange(array.shape[0]):
        for j in xrange(array.shape[1]):
            f.write(str(array[i][j])+' ')
        f.write('\n')

def printarray_2D(array, precise=2):
    assert len(array.shape) == 2, array.shape
    format = '%.'+str(precise)+'f'
    for i in xrange(array.shape[0]):
        for j in xrange(array.shape[1]):
            print format %array[i][j],
        print