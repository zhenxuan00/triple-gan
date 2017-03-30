'''
shortcuts for compsited layers
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
import sys
sys.path.append("..")
from layers.merge import ConvConcatLayer, MLPConcatLayer

# convolutional layer
# following optional batch normalization, pooling and dropout
def convlayer(l,bn,dr,ps,n_kerns,d_kerns,nonlinearity,pad,stride,W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.), name=None):
    l = lasagne.layers.Conv2DLayer(l, num_filters=n_kerns, filter_size=d_kerns, stride=stride, pad=pad, W=W, b=b, nonlinearity=nonlinearity, name=name)
    if bn:
        l = lasagne.layers.batch_norm(l)
    if ps > 1:
        l = lasagne.layers.MaxPool2DLayer(l, pool_size=(ps,ps))
    if dr > 0:
        l = lasagne.layers.DropoutLayer(l, p=dr)
    return l

# mlp layer
# following optional batch normalization and dropout
def mlplayer(l,bn,dr,num_units,nonlinearity,name):
    l = lasagne.layers.DenseLayer(l,num_units=num_units,nonlinearity=nonlinearity,name="MLP-"+name)
    if bn:
        l = lasagne.layers.batch_norm(l, name="BN-"+name)
    if dr > 0:
        l = lasagne.layers.DropoutLayer(l, p=dr, name="Drop-"+name)
    return l
