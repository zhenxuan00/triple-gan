# ZCA and MeanOnlyBNLayer implementations copied from
#   https://github.com/TimSalimans/weight_norm/blob/master/nn.py
#
# Modifications made to MeanOnlyBNLayer:
# - Added configurable momentum.
# - Added 'modify_incoming' flag for weight matrix sharing (not used in this project).
# - Sums and means use float32 datatype.

import numpy as np
import theano as th
import theano.tensor as T
from scipy import linalg
import lasagne

class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = th.shared(np.dot(tmp, U.T).astype(th.config.floatX))
        self.inv_ZCA_mat = th.shared(np.dot(tmp2, U.T).astype(th.config.floatX))
        self.mean = th.shared(m.astype(th.config.floatX))

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0],np.prod(s[1:]))) - self.mean.get_value(), self.ZCA_mat.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return T.dot(x.flatten(2) - self.mean.dimshuffle('x',0), self.ZCA_mat).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")
            
    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (np.dot(x.reshape((s[0],np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return (T.dot(x.flatten(2), self.inv_ZCA_mat) + self.mean.dimshuffle('x',0)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

# T.nnet.relu has some issues with very large inputs, this is more stable
def relu(x):
    return T.maximum(x, 0)

class MeanOnlyBNLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), nonlinearity=relu, modify_incoming=True, momentum=0.9, **kwargs):
        super(MeanOnlyBNLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.momentum = momentum
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        if modify_incoming:
            incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
            if incoming.W_param.ndim==4:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
            else:
                W_axes_to_sum = 0
                W_dimshuffle_args = ['x',0]
            if g is not None:
                incoming.W = incoming.W_param * (self.g/T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,dtype=th.config.floatX, acc_dtype=th.config.floatX))).dimshuffle(*W_dimshuffle_args)
            else:
                incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True,dtype=th.config.floatX,acc_dtype=th.config.floatX))        

    def get_output_for(self, input, deterministic=False, init=False, **kwargs):
        if deterministic:
            activation = input - self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)
        else:
            m = T.mean(input,axis=self.axes_to_sum,dtype=th.config.floatX,acc_dtype=th.config.floatX)
            activation = input - m.dimshuffle(*self.dimshuffle_args)
            self.bn_updates = [(self.avg_batch_mean, self.momentum*self.avg_batch_mean + (1.0-self.momentum)*m)]
            if init:
                stdv = T.sqrt(T.mean(T.square(activation),axis=self.axes_to_sum,dtype=th.config.floatX,acc_dtype=th.config.floatX))
                activation /= stdv.dimshuffle(*self.dimshuffle_args)
                self.init_updates = [(self.g, self.g/stdv)]
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(activation)
        
def mean_only_bn(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    return MeanOnlyBNLayer(layer, name=layer.name+'_n', nonlinearity=nonlinearity, **kwargs)
