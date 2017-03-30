import lasagne
from lasagne import init
from lasagne import nonlinearities

import theano.tensor as T
import theano
import numpy as np
import theano.tensor.extra_ops as Textra

__all__ = [
    "ConvConcatLayer", #
    "MLPConcatLayer", #
]


class ConvConcatLayer(lasagne.layers.MergeLayer):
    '''
    concatenate a tensor and a vector on feature map axis 
    '''
    def __init__(self, incomings, num_cls, **kwargs):
        super(ConvConcatLayer, self).__init__(incomings, **kwargs)
        self.num_cls = num_cls

    def get_output_shape_for(self, input_shapes):
        res = list(input_shapes[0])
        res[1] += self.num_cls
        return tuple(res)

    def get_output_for(self, input, **kwargs):
        x, y = input
        if y.ndim == 1:
            y = T.extra_ops.to_one_hot(y, self.num_cls)
        if y.ndim == 2:
            y = y.dimshuffle(0, 1, 'x', 'x')
        assert y.ndim == 4
        return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)

class MLPConcatLayer(lasagne.layers.MergeLayer):
    '''
    concatenate a matrix and a vector on feature axis 
    '''
    def __init__(self, incomings, num_cls, **kwargs):
        super(MLPConcatLayer, self).__init__(incomings, **kwargs)
        self.num_cls = num_cls

    def get_output_shape_for(self, input_shapes):
        res = list(input_shapes[0])
        res[1] += self.num_cls
        return tuple(res)

    def get_output_for(self, input, **kwargs):
        x, y = input
        if y.ndim == 1:
            y = T.extra_ops.to_one_hot(y, self.num_cls)
        assert y.ndim == 2
        return T.concatenate([x, y], axis=1)