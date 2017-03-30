import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn

# code from ImprovedGAN, Salimans and Goodfellow: https://github.com/openai/improved-gan

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, target_shape, filter_size, stride=(2, 2),
                 W=lasagne.init.Normal(0.05), b=lasagne.init.Constant(0.), nonlinearity=None, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.target_shape = target_shape
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.filter_size = lasagne.layers.dnn.as_tuple(filter_size, 2)
        self.stride = lasagne.layers.dnn.as_tuple(stride, 2)
        self.target_shape = target_shape

        self.W_shape = (incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
        self.W = self.add_param(W, self.W_shape, name="W")
        if b is not None:
            self.b = self.add_param(b, (target_shape[1],), name="b")
        else:
            self.b = None

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.target_shape, kshp=self.W_shape, subsample=self.stride, border_mode='half')
        activation = op(self.W, input, self.target_shape[2:])

        if self.b is not None:
            activation += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return self.target_shape