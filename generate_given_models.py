'''
This code generates data in various ways given a trained Triple-GAN

Note: Due to the effect of Batch Normalization, it is better to generate batch_size_g (see train file, 200 for cifiar10) samples distributed equally across class in each batch.
'''
import gzip, os, cPickle, time, math, argparse, shutil, sys

import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
from lasagne.layers import dnn
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams

from layers.merge import ConvConcatLayer, MLPConcatLayer
from layers.deconv import Deconv2DLayer

from components.shortcuts import convlayer, mlplayer
from components.objectives import categorical_crossentropy_ssl_separated, maximum_mean_discripancy, categorical_crossentropy, feature_matching
from utils.create_ssl_data import create_ssl_data, create_ssl_data_subset
from utils.others import get_nonlin_list, get_pad_list, bernoullisample, printarray_2D, array2file_2D
import utils.paramgraphics as paramgraphics
from utils.checkpoints import load_weights

# global
parser = argparse.ArgumentParser()
parser.add_argument("-oldmodel", type=str, default=argparse.SUPPRESS)
parser.add_argument("-dataset", type=str, default='svhn')
args = parser.parse_args()

filename_script=os.path.basename(os.path.realpath(__file__))
outfolder=os.path.join("results-ssl", os.path.splitext(filename_script)[0])
outfolder+='.'
outfolder+=args.dataset
outfolder+='.'
outfolder+=str(int(time.time()))
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))

# seeds
seed=1234
rng=np.random.RandomState(seed)
theano_rng=MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# G
n_z=100
batch_size_g=200
num_x=50000
# data dependent
if args.dataset == 'svhn' or args.dataset == 'cifar10':
    gen_final_non=ln.tanh
    num_classes=10
    dim_input=(32,32)
    in_channels=3
    colorImg=True
    generation_scale=True
elif args.dataset == 'mnist':
    gen_final_non=ln.sigmoid
    num_classes=10
    dim_input=(28,28)
    in_channels=1
    colorImg=False
    generation_scale=False

'''
models
'''
# symbols
sym_y_g = T.ivector()
sym_z_input = T.matrix()
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))
sym_z_shared = T.tile(theano_rng.uniform((batch_size_g/num_classes, n_z)), (num_classes, 1))

# generator y2x: p_g(x, y) = p(y) p_g(x | y) where x = G(z, y), z follows p_g(z)
gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]
if args.dataset == 'svhn' or args.dataset == 'cifar10':
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-00'))
    gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu, name='gen-01'), g=None, name='gen-02'))
    gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4), name='gen-03'))
    gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-10'))
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12')) # 4 -> 8
    gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-20'))
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-21'), g=None, name='gen-22')) # 8 -> 16
    gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-30'))
    gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=gen_final_non, name='gen-31'), train_g=True, init_stdv=0.1, name='gen-32')) # 16 -> 32
elif args.dataset == 'mnist':
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-1'))
    gen_layers.append(ll.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=ln.softplus, name='gen-2'), name='gen-3'))
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-4'))
    gen_layers.append(ll.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=ln.softplus, name='gen-5'), name='gen-6'))
    gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-7'))
    gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=gen_final_non, name='gen-8')))

# outputs
gen_out_x = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_rand}, deterministic=False)
gen_out_x_shared = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_shared}, deterministic=False)
gen_out_x_interpolation = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_input}, deterministic=False)
generate = theano.function(inputs=[sym_y_g], outputs=gen_out_x)
generate_shared = theano.function(inputs=[sym_y_g], outputs=gen_out_x_shared)
generate_interpolation = theano.function(inputs=[sym_y_g, sym_z_input], outputs=gen_out_x_interpolation)

'''
Load pretrained model
'''
load_weights(args.oldmodel, gen_layers)

# interpolation on latent space (z) class conditionally
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    orignial_z = np.repeat(rng.uniform(size=(num_classes,n_z)), batch_size_g/num_classes, axis=0)
    target_z = np.repeat(rng.uniform(size=(num_classes,n_z)), batch_size_g/num_classes, axis=0)
    alpha = np.tile(np.arange(batch_size_g/num_classes) * 1.0 / (batch_size_g/num_classes-1), num_classes)
    alpha = alpha.reshape(-1,1)
    z = np.float32((1-alpha)*orignial_z+alpha*target_z)
    x_gen_batch = generate_interpolation(sample_y, z)
    x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg=colorImg, tile_shape=(num_classes, 2*num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'interpolation-'+str(i)+'.png'))

# class conditionally generation with shared z and fixed y
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    x_gen_batch = generate_shared(sample_y)
    x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg=colorImg, tile_shape=(num_classes, 2*num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'shared-'+str(i)+'.png'))

# generation with randomly sampled z and y
for i in xrange(10):
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    inds = np.random.permutation(batch_size_g)
    sample_y = sample_y[inds]
    x_gen_batch = generate(sample_y)
    x_gen_batch = x_gen_batch.reshape((batch_size_g,-1))
    image = paramgraphics.mat_to_img(x_gen_batch.T, dim_input, colorImg=colorImg, tile_shape=(num_classes, 2*num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'random-'+str(i)+'.png'))

if args.dataset != 'cifar10':
    exit()

# large number of random generation for inception score computation
x_gen = []
# generation for each class
x_classes = []
for i in xrange(num_classes):
    x_classes.append([]) 
for i in xrange(num_x / batch_size_g):
    print i
    sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))
    x_gen_batch = generate(sample_y)
    x_gen.append(x_gen_batch)
    if i < 5:
        for j in xrange(num_classes):
            x_classes[j].append(x_gen_batch[j*20:(j+1)*20])
    if i == 5:
        for ind in xrange(num_classes):
            x_classes[ind] = np.concatenate(x_classes[ind], axis=0)
            image = paramgraphics.mat_to_img(x_classes[ind].T, dim_input, colorImg=colorImg, tile_shape=(num_classes, num_classes), scale=generation_scale, save_path=os.path.join(outfolder, 'class-'+str(ind)+'.png'))

x_gen=np.concatenate(x_gen, axis=0)
np.save(os.path.join(outfolder,'inception_score'), x_gen)
    


