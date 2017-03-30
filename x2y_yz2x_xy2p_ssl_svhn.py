'''
This code implements a triple GAN for semi-supervised learning on SVHN
'''
import os, time, argparse, shutil

import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
from lasagne.layers import dnn
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
import svhn_data

from layers.merge import ConvConcatLayer, MLPConcatLayer

from components.objectives import categorical_crossentropy_ssl_separated, categorical_crossentropy
import utils.paramgraphics as paramgraphics


'''
parameters
'''
# global
parser = argparse.ArgumentParser()
parser.add_argument("-key", type=str, default=argparse.SUPPRESS)
parser.add_argument("-ssl_seed", type=int, default=1)
parser.add_argument("-nlabeled", type=int, default=1000)
parser.add_argument("-oldmodel", type=str, default=argparse.SUPPRESS)
args = parser.parse_args()
args = vars(args).items()
cfg = {}
for name, val in args:
    cfg[name] = val

filename_script=os.path.basename(os.path.realpath(__file__))
outfolder=os.path.join("results-ssl", os.path.splitext(filename_script)[0])
outfolder+='.'
for item in cfg:
    if item is not 'oldmodel':
        outfolder += item+str(cfg[item])+'.'
    else:
        outfolder += 'oldmodel.'
outfolder+=str(int(time.time()))
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
    sample_path = os.path.join(outfolder, 'sample')
    os.makedirs(sample_path)
logfile=os.path.join(outfolder, 'logfile.log')
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))
# fixed random seeds
ssl_data_seed=cfg['ssl_seed']
num_labelled=cfg['nlabeled']
print ssl_data_seed, num_labelled

seed=1234
rng=np.random.RandomState(seed)
theano_rng=MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# dataset
data_dir='/home/chongxuan/mfs/data/svhn'

# flags
valid_flag=False
objective_flag='argmax' # integrate y, choose argmax

# pre-train C
pre_num_epoch=0
pre_alpha_unlabeled_entropy=.3
pre_alpha_average=1e-3
pre_lr=3e-4
pre_batch_size_lc=200
pre_batch_size_uc=200
# C
alpha_decay=1e-4
alpha_labeled=1.
alpha_unlabeled_entropy=.3
alpha_average=1e-3
alpha_cla=1.
alpha_cla_adversarial=.01
# G
n_z=100
alpha_cla_g=0.03
epoch_cla_g=200
# D
noise_D_data=.3
noise_D=.5
# optimization
b1_g=.5 # mom1 in Adam
b1_d=.5
b1_c=.5
batch_size_g=200
batch_size_l_c=200
batch_size_u_c=200
batch_size_u_d=180
batch_size_l_d=20
lr=3e-4
num_epochs=1000
anneal_lr_epoch=300
anneal_lr_every_epoch=1
anneal_lr_factor=.995
# data dependent
gen_final_non=ln.tanh
num_classes=10
dim_input=(32,32)
in_channels=3
colorImg=True
generation_scale=True
z_generated=num_classes
# evaluation
vis_epoch=10
eval_epoch=1
batch_size_eval=200


'''
data
'''
if valid_flag:
    def rescale(mat):
        return (((-127.5 + mat)/127.5).astype(theano.config.floatX)).reshape((-1, in_channels)+dim_input)
    train_x, train_y, valid_x, valid_y, eval_x, eval_y = load_svhn_small(data_dir, num_val=5000)
    eval_x = valid_x
    eval_y = valid_y
else:
    def rescale(mat):
        return np.transpose(np.cast[theano.config.floatX]((-127.5 + mat)/127.5),(3,2,0,1))
    train_x, train_y = svhn_data.load('/home/chongxuan/mfs/data/svhn/','train')
    eval_x, eval_y = svhn_data.load('/home/chongxuan/mfs/data/svhn/','test')

train_y = np.int32(train_y)
eval_y = np.int32(eval_y)
train_x = rescale(train_x)
eval_x = rescale(eval_x)
x_unlabelled = train_x.copy()

print train_x.shape, eval_x.shape

rng_data = np.random.RandomState(ssl_data_seed)
inds = rng_data.permutation(train_x.shape[0])
train_x = train_x[inds]
train_y = train_y[inds]
x_labelled = []
y_labelled = []
for j in range(num_classes):
    x_labelled.append(train_x[train_y==j][:num_labelled/num_classes])
    y_labelled.append(train_y[train_y==j][:num_labelled/num_classes])
x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)
del train_x

if True:
    print 'Size of training data', x_labelled.shape[0], x_unlabelled.shape[0]
    y_order = np.argsort(y_labelled)
    _x_mean = x_labelled[y_order]
    image = paramgraphics.mat_to_img(_x_mean.T, dim_input, tile_shape=(num_classes, num_labelled/num_classes), colorImg=colorImg, scale=generation_scale, save_path=os.path.join(outfolder, 'x_l_'+str(ssl_data_seed)+'_triple-gan.png'))

pretrain_batches_train_uc = x_unlabelled.shape[0] / pre_batch_size_uc
pretrain_batches_train_lc = x_labelled.shape[0] / pre_batch_size_lc
n_batches_train_u_c = x_unlabelled.shape[0] / batch_size_u_c
n_batches_train_l_c = x_labelled.shape[0] / batch_size_l_c
n_batches_train_u_d = x_unlabelled.shape[0] / batch_size_u_d
n_batches_train_l_d = x_labelled.shape[0] / batch_size_l_d
n_batches_train_g = x_unlabelled.shape[0] / batch_size_g
n_batches_eval = eval_x.shape[0] / batch_size_eval


'''
models
'''
# symbols
sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))
sym_x_u = T.tensor4()
sym_x_u_d = T.tensor4()
sym_x_u_g = T.tensor4()
sym_x_l = T.tensor4()
sym_y = T.ivector()
sym_y_g = T.ivector()
sym_x_eval = T.tensor4()
sym_lr = T.scalar()
sym_alpha_cla_g = T.scalar()
sym_alpha_unlabel_entropy = T.scalar()
sym_alpha_unlabel_average = T.scalar()

shared_unlabel = theano.shared(x_unlabelled, borrow=True)
slice_x_u_g = T.ivector()
slice_x_u_d = T.ivector()
slice_x_u_c = T.ivector()

# classifier x2y: p_c(x, y) = p(x) p_c(y | x)
cla_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
cla_layers = [cla_in_x]
cla_layers.append(ll.DropoutLayer(cla_layers[-1], p=0.2, name='cla-00'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-02'), name='cla-03'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-11'), name='cla-12'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-21'), name='cla-22'))
cla_layers.append(dnn.MaxPool2DDNNLayer(cla_layers[-1], pool_size=(2, 2)))
cla_layers.append(ll.DropoutLayer(cla_layers[-1], p=0.5, name='cla-23'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-31'), name='cla-32'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-41'), name='cla-42'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-51'), name='cla-52'))
cla_layers.append(dnn.MaxPool2DDNNLayer(cla_layers[-1], pool_size=(2, 2)))
cla_layers.append(ll.DropoutLayer(cla_layers[-1], p=0.5, name='cla-53'))
cla_layers.append(ll.batch_norm(dnn.Conv2DDNNLayer(cla_layers[-1], 512, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-61'), name='cla-62'))
cla_layers.append(ll.batch_norm(ll.NINLayer(cla_layers[-1], num_units=256, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-71'), name='cla-72'))
cla_layers.append(ll.batch_norm(ll.NINLayer(cla_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu, name='cla-81'), name='cla-82'))
cla_layers.append(ll.GlobalPoolLayer(cla_layers[-1], name='cla-83'))
cla_layers.append(ll.batch_norm(ll.DenseLayer(cla_layers[-1], num_units=num_classes, W=Normal(0.05), nonlinearity=ln.softmax, name='cla-91'), name='cla-92'))

# generator y2x: p_g(x, y) = p(y) p_g(x | y) where x = G(z, y), z follows p_g(z)
gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]
gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-00'))
gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu, name='gen-01'), g=None, name='gen-02'))
gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (-1,512,4,4), name='gen-03'))
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-10'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-11'), g=None, name='gen-12')) # 4 -> 8
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-20'))
gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu, name='gen-21'), g=None, name='gen-22')) # 8 -> 16
gen_layers.append(ConvConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-30'))
gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=gen_final_non, name='gen-31'), train_g=True, init_stdv=0.1, name='gen-32')) # 16 -> 32

# discriminator xy2p: test a pair of input comes from p(x, y) instead of p_c or p_g
dis_in_x = ll.InputLayer(shape=(None, in_channels) + dim_input)
dis_in_y = ll.InputLayer(shape=(None,))
dis_layers = [dis_in_x]
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-00'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-01'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-02'), name='dis-03'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-20'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 32, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-21'), name='dis-22'))
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-23'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-30'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-31'), name='dis-32'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-40'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 64, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-41'), name='dis-42'))
dis_layers.append(ll.DropoutLayer(dis_layers[-1], p=0.2, name='dis-43'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-50'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-51'), name='dis-52'))
dis_layers.append(ConvConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-60'))
dis_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(dis_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu, name='dis-61'), name='dis-62'))
dis_layers.append(ll.GlobalPoolLayer(dis_layers[-1], name='dis-63'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-70'))
dis_layers.append(nn.weight_norm(ll.DenseLayer(dis_layers[-1], num_units=1, W=Normal(0.05), nonlinearity=ln.sigmoid, name='dis-71'), train_g=True, init_stdv=0.1, name='dis-72'))


'''
objectives
'''
# outputs
gen_out_x = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_rand}, deterministic=False)

cla_out_y_l = ll.get_output(cla_layers[-1], sym_x_l, deterministic=False)
cla_out_y_eval = ll.get_output(cla_layers[-1], sym_x_eval, deterministic=True)
cla_out_y = ll.get_output(cla_layers[-1], sym_x_u, deterministic=False)
cla_out_y_d = ll.get_output(cla_layers[-1], {cla_in_x:sym_x_u_d}, deterministic=False)
cla_out_y_d_hard = cla_out_y_d.argmax(axis=1)
cla_out_y_g = ll.get_output(cla_layers[-1], {cla_in_x:gen_out_x}, deterministic=False)

dis_out_p = ll.get_output(dis_layers[-1], {dis_in_x:T.concatenate([sym_x_l,sym_x_u_d], axis=0),dis_in_y:T.concatenate([sym_y,cla_out_y_d_hard], axis=0)}, deterministic=False)
dis_out_p_g = ll.get_output(dis_layers[-1], {dis_in_x:gen_out_x,dis_in_y:sym_y_g}, deterministic=False)

if objective_flag == 'integrate':
    # integrate
    dis_out_p_c = ll.get_output(dis_layers[-1],
                                {dis_in_x:T.repeat(sym_x_u, num_classes, axis=0),
                                dis_in_y:np.tile(np.arange(num_classes), batch_size_u_c)},
                                deterministic=False)
elif objective_flag == 'argmax':
    # argmax approximation
    cla_out_y_hard = cla_out_y.argmax(axis=1)
    dis_out_p_c = ll.get_output(dis_layers[-1], {dis_in_x:sym_x_u,dis_in_y:cla_out_y_hard}, deterministic=False)
else:
    raise Exception('Unknown objective flags')

image = ll.get_output(gen_layers[-1], {gen_in_y:sym_y_g, gen_in_z:sym_z_image}, deterministic=False) # for generation

accurracy_eval = (lasagne.objectives.categorical_accuracy(cla_out_y_eval, sym_y)) # for evaluation
accurracy_eval = accurracy_eval.mean()

# costs
bce = lasagne.objectives.binary_crossentropy

dis_cost_p = bce(dis_out_p, T.ones(dis_out_p.shape)).mean() # D distincts p
dis_cost_p_g = bce(dis_out_p_g, T.zeros(dis_out_p_g.shape)).mean() # D distincts p_g
gen_cost_p_g = bce(dis_out_p_g, T.ones(dis_out_p_g.shape)).mean() # G fools D

weight_decay_classifier = lasagne.regularization.regularize_layer_params_weighted({cla_layers[-1]:1}, lasagne.regularization.l2) # weight decay

dis_cost_p_c = bce(dis_out_p_c, T.zeros(dis_out_p_c.shape)) # D distincts p_c
cla_cost_p_c = bce(dis_out_p_c, T.ones(dis_out_p_c.shape)) # C fools D

if objective_flag == 'integrate':
    # integrate
    weight_loss_c = T.reshape(cla_cost_p_c, (-1, num_classes)) * cla_out_y
    cla_cost_p_c = T.sum(weight_loss_c, axis=1).mean()
    weight_loss_d = T.reshape(dis_cost_p_c, (-1, num_classes)) * cla_out_y
    dis_cost_p_c = T.sum(weight_loss_d, axis=1).mean()
elif objective_flag == 'argmax':
    # argmax approximation
    p = cla_out_y.max(axis=1)
    cla_cost_p_c = (cla_cost_p_c*p).mean()
    dis_cost_p_c = dis_cost_p_c.mean()

cla_cost_cla = categorical_crossentropy_ssl_separated(predictions_l=cla_out_y_l, targets=sym_y, predictions_u=cla_out_y, weight_decay=weight_decay_classifier, alpha_labeled=alpha_labeled, alpha_unlabeled=sym_alpha_unlabel_entropy, alpha_average=sym_alpha_unlabel_average, alpha_decay=alpha_decay) # classification loss

pretrain_cla_loss = categorical_crossentropy_ssl_separated(predictions_l=cla_out_y_l, targets=sym_y, predictions_u=cla_out_y, weight_decay=weight_decay_classifier, alpha_labeled=alpha_labeled, alpha_unlabeled=pre_alpha_unlabeled_entropy, alpha_average=pre_alpha_average, alpha_decay=alpha_decay) # classification loss
pretrain_cost = pretrain_cla_loss

cla_cost_cla_g = categorical_crossentropy(predictions=cla_out_y_g, targets=sym_y_g)

dis_cost = dis_cost_p + .5*dis_cost_p_g + .5*dis_cost_p_c
gen_cost = .5*gen_cost_p_g
# flag
cla_cost = alpha_cla_adversarial*.5*cla_cost_p_c + cla_cost_cla + sym_alpha_cla_g*cla_cost_cla_g
# fast
cla_cost_fast = alpha_cla_adversarial*.5*cla_cost_p_c + cla_cost_cla

dis_cost_list=[dis_cost, dis_cost_p, .5*dis_cost_p_g, .5*dis_cost_p_c]
gen_cost_list=[gen_cost,]
# flag
cla_cost_list=[cla_cost, alpha_cla_adversarial*.5*cla_cost_p_c, cla_cost_cla, sym_alpha_cla_g*cla_cost_cla_g]
# fast
cla_cost_list_fast=[cla_cost_fast, alpha_cla_adversarial*.5*cla_cost_p_c, cla_cost_cla]

# updates of D
dis_params = ll.get_all_params(dis_layers, trainable=True)
dis_grads = T.grad(dis_cost, dis_params)
dis_updates = lasagne.updates.adam(dis_grads, dis_params, beta1=b1_d, learning_rate=sym_lr)

# updates of G
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_grads = T.grad(gen_cost, gen_params)
gen_updates = lasagne.updates.adam(gen_grads, gen_params, beta1=b1_g, learning_rate=sym_lr)

# updates of C
cla_params = ll.get_all_params(cla_layers, trainable=True)
cla_grads = T.grad(cla_cost, cla_params)
cla_updates_ = lasagne.updates.adam(cla_grads, cla_params, beta1=b1_c, learning_rate=sym_lr)

# fast updates of C
cla_params = ll.get_all_params(cla_layers, trainable=True)
cla_grads_fast = T.grad(cla_cost_fast, cla_params)
cla_updates_fast_ = lasagne.updates.adam(cla_grads_fast, cla_params, beta1=b1_c, learning_rate=sym_lr)

pre_cla_grad = T.grad(pretrain_cost, cla_params)
pretrain_updates_ = lasagne.updates.adam(pre_cla_grad, cla_params, beta1=0.9, beta2=0.999,
                               epsilon=1e-8, learning_rate=pre_lr)

######## avg
avg_params = lasagne.layers.get_all_params(cla_layers)
cla_param_avg=[]
for param in avg_params:
   value = param.get_value(borrow=True)
   cla_param_avg.append(theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable,
                              name=param.name))
cla_avg_updates = [(a,a + 0.01*(p-a)) for p,a in zip(avg_params,cla_param_avg)]
cla_avg_givens = [(p,a) for p,a in zip(avg_params, cla_param_avg)]
cla_updates = cla_updates_.items() + cla_avg_updates
cla_updates_fast = cla_updates_fast_.items() + cla_avg_updates
pretrain_updates = pretrain_updates_.items() + cla_avg_updates

# functions
train_batch_dis = theano.function(inputs=[sym_x_l, sym_y, sym_y_g,
                                          slice_x_u_c, slice_x_u_d, sym_lr],
                                  outputs=dis_cost_list, updates=dis_updates,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c],
                                          sym_x_u_d: shared_unlabel[slice_x_u_d]})
train_batch_gen = theano.function(inputs=[sym_y_g, sym_lr],
                                  outputs=gen_cost_list, updates=gen_updates)
train_batch_cla = theano.function(inputs=[sym_x_l, sym_y, sym_y_g, slice_x_u_c, sym_alpha_cla_g, sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                  outputs=cla_cost_list, updates=cla_updates,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c]})
# fast
train_batch_cla_fast = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c, sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                  outputs=cla_cost_list_fast, updates=cla_updates_fast,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c]})

sym_index = T.iscalar()
bslice = slice(sym_index*pre_batch_size_uc, (sym_index+1)*pre_batch_size_uc)
pretrain_batch_cla = theano.function(inputs=[sym_x_l, sym_y, sym_index],
                                  outputs=[pretrain_cost], updates=pretrain_updates,
                                  givens={sym_x_u: shared_unlabel[bslice]})
generate = theano.function(inputs=[sym_y_g], outputs=image)
# avg
evaluate = theano.function(inputs=[sym_x_eval, sym_y], outputs=[accurracy_eval], givens=cla_avg_givens)

'''
Load pretrained model
'''
if 'oldmodel' in cfg:
    from utils.checkpoints import load_weights
    load_weights(cfg['oldmodel'], dis_layers+cla_layers+gen_layers)
    for (p, a) in zip(ll.get_all_params(cla_layers), avg_params):
        a.set_value(p.get_value())


'''
Pretrain C
'''
for epoch in range(1, 1+pre_num_epoch):
    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    shared_unlabel.set_value(x_unlabelled[p_u])

    for i in range(pretrain_batches_train_uc):
        i_c = i % pretrain_batches_train_lc
        from_l_c = i_c*pre_batch_size_lc
        to_l_c = (i_c+1)*pre_batch_size_lc

        pretrain_batch_cla(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c], i)
    accurracy=[]
    for i in range(n_batches_eval):
        accurracy_batch = evaluate(eval_x[i*batch_size_eval:(i+1)*batch_size_eval], eval_y[i*batch_size_eval:(i+1)*batch_size_eval])
        accurracy += accurracy_batch
    accurracy=np.mean(accurracy)
    print str(epoch) + ':Pretrain accuracy: ' + str(1-accurracy)

'''
train and evaluate
'''
print 'Start training'
for epoch in range(1, 1+num_epochs):
    start = time.time()

    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_g = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    dl = [0.] * len(dis_cost_list)
    gl = [0.] * len(gen_cost_list)
    cl = [0.] * len(cla_cost_list)

    # fast
    dis_t=0.
    gen_t=0.
    cla_t=0.
    for i in range(n_batches_train_u_c):
        from_u_c = i*batch_size_u_c
        to_u_c = (i+1)*batch_size_u_c
        i_c = i % n_batches_train_l_c
        from_l_c = i_c*batch_size_l_c
        to_l_c = (i_c+1)*batch_size_l_c
        i_d = i % n_batches_train_l_d
        from_l_d = i_d*batch_size_l_d
        to_l_d = (i_d+1)*batch_size_l_d
        i_d_ = i % n_batches_train_u_d
        from_u_d = i_d_*batch_size_u_d
        to_u_d = (i_d_+1)*batch_size_u_d

        sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g/num_classes))

        tmp = time.time()
        dl_b = train_batch_dis(x_labelled[from_l_d:to_l_d], y_labelled[from_l_d:to_l_d], sample_y, p_u[from_u_c:to_u_c], p_u_d[from_u_d:to_u_d], lr)
        for j in xrange(len(dl)):
            dl[j] += dl_b[j]

        tmp1 = time.time()

        gl_b = train_batch_gen(sample_y, lr)
        for j in xrange(len(gl)):
            gl[j] += gl_b[j]

        tmp2 = time.time()

        # fast
        if epoch < epoch_cla_g:
            cl_b = train_batch_cla_fast(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c], p_u[from_u_c:to_u_c], lr, alpha_unlabeled_entropy, alpha_average)
            cl_b += [0,]
        else:
            cl_b = train_batch_cla(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c], sample_y, p_u[from_u_c:to_u_c], alpha_cla_g, lr, alpha_unlabeled_entropy, alpha_average)

        for j in xrange(len(cl)):
            cl[j] += cl_b[j]

        tmp3 = time.time()
        dis_t+=(tmp1-tmp)
        gen_t+=(tmp2-tmp1)
        cla_t+=(tmp3-tmp2)

    print 'dis:', dis_t, 'gen:', gen_t, 'cla:', cla_t, 'total', dis_t+gen_t+cla_t

    for i in xrange(len(dl)):
        dl[i] /= n_batches_train_u_c
    for i in xrange(len(gl)):
        gl[i] /= n_batches_train_u_c
    for i in xrange(len(cl)):
        cl[i] /= n_batches_train_u_c

    if (epoch >= anneal_lr_epoch) and (epoch % anneal_lr_every_epoch == 0):
        lr = lr*anneal_lr_factor

    t = time.time() - start

    line = "*Epoch=%d Time=%.2f LR=%.5f\n" %(epoch, t, lr) + "DisLosses: " + str(dl)+"\nGenLosses: "+str(gl)+"\nClaLosses: "+str(cl)

    print line
    with open(logfile,'a') as f:
        f.write(line + "\n")

    # random generation for visualization
    if epoch % vis_epoch == 0:
        import  utils.paramgraphics as paramgraphics
        tail = '-'+str(epoch)+'.png'
        ran_y = np.int32(np.repeat(np.arange(num_classes), num_classes))
        x_gen = generate(ran_y)
        x_gen = x_gen.reshape((z_generated*num_classes,-1))
        image = paramgraphics.mat_to_img(x_gen.T, dim_input, colorImg=colorImg, scale=generation_scale, save_path=os.path.join(sample_path, 'sample'+tail))

    if epoch % eval_epoch == 0:
        accurracy=[]
        for i in range(n_batches_eval):
            accurracy_batch = evaluate(eval_x[i*batch_size_eval:(i+1)*batch_size_eval], eval_y[i*batch_size_eval:(i+1)*batch_size_eval])
            accurracy += accurracy_batch
        accurracy=np.mean(accurracy)
        print ('ErrorEval=%.5f\n' % (1-accurracy,))
        with open(logfile,'a') as f:
            f.write(('ErrorEval=%.5f\n\n' % (1-accurracy,)))

    if epoch % 200 == 0 or (epoch == epoch_cla_g - 1):
        from utils.checkpoints import save_weights
        params = ll.get_all_params(dis_layers+cla_layers+gen_layers)
        save_weights(os.path.join(outfolder, 'model_epoch' + str(epoch) + '.npy'), params, None)
        save_weights(os.path.join(outfolder, 'average'+ str(epoch) +'.npy'), cla_param_avg, None)
