'''
objectives
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne
from theano.tensor.extra_ops import to_one_hot

def categorical_crossentropy(predictions, targets, epsilon=1e-6):
    # avoid overflow
    predictions = T.clip(predictions, epsilon, 1-epsilon)
    # check shape of targets
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    return lasagne.objectives.categorical_crossentropy(predictions, targets).mean()

def entropy(predictions):
    return categorical_crossentropy(predictions, predictions)

def negative_entropy_of_mean(predictions):
    return -entropy(predictions.mean(axis=0, keepdims=True))

def categorical_crossentropy_of_mean(predictions):
    num_cls = predictions.shape[1]
    uniform_targets = T.ones((1, num_cls)) / num_cls
    return categorical_crossentropy(predictions.mean(axis=0, keepdims=True), uniform_targets)

def categorical_crossentropy_ssl_alternative(predictions, targets, num_labelled, weight_decay, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions[:num_labelled], targets)
    en_loss = entropy(predictions[num_labelled:])
    av_loss = negative_entropy_of_mean(predictions[num_labelled:])
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def categorical_crossentropy_ssl(predictions, targets, num_labelled, weight_decay, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions[:num_labelled], targets)
    en_loss = entropy(predictions[num_labelled:])
    av_loss = categorical_crossentropy_of_mean(predictions[num_labelled:])
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def categorical_crossentropy_ssl_separated(predictions_l, targets, predictions_u, weight_decay, alpha_labeled=1., alpha_unlabeled=.3, alpha_average=1e-3, alpha_decay=1e-4):
    ce_loss = categorical_crossentropy(predictions_l, targets)
    en_loss = entropy(predictions_u)
    av_loss = categorical_crossentropy_of_mean(predictions_u)
    return alpha_labeled*ce_loss + alpha_unlabeled*en_loss + alpha_average*av_loss + alpha_decay*weight_decay

def maximum_mean_discripancy(sample, data, sigma=[2. , 5., 10., 20., 40., 80.]):
    sample = sample.flatten(2)
    data = data.flatten(2)

    x = T.concatenate([sample, data], axis=0)
    xx = T.dot(x, x.T)
    x2 = T.sum(x*x, axis=1, keepdims=True)
    exponent = xx - .5*x2 - .5*x2.T
    s_samples = T.ones([sample.shape[0], 1])*1./sample.shape[0]
    s_data = -T.ones([data.shape[0], 1])*1./data.shape[0]
    s_all = T.concatenate([s_samples, s_data], axis=0)
    s_mat = T.dot(s_all, s_all.T)
    mmd_loss = 0.
    for s in sigma:
        kernel_val = T.exp((1./s) * exponent)
        mmd_loss += T.sum(s_mat*kernel_val)
    return T.sqrt(mmd_loss)

def feature_matching(f_sample, f_data, norm='l2'):
    if norm == 'l2':
        return T.mean(T.square(T.mean(f_sample,axis=0)-T.mean(f_data,axis=0)))
    elif norm == 'l1':
        return T.mean(abs(T.mean(f_sample,axis=0)-T.mean(f_data,axis=0)))
    else:
        raise NotImplementedError

def multiclass_s3vm_loss(predictions_l, targets, predictions_u, weight_decay, alpha_labeled=1., alpha_unlabeled=1., alpha_average=1., alpha_decay=1., delta=1., norm_type=2, form ='mean_class', entropy_term=False):
    '''
    predictions: 
        size L x nc
             U x nc
    targets: 
        size L x nc

    output:
        weighted sum of hinge loss, hat loss, balance constraint and weight decay
    '''
    num_cls = predictions_l.shape[1]
    if targets.ndim == predictions_l.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions_l.ndim:
        raise TypeError('rank mismatch between targets and predictions')

    hinge_loss = multiclass_hinge_loss_(predictions_l, targets, delta)
    hat_loss = multiclass_hat_loss(predictions_u, delta)
    regularization = balance_constraint(predictions_l, targets, predictions_u, norm_type, form)
    if not entropy_term:
        return alpha_labeled*hinge_loss.mean() + alpha_unlabeled*hat_loss.mean() + alpha_average*regularization + alpha_decay*weight_decay
    else:
        # given an unlabeled data, when treat hat loss as the entropy term derived from a lowerbound, it should conflict to current prediction, which is quite strange but true ... the entropy term enforce the discriminator to predict unlabeled data uniformly as a regularization
        # max entropy regularization provides a tighter lowerbound but hurt the semi-supervised learning performance as it conflicts to the hat loss ...
        return alpha_labeled*hinge_loss.mean() - alpha_unlabeled*hat_loss.mean() + alpha_average*regularization + alpha_decay*weight_decay


def multiclass_hinge_loss_(predictions, targets, delta=1):
    return lasagne.objectives.multiclass_hinge_loss(predictions, targets, delta)

def multiclass_hinge_loss(predictions, targets, weight_decay, alpha_decay=1., delta=1):
    return multiclass_hinge_loss_(predictions, targets, delta).mean() + alpha_decay*weight_decay

def multiclass_hat_loss(predictions, delta=1):
    targets = T.argmax(predictions, axis=1)
    return multiclass_hinge_loss(predictions, targets, delta)

def balance_constraint(p_l, t_l, p_u, norm_type=2, form='mean_class'):
    '''
    balance constraint
    ------
    norm_type: type of norm 
            l2 or l1
    form: form of regularization
            mean_class: average mean activation of u and l data should be the same over each class
            mean_all: average mean activation of u and l data should be the same over all data
            ratio: 

    '''
    t_u = T.argmax(p_u, axis=1)
    num_cls = p_l.shape[1]
    t_u = theano.tensor.extra_ops.to_one_hot(t_u, num_cls)
    if form == 'mean_class':
        res = (p_l*t_l).mean(axis=0) - (p_u*t_u).mean(axis=0)
    elif form == 'mean_all':
        res = p_l.mean(axis=0) - p_u.mean(axis=0)
    elif form == 'ratio':
        pass

    # res should be a vector with length number_class
    return res.norm(norm_type)
