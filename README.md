# Triple Generative Adversarial Nets
## [Chongxuan Li](https://github.com/zhenxuan00), Kun Xu, Jun Zhu and Bo Zhang

Full [paper](https://arxiv.org/abs/1703.02291). A unified GAN model for classification and class-conditional generation in semi-supervised learning.

## Some libs we used in our experiments
> Python
> Numpy
> Scipy
> [Theano](https://github.com/Theano/Theano)
> [Lasagne](https://github.com/Lasagne/Lasagne)
> [Parmesan](https://github.com/casperkaae/parmesan)

## We thank the authors of [Improved-GAN](https://github.com/openai/improved-gan) and [Temporal Ensemble](https://github.com/smlaine2/tempens) for providing their code. Our code is widely adapted from their repositories.

## Excellent classification results on MNIST, SVHN and CIFAR10 datasets, see the paper for a comparison with previous state-of-the-art.

## Generation

### Comparing with GAN trained with [feature matching](https://arxiv.org/abs/1606.03498)
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/mnist_fm.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/mnist_random.png" width="320">

### Generating images in a specific class
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-0.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-2.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-7.png" width="320">

### Disentangle styles from classes
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/svhn_data.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/svhn_share_z.png" width="320">

### Class-conditional linear interpolation on latent space
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/svhn_linear.png" width="640">
