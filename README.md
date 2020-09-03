# Triple Generative Adversarial Nets (Triple-GAN)
## [Chongxuan Li](https://github.com/zhenxuan00), [Kun Xu](https://github.com/taufikxu), Jun Zhu and Bo Zhang

Code for reproducing most of the results in the [paper](https://arxiv.org/abs/1703.02291). Triple-GAN: a unified GAN model for classification and class-conditional generation in semi-supervised learning.

Warning: the code is still under development.


We propose Triple-GAN-V2 built upon mean teacher classifier and projection discriminator with spectral norm and implement Triple-GAN in Pytorch. See the source code at https://github.com/taufikxu/Triple-GAN


## Envoronment settings and libs we used in our experiments

This project is tested under the following environment setting.
- OS: Ubuntu 16.04.3
- GPU: Geforce 1080 Ti or Titan X(Pascal or Maxwell)
- Cuda: 8.0, Cudnn: v5.1 or v7.03
- Python: 2.7.14(setup with Miniconda2)
- Theano: 0.9.0.dev-c697eeab84e5b8a74908da654b66ec9eca4f1291
- Lasagne: 0.2.dev1
- Parmesan: 0.1.dev1

> Python
> Numpy
> Scipy
> [Theano](https://github.com/Theano/Theano)
> [Lasagne](https://github.com/Lasagne/Lasagne)(version 0.2.dev1)
> [Parmesan](https://github.com/casperkaae/parmesan)

Thank the authors of these libs. We also thank the authors of [Improved-GAN](https://github.com/openai/improved-gan) and [Temporal Ensemble](https://github.com/smlaine2/tempens) for providing their code. Our code is widely adapted from their repositories.

## Results

Triple-GAN can achieve excellent classification results on MNIST, SVHN and CIFAR10 datasets, see the [paper](https://arxiv.org/abs/1703.02291) for a comparison with the previous state-of-the-art. See generated images as follows:

### Comparing Triple-GAN (right) with GAN trained with [feature matching](https://arxiv.org/abs/1606.03498) (left)
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/mnist_fm.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/mnist_random.png" width="320">

### Generating images in four specific classes (airplane, automobile, bird, horse)
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-0.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-1.png" width="320">
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-2.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/class-7.png" width="320">

### Disentangling styles from classes (left: data, right: Triple-GAN)
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/svhn_data.png" width="320">  <img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/svhn_share_z.png" width="320">

### Class-conditional linear interpolation on latent space
<img src="https://github.com/zhenxuan00/triple-gan/blob/master/images/svhn_linear.png" width="640">
