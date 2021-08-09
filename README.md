# Energy-Based Models for Continual Learning

This project aims at classification continual learning problems using Energy-Based Models. Mainly based on our paper [Energy-Based Models for Continual Learning](https://arxiv.org/pdf/2011.12216.pdf).

- [Project Page](https://energy-based-model.github.io/Energy-Based-Models-for-Continual-Learning/)

- [Code] This code is the basic version of our paper. We will release the final version soon.

------------------------------------------------------------------------------------------------------------------------
## Requirements
The current version of the code has been tested with:
* `pytorch 1.4.0`
* `torchvision 0.2.1`


------------------------------------------------------------------------------------------------------------------------
## Training (Boundary-aware setting)

### Split MNIST:

#### EBM:
``sh scripts/boundary_aware/train_ebm_splitmnist.sh``

#### Softmax-based classifier:
``sh scripts/boundary_aware/train_sbc_splitmnist.sh``


### Permuted MNIST:

#### EBM:
``sh scripts/boundary_aware/train_ebm_permmnist.sh``

#### Softmax-based classifier:
``sh scripts/boundary_aware/train_sbc_permmnist.sh``


### CIFAR-10:

#### EBM:
``sh scripts/boundary_aware/train_ebm_cifar10.sh``

#### Softmax-based classifier:
``sh scripts/boundary_aware/train_sbc_cifar10.sh``


### CIFAR-100:

#### EBM:
``sh scripts/boundary_aware/train_ebm_cifar100.sh``

#### Softmax-based classifier:
``sh scripts/boundary_aware/train_sbc_cifar100.sh``



------------------------------------------------------------------------------------------------------------------------
## Training (Boundary-agnostic setting)

### Split MNIST:

#### EBM:
``sh scripts/boundary_agnostic/train_ebm_splitmnist.sh``

#### Softmax-based classifier:
``sh scripts/boundary_agnostic/train_sbc_splitmnist.sh``


### Permuted MNIST:

#### EBM:
``sh scripts/boundary_agnostic/train_ebm_permmnist.sh``

#### Softmax-based classifier:
``sh scripts/boundary_agnostic/train_sbc_permmnist.sh``


### CIFAR-10:

#### EBM:
``sh scripts/boundary_agnostic/train_ebm_cifar10.sh``

#### Softmax-based classifier:
``sh scripts/boundary_agnostic/train_sbc_cifar10.sh``


### CIFAR-100:

#### EBM:
``sh scripts/boundary_agnostic/train_ebm_cifar100.sh``

#### Softmax-based classifier:
``sh scripts/boundary_agnostic/train_sbc_cifar100.sh``




------------------------------------------------------------------------------------------------------------------------
### Acknowledgements
Parts of the code were based on the implementation of https://github.com/GMvandeVen/continual-learning.




------------------------------------------------------------------------------------------------------------------------
### Citation
Please consider citing our papers if you use this code in your research:
```
@article{li2020energy,
  title={Energy-Based Models for Continual Learning},
  author={Li, Shuang and Du, Yilun and van de Ven, Gido M and Torralba, Antonio and Mordatch, Igor},
  journal={arXiv preprint arXiv:2011.12216},
  year={2020}
}
```













