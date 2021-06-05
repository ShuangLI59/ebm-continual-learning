import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
from data_loader.data_config import AVAILABLE_DATASETS, DATASET_CONFIGS, AVAILABLE_TRANSFORMS, labels_per_tasks_overlap, labels_per_tasks_nooverlap, labels_per_tasks_standard
from data_loader.data_utils import _permutate_image_pixels, _create_task_probs, SubDataset, ContinuousMultinomialSampler
import torch
import pdb
import os
import pickle
import random
from copy import deepcopy

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir=None,
                verbose=False, target_transform=None):
    
    data_name = 'mnist' if name=='mnist28' else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    if name=='cifar100':
        if type=='train':
            dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS['cifar100_train'],
                transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
            ])
        else:
            dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS['cifar100_test'],
                transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
            ])

        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)
    elif name=='cifar10':
        if type=='train':
            dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS['cifar10_train'],
                transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
            ])
        else:
            dataset_transform = transforms.Compose([
                *AVAILABLE_TRANSFORMS['cifar10_test'],
                transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
            ])

        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)
    else:
        dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[name],
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
        ])

        
        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    return dataset




def get_multitask_experiment(args, name, scenario, tasks, data_dir=None, only_config=False, verbose=False, exception=False):
    
    if name == 'permMNIST':
        config = DATASET_CONFIGS['mnist']
        
        labels_per_task = labels_per_tasks_standard['permMNIST'][args.seed]
        classes_per_task = 10
        task_dict = {k:int(k/10)+1 for k in range(100)}

        if not only_config:
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(len(labels_per_task)-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(len(labels_per_task))]
            # prepare datasets
            train_datasets = []
            test_datasets = []
            for task_id, p in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class') else None
                train_datasets.append(get_dataset('mnist', type="train", permutation=p, dir=data_dir,
                                                  target_transform=target_transform, verbose=verbose))
                test_datasets.append(get_dataset('mnist', type="test", permutation=p, dir=data_dir,
                                                 target_transform=target_transform, verbose=verbose))


    elif name == 'splitMNIST':
        config = DATASET_CONFIGS['mnist28']
        
        if not only_config:
            permutation = np.array(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)

            ## ---------------------------------------------------------------------------------------------------------
            ## generated data splits
            ## ---------------------------------------------------------------------------------------------------------
            labels_per_task = labels_per_tasks_standard['splitMNIST'][args.seed]
            task_dict = {k:int(k/2)+1 for k in range(10)}

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []

            if args.random_task_order:
                print('random task order')
                labels_per_task = deepcopy(labels_per_task)
                random.shuffle(labels_per_task)

            for task_id, labels in enumerate(labels_per_task):
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(args, mnist_train, labels, class_per_task=2, task_id=task_id, target_transform=target_transform))
                test_datasets.append(SubDataset(args, mnist_test, labels, class_per_task=2, task_id=task_id, target_transform=target_transform))

            
            labels_per_task = labels_per_tasks_standard['splitMNIST'][args.seed]


    elif name == 'cifar10':
        config = DATASET_CONFIGS['cifar10']
        
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            cifar10_train = get_dataset('cifar10', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            cifar10_test = get_dataset('cifar10', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            
            ## ---------------------------------------------------------------------------------------------------------
            ## generated data splits
            ## ---------------------------------------------------------------------------------------------------------
            labels_per_task = labels_per_tasks_standard['cifar10'][args.seed]
            task_dict = {k:int(k/2)+1 for k in range(10)}

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []

            all_train_labels = [cifar10_train[index][1] for index in range(len(cifar10_train))]
            all_test_labels = [cifar10_test[index][1] for index in range(len(cifar10_test))]

            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(args, cifar10_train, labels, target_transform=target_transform, all_labels=all_train_labels, cifar=True))
                test_datasets.append(SubDataset(args, cifar10_test, labels, target_transform=target_transform, all_labels=all_test_labels, cifar=True))
    
    elif name == 'cifar100':
        # configurations
        config = DATASET_CONFIGS['cifar100']
        
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(100)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, target_transform=target_transform,
                                      verbose=verbose)
            cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            
            ## ---------------------------------------------------------------------------------------------------------
            ## generated data splits
            ## ---------------------------------------------------------------------------------------------------------
            labels_per_task = labels_per_tasks_standard['cifar100'][args.seed]
            task_dict = {k:int(k/10)+1 for k in range(100)}

            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []

            all_train_labels = [cifar100_train[index][1] for index in range(len(cifar100_train))]
            all_test_labels = [cifar100_test[index][1] for index in range(len(cifar100_test))]

            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                train_datasets.append(SubDataset(args, cifar100_train, labels, target_transform=target_transform, all_labels=all_train_labels, cifar=True))
                test_datasets.append(SubDataset(args, cifar100_test, labels, target_transform=target_transform, all_labels=all_test_labels, cifar=True))
    
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    

    if not args.task_boundary:
        
        iterations_per_virtual_epc = np.min([len(train_dataset) for train_dataset in train_datasets])
        iterations_per_virtual_epc = int(iterations_per_virtual_epc/args.batch)
        
        total_num_epochs = args.epc_per_virtual_task * len(labels_per_task)
        total_iters = total_num_epochs*iterations_per_virtual_epc
        
        beta = 4
        # Create probabilities of tasks over iterations
        tasks_probs_over_iterations = [_create_task_probs(total_iters, len(labels_per_task), task_id, beta=beta) for task_id in range(len(labels_per_task))]
        normalize_probs = torch.zeros_like(tasks_probs_over_iterations[0])
        for probs in tasks_probs_over_iterations:
            normalize_probs.add_(probs)
        for probs in tasks_probs_over_iterations:
            probs.div_(normalize_probs)
        tasks_probs_over_iterations = torch.cat(tasks_probs_over_iterations).view(-1, tasks_probs_over_iterations[0].shape[0])
        tasks_probs_over_iterations_lst = []
        for col in range(tasks_probs_over_iterations.shape[1]):
            tasks_probs_over_iterations_lst.append(tasks_probs_over_iterations[:, col])
        tasks_probs_over_iterations = tasks_probs_over_iterations_lst


    
        tasks_samples_indices = []
        total_len = 0
        for train_dataset in train_datasets:
            tasks_samples_indices.append(torch.tensor(range(total_len, total_len+len(train_dataset)), dtype=torch.int32))
            total_len += len(train_dataset)

    
        all_datasets = torch.utils.data.ConcatDataset(train_datasets)
        train_sampler = ContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=args.batch,
                                                     tasks_samples_indices=tasks_samples_indices,
                                                     tasks_probs_over_iterations=tasks_probs_over_iterations,
                                                     num_of_batches=iterations_per_virtual_epc)

        pin_memory = True if torch.cuda.is_available() else False
        train_loader = torch.utils.data.DataLoader(all_datasets, batch_size=args.batch,
                                                        num_workers=1, sampler=train_sampler, pin_memory=pin_memory)

        
        # test_loaders = []
        # for test_dataset in test_datasets:
        #     test_loaders.append(torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=1, shuffle=False, pin_memory=pin_memory))


        config['num_classes'] = args.num_classes
        config['labels_per_task'] = labels_per_task
        config['iterations_per_virtual_epc'] = iterations_per_virtual_epc
        config['task_dict'] = task_dict
        return ((train_loader, test_datasets), config)
    
    else:
        config['num_classes'] = args.num_classes
        config['labels_per_task'] = labels_per_task
        return ((train_datasets, test_datasets), config)


