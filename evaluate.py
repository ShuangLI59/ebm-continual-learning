import numpy as np
import torch
import visual_plt
import utils
import pdb
import gc
from torch.nn import functional as F
import random
import os
import pickle
import time
####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(args, model, dataset, batch_size=128, test_size=1024, verbose=True,
             with_exemplars=False, no_task_mask=False, task=None, current_task=None, device="cuda"):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0

    labels_per_task = args.labels_per_task
    seen_classes_list = []
    for i in range(current_task):
        seen_classes_list += labels_per_task[i]
    seen_classes = np.array(seen_classes_list)

    tic = time.time()
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break

        
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(model._device()), labels.to(model._device())
        batch_size = data.shape[0]
        for tem in labels: assert tem in seen_classes_list ## y shoud be in current classes
        
        with torch.no_grad():
            # Run model
            y_hat = model(data)

                        
            y_hat = y_hat[:, seen_classes]

            ## accuracy
            label_tem = torch.tensor([seen_classes_list.index(tem) for tem in labels]).long().to(device)
            _, precision = torch.max(y_hat, 1)

            predicted = (precision == label_tem).sum().item()

        total_correct += predicted
        total_tested += len(data)

    toc = time.time()
    print('time: ', toc - tic)
    precision = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    verbose = True
    if verbose:
        print('=> {}: Task {} precision: {:.3f}'.format(args.save_dir, task, precision))
        
    return precision




def validate_ebm(args, model, dataset, batch_size=128, test_size=1024, verbose=True,
             with_exemplars=False, no_task_mask=False, task=None, current_task=None, device="cuda"):

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0

    labels_per_task = args.labels_per_task
    seen_classes_list = []
    for i in range(current_task):
        seen_classes_list += labels_per_task[i]
    seen_classes = np.array(seen_classes_list)

    tic = time.time()
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break


        data, labels = data.to(model._device()), labels.to(model._device())
        batch_size = data.shape[0]
        for tem in labels: assert tem in list(seen_classes_list) ## y shoud be in current classes
        
        if args.experiment=='cifar100':
            ys_to_test = torch.LongTensor(batch_size, len(seen_classes_list)).to(device)
            for i in range(batch_size):
                ys_to_test[i] = torch.tensor(seen_classes_list).to(device)
            energy = model(data, ys_to_test)

            # accuracy
            _, predicted = torch.max(energy, 1) ## cifar100 model predict negative energy
            label_tem = torch.tensor([seen_classes_list.index(tem) for tem in labels]).long().to(device)
            predicted = (predicted == label_tem).sum().item()
            
        else:
            ## get negatives+positive labels
            joint_targets = torch.tensor(seen_classes).view(1, -1).expand(batch_size, len(seen_classes))
            joint_targets = joint_targets.long().to(device)

            with torch.no_grad():
                # Run model
                # print(task, current_task)
                if args.task_info_input:
                    task_id = (torch.ones([batch_size])*(task-1)).long().to(device)
                    energy = model(data, joint_targets, task_id)
                else:
                    energy = model(data, joint_targets)

            
                # accuracy
                _, predicted = torch.min(energy, 1)
                label_tem = torch.tensor([seen_classes_list.index(tem) for tem in labels]).long().to(device)

                predicted = (predicted == label_tem).sum().item()




        total_correct += predicted
        total_tested += len(data)

    toc = time.time()
    print('ebm time: ', toc - tic)
    precision = total_correct / total_tested


    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    verbose = True
    if verbose:
        print('=> {}: Task {} precision: {:.3f}'.format(args.save_dir, task, precision))
    
    return precision



def precision(args, model, datasets, current_task, iteration, labels_per_task=None, scenario="class",
              precision_dict=None, test_size=None, visdom=None, verbose=False, summary_graph=True,
              with_exemplars=False, no_task_mask=False, device="cuda"):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [precision_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [scenario]          <str> how to decide which classes to include during evaluating precision
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    
    n_tasks = len(datasets)
    precs = []
    for i in range(n_tasks):
        if i+1 <= current_task:
            if args.ebm:
                precs.append(validate_ebm(args, model, datasets[i], test_size=test_size, verbose=verbose,
                                      with_exemplars=with_exemplars,
                                      no_task_mask=no_task_mask, task=i+1, current_task=current_task, device=device))
            else:
                precs.append(validate(args, model, datasets[i], test_size=test_size, verbose=verbose,
                                      with_exemplars=with_exemplars,
                                      no_task_mask=no_task_mask, task=i+1, current_task=current_task, device=device))
        else:
            precs.append(0)
    
    average_precs = sum([precs[task_id] for task_id in range(current_task)]) / current_task

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    # Append results to [progress]-dictionary and return
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if precision_dict is not None:
        for task_id, _ in enumerate(names):
            precision_dict["all_tasks"][task_id].append(precs[task_id])
        precision_dict["average"].append(average_precs)
        precision_dict["x_iteration"].append(iteration)
        precision_dict["x_task"].append(current_task)
    return precision_dict



def initiate_precision_dict(n_tasks):
    '''Initiate <dict> with all precision-measures to keep track of.'''
    precision = {}
    precision["all_tasks"] = [[] for _ in range(n_tasks)]
    precision["average"] = []
    precision["x_iteration"] = []
    precision["x_task"] = []
    return precision


