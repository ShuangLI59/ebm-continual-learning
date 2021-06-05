import torch
import numpy as np
import tqdm
import copy
import utils
import time
import pdb
import sys
import random
import pickle
import os
from continual_learner import ContinualLearner



def train_cl(args, model, train_datasets, scenario="class",labels_per_task=None,iters=2000,batch_size=32,
             loss_cbs=list(), eval_cbs=list()):
   
    model.train()
    cuda = model._is_on_cuda()
    device = model._device()


    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())

    REAL_IMG_REPLAY = None
    pre_data_img_list = []
    pre_data_lab_list = []
    

    for task, train_dataset in enumerate(train_datasets, 1):

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        
        if REAL_IMG_REPLAY:
            i = task-2
            previous_dataset = train_datasets[i]
            previous_data_loader = iter(utils.get_data_loader(previous_dataset, batch_size, cuda=cuda, drop_last=True))
            
            tem_img = []
            tem_lab = []
            while(1):
                try:
                    previous_x, previous_y = next(previous_data_loader)
                    tem_img.append(previous_x)
                    tem_lab.append(previous_y)
                except:
                    # print(storage.imgs_step)
                    break

            
            if args.experiment=='cifar10' or args.experiment=='cifar100':
                tem_img = torch.stack(tem_img).view(-1, 3, 32, 32)
            elif args.experiment=='splitMNIST':
                tem_img = torch.stack(tem_img).view(-1, 1, 28, 28)
            elif args.experiment=='permMNIST':
                tem_img = torch.stack(tem_img).view(-1, 1, 32, 32)
            tem_lab = torch.stack(tem_lab).view(-1)


            
            # classes_per_task = len(args.labels_per_task[0])
            # pre_data_img_list.append(tem_img[pre_data_i_index[:args.budget*classes_per_task]])
            # pre_data_lab_list.append(tem_lab[pre_data_i_index[:args.budget*classes_per_task]])

            replay_way = 'v2'

            
            if replay_way == 'v1':
                pre_data_i_index = list(range(len(tem_img)))
                random.shuffle(pre_data_i_index)
                pre_data_img_list.append(tem_img[pre_data_i_index[:args.budget]])
                pre_data_lab_list.append(tem_lab[pre_data_i_index[:args.budget]])
            
            elif replay_way == 'v2':
                num_sample_each_task = int(args.budget/(task-1))
                if task>2:
                    pre_data_img_list = [tem[:num_sample_each_task] for tem in pre_data_img_list]
                    pre_data_lab_list = [tem[:num_sample_each_task] for tem in pre_data_lab_list]

                pre_data_i_index = list(range(len(tem_img)))
                random.shuffle(pre_data_i_index)
                pre_data_img_list.append(tem_img[pre_data_i_index[:num_sample_each_task]])
                pre_data_lab_list.append(tem_lab[pre_data_i_index[:num_sample_each_task]])


            if args.experiment=='cifar10' or args.experiment=='cifar100':
                pre_data_img = torch.stack(pre_data_img_list).view(-1, 3, 32, 32)
            elif args.experiment=='splitMNIST':
                pre_data_img = torch.stack(pre_data_img_list).view(-1, 1, 28, 28)
            elif args.experiment=='permMNIST':
                pre_data_img = torch.stack(pre_data_img_list).view(-1, 1, 32, 32)
            pre_data_lab = torch.stack(pre_data_lab_list).view(-1)

            # print(pre_data_img.shape)
            # print(pre_data_lab.shape)
            # print(pre_data_lab.max(), pre_data_lab.min())
            

        progress = tqdm.tqdm(range(1, iters+1))
        iters_left = 1

        for batch_index in range(1, iters+1):
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            x, y = next(data_loader)
            x, y = x.to(device), y.to(device)
            

            x_, y_ = None, None
            if REAL_IMG_REPLAY:
                index = list(range(len(pre_data_img)))
                random.shuffle(index)
                index = index[:batch_size]
                # index = index[:batch_size*(task-1)]
                x_ = pre_data_img[index]
                y_ = pre_data_lab[index]
                x_ = x_.to(device)
                y_ = y_.to(device)


            if batch_index <= iters:
                loss_dict = model.train_a_batch(args, x, y, x_, y_, task=task)


                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (model.si_c>0):
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(args, model, batch_index, task=task)

        progress.close()
        
        
        ## EWC and SI is only for softmax-based classifier
        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty    
        if isinstance(model, ContinualLearner) and (model.ewc_lambda>0):
            # -estimate FI-matrix
            model.estimate_fisher(args, train_dataset, task)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            model.update_omega(W, model.epsilon)


        if args.replay_mode == 'real_img_replay':
            REAL_IMG_REPLAY = True



def train_cl_noboundary(args, model, train_datasets, scenario="class", labels_per_task=None, iters=2000, batch_size=32, loss_cbs=list(), eval_cbs=list()):
   
    model.train()
    cuda = model._is_on_cuda()
    device = model._device()


    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
    

    for epoch in range(args.epc_per_virtual_task*args.tasks):
        
        ## training
        progress = tqdm.tqdm(range(1, args.iterations_per_virtual_epc+1))
        
        for batch_index, (x, y) in enumerate(train_datasets):

            # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
            if isinstance(model, ContinualLearner) and (model.si_c>0):
                W = {}
                p_old = {}
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        W[n] = p.data.clone().zero_()
                        p_old[n] = p.data.clone()


            x, y = x.to(device), y.to(device)
            x_, y_ = None, None
            loss_dict = model.train_a_batch(args, x, y, x_, y_)

            
            
            # Update running parameter importance estimates in W
            if isinstance(model, ContinualLearner) and (model.si_c>0):
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad*(p.detach()-p_old[n]))
                        p_old[n] = p.detach().clone()


            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, epoch)

            
            # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
            if isinstance(model, ContinualLearner) and (model.ewc_lambda>0):
                # -estimate FI-matrix
                model.estimate_fisher(args, x, y)

            # SI: calculate and update the normalized path integral
            if isinstance(model, ContinualLearner) and (model.si_c>0):
                model.update_omega(W, model.epsilon)


        progress.close()


        ## testing
        if epoch % args.epc_per_virtual_task==0:
            task = args.task_dict[int(y.max())]
            
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(args, model, batch_index, task=task)
            
        

