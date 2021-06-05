#!/usr/bin/env python3
import argparse
import os
import numpy as np
import time
import torch
from torch import optim
import visual_plt
import utils
from param_stamp import get_param_stamp
import evaluate
from continual_learner import ContinualLearner
from data_loader.data_loader_online import get_multitask_experiment
from network.classifier import Classifier
from network.ebm import EBM
import callbacks as cb
from train import train_cl, train_cl_noboundary
from param_values import set_default_values
import pdb
import random
import utils

parser = argparse.ArgumentParser('main.py', description='Energy-based Model for Continual Learning.')
parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str, default='../datasets', dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

parser.add_argument('--cls-standard', type=int, default=1)
parser.add_argument('--single-neg', type=int, default=0)
parser.add_argument('--pretrain', type=str, default='')
parser.add_argument('--random-task-order', type=int, default=0)
parser.add_argument('--task-info-input', type=int, default=0)


# expirimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, default='splitMNIST', choices=['permMNIST', 'splitMNIST', 'cifar10', 'cifar100', 'splitMNISToneclass'])
task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
task_params.add_argument('--task-boundary', action='store_true', help='task boundaries are provided')
task_params.add_argument('--epc-per-virtual-task', type=int, default=50)

task_params.add_argument('--tasks', type=int, help='number of tasks')
task_params.add_argument('--model-name', type=str, default='')
task_params.add_argument('--save-dir', type=str, default='')
task_params.add_argument('--res-net-num-layer', type=int, default=8)

# "EBM" parameters
ebm_params = parser.add_argument_group('EBM Parameters')
ebm_params.add_argument('--ebm', action="store_true", help="ebm model")


# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")


# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")

cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")

cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")

cl_params.add_argument('--xdg', action='store_true', help="Use 'Context-dependent Gating' (Masse et al, 2018)")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")

# exact replay
cl_params.add_argument('--replay-mode', type=str, default='')
cl_params.add_argument('--budget', type=int, default=1000)

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--iters', type=int, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, help="learning rate")
train_params.add_argument('--batch', type=int, default=128, help="batch-size")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='adam')


# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
eval_params.add_argument('--loss-log', type=int, default=1000, metavar="N", help="# iters after which to plot loss")
eval_params.add_argument('--prec-log', type=int, default=1000, metavar="N", help="# iters after which to plot precision")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--neg-img', type=int, default=0)

parser.add_argument('--single-test', action='store_false')



def run(args):

    if not args.single_test:
        import pidfile
        resfile = pidfile.exclusive_dirfn(os.path.join(args.r_dir, args.save_dir))
    

    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
    
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)
    
    # set cuda
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    scenario = args.scenario
    

    #-------------------------------------------------------------------------------------------------
    # DATA
    #-------------------------------------------------------------------------------------------------
    (train_datasets, test_datasets), config = get_multitask_experiment(args,
        name=args.experiment, scenario=scenario, tasks=args.tasks, data_dir=args.d_dir,
        verbose=True, exception=True if args.seed==0 else False,
    )
    args.tasks = len(config['labels_per_task'])
    args.labels_per_task = config['labels_per_task']
    if not args.task_boundary:
        args.iterations_per_virtual_epc = config['iterations_per_virtual_epc']
        args.task_dict = config['task_dict']



    #-------------------------------------------------------------------------------------------------
    # MODEL
    #-------------------------------------------------------------------------------------------------
    if args.ebm:
        model = EBM(args, image_size=config['size'], image_channels=config['channels'], classes=config['num_classes'], fc_units=args.fc_units).to(device)
    else:
        model = Classifier(args, image_size=config['size'], image_channels=config['channels'], classes=config['num_classes'], fc_units=args.fc_units).to(device)

    if args.experiment=='cifar100':
        model = utils.init_params(model, args)
        for param in model.convE.parameters():
            param.requires_grad = False


    if args.pretrain:
        checkpoint = torch.load(args.pretrain)
        best_acc = checkpoint['best_acc']
        checkpoint_state = checkpoint['state_dict']
        
        print('-----------------------------------------------------------------------------')
        print('load pretrained model %s' % args.pretrain)
        print('best_acc', best_acc)
        print('-----------------------------------------------------------------------------')
        

        model_dict = model.fcE.state_dict()
        checkpoint_state = {k[7:]: v for k, v in checkpoint_state.items() if k[7:] in model_dict} ## remove module.
        del checkpoint_state['classifier.weight']
        del checkpoint_state['classifier.bias']
        if 'y_ebm.weight' in checkpoint_state:
            del checkpoint_state['y_ebm.weight']
        model_dict.update(checkpoint_state) 
        model.fcE.load_state_dict(model_dict)

        for param in model.fcE.model.parameters():
            param.requires_grad = False

    
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type=="sgd":
        model.optimizer = optim.SGD(model.optim_list)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))

    
    #-------------------------------------------------------------------------------------------------
    # CL-STRATEGY: ALLOCATION
    #-------------------------------------------------------------------------------------------------

    # Elastic Weight Consolidation (EWC)
    if isinstance(model, ContinualLearner):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        if args.ewc:
            model.fisher_n = args.fisher_n
            model.gamma = args.gamma
            model.online = args.online
            model.emp_FI = args.emp_fi

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner):
        model.si_c = args.si_c if args.si else 0
        if args.si:
            model.epsilon = args.epsilon


    #-------------------------------------------------------------------------------------------------
    # Get parameter-stamp (and print on screen)
    #-------------------------------------------------------------------------------------------------
    param_stamp = get_param_stamp(args, model.name, verbose=True)
    param_stamp = param_stamp + '--' + args.model_name
    
    
    # -define [precision_dict] to keep track of performance during training for storing and for later plotting in pdf
    precision_dict = evaluate.initiate_precision_dict(args.tasks)
    

    
    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#
    solver_loss_cbs = [cb._solver_loss_cb(log=args.loss_log, model=model, tasks=args.tasks, iters_per_task=args.iters)]
    
    eval_cb = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, visdom=args.visdom, precision_dict=None, iters_per_task=args.iters,
        test_size=args.prec_n, labels_per_task=config['labels_per_task'], scenario=scenario)
    eval_cb_full = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, precision_dict=precision_dict,
        iters_per_task=args.iters, labels_per_task=config['labels_per_task'], scenario=scenario)
    eval_cbs = [eval_cb, eval_cb_full]
    



    #-------------------------------------------------------------------------------------------------
    # TRAINING
    #-------------------------------------------------------------------------------------------------
    print("--> Training:")
    start = time.time()

    if args.task_boundary:
        train_cl(
            args, model, train_datasets, scenario=scenario, labels_per_task=config['labels_per_task'],
            iters=args.iters, batch_size=args.batch,
            eval_cbs=eval_cbs, loss_cbs=solver_loss_cbs)
    else:
        train_cl_noboundary(
            args, model, train_datasets, scenario=scenario, labels_per_task=config['labels_per_task'],
            iters=args.iters, batch_size=args.batch,
            eval_cbs=eval_cbs, loss_cbs=solver_loss_cbs)

    training_time = time.time() - start
    


    #-------------------------------------------------------------------------------------------------
    # EVALUATION
    #-------------------------------------------------------------------------------------------------
    print("\n\n--> Evaluation ({}-incremental learning scenario):".format(args.scenario))
    if args.ebm:
        precs = [evaluate.validate_ebm(
            args, model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=False,
            current_task = args.tasks) for i in range(args.tasks)]
    else:
        precs = [evaluate.validate(
            args, model, test_datasets[i], verbose=False, test_size=None, task=i+1, with_exemplars=False,
            current_task = args.tasks) for i in range(args.tasks)]

    print("\n Precision on test-set (softmax classification):")
    for i in range(args.tasks):
        print(" - Task {}: {:.4f}".format(i + 1, precs[i]))
    average_precs = sum(precs) / args.tasks
    print('average precision over all {} tasks: {:.4f}'.format(args.tasks, average_precs))


    #-------------------------------------------------------------------------------------------------
    # OUTPUT
    #-------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(args.r_dir, args.save_dir)):
        os.makedirs(os.path.join(args.r_dir, args.save_dir))
    
    output_file = open("{}/{}/{}.txt".format(args.r_dir, args.save_dir, param_stamp), 'w')
    output_file.write("Training time {} \n".format(training_time))
    for i in range(args.tasks):
        output_file.write(" - Task {}: {:.4f}".format(i + 1, precs[i]))
        output_file.write("\n")
    output_file.write(' - Average {}\n'.format(average_precs))
    output_file.close()
    file_name = "{}/{}/{}".format(args.r_dir, args.save_dir, param_stamp)
    utils.save_object(precision_dict, file_name)


    if args.pdf:
        pp = visual_plt.open_pdf("{}/{}/{}.pdf".format(args.r_dir, args.save_dir, param_stamp))
        # -show metrics reflecting progression during training
        figure_list = []  #-> create list to store all figures to be plotted
        # -generate all figures (and store them in [figure_list])
        figure = visual_plt.plot_lines(
            precision_dict["all_tasks"], x_axes=precision_dict["x_task"],
            line_names=['task {}'.format(i + 1) for i in range(args.tasks)]
        )
        figure_list.append(figure)
        figure = visual_plt.plot_lines(
            [precision_dict["average"]], x_axes=precision_dict["x_task"],
            line_names=['average all tasks so far']
        )
        figure_list.append(figure)
        # -add figures to pdf (and close this pdf).
        for figure in figure_list:
            pp.savefig(figure)

        pp.close()

    if not args.single_test:
        resfile.done()


if __name__ == '__main__':    
    args = parser.parse_args()
    args = set_default_values(args)
    run(args)



