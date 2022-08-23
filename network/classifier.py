import torch
from torch.nn import functional as F
from network.classifier_layers import CLS_MLP, CLS_CONV, CLS_net_cifar100
from network.conv_layers import ConvLayers
from network.fc.layers import fc_layer
import utils
import pdb
import random
import numpy as np
import torch.nn as nn
from continual_learner import ContinualLearner

class Classifier(ContinualLearner):
    def __init__(self, args, image_size, image_channels, classes, fc_units=1000):
        super().__init__()

        self.label = "Classifier"
        self.args = args
        self.num_classes = classes
        self.class_entries = list(range(self.num_classes))
        self.labels_per_task = args.labels_per_task
        
        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        # fully connected hidden layers
        if args.experiment=='splitMNIST' or args.experiment=='permMNIST' or args.experiment=='splitMNISToneclass':
            self.fcE = CLS_MLP(args, num_classes=self.num_classes, input_size=image_channels*image_size**2, hid_size=fc_units)
            
        elif args.experiment=='cifar10':
            self.fcE = CLS_CONV(args, num_classes=self.num_classes, input_size=image_channels*image_size**2, hid_size=fc_units)

        elif args.experiment=='cifar100':
            self.convE = ConvLayers(
                conv_type='standard', block_type="basic", num_blocks=2, image_channels=image_channels,
                depth=5, start_channels=16, reducing_layers=4, batch_norm=True, nl='relu',
                global_pooling=False, gated=False, output="none",
            )
            #------------------------------calculate input/output-sizes--------------------------------#
            fc_units = 2000
            h_dim = 2000
            fc_layers = 3
            self.conv_out_units = self.convE.out_units(image_size)
            self.conv_out_size = self.convE.out_size(image_size)
            self.conv_out_channels = self.convE.out_channels
            if fc_layers<2:
                self.fc_layer_sizes = [self.conv_out_units]  #--> this results in self.fcE = modules.Identity()
            elif fc_layers==2:
                self.fc_layer_sizes = [self.conv_out_units, h_dim]
            else:
                self.fc_layer_sizes = [self.conv_out_units]+[int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers-1)]
            self.units_before_classifier = h_dim if fc_layers>1 else self.conv_out_units
            #------------------------------------------------------------------------------------------#
            self.fcE = CLS_net_cifar100(size_per_layer=self.fc_layer_sizes, drop=0, batch_norm=False, nl='relu', bias=True,
                       excitability=False, excit_buffer=True, gated=False)
            self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none', drop=0)


    def forward(self, x):
        if self.args.experiment=='splitMNIST' or self.args.experiment=='permMNIST' or self.args.experiment=='splitMNISToneclass':
            final_features = self.fcE(self.flatten(x))
        
        elif self.args.experiment=='cifar10':
            final_features = self.fcE(x)

        elif self.args.experiment=='cifar100':
            hidden_rep = self.convE(x)
            final_features = self.fcE(self.flatten(hidden_rep))
            final_features = self.classifier(final_features)

        return final_features

    
    @property
    def name(self):
        return 'Classifier'

    def train_a_batch(self, args, x, y, x_, y_, task=1, device="gpu"):
        self.train()
        self.optimizer.zero_grad()
        batch_size = x.shape[0]


        if x_ is None:
            if args.task_boundary:
                cur_classes = self.labels_per_task[task-1]
            else:
                cur_classes=list(y.unique())
            for tem in y: assert tem in cur_classes ## y shoud be in current classes
        else:
            if args.task_boundary:
                cur_classes = np.stack(self.labels_per_task[:task])
                cur_classes = cur_classes.reshape(-1)
                x = torch.cat([x, x_], dim=0)
                y = torch.cat([y, y_], dim=0)
                
    
        if args.cls_standard:
            y_hat = self(x)

            over_seen_classes = True
            if over_seen_classes:
                seen_classes_list = []

                if not args.task_boundary:
                    task = args.task_dict[int(y.max())]

                for i in range(task):
                    seen_classes_list += self.labels_per_task[i]

                y_hat = y_hat[:, seen_classes_list]


                ## compute loss
                y_tem = torch.tensor([seen_classes_list.index(tem) for tem in y]).long().to(device)
                
                if y_ is not None:
                    predL = F.cross_entropy(input=y_hat, target=y_tem, reduction='none')
                    predL = 1/task * predL[:batch_size].mean() + (1-1/task) * predL[batch_size:].mean()
                else:
                    predL = F.cross_entropy(input=y_hat, target=y_tem, reduction='mean')
                loss_cur = predL

                ## compuate accuracy
                _, precision = torch.max(y_hat, 1)
                precision = 1.* (precision == y_tem).sum().item() / x.size(0)
            
            else:
                ## compute loss over all classes
                predL = F.cross_entropy(input=y_hat, target=y, reduction='mean')
                loss_cur = predL

                ## compuate accuracy
                _, precision = torch.max(y_hat, 1)
                precision = 1.* (precision == y).sum().item() / x.size(0)

        else:
            # single_neg = args.single_neg
            single_neg = True
            if single_neg:
                joint_targets = torch.LongTensor(batch_size, 2).cuda()
                for i in range(batch_size):
                    while True:
                        neg_sample = random.choice(cur_classes)
                        if not neg_sample == y[i]:
                            break
                    joint_targets[i] = torch.tensor([y[i], neg_sample]).cuda()
                
                y_hat = self(x)
                y_hat = y_hat.gather(dim=1, index=joint_targets)

                ## compute loss
                predL = F.cross_entropy(input=y_hat, target=torch.zeros(batch_size).long().cuda(), reduction='mean')
                loss_cur = predL

                ## compuate accuracy
                _, precision = torch.max(y_hat, 1)
                precision = 1.* (precision == torch.zeros(batch_size).long().cuda()).sum().item() / x.size(0)

            else:
                y_hat = self(x)
                y_hat = y_hat[:, cur_classes]

                ## compute loss
                y_tem = torch.tensor([cur_classes.index(tem) for tem in y]).long().cuda()
                predL = F.cross_entropy(input=y_hat, target=y_tem, reduction='mean')
                loss_cur = predL

                ## compuate accuracy
                _, precision = torch.max(y_hat, 1)
                precision = 1.* (precision == y_tem).sum().item() / x.size(0)

        ## other losses
        loss_total = loss_cur
        
        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss


        loss_total.backward()
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'ewc': ewc_loss.item(), 
            'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }



