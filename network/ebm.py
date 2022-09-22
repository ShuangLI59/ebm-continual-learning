import torch
from torch.nn import functional as F
from network.ebm_layers import EBM_MLP, EBM_CONV, EBM_net_cifar100
from network.conv_layers import ConvLayers
import utils
import pdb
import torch.nn as nn
import numpy as np
import random
import gc
from continual_learner import ContinualLearner
from network.energy_loss import nll, mee, square_exponential, square_square, log_loss, hinge_loss


class EBM(ContinualLearner):
    def __init__(self, args, image_size, image_channels, classes, fc_units=1000):
        super().__init__()

        self.label = "EBM"        
        self.args = args
        self.num_classes = classes
        self.class_entries = list(range(self.num_classes))
        self.labels_per_task = args.labels_per_task
        self.device=args.device
        
        # flatten image to 2D-tensor
        self.flatten = utils.Flatten()

        
        # fully connected hidden layers
        if args.experiment=='splitMNIST' or args.experiment=='permMNIST':
            self.fcE = EBM_MLP(args, num_classes=self.num_classes, input_size=image_channels*image_size**2, hid_size=fc_units)    

        elif args.experiment=='cifar10':
            self.fcE = EBM_CONV(args, num_classes=self.num_classes, input_size=image_channels*image_size**2, hid_size=fc_units)

        elif args.experiment=='cifar100':
            # self.fcE = EBM_CONV(args, num_classes=self.num_classes, input_size=image_channels*image_size**2, hid_size=fc_units)

            self.convE = ConvLayers(
                conv_type='standard', block_type="basic", num_blocks=2, image_channels=3,
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
            self.fcE = EBM_net_cifar100(num_classes=100, size_per_layer=self.fc_layer_sizes, drop=0, batch_norm=False,
                           nl='relu', bias=True, excitability=False, excit_buffer=True,
                           fixed_mask=True, mask_prob=0.85, only_first=False, with_skip=False,device=self.device)
            self.classifier = nn.Linear(self.units_before_classifier, 1, bias=True)

                
    def forward(self, x, y, task_id=None):
        if self.args.experiment=='splitMNIST' or self.args.experiment=='permMNIST' or self.args.experiment=='splitMNISToneclass':
            if self.args.task_info_input:
                final_features = self.fcE(self.flatten(x), y, task_id)
            else:
                final_features = self.fcE(self.flatten(x), y)
        
        elif self.args.experiment=='cifar10':
            final_features = self.fcE(x, y)

        elif self.args.experiment=='cifar100':
            # final_features = self.fcE(x, y)

            batch_size = x.shape[0]
            hidden_rep = self.flatten(self.convE(x))
            features = self.fcE(hidden_rep, y)
            final_features = self.classifier(features)
            final_features = final_features.view(batch_size, -1)

        return final_features


    @property
    def name(self):
        return 'EBM'


    def forward_cifar100(self, args, x, y, x_, y_, task):
        single_neg = True
        self.neg_energy = True
        self.only_classes_in_current_task = True
        
        batch_size_ori, c, w, h = x.shape



        if x_ is not None:
            x = torch.cat([x, x_], dim=0)
            y = torch.cat([y, y_], dim=0)

        y_list = list(y.cpu().numpy())
        y_list_to_sample_from = y_list
        batch_size, c, w, h = x.shape

        # print(x.shape, y.shape)
        # print(np.max(y_list_to_sample_from), np.min(y_list_to_sample_from))

        if single_neg:
            ys_to_test = torch.LongTensor(batch_size, 2).to(self._device())
            for i in range(len(y_list)):
                if len(np.unique(y_list_to_sample_from))==1:
                    # the below would break down if all entries in the batch have the same label!!!
                    raise RuntimeError('All samples in this batch have the same label!!')
                else:
                    while True:
                        neg_sample = random.choice(y_list_to_sample_from)
                        if not neg_sample == y_list[i]:
                            break
                ys_to_test[i] = torch.tensor([y_list[i], neg_sample]).to(self._device())
        else:
            # -select all labels to use as negative samples (NOTE: one of them should be the positive sample!)
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                if self.only_classes_in_current_task:
                    class_entries = class_entries[-self.classes_per_task:]
            else:
                class_entries = list(range(self.classes))
            # -convert them to proper shape
            ys_to_test = torch.LongTensor(batch_size, len(class_entries)).to(self._device())
            for i in range(batch_size):
                ys_to_test[i] = torch.tensor(class_entries).to(self._device())


        
        y_neg_energies = self(x, y=ys_to_test) if self.neg_energy else -1 * self(x, y=ys_to_test)


        # Calculate multiclass prediction loss
        if single_neg:
            ne_pos_sample = y_neg_energies[:, 0]
            ne_neg_samples = torch.logsumexp(y_neg_energies, dim=1, keepdim=False)
            predL = -ne_pos_sample + ne_neg_samples
            
            if y_ is not None:
                predL = 1/task * predL[:batch_size_ori].mean() + (1-1/task) * predL[batch_size_ori:].mean()
            else:
                predL = predL.mean()
        else:
            if y is not None and len(y.size()) == 0:
                y = y.expand(1)  # --> hack to make it work if batch-size is 1
            if y is not None:
                if self.only_classes_in_current_task:
                    y = y - class_entries[0]
                # predL = F.cross_entropy(input=y_neg_energies, target=y, reduction='none') #-> summing over classes implicit
                # predL = lf.weighted_average(predL, weights=None, dim=0)                   #-> average over batch
                # NOTE: above two lines are similar to below 5 lines!
                tem_y = y.view(y.shape[0], 1)
                ne_pos_sample = y_neg_energies.gather(dim=1, index=tem_y)
                ne_neg_samples = torch.logsumexp(y_neg_energies, dim=1, keepdim=True)
                predL = -ne_pos_sample + ne_neg_samples
                predL = predL.mean()

        # Weigh losses
        loss_cur = None if y is None else predL

        # Calculate training-precision
        if single_neg:
            precision = None if y is None else (y_neg_energies.max(1)[1]==0).sum().item() / x.size(0)
        else:
            precision = None if y is None else (y == y_neg_energies.max(1)[1]).sum().item() / x.size(0)

        return loss_cur, precision


        

    def train_a_batch(self, args, x, y, x_, y_, task=1, device="gpu", loss="nll"):
        self.train()
        self.optimizer.zero_grad()

        # print(y.min(), y.max(), task)

        if args.experiment=='cifar100':
            loss_cur, precision = self.forward_cifar100(args, x, y, x_, y_, task)
        else:
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
                

            batch_size, c, w, h = x.shape

            
            single_neg = True
            if self.args.experiment=='splitMNISToneclass':
                joint_targets = torch.cat([y[:,None], (torch.ones([batch_size,1])*99).to(device)], dim=-1)
                joint_targets = joint_targets.to(device).long()
            else:
                if single_neg:
                    joint_targets = torch.LongTensor(batch_size, 2).to(device)
                    for i in range(batch_size):
                        while True:
                            neg_sample = random.choice(cur_classes)
                            if not neg_sample == y[i]:
                                break
                        joint_targets[i] = torch.tensor([y[i], neg_sample]).to(device)
                else:
                    joint_targets = torch.tensor(np.array(cur_classes)).view(1, -1).expand(batch_size, len(cur_classes))
                    joint_targets = joint_targets.to(device).long()



    
            if self.args.task_info_input:
                task_id = (torch.ones([batch_size])*(task-1)).long().to(device)
                energy = self(x, joint_targets, task_id) # [128, 4]
            else:
                energy = self(x, joint_targets) # [128, 4]


            ## compute loss
            if single_neg:
                energy_pos = energy[:, 0].view(batch_size, -1)
                energy_neg = energy[:, 1].view(batch_size, -1)
            else:
                y_tem = torch.tensor([cur_classes.index(tem) for tem in y]).long().to(device)
                y_tem = y_tem.view(batch_size, 1)
                energy_pos = energy.gather(dim=1, index=y_tem)


            if loss == "nll":
                predL = nll(energy, batch_size)
            elif loss == "mee":
                predL = mee(energy, batch_size)
            elif loss == "square_exponential":
                predL = square_exponential(energy, batch_size)
            elif loss == "square_square":
                predL = square_square(energy, batch_size)
            elif loss == "log_loss":
                predL = log_loss(energy, batch_size)
            elif loss == "hinge_loss":
                predL = hinge_loss(energy, batch_size)
            else:
                raise Exception("loss fx out of choice.")

            L2_loss = energy_pos.pow(2).mean()
            loss_cur = predL

            ## compuate accuracy
            if single_neg:
                _, precision = torch.min(energy, 1)
                precision = 1.* (precision == torch.zeros(batch_size).long().to(device)).sum() / x.size(0)
            else:
                _, precision = torch.min(energy, 1)
                precision = 1.* (precision == y_tem.view(-1)).sum() / x.size(0)


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



