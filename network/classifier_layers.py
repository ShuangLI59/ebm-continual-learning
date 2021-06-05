from torch import nn
import numpy as np
import utils
import pdb
import torch
from torch.nn import functional as F

class CLS_MLP(nn.Module):
    
    def __init__(self, args, num_classes, input_size=1000, hid_size=1000):
        super().__init__()

        self.args = args
        self.num_classes = num_classes
        self.hid_size = hid_size
        
        self.x_fc1 = nn.Linear(input_size, hid_size)
        self.x_fc2 = nn.Linear(hid_size, hid_size)
        self.classifier = nn.Linear(hid_size, self.num_classes)


    def forward(self, x):
        x = self.x_fc1(x)
        x = F.relu(x)
        x = self.x_fc2(x)
        x = F.relu(x)
        x = self.classifier(x)

        return x



class CLS_CONV(nn.Module):
    def __init__(self, args, num_classes, input_size=1000, hid_size=1000):
        super().__init__()
        
        self.args = args
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # nn.Dropout(p=0.25),
            )


        # self.hid_size = 512
        self.hid_size = 1024
        self.fc1 = nn.Linear(2304, self.hid_size)
        # self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)
        self.classifier = nn.Linear(self.hid_size, num_classes)



    def forward(self, x):

        x = self.model(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.classifier(x)

        return x




from network.utils import modules
from network.fc.layers import fc_layer, fc_layer_fixed_gates
class CLS_net_cifar100(nn.Module):
    '''Module for a multi-layer perceptron (MLP). Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None, size_per_layer=None,
                 drop=0, batch_norm=True, nl="relu", bias=True, excitability=False, excit_buffer=False, gated=False,
                 output='normal'):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gated]            <bool>; if True, each linear layer has an additional learnable gate
                                    (whereby the gate is controlled by the same input as that goes through the gate)
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity'''

        super().__init__()
        self.output = output

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if (hid_smooth is not None):
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers-1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size] if layers>0 else [input_size]
        self.layers = len(size_per_layer)-1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{gate}".format(
            drop="" if drop==0 else "d{}".format(drop),
            bias="" if bias else "n", exc="e" if excitability else "", bn="b" if batch_norm else "",
            nl="l" if nl=="leakyrelu" else "", gate="g" if gated else "",
        )
        nd_label = "{}{}".format("" if nd_label=="" else "-{}".format(nd_label),
                                 "" if output=="normal" else "-{}".format(output))
        # -set label
        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement=="" else "x", i)
        self.label = "F{}{}".format(size_statement, nd_label) if self.layers>0 else ""

        # set layers
        for lay_id in range(1, self.layers+1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id-1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            layer = fc_layer(
                in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer,
                batch_norm=False if (lay_id==self.layers and not output=="normal") else batch_norm, gated=gated,
                nl=("none" if output=="none" else nn.Sigmoid()) if (
                    lay_id==self.layers and not output=="normal"
                ) else nl, drop=drop if lay_id>1 else 0.,
            )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers<1:
            self.noLayers = modules.Identity()

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False, **kwargs):
        # Initiate <list> for keeping track of intermediate hidden-(pre)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all fc-layers
        for lay_id in range(skip_first+1, self.layers+1-skip_last):
            (x, pre_act) = getattr(self, 'fcLayer{}'.format(lay_id))(x, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)  #-> for each layer, store pre-activations
                if lay_id<(self.layers-skip_last):
                    hidden_act_list.append(x) #-> for all but last layer, store hidden activations
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x


    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.layers+1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list



    