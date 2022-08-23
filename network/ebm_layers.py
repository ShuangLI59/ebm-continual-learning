from torch import nn
import numpy as np
import utils
import pdb
import torch
from torch.nn import functional as F



class EBM_MLP(nn.Module):
    
    def __init__(self, args, num_classes, input_size=1000, hid_size=1000):
        super().__init__()

        self.args = args
        self.num_classes = num_classes
        self.hid_size = hid_size
       
        self.y_ebm = nn.Embedding(self.num_classes, hid_size)
        
        self.x_fc1 = nn.Linear(input_size, hid_size)
        self.x_fc2 = nn.Linear(hid_size, hid_size)    
        self.classifier = nn.Linear(hid_size, 1)
        

    def forward(self, x, y):
        bs = x.shape[0]
        
        y = self.y_ebm(y)

        x = self.x_fc1(x)
        x = x[:,None,:].expand_as(y)
    
        y = F.normalize(y, p=2, dim=-1)
        z = x * y
        x = x + z
        x = F.relu(x)

        if not self.args.task_boundary:
            x = self.x_fc2(x)
            z = x * y
            x = x + z
            x = F.relu(x)
        
        x = self.classifier(x)
        x = x.view(bs, -1)
        
        return x





class EBM_CONV(nn.Module):
    def __init__(self, args, num_classes, input_size=1000, hid_size=1000):
        super().__init__()

        self.args = args
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.avgpool = nn.MaxPool2d(2, stride=2)
        self.avgpool2 = nn.MaxPool2d(2, stride=2)


        self.hid_size = 1024
        self.fc1 = nn.Linear(2304, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)
        self.classifier = nn.Linear(self.hid_size, 1)
        
        
        self.num_classes = num_classes
        self.y_ebm = nn.Embedding(self.num_classes, self.hid_size)
        

    def forward(self, x, y):
        bs = x.shape[0]

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)

        y = self.y_ebm(y)
        y = F.softmax(y, dim=-1) * y.shape[-1]

        x = self.fc1(x)
        x = x[:,None,:].expand_as(y)
        x = x * y


        x = self.classifier(x)
        x = x.view(bs, -1)
        
        return x



## ------------------------------------------------------------------------------------------------------------------------
## cifar100
## ------------------------------------------------------------------------------------------------------------------------
from network.fc.layers import fc_layer,fc_layer_fixed_gates
class EBM_net_cifar100(nn.Module):
    '''Module for a multi-layer, fully-connected energy-based model (EBM).

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor'''

    def __init__(self, num_classes=10, input_size=1000, output_size=10, layers=2, hid_size=1000, hid_smooth=None,
                 size_per_layer=None,
                 drop=0, batch_norm=True, nl="relu", bias=True, excitability=False, excit_buffer=False, output='normal',
                 fixed_mask=True, mask_prob=0.8, only_first=False, with_skip=False, device="gpu"):
        '''sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [num_classes]      # of classes
        [input_size]       # of inputs
        [output_size]      # of output units
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity
        EBM-related parameters
        [fixed_mask]       <bool>; whether to use fixed masks instead of learnable gates
        [mask_prop]        <float>; probability of each node being gated for particular class (if using `fixed_mask`)
        [only_first]       <bool>; whether learnable gate is only used for first layer (only if not using `fixed_mask`)
                              NOTE: if set to ``False``, all layers must have same number of units!
        [with_skip]        <bool>; whehter there should be a skip-connection around the learnable gate
        '''

        super().__init__()
        self.output = output
        self.fixed_mask = fixed_mask
        self.only_first = only_first
        self.num_classes = num_classes
        self.with_skip = with_skip

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if (hid_smooth is not None):
                    hidden_sizes = [int(x) for x in np.linspace(hid_size, hid_smooth, num=layers - 1)]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]

            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer) - 1
        self.output_size = size_per_layer[-1]

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}".format(
            drop="" if drop == 0 else "d{}".format(drop),
            bias="" if bias else "n", exc="e" if excitability else "", bn="b" if batch_norm else "",
            nl="l" if nl == "leakyrelu" else "",
        )
        nd_label = "{}{}".format("" if nd_label == "" else "-{}".format(nd_label),
                                 "" if output == "normal" else "-{}".format(output))
        # -set label
        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement == "" else "x", i)
        self.label = "EBM{}{}{}{}".format(
            "fm{}".format(mask_prob) if fixed_mask else ("sk" if with_skip else ""),
            "-of" if only_first else "", size_statement, nd_label
        ) if self.layers > 0 else ""

        # set layers
        for lay_id in range(1, self.layers + 1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id - 1]
            out_size = size_per_layer[lay_id]

            # embedding of y
            if not fixed_mask:
                self.goal_ebm = nn.Embedding(self.num_classes, size_per_layer[1])

            # define and set the fully connected layer
            if fixed_mask and (lay_id==1 or not self.only_first):
                layer = fc_layer_fixed_gates(
                    in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop,
                    batch_norm=False if (lay_id == self.layers and not output == "normal") else batch_norm,
                    nl=nn.Sigmoid() if (lay_id == self.layers and not output == "normal") else nl,
                    gate_size=num_classes, gating_prop=mask_prob, device=device
                )
            else:
                layer = fc_layer(
                    in_size, out_size, bias=bias, excitability=excitability, excit_buffer=excit_buffer, drop=drop,
                    batch_norm=False if (lay_id == self.layers and not output == "normal") else batch_norm,
                    nl=nn.Sigmoid() if (lay_id == self.layers and not output == "normal") else nl,
                )
            setattr(self, 'fcLayer{}'.format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers < 1:
            self.noLayers = utils.Identity()



    def forward(self, x, y, **kwargs):
        '''Returns energies for each batch-sample in [x] according to each class in corresponding batch-entry of [y].

        Args:
            x (tensor: [batch]x[input_units])
            y (tensor: [batch]x[classes_to_test])

        Returns:
            features_per_class (tensor: [batch]x[classes_to_test]x[output_units])
        '''

        batch_size = x.shape[0]

        # Reshape `x` to [batch]x[classes_to_test]x[input_units]
        #-> create multiple copies of [x], one for each class to compute energy for
        classes_to_test = y.shape[1]
        x = x[:, None, :].expand([batch_size, classes_to_test, x.shape[-1]])

        # Deal with `y`
        if self.fixed_mask:
            # -reshape `y` to one-hot-tensor of shape [batch]x[classes_to_test]x[classes]
            y_one_hot = torch.zeros(batch_size, classes_to_test, self.num_classes).to(next(self.parameters()).device)
            y_one_hot.scatter_(dim=2, index=y.view(batch_size, classes_to_test, 1), value=1.)
        else:
            # -embed `y` and put through softmax for the learnable gate
            y = self.goal_ebm(y)  # -> shape: [batch]x[classes_to_test]x[units]
            y = F.softmax(y, dim=-1)

        # Sequentially pass [x] through all fc-layers
        for lay_id in range(1, self.layers+1):
            if self.fixed_mask:
                x = getattr(self, 'fcLayer{}'.format(lay_id))(x, gate_input=y_one_hot)
            else:
                x = getattr(self, 'fcLayer{}'.format(lay_id))(x)
                # -apply the learnable gate, if applicable
                if lay_id == 1 or (not self.only_first):
                    if self.with_skip:
                        x = x * y * y.shape[-1] + x
                        x = F.relu(x)
                    else:
                        x = x * y * y.shape[-1]

        return x                 #-> shape: [batch]x[classes_to_test]x[output_units]


    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        for layer_id in range(1, self.layers + 1):
            list += getattr(self, 'fcLayer{}'.format(layer_id)).list_init_layers()
        return list
