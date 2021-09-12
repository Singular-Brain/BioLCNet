import torch
import os
import numpy as np

from biolcnet import BioLCNet
from utils.reward import DynamicDopamineInjection
from utils.dataset import ClassSelector, load_datasets

import matplotlib.pyplot as plt

gpu = True

### For reproducibility
seed = 2045 # The Singularity is Near!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on", device)

n_neurons = 1000
n_classes = 10
neuron_per_class = int(n_neurons/n_classes)


train_hparams = {
    'n_train' : 10000,
    'n_test' : 10000,
    'n_val' : 250,
    'val_interval' : 250,
    'running_window_length': 250,
}

network_hparams = {
    # net structure
    'crop_size': 22,
    'neuron_per_c': neuron_per_class,
    'in_channels':1,
    'n_channels_lc': 100,
    'filter_size': 15,
    'stride': 4,
    'n_neurons' : n_neurons,
    'n_classes': n_classes,
    
    # time & Phase
    'dt' : 1,
    'pre_observation': True,
    'has_decision_period': True,
    'observation_period': 256,
    'decision_period': 256,
    'time': 256*3,
    'online_rewarding': False,

    # Nodes
    'theta_plus': 0.05,
    'tc_theta_decay': 1e6,
    'tc_trace':20,
    'trace_additive' : False,
    
    # Learning
    'nu_LC': (0.0001,0.01),
    'nu_Output':0.1,

    # weights
    'wmin': 0.0,
    'wmax': 1.0,
    
    # Inhibition
    'inh_type_FC': 'between_layers',
    'inh_factor_FC': 100,
    'inh_LC': True,
    'inh_factor_LC': 100,
    
    # Normalization
    'norm_factor_LC': 0.25*15*15,
    
    # clamping
    'clamp_intensity': None,

    # Save
    'save_path': None,  # Specify for saving the model (Especially for pre-training the lc layer)
    'load_path': None,
    'LC_weights_path': None, # Specify for loading the pre-trained lc weights

    # Plot:
    'confusion_matrix' : False,
    'lc_weights_vis': False,
    'out_weights_vis': False,
    'lc_convergence_vis': False,
    'out_convergence_vis': False,
}

reward_hparams= {
    'n_labels': n_classes,
    'neuron_per_class': neuron_per_class,
    'variant': 'scalar',
    'tc_reward':0,
    'dopamine_base': 0.0,
    'reward_base': 1.,
    'punishment_base': 1.,
    'sub_variant': 'RPE',
    'td_nu': 0.0005,  #RPE
    'ema_window': 10, #RPE
    }

# Dataset Hyperparameters
target_classes = None, ##(0,1) for Pavlovian condistioning
if target_classes:
    ## For Pavlovian conditioning
    npz_file = np.load('utils\mnist_mask_5.npz') 
    mask, mask_test = torch.from_numpy(npz_file['arr_0']), torch.from_numpy(npz_file['arr_1'])
    n_classes = len(target_classes)
else:
    mask = None
    mask_test = None
    n_classes = 10

data_hparams = { 
    'intensity': 128,
    'time': 256*3,
    'crop_size': 22,
    'round_input': False,
}

dataloader, val_loader, test_loader = load_datasets(data_hparams, target_classes=target_classes, mask=mask, mask_test=mask_test)

hparams = {**reward_hparams, **network_hparams, **train_hparams, **data_hparams}
net = BioLCNet(**hparams, reward_fn = DynamicDopamineInjection)
net.fit(dataloader = dataloader, val_loader = val_loader, reward_hparams = reward_hparams, **train_hparams)



### Test
data_hparams = { 
    'intensity': 128,
    'time': 256*2,
    'crop_size': 22,
    'round_input': False,
}
network_hparams['time'] = 256*2
net.time = 256*2
_, _, test_loader = load_datasets(data_hparams, target_classes=target_classes, mask=mask, mask_test=mask_test)

net.evaluate(test_loader, train_hparams['n_test'], **reward_hparams)