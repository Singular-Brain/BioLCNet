# Paper 
### BioLCNet: Reward-modulated Locally Connected Spiking Neural Networks

Hafez Ghaemi, Erfan Mirzaei, Mahbod Nouri, Saeed Reza Kheradpisheh

arXiv: https:// 

# Requirements 
Python 3.7 , PyTorch == 1.8, Bindsnet

Note: Older versions of PyTorch will not work because we use 'container_abcs' from 'torch._six',
 which isn't compatible with 1.9.

To install bindsnet you should use: 
```
!pip install -q git+https://github.com/bindsnet/bindsnet
``` 

# Main Experiments 

### Feature Extraction 
In this part, we trained our hidden layer to extract features from the
MNISt Images, and then use these features as a fixed pretraining in the network. 

### Image Classification 
After transfer weights from pretrained features from the previous section, we 
train the network to classify the MNIST images. 
 
### Classical(Pavlovian) Conditioning
In this experiment, we present the network with images belonging
to one class of the MNIST dataset as the neutral stimuli, and 
give reward such that it conditioned to the desired response at
each task. The purpose of this experiment is to show the effectiveness
of our rewarding mechanism. 
