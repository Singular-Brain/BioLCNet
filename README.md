# Paper 
### BioLCNet: Reward-modulated Locally Connected Spiking Neural Networks

Hafez Ghaemi, Erfan Mirzaei, Mahbod Nouri, Saeed Reza Kheradpisheh

arXiv: https://https://arxiv.org/abs/2109.05539

# Requirements 

To install bindsnet you should only use the following command:
```
!pip install -q git+https://github.com/bindsnet/bindsnet
``` 
This will downgrade some of your installed packages includin PyTorch.

# Main Experiments 

### Feature Extraction 
In this part, we trained our hidden layer to extract features from the
MNIST images. We use these features as pre-trained weights in the classification task.

### Image Classification 
After transfering the weights from the pretrained features from the previous section, we 
train the network to classify the MNIST dataset images. You can also run "main.ipynb" for this experiment.
 
### Classical(Pavlovian) Conditioning
In this experiment, we present the network with images belonging
to one arbitrary class of the MNIST dataset as the neutral stimuli, and 
give reward such that it becomes conditioned to the desired response during
each task. The purpose of this experiment is to show the effectiveness
of our rewarding mechanism. 
