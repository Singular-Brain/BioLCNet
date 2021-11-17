
import torch
from bindsnet.encoding import PoissonEncoder

from bindsnet.datasets import MNIST

import os

from torchvision import transforms



class ClassSelector(torch.utils.data.sampler.Sampler):
    """Select target classes from the dataset"""
    def __init__(self, target_classes, data_source, mask = None):
        if mask is not None:
            self.mask = mask
        else:
            self.mask = torch.tensor([1 if data_source[i]['label'] in target_classes else 0 for i in range(len(data_source))])
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)

    
# Load MNIST data.
def load_datasets(data_hparams, target_classes=None, mask=None, mask_test=None):
    dataset = MNIST(
        PoissonEncoder(time=data_hparams['time'], dt=1),
        None,
        root=os.path.join("MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * data_hparams['intensity'])),
            transforms.CenterCrop(data_hparams['crop_size'])]
        ),
    )

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            sampler = ClassSelector(
                                                    target_classes = target_classes,
                                                    data_source = dataset,
                                                    mask = mask,
                                                    ) if target_classes else None
                                            )

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            sampler = ClassSelector(
                                                    target_classes = target_classes,
                                                    data_source = dataset,
                                                    mask = mask_test,
                                                    ) if target_classes else None
                                            )

    # Load test dataset
    test_dataset = MNIST(   
        PoissonEncoder(time=data_hparams['time'], dt=1),
        None,
        root=os.path.join("MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * data_hparams['intensity'])),
            transforms.CenterCrop(data_hparams['crop_size'])]
        ),
    )


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                        sampler = ClassSelector(
                                                target_classes = target_classes,
                                                data_source = dataset,
                                                mask = mask_test,
                                                ) if target_classes else None
                                        )

    return dataloader, val_loader, test_loader