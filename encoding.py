import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
import snntorch.spikeplot as splt
import torch.backends.cudnn as cudnn
import torchvision
from snntorch import spikegen

import utils.checkpoint
from data.mnist_datasets import MNISTDataset, SequentialMNISTDataset, HeteroAssociativeMNISTDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from layers.encoding import EncodingLayer
from models.network_models import MNISTOneShot, BackUp
from models.neuron_models import IafPscDelta
from utils.utils import salt_pepper_noise, apply_mask
# from models.protonet_models import SpikingProtoNet
from models.spiking_model import SpikingProtoNet

image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

test_set = SequentialMNISTDataset(root='/usr/common/datasets/MNIST', train=False, classes=3,
                                      dataset_size=1000, image_transform=image_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=1, prefetch_factor=2)

direct_encoding = EncodingLayer(1, 784, False, False, 50,
                                IafPscDelta(thr=0.05, refractory_time_steps=3,
                                            tau_mem=20.0, spike_function=SpikeFunction))
for i, sample in enumerate(test_loader):
    image_sequence, labels, image_query, targets = sample
    break

spike_data = spikegen.latency(image_query, num_steps=50, tau=2, threshold=0.05,
                              normalize=True, linear=True, clip=True)

encoded_query, _ = direct_encoding(torch.flatten(image_query, -2, -1).unsqueeze(2)) # [batch_size, time_steps, height*width]
encoded_query_array = encoded_query.clone().detach().numpy()
encoded_query = encoded_query.squeeze(0)

query_image = image_query[0].numpy()
fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
ax.imshow(np.transpose(query_image, (1, 2, 0)), aspect='equal', cmap='gray', vmin=0, vmax=1)
ax.set_axis_off()
plt.tight_layout()

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(encoded_query, ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(50, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()


encoded_data = spike_data[:, 0].view(50, -1).detach().numpy()

fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
ax.set_ylabel('Neuron Index')
ax.set_xlabel('Time Step')
ax.pcolormesh(encoded_data.T, cmap='binary')
plt.tight_layout()
