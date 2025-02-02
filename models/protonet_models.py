"""Spiking Prototypical network"""

from collections import OrderedDict
from typing import List

import torch.nn.functional
from torch.nn import Parameter

from layers.convolution import *
from layers.dense import DenseLayer
from layers.encoding import EncodingLayer
from models.neuron_models import NeuronModel, IafPscDelta


# 脉冲卷积块：[卷积层，最大池化层]，使用IF神经元
def spiking_conv_block(in_channels: int, out_channels: int, use_bias: bool) -> torch.nn.Module:
    return torch.nn.Sequential(
        Conv2DLayer(in_channels,
                    out_channels,
                    k_size=3,
                    padding=1,
                    stride=1,
                    use_bias=use_bias,
                    dynamics=NonLeakyIafPscDelta(thr=0.1,
                                                 perfect_reset=False,
                                                 refractory_time_steps=3,
                                                 spike_function=SpikeFunction,
                                                 dampening_factor=1.)),
        MaxPool2DLayer(k_size=2, stride=2, padding=0)
    )


# 卷积块：[卷积层，BN层，激活函数，最大池化层]
def conv_block(in_channels: int, out_channels: int, use_batch_norm: bool = True) -> torch.nn.Module:
    if use_batch_norm:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )


# SpikingCNN编码层
class SpikingProtoNet(torch.nn.Module):

    def __init__(self, dynamics: NeuronModel, weight_dict: dict = None, num_time_steps: int = 100,
                 refractory_time_steps: int = 0, input_depth: int = 1, input_size: int = 784, hidden_size: int = 64,
                 output_size: int = 64, use_bias: bool = False) -> None:
        super().__init__()
        self.input_depth = input_depth
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.refractory_time_steps = refractory_time_steps
        self.layer_v_th = [0., 0., 0., 0.]
        self.aux_encoding = EncodingLayer(1, input_size, False, False, num_time_steps, dynamics)

        self.encoder = torch.nn.Sequential(
            spiking_conv_block(input_depth, hidden_size, use_bias),
            spiking_conv_block(hidden_size, hidden_size, use_bias),
            spiking_conv_block(hidden_size, hidden_size, use_bias),
            spiking_conv_block(hidden_size, output_size, use_bias),
        )

        if weight_dict:
            new_state_dict = OrderedDict()
            for k, v in weight_dict.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]  # remove `module.`
                new_state_dict[k] = v
            for i in range(len(self.encoder)):
                self.encoder[i][0].conv2d.weight = Parameter(new_state_dict['encoder.' + str(i) + '.0.weight'])
                self.encoder[i][0].conv2d.weight.required_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size()
        h = 30 if self.input_size == 600 else 28
        w = 20 if self.input_size == 600 else 28
        x = torch.reshape(x, [batch_size, sequence_length, self.input_depth, h, w])
        x_array = x.clone().detach().to('cpu').numpy()
        x = self.encoder(x)
        x1 = x.clone()
        x1 = x1.view(batch_size, sequence_length, -1)
        return torch.reshape(x, [batch_size, sequence_length, self.output_size])

    # 阈值平衡算法
    def threshold_balancing(self, assign_v_th: List = None, aux_train_image: torch.Tensor = None) -> None:
        if assign_v_th is not None:
            self.layer_v_th = assign_v_th

            for i in range(len(self.encoder)):
                self.encoder[i][0].dynamics.thr = self.layer_v_th[i]
                self.encoder[i][0].dynamics.refractory_time_steps = self.refractory_time_steps
        else:
            if aux_train_image is None:
                raise ValueError('Need some aux_train_image to calculate layer-wise firing thresholds.')
            else:
                max_activation = {}

                def get_activation(name):
                    def hook(model, input, output):  # noqa
                        max_activation[name] = output[1].detach()
                    return hook

                for i in range(len(self.encoder)):
                    self.encoder[i][0].register_forward_hook(get_activation('conv' + str(i)))

                aux_x, _ = self.aux_encoding(torch.flatten(aux_train_image, -2, -1).unsqueeze(2))
                for i in range(len(self.encoder)):
                    _ = self.forward(aux_x)
                    v_th = max_activation['conv' + str(i)]
                    self.layer_v_th[i] = v_th if self.layer_v_th[i] < v_th else self.layer_v_th[i]
                    self.encoder[i][0].dynamics.thr = self.layer_v_th[i]
                    self.encoder[i][0].dynamics.refractory_time_steps = self.refractory_time_steps


# CNN编码层：四个卷积块
class ProtoNet(torch.nn.Module):

    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 64,
                 use_batch_norm: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = torch.nn.Sequential(
            conv_block(input_size, hidden_size, use_batch_norm),
            conv_block(hidden_size, hidden_size, use_batch_norm),
            conv_block(hidden_size, hidden_size, use_batch_norm),
            conv_block(hidden_size, output_size, use_batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x.view(x.size(0), -1)
