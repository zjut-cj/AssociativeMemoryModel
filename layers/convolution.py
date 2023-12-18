"""Spiking 2D convolution and pooling layers"""

import math
from typing import Tuple, Optional, List

import torch
import torch.nn.functional
from torch import nn

from functions.autograd_functions import SpikeFunction
from models.neuron_models import NeuronModel, NonLeakyIafPscDelta


class Conv2DLayer(torch.nn.Module):

    def __init__(self, fan_in: int, fan_out: int, k_size: int, padding: int, stride: int,
                 dynamics: NeuronModel, use_bias: bool = False) -> None:
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.k_size = k_size
        self.padding = padding
        self.stride = stride
        self.conv2d = torch.nn.Conv2d(fan_in, fan_out, (k_size, k_size), stride=(stride, stride),
                                      padding=(padding, padding), bias=use_bias)
        self.dynamics = dynamics
        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
        batch_size, sequence_length, c, h, w = x.size()
        new_h = int((h - self.k_size + 2 * self.padding) / self.stride + 1)
        new_w = int((w - self.k_size + 2 * self.padding) / self.stride + 1)
        hidden_size = self.fan_out * new_h * new_w
        assert self.fan_in == c

        if states is None:
            states = self.dynamics.initial_states(batch_size, hidden_size, x.dtype, x.device)

        output_sequence, max_activation = [], [-float('inf')]
        for t in range(sequence_length):
            output = torch.flatten(self.conv2d(x.select(1, t)), -3, -1)
            max_activation.append(torch.max(output))
            output, states = self.dynamics(output, states)
            output_sequence.append(output)

        output = torch.reshape(torch.stack(output_sequence, dim=1),
                               [batch_size, sequence_length, self.fan_out, new_h, new_w])

        return output, max(max_activation)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.conv2d.weight, gain=math.sqrt(2))


class MaxPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.k_size, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0] if isinstance(x, Tuple) else x
        batch_size, sequence_length, c, h, w = x.size()

        output_sequence = []
        for t in range(sequence_length):
            output_sequence.append(self.max_pool(x.select(1, t)))

        return torch.stack(output_sequence, dim=1)


class AvgPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.k_size, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0] if isinstance(x, Tuple) else x
        batch_size, sequence_length, c, h, w = x.size()

        output_sequence = []
        for t in range(sequence_length):
            output_sequence.append(self.avg_pool(x.select(1, t)))

        return torch.stack(output_sequence, dim=1)


# class TextConv2DLayer(torch.nn.Module):
#
#     def __init__(self, fan_in: int, fan_out: int, k_size: List[int], hidden_dim: int,
#                  dynamics: NeuronModel, use_bias: bool = False) -> None:
#         super().__init__()
#         self.fan_in = fan_in
#         self.fan_out = fan_out
#         self.k_size = k_size
#         self.hidden_dim = hidden_dim
#         self.conv2d_0 = torch.nn.Conv2d(fan_in, fan_out, (k_size[0], hidden_dim), bias=use_bias)
#         self.conv2d_1 = torch.nn.Conv2d(fan_in, fan_out, (k_size[1], hidden_dim), bias=use_bias)
#         self.conv2d_2 = torch.nn.Conv2d(fan_in, fan_out, (k_size[2], hidden_dim), bias=use_bias)
#         self.dynamics = dynamics
#         self.reset_parameters()
#
#     def forward(self, x: torch.Tensor, states_0: Optional[Tuple[torch.Tensor, ...]] = None,
#                 states_1: Optional[Tuple[torch.Tensor, ...]] = None,
#                 states_2: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
#         x = x.unsqueeze(1)
#         batch_size, channel, sentence_length, embedding_length = x.size()
#         hidden_size_0 = self.fan_out * (sentence_length - self.k_size[0] + 1)
#         hidden_size_1 = self.fan_out * (sentence_length - self.k_size[1] + 1)
#         hidden_size_2 = self.fan_out * (sentence_length - self.k_size[2] + 1)
#         assert self.fan_in == channel
#
#         if states_0 is None:
#             states_0 = self.dynamics.initial_states(batch_size, hidden_size_0, x.dtype, x.device)
#         if states_1 is None:
#             states_1 = self.dynamics.initial_states(batch_size, hidden_size_1, x.dtype, x.device)
#         if states_2 is None:
#             states_2 = self.dynamics.initial_states(batch_size, hidden_size_2, x.dtype, x.device)
#
#         output_sequence_0 = []
#         for t in range(sentence_length):
#             outputs = torch.flatten(self.conv2d_0(x), -3, -1)
#             output, states_0 = self.dynamics(outputs, states_0)
#             output_sequence_0.append(output)
#         output_sequence_1 = []
#         for t in range(sentence_length):
#             outputs = torch.flatten(self.conv2d_1(x), -3, -1)
#             output, states_1 = self.dynamics(outputs, states_1)
#             output_sequence_1.append(output)
#         output_sequence_2 = []
#         for t in range(sentence_length):
#             outputs = torch.flatten(self.conv2d_2(x), -3, -1)
#             output, states_2 = self.dynamics(outputs, states_2)
#             output_sequence_2.append(output)
#
#         output_0 = torch.reshape(torch.stack(output_sequence_0, dim=1), [batch_size, sentence_length, self.fan_out,
#                                                                          sentence_length - self.k_size[0] + 1, 1])
#         output_1 = torch.reshape(torch.stack(output_sequence_1, dim=1), [batch_size, sentence_length, self.fan_out,
#                                                                          sentence_length - self.k_size[1] + 1, 1])
#         output_2 = torch.reshape(torch.stack(output_sequence_2, dim=1), [batch_size, sentence_length, self.fan_out,
#                                                                          sentence_length - self.k_size[2] + 1, 1])
#         return output_0, output_1, output_2
#
#     def reset_parameters(self) -> None:
#         torch.nn.init.xavier_uniform_(self.conv2d_0.weight, gain=math.sqrt(2))
#         torch.nn.init.xavier_uniform_(self.conv2d_1.weight, gain=math.sqrt(2))
#         torch.nn.init.xavier_uniform_(self.conv2d_2.weight, gain=math.sqrt(2))
#
#
# class TextMaxPool2DLayer(torch.nn.Module):
#
#     def __init__(self, k_size: List[int], sentence_length: int) -> None:
#         super().__init__()
#         self.k_size = k_size
#         self.sentence_length = sentence_length
#         self.max_pool_0 = torch.nn.MaxPool2d((sentence_length - k_size[0] + 1, 1))
#         self.max_pool_1 = torch.nn.MaxPool2d((sentence_length - k_size[1] + 1, 1))
#         self.max_pool_2 = torch.nn.MaxPool2d((sentence_length - k_size[2] + 1, 1))
#
#     def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
#         x_0, x_1, x_2 = x
#         x_0 = x_0[0] if isinstance(x_0, Tuple) else x_0
#         batch_size, sequence_length, c, h, w = x_0.size()
#
#         output_sequence_0 = []
#         for t in range(sequence_length):
#             output_sequence_0.append(self.max_pool_0(x_0.select(1, t)))
#
#         x_1 = x_1[0] if isinstance(x_1, Tuple) else x_1
#         batch_size, sequence_length, c, h, w = x_1.size()
#
#         output_sequence_1 = []
#         for t in range(sequence_length):
#             output_sequence_1.append(self.max_pool_1(x_1.select(1, t)))
#
#         x_2 = x_2[0] if isinstance(x_2, Tuple) else x_2
#         batch_size, sequence_length, c, h, w = x_2.size()
#
#         output_sequence_2 = []
#         for t in range(sequence_length):
#             output_sequence_2.append(self.max_pool_2(x_2.select(1, t)))
#
#         output_0 = torch.stack(output_sequence_0, dim=1)
#         output_1 = torch.stack(output_sequence_1, dim=1)
#         output_2 = torch.stack(output_sequence_2, dim=1)
#
#         return torch.cat([output_0, output_1, output_2], dim=2)
#
#
# class TextAvgPool2DLayer(torch.nn.Module):
#
#     def __init__(self, k_size: List[int], sentence_length: int) -> None:
#         super().__init__()
#         self.k_size = k_size
#         self.avg_pool_0 = torch.nn.AvgPool2d((sentence_length - k_size[0] + 1, 1))
#         self.avg_pool_1 = torch.nn.AvgPool2d((sentence_length - k_size[1] + 1, 1))
#         self.avg_pool_2 = torch.nn.AvgPool2d((sentence_length - k_size[2] + 1, 1))
#
#     def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
#         x_0, x_1, x_2 = x
#         x_0 = x_0[0] if isinstance(x_0, Tuple) else x_0
#         batch_size, sequence_length, c, h, w = x_0.size()
#
#         output_sequence_0 = []
#         for t in range(sequence_length):
#             output_sequence_0.append(self.avg_pool_0(x_0.select(1, t)))
#
#         x_1 = x_1[0] if isinstance(x_1, Tuple) else x_1
#         batch_size, sequence_length, c, h, w = x_1.size()
#
#         output_sequence_1 = []
#         for t in range(sequence_length):
#             output_sequence_1.append(self.avg_pool_1(x_1.select(1, t)))
#
#         x_2 = x_2[0] if isinstance(x_2, Tuple) else x_2
#         batch_size, sequence_length, c, h, w = x_2.size()
#
#         output_sequence_2 = []
#         for t in range(sequence_length):
#             output_sequence_2.append(self.avg_pool_2(x_2.select(1, t)))
#
#         output_0 = torch.stack(output_sequence_0, dim=1)
#         output_1 = torch.stack(output_sequence_1, dim=1)
#         output_2 = torch.stack(output_sequence_2, dim=1)
#
#         return torch.cat([output_0, output_1, output_2], dim=2)


class TextConv2DLayer(torch.nn.Module):
    def __init__(self, fan_in: int, fan_out: int, k_size: List[int], hidden_dim: int,
                 dynamics: NeuronModel, use_bias: bool = False) -> None:
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.k_size = k_size
        self.hidden_dim = hidden_dim

        self.conv2d_0 = torch.nn.Conv2d(fan_in, fan_out, (k_size[0], hidden_dim), bias=use_bias)
        self.conv2d_1 = torch.nn.Conv2d(fan_in, fan_out, (k_size[1], hidden_dim), bias=use_bias)
        self.conv2d_2 = torch.nn.Conv2d(fan_in, fan_out, (k_size[2], hidden_dim), bias=use_bias)

        self.dynamics = dynamics
        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
        x = x.unsqueeze(1)
        batch_size, channel, sentence_length, embedding_length = x.size()
        assert self.fan_in == channel

        hidden_sizes = [sentence_length - k + 1 for k in self.k_size]

        if states is None:
            states = [self.dynamics.initial_states(batch_size, self.fan_out * hidden_size, x.dtype, x.device)
                      for hidden_size in hidden_sizes]

        conv2d_layers = [self.conv2d_0, self.conv2d_1, self.conv2d_2]

        output_sequences = []

        for i, conv2d_layer in enumerate(conv2d_layers):
            hidden_size = hidden_sizes[i]
            states_i = states[i]
            output_sequence = []

            for t in range(sentence_length):
                outputs = torch.flatten(conv2d_layer(x), -3, -1)
                output, states_i = self.dynamics(outputs, states_i)
                output_sequence.append(output)

            output_i = torch.reshape(torch.stack(output_sequence, dim=1),
                                     [batch_size, sentence_length, self.fan_out, hidden_size, 1])
            output_sequences.append(output_i)

        return tuple(output_sequences)

    def reset_parameters(self) -> None:
        for conv2d_layer in [self.conv2d_0, self.conv2d_1, self.conv2d_2]:
            torch.nn.init.xavier_uniform_(conv2d_layer.weight, gain=math.sqrt(2))


class TextAvgPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: List[int], sentence_length: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.avg_pool_0 = torch.nn.AvgPool2d((sentence_length - k_size[0] + 1, 1))
        self.avg_pool_1 = torch.nn.AvgPool2d((sentence_length - k_size[1] + 1, 1))
        self.avg_pool_2 = torch.nn.AvgPool2d((sentence_length - k_size[2] + 1, 1))

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x_0, x_1, x_2 = x

        def apply_avg_pool(avg_pool, x):
            batch_size, sequence_length, c, h, w = x.size()
            output_sequence = [avg_pool(x.select(1, t)) for t in range(sequence_length)]
            return torch.stack(output_sequence, dim=1)

        output_0 = apply_avg_pool(self.avg_pool_0, x_0[0] if isinstance(x_0, Tuple) else x_0)
        output_1 = apply_avg_pool(self.avg_pool_1, x_1[0] if isinstance(x_1, Tuple) else x_1)
        output_2 = apply_avg_pool(self.avg_pool_2, x_2[0] if isinstance(x_2, Tuple) else x_2)

        return torch.cat([output_0, output_1, output_2], dim=2)
