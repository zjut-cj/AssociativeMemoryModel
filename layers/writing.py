"""Writing layer"""

import math
from typing import Optional, Tuple, Callable, List

import torch
import torch.nn.functional

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel

# 记忆层的key-value层
class WritingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, plasticity_rule: Callable, tau_trace: float,
                 dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size                # key层的数据维度
        self.hidden_size = hidden_size              # value层的数据维度
        self.plasticity_rule = plasticity_rule      # 可塑性规则
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和key层之间的权重以及输入和value层的输入
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size + hidden_size, input_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, states: Optional[Tuple[
            List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]] = None) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # 输入x的形状：(batch_size, sequence_length, input_size)，如（128，1000，160）
        batch_size, sequence_length, _ = x.size()

        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            key_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            val_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, key_trace, val_trace = states

        if mem is None:
            mem = torch.zeros(batch_size, self.hidden_size, self.hidden_size, dtype=x.dtype, device=x.device)

        # 得到key层和value层的输入，(batch_size, sequence_length, 2*hidden_size)
        i = torch.nn.functional.linear(x, self.W)
        # 分离key层和value层的输入
        ik, iv = i.chunk(2, dim=2)

        # 对每个时间步计算key层和value层的输出
        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):

            # Key-layer
            key, key_states = self.dynamics(ik.select(1, t), key_states)

            # Current from key-layer to value-layer ('bij,bj->bi', mem, key)
            # key层到value层的输入电流
            ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)

            # Value-layer
            val, val_states = self.dynamics(iv.select(1, t) + ikv_t, val_states)

            # Update traces 更新key层和value层的迹
            key_trace = exp_convolve(key, key_trace, self.decay_trace)
            val_trace = exp_convolve(val, val_trace, self.decay_trace)

            # Update memory 并根据可塑性规则更新记忆 mem
            # mem表示key层神经元到value层神经元之间的突触权重
            delta_mem = self.plasticity_rule(key_trace, val_trace, mem)
            mem = mem + delta_mem

            key_output_sequence.append(key)
            val_output_sequence.append(val)

        # 最终的神经元的膜电位和迹
        states = [key_states, val_states, key_trace, val_trace]

        # 返回key-value突触权重和两层神经元的输出
        return mem, torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))


class WritingLayerReLU(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, plasticity_rule: Callable) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.plasticity_rule = plasticity_rule

        self.W = torch.nn.Parameter(torch.Tensor(hidden_size + hidden_size, input_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[torch.Tensor] = None) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            mem = torch.zeros(batch_size, self.hidden_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            mem = states

        i = torch.nn.functional.linear(x, self.W)
        ik, iv = i.chunk(2, dim=2)

        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):

            # Key-layer
            key = torch.nn.functional.relu(ik.select(1, t))

            # Value-layer
            val = torch.nn.functional.relu(iv.select(1, t))

            # Update memory
            delta_mem = self.plasticity_rule(key, val, mem)
            mem = mem + delta_mem

            key_output_sequence.append(key)
            val_output_sequence.append(val)

        return mem, torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
