"""Reading layer"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional

from models.neuron_models import NeuronModel

# 读取记忆，根据查询信息从存储的记忆中检索信息
class ReadingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, feedback_delay: int,
                 dynamics: NeuronModel, learn_feedback_delay: bool = False) -> None:
        super().__init__()
        self.input_size = input_size            # 输入数据的维度
        self.hidden_size = hidden_size          # 隐藏层的维度，也就是输出层的维度memory_size
        self.feedback_delay = feedback_delay    # 反馈延迟，在检索时从value层到key层的反馈连接
        self.dynamics = dynamics

        self.W = torch.nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: torch.Tensor, states: Optional[Tuple[List[torch.Tensor],
                List[torch.Tensor], torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()
        # 输入x的形状为：(batch_size, sequence_length, input_size)
        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_buffer = torch.zeros(batch_size, self.feedback_delay, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, val_buffer = states

        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):
            # Compute current from input and previous value to key-layer
            # 计算当前输入数据和前一个时间步到达key层的电流
            i = torch.nn.functional.linear(torch.cat([x.select(1, t),
                                                      val_buffer.select(1, t % self.feedback_delay)], dim=-1), self.W)

            # Key-layer
            key, key_states = self.dynamics(i, key_states)

            # Current from key-layer to value-layer ('bij,bj->bi', mem, key)
            # key层神经元到value层神经元的电流
            ikv_t = (key.unsqueeze(1) * mem).sum(2)

            # Value-layer
            val, val_states = self.dynamics(ikv_t, val_states)

            # Update value buffer
            # 更新反馈缓冲区：反馈缓冲区 val_buffer，用于存储过去的 Value 层输出。这个缓冲区在每个时间步都会更新。
            val_buffer[:, t % self.feedback_delay, :] = val

            key_output_sequence.append(key)
            val_output_sequence.append(val)

        states = [key_states, val_states, val_buffer]
        # 返回key层和value层的输出序列
        return torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))


class ReadingLayerReLU(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.size()

        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):

            i = torch.nn.functional.linear(x.select(1, t), self.W)

            key = torch.nn.functional.relu(i)

            val = (key.unsqueeze(1) * mem).sum(2)

            key_output_sequence.append(key)
            val_output_sequence.append(val)

        return torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
