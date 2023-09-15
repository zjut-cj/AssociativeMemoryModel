import math
from typing import Optional, Tuple, Callable, List

import torch
import torch.nn.functional

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel


# 记忆层的key-value层
class MemoryLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, plasticity_rule: Callable, tau_trace: float,
                 feedback_delay: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # key层的数据维度
        self.hidden_size = hidden_size  # value层的数据维度
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.feedback_delay = feedback_delay  # 反馈延迟
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和key层之间的权重以及输入和value层的输入
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size + hidden_size, input_size))
        # value层到key层的反馈权重
        self.feedback_weights = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, recall=False, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        x_array = x.clone().to('cpu').detach().numpy()
        batch_size, sequence_length, _ = x.size()

        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            key_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            val_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            # key_buffer = torch.zeros(batch_size, self.feedback_delay, self.hidden_size, dtype=x.dtype, device=x.device)
            val_buffer = torch.zeros(batch_size, self.feedback_delay, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, key_trace, val_trace, val_buffer = states

        if mem is None:
            mem = torch.zeros(batch_size, self.hidden_size, self.hidden_size, dtype=x.dtype, device=x.device)

        # 编码后输入到key-value层的输入电流
        i = torch.nn.functional.linear(x, self.W)
        i_array = i.clone().to('cpu').detach().numpy()
        # 划分key和value层的输入
        ik, iv = i.chunk(2, dim=2)
        ik_array = ik.clone().to('cpu').detach().numpy()
        iv_array = iv.clone().to('cpu').detach().numpy()

        # key和value层的脉冲输出
        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):
            if recall:
                with torch.no_grad():
                    # key层神经元在t时刻接收到上一个时间步的value层的反馈
                    # feedback_weights_copy = self.feedback_weights.clone()
                    # feedback_input = torch.nn.functional.linear(val_buffer.select(1, t % self.feedback_delay),
                    #                                             self.feedback_weights)
                    feedback_input = torch.nn.functional.linear(torch.cat([ik.select(1, t),
                                                                          val_buffer.select(1, t % self.feedback_delay)]
                                                                          , dim=-1), self.feedback_weights)
                    # key层神经元在t时刻的脉冲key和当前时刻的神经元状态key_states
                    key, key_states = self.dynamics(feedback_input, key_states)
                    # key, key_states = self.dynamics(ik.select(1, t), key_states)

                    # ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)
                    ikv_t = (key.unsqueeze(1) * mem).sum(2)

                    # val层神经元在t时刻的脉冲val和当前时刻的神经元状态val_states
                    # val, val_states = self.dynamics(0.5 * iv.select(1, t) + ikv_t, val_states)
                    val, val_states = self.dynamics(ikv_t, val_states)

                    # 更新key层和value层的迹
                    key_trace = exp_convolve(key, key_trace, self.decay_trace)
                    val_trace = exp_convolve(val, val_trace, self.decay_trace)
                    # 更新缓冲区中用于存储过去的 Value 层输出。这个缓冲区在每个时间步都会更新
                    val_buffer[:, t % self.feedback_delay, :] = val

                    key_output_sequence.append(key)
                    val_output_sequence.append(val)
            else:
                # key层神经元在t时刻接收到上一个时间步的value层的反馈
                # feedback_weights_copy = self.feedback_weights.clone()
                # feedback_input = torch.nn.functional.linear(val_buffer.select(1, t % self.feedback_delay),
                #                                             self.feedback_weights)

                # key层神经元在t时刻的脉冲key和当前时刻的神经元状态key_states
                # key, key_states = self.dynamics(ik.select(1, t) + feedback_input, key_states)
                key, key_states = self.dynamics(ik.select(1, t), key_states)

                ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)

                # val层神经元在t时刻的脉冲val和当前时刻的神经元状态val_states
                val, val_states = self.dynamics(iv.select(1, t) + ikv_t, val_states)

                # 更新key层和value层的迹
                key_trace = exp_convolve(key, key_trace, self.decay_trace)
                val_trace = exp_convolve(val, val_trace, self.decay_trace)
                # 更新缓冲区中用于存储过去的 Value 层输出。这个缓冲区在每个时间步都会更新
                # val_buffer[:, t % self.feedback_delay, :] = val
                # 更新key到value层的连接mem
                delta_mem = self.plasticity_rule(key_trace, val_trace, mem)
                mem = mem + delta_mem

                key_output_sequence.append(key)
                val_output_sequence.append(val)

        states = [key_states, val_states, key_trace, val_trace, val_buffer]

        return mem, torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.feedback_weights, gain=math.sqrt(2))
