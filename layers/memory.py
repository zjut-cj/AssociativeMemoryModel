import math
from typing import Optional, Tuple, Callable, List

import torch
import torch.nn.functional

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


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
        self.feedback_weights = torch.nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, recall=False, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
        # 划分key和value层的输入
        ik, iv = i.chunk(2, dim=2)
        # key和value层的脉冲输出
        key_output_sequence = []
        val_output_sequence = []
        # excitatory_mem = []
        if recall:
            for t in range(sequence_length):
                # key层神经元在t时刻接收到上一个时间步的value层的反馈
                # feedback_weights_copy = self.feedback_weights.clone()
                # feedback_input = 0.2 * torch.nn.functional.linear(val_buffer.select(1, t % self.feedback_delay),
                #                                                   self.feedback_weights)
                # feedback_input = torch.nn.functional.linear(torch.cat([x.select(1, t),
                #                                                       val_buffer.select(1, t % self.feedback_delay)]
                #                                                       , dim=-1), self.feedback_weights)
                # key层神经元在t时刻的脉冲key和当前时刻的神经元状态key_states
                key, key_states = self.dynamics(ik.select(1, t), key_states)
                # key, key_states = self.dynamics(feedback_input, key_states)

                # ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)
                ikv_t = (key.unsqueeze(1) * mem).sum(2)

                # val层神经元在t时刻的脉冲val和当前时刻的神经元状态val_states
                # val, val_states = self.dynamics(0.5 * iv.select(1, t) + ikv_t, val_states)
                val, val_states = self.dynamics(ikv_t, val_states)
                # excitatory_mem.append(ikv_t)

                # 更新key层和value层的迹
                # key_trace = exp_convolve(key, key_trace, self.decay_trace)
                # val_trace = exp_convolve(val, val_trace, self.decay_trace)
                # 更新缓冲区中用于存储过去的 Value 层输出。这个缓冲区在每个时间步都会更新
                val_buffer[:, t % self.feedback_delay, :] = val

                key_output_sequence.append(key)
                val_output_sequence.append(val)
            # excitatory_mem = torch.stack(excitatory_mem, dim=1)
            # excitatory_mem_avg = torch.sum(excitatory_mem, dim=2) / 100.0
            # excitatory_mem_array = excitatory_mem.detach().to('cpu').numpy()
            # excitatory_mem_avg_array = excitatory_mem_avg.detach().to('cpu').numpy()
            #
            # # 创建时间步数组（横坐标）
            # time_steps = range(1, 51)
            # # 计算 excitatory_data 的平均值
            # excitatory_mean = np.mean(excitatory_mem_avg_array[0])
            # # 绘制折线图
            # plt.plot(time_steps, excitatory_mem_avg_array[0], label='Excitatory', color='blue')
            # # 绘制 excitatory_data 平均值的虚线
            # plt.axhline(y=excitatory_mean, linestyle='--', color='blue', label='Excitatory Mean')
            # # 添加图例
            # plt.legend()
            # # 添加坐标轴标题
            # plt.xlabel('Time steps')
            # plt.ylabel('Avg Mem')
            # # 显示图形
            # plt.show()
        else:
            for t in range(sequence_length):
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

                # excitatory_mem.append(val_states[1])

                # 更新key层和value层的迹
                key_trace = exp_convolve(key, key_trace, self.decay_trace)
                val_trace = exp_convolve(val, val_trace, self.decay_trace)
                # 更新缓冲区中用于存储过去的 Value 层输出。这个缓冲区在每个时间步都会更新
                # val_buffer[:, t % self.feedback_delay, :] = val
                # 更新key到value层的连接mem
                delta_mem = self.plasticity_rule(key_trace, val_trace, mem)
                mem = mem + delta_mem

                # if t == 220:
                #     noise = torch.rand_like(mem) * 0.5
                #     mem = mem + noise
                #     mem = torch.clamp(mem, 0, 1)


                key_output_sequence.append(key)
                val_output_sequence.append(val)
            # excitatory_mem = torch.stack(excitatory_mem, dim=1)
            # excitatory_mem_avg = torch.sum(excitatory_mem, dim=2) / 100.0
            # excitatory_mem_array = excitatory_mem.detach().to('cpu').numpy()
            # excitatory_mem_avg_array = excitatory_mem_avg.detach().to('cpu').numpy()
            #
            # # 创建时间步数组（横坐标）
            # time_steps = range(1, 501)
            # # 计算 excitatory_data 的平均值
            # excitatory_mean = np.mean(excitatory_mem_avg_array[0])
            # # 绘制折线图
            # plt.plot(time_steps, excitatory_mem_avg_array[0], label='Excitatory', color='blue')
            # # 绘制 excitatory_data 平均值的虚线
            # plt.axhline(y=excitatory_mean, linestyle='--', color='blue', label='Excitatory Mean')
            # # 添加图例
            # plt.legend()
            # # 添加坐标轴标题
            # plt.xlabel('Time steps')
            # plt.ylabel('Avg Input Current')
            # # 显示图形
            # plt.show()

        states = [key_states, val_states, key_trace, val_trace, val_buffer]

        return mem, torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2.0))
        # torch.nn.init.xavier_uniform_(self.feedback_weights, gain=math.sqrt(2))


class InhibitionMemoryLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, plasticity_rule: Callable, tau_trace: float,
                 feedback_delay: int, inhibition_size: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # key层的数据维度
        self.hidden_size = hidden_size  # value层的数据维度
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.feedback_delay = feedback_delay  # 反馈延迟
        self.inhibition_size = inhibition_size
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和key层之间的权重以及输入和value层的输入
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size + hidden_size, input_size))
        # self.alpha = torch.nn.Parameter(torch.as_tensor(5.0))
        # self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None,
                excitatory_inhibitory: Optional[torch.Tensor] = None,
                inhibitory_excitatory: Optional[torch.Tensor] = None, recall=False, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            inhibitory_states = self.dynamics.initial_states(batch_size, self.inhibition_size, x.dtype, x.device)
            key_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            val_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            inhibitory_trace = torch.zeros(batch_size, self.inhibition_size, dtype=x.dtype, device=x.device)
            inhibitory_buffer = torch.zeros(batch_size, self.feedback_delay, self.inhibition_size,
                                            dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, inhibitory_states, key_trace, val_trace, \
            inhibitory_trace, inhibitory_buffer = states

        if mem is None:
            mem = torch.zeros(batch_size, self.hidden_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if excitatory_inhibitory is None:
            excitatory_inhibitory = torch.rand(batch_size, self.hidden_size, self.inhibition_size,
                                                dtype=x.dtype, device=x.device)
            # excitatory_inhibitory = torch.zeros(batch_size, self.hidden_size, self.inhibition_size,
            #                                     dtype=x.dtype, device=x.device)
            # torch.nn.init.xavier_uniform_(excitatory_inhibitory, gain=math.sqrt(2.0))
            mask = torch.eye(excitatory_inhibitory.size(-1), dtype=torch.bool, device=x.device).unsqueeze(0)
            excitatory_inhibitory_nondiag = excitatory_inhibitory.masked_fill(mask, 0.0)
            excitatory_inhibitory = excitatory_inhibitory - excitatory_inhibitory_nondiag
        if inhibitory_excitatory is None:
            inhibitory_excitatory = torch.zeros(batch_size, self.inhibition_size, self.hidden_size,
                                                dtype=x.dtype, device=x.device)
            # torch.nn.init.xavier_uniform_(inhibitory_excitatory, gain=math.sqrt(2.0))
            # mask = torch.eye(inhibitory_excitatory.size(-1), dtype=torch.bool, device=x.device).unsqueeze(0)
            # inhibitory_excitatory = inhibitory_excitatory.masked_fill(mask, 0.0)

        # 编码后输入到key-value层的输入电流
        i = torch.nn.functional.linear(x, self.W)
        # 划分key和value层的输入
        ik, iv = i.chunk(2, dim=2)
        # key和value层的脉冲输出
        key_output_sequence = []
        val_output_sequence = []
        # excitatory_mem = []
        # inhibitory_mem = []
        # excitatory_inhibitory_array = excitatory_inhibitory.clone().detach().to('cpu').numpy()
        # inhibitory_excitatory_array = inhibitory_excitatory.clone().detach().to('cpu').numpy()
        if recall:
            for t in range(sequence_length):
                excitatory_inhibitory_mask = torch.eye(excitatory_inhibitory.size(-1),
                                                       dtype=torch.bool, device=x.device).unsqueeze(0)
                excitatory_inhibitory_nondiag = excitatory_inhibitory.masked_fill(excitatory_inhibitory_mask, 0.0)
                excitatory_inhibitory = excitatory_inhibitory - excitatory_inhibitory_nondiag

                inhibitory_excitatory_mask = torch.eye(inhibitory_excitatory.size(-1),
                                                       dtype=torch.bool, device=x.device).unsqueeze(0)
                inhibitory_excitatory = inhibitory_excitatory.masked_fill(inhibitory_excitatory_mask, 0.0)

                # key层神经元在t时刻的脉冲key和当前时刻的神经元状态key_states
                key, key_states = self.dynamics(ik.select(1, t), key_states)

                ikv_t = (key.unsqueeze(1) * mem).sum(2)

                inhibitory_output = (inhibitory_buffer.select(1, t % self.feedback_delay).clone().unsqueeze(1)
                                     * inhibitory_excitatory).sum(2)

                val, val_states = self.dynamics(ikv_t - 1.0 * inhibitory_output, val_states)

                inhibitory_input = (val.unsqueeze(1) * excitatory_inhibitory).sum(1)
                inhibitory, inhibitory_states = self.dynamics(inhibitory_input, inhibitory_states)

                # excitatory_mem.append(ikv_t - 1.0 * inhibitory_output)
                # inhibitory_mem.append(inhibitory_input)

                val_trace = exp_convolve(val, val_trace, self.decay_trace)
                inhibitory_trace = exp_convolve(inhibitory, inhibitory_trace, self.decay_trace)
                delta_excitatory_inhibitory = self.plasticity_rule(val_trace, inhibitory_trace, excitatory_inhibitory)
                delta_inhibitory_excitatory = self.plasticity_rule(inhibitory_trace, val_trace, inhibitory_excitatory)
                excitatory_inhibitory = excitatory_inhibitory + delta_excitatory_inhibitory
                inhibitory_excitatory = inhibitory_excitatory + delta_inhibitory_excitatory

                # 更新缓冲区中用于存储过去的 inhibitory 层输出。这个缓冲区在每个时间步都会更新
                inhibitory_buffer[:, t % self.feedback_delay, :] = inhibitory

                key_output_sequence.append(key)
                val_output_sequence.append(val)
            # excitatory_mem = torch.stack(excitatory_mem, dim=1)
            # inhibitory_mem = torch.stack(inhibitory_mem, dim=1)
            # excitatory_mem_avg = torch.sum(excitatory_mem, dim=2) / 100.0
            # inhibitory_mem_avg = torch.sum(inhibitory_mem, dim=2) / 100.0
            # excitatory_mem_array = excitatory_mem.detach().to('cpu').numpy()
            # inhibitory_mem_array = inhibitory_mem.detach().to('cpu').numpy()
            # excitatory_mem_avg_array = excitatory_mem_avg.detach().to('cpu').numpy()
            # inhibitory_mem_avg_array = inhibitory_mem_avg.detach().to('cpu').numpy()
            #
            # # 创建时间步数组（横坐标）
            # time_steps = range(1, 51)
            # # 计算 excitatory_data 的平均值
            # excitatory_mean = np.mean(excitatory_mem_avg_array[0])
            # # 绘制折线图
            # plt.plot(time_steps, excitatory_mem_avg_array[0], label='Excitatory', color='blue')
            # plt.plot(time_steps, inhibitory_mem_avg_array[0], label='Inhibitory', color='red')
            # # 绘制 excitatory_data 平均值的虚线
            # plt.axhline(y=excitatory_mean, linestyle='--', color='blue', label='Excitatory Mean')
            # # 添加图例
            # plt.legend()
            # # 添加坐标轴标题
            # plt.xlabel('Time steps')
            # plt.ylabel('Avg Mem')
            # # 显示图形
            # plt.show()
        else:
            for t in range(sequence_length):
                excitatory_inhibitory_mask = torch.eye(excitatory_inhibitory.size(-1),
                                                       dtype=torch.bool, device=x.device).unsqueeze(0)
                excitatory_inhibitory_nondiag = excitatory_inhibitory.masked_fill(excitatory_inhibitory_mask, 0.0)
                excitatory_inhibitory = excitatory_inhibitory - excitatory_inhibitory_nondiag

                inhibitory_excitatory_mask = torch.eye(inhibitory_excitatory.size(-1),
                                                       dtype=torch.bool, device=x.device).unsqueeze(0)
                inhibitory_excitatory = inhibitory_excitatory.masked_fill(inhibitory_excitatory_mask, 0.0)

                key, key_states = self.dynamics(ik.select(1, t), key_states)

                ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)

                inhibitory_output = (inhibitory_buffer.select(1, t % self.feedback_delay).clone().unsqueeze(1)
                                     * inhibitory_excitatory).sum(2)

                val, val_states = self.dynamics(iv.select(1, t) + ikv_t - 1.0 * inhibitory_output, val_states)

                inhibitory_input = (val.unsqueeze(1) * excitatory_inhibitory).sum(1)
                inhibitory, inhibitory_states = self.dynamics(inhibitory_input, inhibitory_states)
                # 更新缓冲区中用于存储过去的 inhibitory 层输出。这个缓冲区在每个时间步都会更新
                inhibitory_buffer[:, t % self.feedback_delay, :] = inhibitory

                # excitatory_mem.append(val_states[1])
                # inhibitory_mem.append(inhibitory_states[1])

                # 更新key层和value层的迹
                key_trace = exp_convolve(key, key_trace, self.decay_trace)
                val_trace = exp_convolve(val, val_trace, self.decay_trace)
                # 更新缓冲区中用于存储过去的 Value 层输出。这个缓冲区在每个时间步都会更新
                delta_mem = self.plasticity_rule(key_trace, val_trace, mem)
                mem = mem + delta_mem

                # if t == 220:
                #     noise = torch.rand_like(mem) * 0.5
                #     mem = mem + noise
                #     mem = torch.clamp(mem, 0, 1)

                inhibitory_trace = exp_convolve(inhibitory, inhibitory_trace, self.decay_trace)
                delta_excitatory_inhibitory = self.plasticity_rule(val_trace, inhibitory_trace, excitatory_inhibitory)
                delta_inhibitory_excitatory = self.plasticity_rule(inhibitory_trace, val_trace, inhibitory_excitatory)
                excitatory_inhibitory = excitatory_inhibitory + delta_excitatory_inhibitory
                inhibitory_excitatory = inhibitory_excitatory + delta_inhibitory_excitatory

                key_output_sequence.append(key)
                val_output_sequence.append(val)
            # excitatory_mem = torch.stack(excitatory_mem, dim=1)
            # inhibitory_mem = torch.stack(inhibitory_mem, dim=1)
            # excitatory_mem_avg = torch.sum(excitatory_mem, dim=2) / 100.0
            # inhibitory_mem_avg = torch.sum(inhibitory_mem, dim=2) / 100.0
            # excitatory_mem_array = excitatory_mem.detach().to('cpu').numpy()
            # inhibitory_mem_array = inhibitory_mem.detach().to('cpu').numpy()
            # excitatory_mem_avg_array = excitatory_mem_avg.detach().to('cpu').numpy()
            # inhibitory_mem_avg_array = inhibitory_mem_avg.detach().to('cpu').numpy()
            #
            # # 创建时间步数组（横坐标）
            # time_steps = range(1, 501)
            # # 计算 excitatory_data 的平均值
            # excitatory_mean = np.mean(excitatory_mem_avg_array[0])
            # # 绘制折线图
            # plt.plot(time_steps, excitatory_mem_avg_array[0], label='Excitatory', color='blue')
            # plt.plot(time_steps, inhibitory_mem_avg_array[0], label='Inhibitory', color='red')
            # # 绘制 excitatory_data 平均值的虚线
            # plt.axhline(y=excitatory_mean, linestyle='--', color='blue', label='Excitatory Mean')
            # # 添加图例
            # plt.legend()
            # # 添加坐标轴标题
            # plt.xlabel('Time steps')
            # plt.ylabel('Avg Input Current')
            # # 显示图形
            # plt.show()

        states = [key_states, val_states, inhibitory_states, key_trace, val_trace, inhibitory_trace, inhibitory_buffer]
        # excitatory_inhibitory_array = excitatory_inhibitory.clone().detach().to('cpu').numpy()
        # inhibitory_excitatory_array = inhibitory_excitatory.clone().detach().to('cpu').numpy()
        # mem_array = mem.clone().detach().to('cpu').numpy()

        return mem, excitatory_inhibitory, inhibitory_excitatory, torch.stack(key_output_sequence, dim=1), \
               torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2.0))


class DualInhibitionMemoryLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, plasticity_rule: Callable, tau_trace: float,
                 feedback_delay: int, inhibition_size: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size  # key层的数据维度
        self.hidden_size = hidden_size  # value层的数据维度
        self.plasticity_rule = plasticity_rule  # 可塑性规则
        self.feedback_delay = feedback_delay  # 反馈延迟
        self.inhibition_size = inhibition_size
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        # 表示输入和key层之间的权重以及输入和value层的输入
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size + hidden_size, input_size))
        # value层到key层的反馈权重
        # self.feedback_weights = torch.nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        # self.key_excitatory_inhibitory = torch.nn.Parameter(torch.Tensor(inhibition_size, hidden_size))
        # self.key_inhibitory_excitatory = torch.nn.Parameter(torch.Tensor(hidden_size, inhibition_size))
        self.value_excitatory_inhibitory = torch.nn.Parameter(torch.Tensor(inhibition_size, hidden_size))
        self.value_inhibitory_excitatory = torch.nn.Parameter(torch.Tensor(hidden_size, inhibition_size))
        # self.alpha = torch.nn.Parameter(torch.as_tensor(-2.2))
        # self.sigmoid = torch.nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, recall=False, states: Optional[Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            key_inhibitory_states = self.dynamics.initial_states(batch_size, self.inhibition_size, x.dtype, x.device)
            value_inhibitory_states = self.dynamics.initial_states(batch_size, self.inhibition_size, x.dtype, x.device)
            key_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            val_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            value_inhibitory_buffer = torch.zeros(batch_size, self.feedback_delay, self.inhibition_size,
                                            dtype=x.dtype, device=x.device)
            key_inhibitory_buffer = torch.zeros(batch_size, self.feedback_delay, self.inhibition_size,
                                                  dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, key_inhibitory_states, value_inhibitory_states, \
            key_trace, val_trace, value_inhibitory_buffer, key_inhibitory_buffer = states

        if mem is None:
            mem = torch.zeros(batch_size, self.hidden_size, self.hidden_size, dtype=x.dtype, device=x.device)

        # 编码后输入到key-value层的输入电流
        i = torch.nn.functional.linear(x, self.W)
        # 划分key和value层的输入
        ik, iv = i.chunk(2, dim=2)
        # key和value层的脉冲输出
        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):
            if recall:
                # key_inhibitory_output = torch.nn.functional.linear(key_inhibitory_buffer.select
                #                                                    (1, t % self.feedback_delay).clone(),
                #                                                    self.key_inhibitory_excitatory)
                # key_inhibitory_output_array = key_inhibitory_output.clone().detach().to('cpu').numpy()
                # key层神经元在t时刻的脉冲key和当前时刻的神经元状态key_states
                # key, key_states = self.dynamics(ik.select(1, t) - 0.1 * key_inhibitory_output, key_states)
                # key_inhibitory_input = torch.nn.functional.linear(key, self.key_excitatory_inhibitory)
                # key_inhibitory, key_inhibitory_states = self.dynamics(key_inhibitory_input,
                #                                                       key_inhibitory_states)
                # key_inhibitory_array = key_inhibitory.clone().detach().to('cpu').numpy()

                key, key_states = self.dynamics(ik.select(1, t), key_states)

                ikv_t = (key.unsqueeze(1) * mem).sum(2)

                # val层神经元在t时刻的脉冲val和当前时刻的神经元状态val_states
                value_inhibitory_output = torch.nn.functional.linear(value_inhibitory_buffer.select
                                                                     (1, t % self.feedback_delay).clone(),
                                                                     self.value_inhibitory_excitatory)
                # value_inhibitory_output_array = value_inhibitory_output.clone().detach().to('cpu').numpy()
                val, val_states = self.dynamics(ikv_t + value_inhibitory_output, val_states)


                value_inhibitory_input = torch.nn.functional.linear(val, self.value_excitatory_inhibitory)
                value_inhibitory, value_inhibitory_states = self.dynamics(value_inhibitory_input,
                                                                          value_inhibitory_states)
                # value_inhibitory_array = value_inhibitory.clone().detach().to('cpu').numpy()

                # 更新缓冲区中用于存储过去的 inhibitory 层输出。这个缓冲区在每个时间步都会更新
                # key_inhibitory_buffer[:, t % self.feedback_delay, :] = key_inhibitory
                value_inhibitory_buffer[:, t % self.feedback_delay, :] = value_inhibitory

                key_output_sequence.append(key)
                val_output_sequence.append(val)
            else:
                # key_inhibitory_output = torch.nn.functional.linear(key_inhibitory_buffer.select
                #                                                    (1, t % self.feedback_delay).clone(),
                #                                                    self.key_inhibitory_excitatory)
                # # key层神经元在t时刻接收到上一个时间步的value层的反馈
                # key, key_states = self.dynamics(ik.select(1, t) - 0.1 * key_inhibitory_output, key_states)
                # key_inhibitory_input = torch.nn.functional.linear(key, self.key_excitatory_inhibitory)
                # key_inhibitory, key_inhibitory_states = self.dynamics(key_inhibitory_input,
                #                                                       key_inhibitory_states)

                key, key_states = self.dynamics(ik.select(1, t), key_states)

                ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)

                value_inhibitory_output = torch.nn.functional.linear(value_inhibitory_buffer.select
                                                                     (1, t % self.feedback_delay).clone(),
                                                                     self.value_inhibitory_excitatory)
                # val层神经元在t时刻的脉冲val和当前时刻的神经元状态val_states
                val, val_states = self.dynamics(iv.select(1, t) + ikv_t + value_inhibitory_output, val_states)
                value_inhibitory_input = torch.nn.functional.linear(val, self.value_excitatory_inhibitory)
                value_inhibitory, value_inhibitory_states = self.dynamics(value_inhibitory_input,
                                                                          value_inhibitory_states)

                # 更新key层和value层的迹
                key_trace = exp_convolve(key, key_trace, self.decay_trace)
                val_trace = exp_convolve(val, val_trace, self.decay_trace)
                # 更新key到value层的连接mem
                delta_mem = self.plasticity_rule(key_trace, val_trace, mem)
                mem = mem + delta_mem

                # 更新缓冲区中用于存储过去的 inhibitory 层输出。这个缓冲区在每个时间步都会更新
                # key_inhibitory_buffer[:, t % self.feedback_delay, :] = key_inhibitory
                value_inhibitory_buffer[:, t % self.feedback_delay, :] = value_inhibitory

                key_output_sequence.append(key)
                val_output_sequence.append(val)

        states = [key_states, val_states, key_inhibitory_states, value_inhibitory_states,
                  key_trace, val_trace, key_inhibitory_buffer, key_inhibitory_buffer]

        return mem, torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2.0))
        # torch.nn.init.xavier_uniform_(self.key_excitatory_inhibitory, gain=math.sqrt(2))
        # torch.nn.init.xavier_uniform_(self.key_inhibitory_excitatory, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.value_excitatory_inhibitory, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.value_inhibitory_excitatory, gain=math.sqrt(2))

        # torch.nn.init.uniform_(self.key_excitatory_inhibitory, a=0, b=1)
        # torch.nn.init.uniform_(self.key_inhibitory_excitatory, a=0, b=1)
        # torch.nn.init.uniform_(self.value_excitatory_inhibitory, a=0, b=1)
        # torch.nn.init.uniform_(self.value_inhibitory_excitatory, a=0, b=1)
