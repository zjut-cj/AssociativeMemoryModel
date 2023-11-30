"""Network models"""

import math
from typing import Tuple, Callable, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional

from layers.dense import DenseLayer, AttentionDenseLayer
from layers.embedding import EmbeddingLayer
from layers.encoding import EncodingLayer
from layers.attention import AttentionLayer, SpatioAttentionLayer
from layers.reading import ReadingLayer, ReadingLayerReLU
from layers.writing import WritingLayer, WritingLayerReLU
from layers.memory import MemoryLayer, InhibitionMemoryLayer, DualInhibitionMemoryLayer
from models.neuron_models import NeuronModel
from models.protonet_models import SpikingProtoNet, ProtoNet
from policies import policy


class BackUp(torch.nn.Module):
    def __init__(self, output_size: int, memory_size: int, num_time_steps: int, readout_delay: int, tau_trace: float,
                 image_embedding_layer: torch.nn.Module, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        spiking_image_size = image_embedding_layer.input_size
        image_feature_size = image_embedding_layer.output_size
        writing_layer_input_size = image_feature_size + image_feature_size

        # encoding layer输出图像的脉冲序列形状为[1, 100, 784]
        # self.image_encoding_layer = EncodingLayer(1, spiking_image_size, False, False, num_time_steps, dynamics)
        # embedding layer输出图像的嵌入编码的形状为[1, 100, 64]
        self.image_embedding_layer = image_embedding_layer
        # self.memory_layer = MemoryLayer(writing_layer_input_size, memory_size, plasticity_rule,
        #                                 tau_trace, readout_delay, dynamics)
        self.memory_layer = DualInhibitionMemoryLayer(writing_layer_input_size, memory_size, plasticity_rule,
                                                      tau_trace, readout_delay, memory_size, dynamics)
        self.decoder_l1 = DenseLayer(memory_size, 256, dynamics)
        self.decoder_l2 = DenseLayer(256, output_size, dynamics)

    def forward(self, images: torch.Tensor, query: torch.Tensor) -> Tuple:
        batch_size, sequence_length, *CHW = images.size()

        images_encoded_sequence = []
        # images_spiking_sequence = []
        for t in range(sequence_length):
            # images_spiking, _ = self.image_encoding_layer(torch.flatten(images.select(1, t), -2, -1).unsqueeze(2))
            # images_embedded = self.image_embedding_layer(images_spiking)

            images_embedded = self.image_embedding_layer(images.select(1, t))

            images_encoded_sequence.append(images_embedded)

            # images_spiking_sequence.append(images_spiking)
        images_encoded = torch.cat(images_encoded_sequence, dim=1)
        # images_spiking_list = torch.cat(images_spiking_sequence, dim=1)
        # images_spiking_list_array = images_spiking_list.clone().to('cpu').detach().numpy()
        # images_encoded_array = images_encoded.clone().to('cpu').detach().numpy()

        # query_spiking, _ = self.image_encoding_layer(torch.flatten(query, -2, -1).unsqueeze(2))
        # query_encoded = self.image_embedding_layer(query_spiking)

        query_encoded = self.image_embedding_layer(query)

        # query_array = query.clone().detach().to('cpu').numpy()
        # query_spiking_array = query_spiking.clone().to('cpu').detach().numpy()
        # query_encoded_array = query_encoded.clone().to('cpu').detach().numpy()

        mem, write_key, write_val, _ = self.memory_layer(torch.cat((images_encoded, images_encoded), dim=-1))

        mem, read_key, read_val, _ = self.memory_layer(torch.cat((query_encoded, query_encoded), dim=-1),
                                                       mem=mem, recall=True)

        decoder_output_l1, _, _ = self.decoder_l1(read_val)
        decoder_output_l2, _, _ = self.decoder_l2(decoder_output_l1)

        outputs = torch.sum(decoder_output_l2, dim=1).squeeze() / 15

        encoding_outputs = [images_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]
        decoder_outputs = [decoder_output_l1, decoder_output_l2]

        return outputs, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs


class AttentionMemoryModel(torch.nn.Module):
    def __init__(self, output_size: int, memory_size: int, num_time_steps: int, readout_delay: int, tau_trace: float,
                 image_embedding_layer: torch.nn.Module, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        spiking_image_size = image_embedding_layer.input_size
        image_feature_size = image_embedding_layer.output_size
        writing_layer_input_size = image_feature_size + image_feature_size

        # encoding layer输出图像的脉冲序列形状为[1, 100, 784]
        # self.image_encoding_layer = EncodingLayer(1, spiking_image_size, False, False, num_time_steps, dynamics)
        # embedding layer输出图像的嵌入编码的形状为[1, 100, 64]
        self.image_embedding_layer = image_embedding_layer
        self.memory_layer = MemoryLayer(writing_layer_input_size, memory_size, plasticity_rule,
                                        tau_trace, readout_delay, dynamics)
        self.attention_layer = AttentionLayer(image_feature_size, memory_size, memory_size,
                                              memory_size, memory_size, dynamics)
        # self.attention_layer = SpatioAttentionLayer(1, 1, num_time_steps=num_time_steps, input_size=image_feature_size,
        #                                             dynamics=dynamics)
        self.decoder_l1 = DenseLayer(memory_size, 256, dynamics)
        self.decoder_l2 = DenseLayer(256, output_size, dynamics)

    def forward(self, images: torch.Tensor, query: torch.Tensor) -> Tuple:
        batch_size, sequence_length, *CHW = images.size()

        images_encoded_sequence = []
        # images_spiking_sequence = []
        for t in range(sequence_length):
            # images_spiking, _ = self.image_encoding_layer(torch.flatten(images.select(1, t), -2, -1).unsqueeze(2))
            # images_embedded = self.image_embedding_layer(images_spiking)

            images_embedded = self.image_embedding_layer(images.select(1, t))

            images_encoded_sequence.append(images_embedded)

            # images_spiking_sequence.append(images_spiking)
        images_encoded = torch.cat(images_encoded_sequence, dim=1)
        # images_spiking_list = torch.cat(images_spiking_sequence, dim=1)
        # images_spiking_list_array = images_spiking_list.clone().to('cpu').detach().numpy()

        # query_spiking, _ = self.image_encoding_layer(torch.flatten(query, -2, -1).unsqueeze(2))
        # query_encoded = self.image_embedding_layer(query_spiking)

        query_encoded = self.image_embedding_layer(query)

        # images_attention_encoded = self.attention_layer(images_encoded, images_encoded, images_encoded)
        query_attention_encoded = self.attention_layer(query_encoded, query_encoded, query_encoded)

        # query_array = query.clone().detach().to('cpu').numpy()
        # query_spiking_array = query_spiking.clone().to('cpu').detach().numpy()
        # images_encoded_array = images_encoded.clone().to('cpu').detach().numpy()
        # query_encoded_array = query_encoded.clone().to('cpu').detach().numpy()
        # images_attention_encoded_array = images_attention_encoded.clone().to('cpu').detach().numpy()
        # query_attention_encoded_array = query_attention_encoded.clone().to('cpu').detach().numpy()

        mem, write_key, write_val, _ = \
            self.memory_layer(torch.cat((images_encoded, images_encoded), dim=-1))

        mem, read_key, read_val, _ = \
            self.memory_layer(torch.cat((query_attention_encoded, query_encoded), dim=-1),
                              mem=mem, recall=True)

        # attention_output = self.attention_layer(query_encoded, read_key, read_val)

        # attention_array = attention_output.clone().detach().to('cpu').numpy()
        # read_val_array = read_val.clone().detach().to('cpu').numpy()
        # mem_output_array = mem_output.clone().detach().to('cpu').numpy()

        decoder_output_l1, _, _ = self.decoder_l1(read_val)
        decoder_output_l2, _, _ = self.decoder_l2(decoder_output_l1)

        outputs = torch.sum(decoder_output_l2, dim=1).squeeze() / 15
        decoder_output_l1_array = decoder_output_l1.clone().detach().to('cpu').numpy()
        decoder_output_l2_array = decoder_output_l2.clone().detach().to('cpu').numpy()
        outputs_array = outputs.clone().detach().to('cpu').numpy()

        encoding_outputs = [images_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]
        decoder_outputs = [decoder_output_l1, decoder_output_l2]

        return outputs, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs


class bAbIModel(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, num_embeddings: int, embedding_size: int, memory_size: int,
                 mask_time_words: bool, learn_encoding: bool, num_time_steps: int, readout_delay: int,
                 learn_readout_delay: bool, tau_trace: float, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        self.readout_delay = readout_delay

        self.embedding_layer = EmbeddingLayer(num_embeddings, embedding_size, padding_idx=0)
        self.encoding_layer = EncodingLayer(input_size, embedding_size, mask_time_words, learn_encoding,
                                            num_time_steps, dynamics)
        # self.writing_layer = WritingLayer(embedding_size, memory_size, plasticity_rule, tau_trace, dynamics)
        # self.reading_layer = ReadingLayer(embedding_size, memory_size, readout_delay, dynamics,
        #                                   learn_feedback_delay=learn_readout_delay)
        # self.attention_layer = DenseLayer(memory_size, memory_size, dynamics)
        self.memory_layer = MemoryLayer(embedding_size, memory_size, plasticity_rule,
                                        tau_trace, readout_delay, dynamics)
        self.output_layer = torch.nn.Linear(memory_size, output_size, bias=False)

    def forward(self, story: torch.Tensor, query: torch.Tensor) -> Tuple:

        # story_array = story.clone().detach().to('cpu').numpy()
        # query_array = query.clone().detach().to('cpu').numpy()

        story_embedded = self.embedding_layer(story)
        query_embedded = self.embedding_layer(query)

        # story_embedded_array = story_embedded.clone().detach().to('cpu').numpy()
        # query_embedded_array = query_embedded.clone().detach().to('cpu').numpy()

        story_encoded, _ = self.encoding_layer(story_embedded)
        query_encoded, _ = self.encoding_layer(query_embedded.unsqueeze(1))

        # mem, write_key, write_val, _ = self.writing_layer(story_encoded)

        # read_key, read_val, _ = self.reading_layer(query_encoded, mem)

        mem, write_key, write_val, _ = self.memory_layer(story_encoded)

        mem, read_key, read_val, _ = self.memory_layer(query_encoded, mem=mem, recall=True)

        # attention = torch.bmm(query_encoded.transpose(1, 2), read_key)
        # attention_value = torch.bmm(attention, read_val.transpose(1, 2)).transpose(1, 2)
        # mem_output, _, _ = self.attention_layer(attention_value)

        outputs = torch.sum(read_val[:, -30:, :], dim=1)
        outputs = self.output_layer(outputs)

        encoding_outputs = [story_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs


class SquadModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, memory_size: int, readout_delay: int,
                 tau_trace: float, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()

        self.encoding_layer = DenseLayer(input_size, memory_size, dynamics)

        self.dense_layer = DenseLayer(memory_size, memory_size, dynamics)

        self.memory_layer = MemoryLayer(input_size, memory_size, plasticity_rule,
                                        tau_trace, readout_delay, dynamics)

        self.decoding_layer = DenseLayer(memory_size, memory_size, dynamics)
        self.output_layer = torch.nn.Linear(memory_size, output_size, bias=False)
        # self.output_layer = DenseLayer(memory_size, output_size, dynamics)

    def forward(self, story: torch.Tensor, query: torch.Tensor) -> Tuple:

        context_encoded, _, _ = self.encoding_layer(story)
        questions_encoded, _, _ = self.encoding_layer(query)

        context_dense_output, _, _ = self.dense_layer(context_encoded)
        questions_dense_output, _, _ = self.dense_layer(questions_encoded)

        mem, write_key, write_val, _ = self.memory_layer(context_dense_output)

        mem_array = mem.clone().detach().to('cpu').numpy()
        write_key_array = write_key.clone().detach().to('cpu').numpy()
        write_val_array = write_val.clone().detach().to('cpu').numpy()

        mem, read_key, mem_output, _ = self.memory_layer(questions_dense_output, mem=mem, recall=True)

        read_val, _, _ = self.decoding_layer(mem_output)

        read_key_array = read_key.clone().detach().to('cpu').numpy()
        read_val_array = read_val.clone().detach().to('cpu').numpy()
        output_sum = torch.sum(read_val, dim=1)
        outputs = self.output_layer(output_sum)
        outputs_array = outputs.clone().detach().to('cpu').numpy()

        encoding_outputs = [context_encoded, questions_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs


class TextCNNSquadModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, memory_size: int, readout_delay: int,
                 tau_trace: float, context_embedding_layer: torch.nn.Module, question_embedding_layer: torch.nn.Module,
                 plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()

        self.context_embedding_layer = context_embedding_layer
        self.question_embedding_layer = question_embedding_layer

        self.memory_layer = MemoryLayer(input_size, memory_size, plasticity_rule,
                                        tau_trace, readout_delay, dynamics)

        self.decoding_layer = DenseLayer(memory_size, memory_size, dynamics)
        self.output_layer = torch.nn.Linear(memory_size, output_size, bias=False)
        # self.output_layer = DenseLayer(memory_size, output_size, dynamics)

    def forward(self, story: torch.Tensor, query: torch.Tensor) -> Tuple:
        story_array = story.clone().detach().to('cpu').numpy()
        query_array = query.clone().detach().to('cpu').numpy()

        # context_encoded = self.text_embedding_layer(story)
        # questions_encoded = self.text_embedding_layer(query, recall=True)

        context_encoded = self.context_embedding_layer(story)
        questions_encoded = self.question_embedding_layer(query)

        context_encoded_array = context_encoded.clone().detach().to('cpu').numpy()
        questions_encoded_array = questions_encoded.clone().detach().to('cpu').numpy()

        mem, write_key, write_val, _ = self.memory_layer(context_encoded)

        mem_array = mem.clone().detach().to('cpu').numpy()
        write_key_array = write_key.clone().detach().to('cpu').numpy()
        write_val_array = write_val.clone().detach().to('cpu').numpy()

        mem, read_key, mem_output, _ = self.memory_layer(questions_encoded,
                                                         mem=mem, recall=True)

        read_val, _, _ = self.decoding_layer(mem_output)

        read_key_array = read_key.clone().detach().to('cpu').numpy()
        read_val_array = read_val.clone().detach().to('cpu').numpy()
        output_sum = torch.sum(read_val, dim=1)
        outputs = self.output_layer(output_sum)
        outputs_array = outputs.clone().detach().to('cpu').numpy()

        encoding_outputs = [context_encoded, questions_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs


class TextImageAssociation(torch.nn.Module):
    def __init__(self, output_size: int, memory_size: int, num_time_steps: int, readout_delay: int, tau_trace: float,
                 image_embedding_layer: torch.nn.Module, text_embedding_layer: torch.nn.Module,
                 plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        image_feature_size = image_embedding_layer.output_size
        text_feature_size = text_embedding_layer.output_size
        writing_layer_input_size = text_feature_size + image_feature_size

        self.image_embedding_layer = image_embedding_layer
        self.text_embedding_layer = text_embedding_layer
        self.memory_layer = MemoryLayer(writing_layer_input_size, memory_size, plasticity_rule,
                                        tau_trace, readout_delay, dynamics)
        self.decoder_l1 = DenseLayer(memory_size, 256, dynamics)
        self.decoder_l2 = DenseLayer(256, output_size, dynamics)

    def forward(self, images: torch.Tensor, text: torch.Tensor, query: torch.Tensor) -> Tuple:
        batch_size, sequence_length, *CHW = images.size()

        images_encoded_sequence = []
        text_encoded_sequence = []
        for t in range(sequence_length):

            images_embedded = self.image_embedding_layer(images.select(1, t))
            text_embedded = self.text_embedding_layer(text.select(1, t))

            images_encoded_sequence.append(images_embedded)
            text_encoded_sequence.append(text_embedded)

        images_encoded = torch.cat(images_encoded_sequence, dim=1)
        text_encoded = torch.cat(text_encoded_sequence, dim=1)

        query_encoded = self.text_embedding_layer(query)

        # images_encoded_array = images_encoded.clone().detach().to('cpu').numpy()
        # text_encoded_array = text_encoded.clone().detach().to('cpu').numpy()
        # query_encoded_array = query_encoded.clone().detach().to('cpu').numpy()

        mem, write_key, write_val, _ = self.memory_layer(torch.cat((text_encoded, images_encoded), dim=-1))

        mem, read_key, read_val, _ = self.memory_layer(torch.cat((query_encoded, query_encoded), dim=-1),
                                                       mem=mem, recall=True)

        decoder_output_l1, _, _ = self.decoder_l1(read_val)
        decoder_output_l2, _, _ = self.decoder_l2(decoder_output_l1)

        outputs = torch.sum(decoder_output_l2, dim=1).squeeze() / 15

        encoding_outputs = [text_encoded, images_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]
        decoder_outputs = [decoder_output_l1, decoder_output_l2]

        return outputs, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs
