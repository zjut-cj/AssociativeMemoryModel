"""Plot layer outputs of the model for the question answering tasks"""

import argparse
import random
import sys
from collections import OrderedDict
from scipy.special import softmax

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils.checkpoint
from data.babi_dataset import BABIDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import SquadModel, TextCNNSquadModel
from models.neuron_models import IafPscDelta
from data.squad_datasets import SquadDatasets
from models.text_cnn import SpikingTextCNN


def main():
    parser = argparse.ArgumentParser(description='Question answering task plotting')
    parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')
    parser.add_argument('--check_params', default=1, type=int, choices=[0, 1], metavar='CHECK_PARAMS',
                        help='When loading from a checkpoint check if the model was trained with the same parameters '
                             'as requested now (default: 1)')

    parser.add_argument('--qas_pairs', default=200, type=int, metavar='N',
                        help='Number of context-question-answer pairs (default: 200)')
    parser.add_argument('--train_dir', default='./data', type=str, metavar='DIR',
                        help='Path to dataset (default: ./data)')
    parser.add_argument('--test_dir', default='./data', type=str, metavar='DIR',
                        help='Path to dataset (default: ./data)')
    parser.add_argument('--example', default=4, type=int, metavar='EXAMPLE',
                        help='The example of the bAbI task (default: 4)')
    parser.add_argument('--ten_k', default=1, choices=[0, 1], type=int, metavar='TEN_K',
                        help='Use 10k examples (default: 1')
    parser.add_argument('--add_time_words', default=0, choices=[0, 1], type=int, metavar='ADD_TIME_WORDS',
                        help='Add time word to sentences (default: 0)')
    parser.add_argument('--sentence_duration', default=200, type=int, metavar='N',
                        help='Number of time steps for each sentence (default: 200)')
    parser.add_argument('--max_num_sentences', default=50, type=int, metavar='N',
                        help='Extract only stories with no more than max_num_sentences. '
                             'If None extract all sentences of the stories (default: 50)')
    parser.add_argument('--padding', default='pre', choices=['pre', 'post'], type=str, metavar='PADDING',
                        help='Where to pad (default: pre)')
    parser.add_argument('--dampening_factor', default=1.0, type=float, metavar='N',
                        help='Scale factor for spike pseudo-derivative (default: 1.0)')

    parser.add_argument('--embedding_size', default=96, type=int, metavar='N',
                        help='Embedding size (default: 100)')
    parser.add_argument('--memory_size', default=100, type=int, metavar='N',
                        help='Size of the memory matrix (default: 100)')
    parser.add_argument('--w_max', default=1.0, type=float, metavar='N',
                        help='Soft maximum of Hebbian weights (default: 1.0)')
    parser.add_argument('--gamma_pos', default=0.3, type=float, metavar='N',
                        help='Write factor of Hebbian rule (default: 0.3)')
    parser.add_argument('--gamma_neg', default=0.3, type=float, metavar='N',
                        help='Forget factor of Hebbian rule (default: 0.3)')
    parser.add_argument('--tau_trace', default=20.0, type=float, metavar='N',
                        help='Time constant of key- and value-trace (default: 20.0)')
    parser.add_argument('--readout_delay', default=30, type=int, metavar='N',
                        help='Synaptic delay of the feedback-connections from value-neurons to key-neurons in the '
                             'reading layer (default: 30)')

    parser.add_argument('--thr', default=0.1, type=float, metavar='N',
                        help='Spike threshold (default: 0.1)')
    parser.add_argument('--perfect_reset', action='store_true',
                        help='Set the membrane potential to zero after a spike')
    parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                        help='The number of time steps the neuron is refractory (default: 3)')
    parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                        help='Neuron membrane time constant (default: 20.0)')

    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='Seed for initializing (default: none)')
    args = parser.parse_args()

    args.ten_k = True if args.ten_k else False
    args.add_time_words = True if args.add_time_words else False

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    squad_datasets = SquadDatasets()
    data_set, paragraph_set, split_data_set, cleaned_data_set, word2index_set, vocab_size = \
        squad_datasets.process_data(train_file_path=args.train_dir, valid_file_path=args.test_dir,
                                    qas_pairs=args.qas_pairs)

    train_set, test_set = data_set
    train_paragraph_set, test_paragraph_set = paragraph_set
    train_split_data_set, test_split_data_set = split_data_set
    train_cleaned_data_set, test_cleaned_data_set = cleaned_data_set
    train_word2index_set, test_word2index_set = word2index_set
    train_vocab_size, test_vocab_size = vocab_size

    image_embedding_layer = SpikingTextCNN(IafPscDelta(thr=args.thr,
                                                       perfect_reset=args.perfect_reset,
                                                       refractory_time_steps=args.refractory_time_steps,
                                                       tau_mem=args.tau_mem,
                                                       spike_function=SpikeFunction,
                                                       dampening_factor=args.dampening_factor),
                                           output_size=80,
                                           refractory_time_steps=args.refractory_time_steps)

    print("=> creating model '{model_name}'".format(model_name=SquadModel.__name__))
    # model = SquadModel(
    #     input_size=100,
    #     output_size=100,
    #     memory_size=args.memory_size,
    #     readout_delay=args.readout_delay,
    #     tau_trace=args.tau_trace,
    #     plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
    #                                                   gamma_pos=args.gamma_pos,
    #                                                   gamma_neg=args.gamma_neg),
    #     dynamics=IafPscDelta(thr=args.thr,
    #                          perfect_reset=args.perfect_reset,
    #                          refractory_time_steps=args.refractory_time_steps,
    #                          tau_mem=args.tau_mem,
    #                          spike_function=SpikeFunction,
    #                          dampening_factor=args.dampening_factor))

    model = TextCNNSquadModel(
        input_size=96,
        output_size=100,
        text_embedding_layer=image_embedding_layer,
        memory_size=args.memory_size,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg),
        dynamics=IafPscDelta(thr=args.thr,
                             perfect_reset=args.perfect_reset,
                             refractory_time_steps=args.refractory_time_steps,
                             tau_mem=args.tau_mem,
                             spike_function=SpikeFunction,
                             dampening_factor=args.dampening_factor))

    # Load checkpoint
    if args.checkpoint_path:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = utils.checkpoint.load_checkpoint(args.checkpoint_path, device)
        best_acc = checkpoint['best_acc']
        epoch = checkpoint['epoch']
        print("Best accuracy {}".format(best_acc))
        print("Epoch {}".format(epoch))
        if args.check_params:
            for key, val in vars(args).items():
                if key not in ['check_params', 'seed', 'data_seed', 'checkpoint_path']:
                    if vars(checkpoint['params'])[key] != val:
                        print("=> You tried to load a model that was trained on different parameters as you requested "
                              "now. You may disable this check by setting `check_params` to 0. Aborting...")
                        sys.exit()

        new_state_dict = OrderedDict()
        # print("checkpoint['state_dict']:", checkpoint['state_dict'])
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[len('module.'):]  # remove `module.`
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # Switch to evaluate mode
    model.eval()

    sample = None
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    data_iter = iter(train_loader)
    for _ in range(args.example + 1):
        sample = next(data_iter)


    paragraph = train_paragraph_set[args.example]
    split_data = train_split_data_set[args.example]
    cleaned_data = train_cleaned_data_set[args.example]
    word2index = train_word2index_set[args.example]

    print("Paragraph: ", paragraph)
    print("Split data: ", split_data)
    print("Cleaned data: ", cleaned_data)
    print("Word2index: ", word2index)

    context, questions, answers = sample
    outputs, encoding_outputs, writing_outputs, reading_outputs = model(context, questions)

    # Get the outputs
    context_one_hot = context.detach().numpy()
    questions_one_hot = questions.detach().numpy()
    context_encoded = encoding_outputs[0].detach().numpy()
    questions_encoded = encoding_outputs[1].detach().numpy()
    mem = writing_outputs[0].detach().numpy()
    write_key = writing_outputs[1].detach().numpy()
    write_val = writing_outputs[2].detach().numpy()
    read_key = reading_outputs[0].detach().numpy()
    read_val = reading_outputs[1].detach().numpy()
    outputs = outputs.detach().numpy()

    # 解析 one-hot 编码为文本序列
    # 将 one-hot 编码的张量还原成文本向量
    text_vectors = []
    for one_hot_vector in answers:
        # 获取非零元素的索引
        non_zero_indices = torch.nonzero(one_hot_vector).squeeze()

        # 使用字典将索引还原成单词
        words = [word for word, index in word2index.items() if index in non_zero_indices]

        text_vectors.append(words)

    print("Answers after parse: ", text_vectors)

    answers_array = answers.clone().detach().numpy()

    mean_rate_story_encoding = np.sum(context_encoded, axis=1) / (1e-3 * args.sentence_duration)
    mean_rate_query_encoding = np.sum(questions_encoded, axis=1) / (1e-3 * args.sentence_duration)
    mean_rate_write_key = np.sum(write_key, axis=1) / (1e-3 * args.sentence_duration)
    mean_rate_write_val = np.sum(write_val, axis=1) / (1e-3 * args.sentence_duration)
    mean_rate_read_key = np.sum(read_key, axis=1) / (1e-3 * args.sentence_duration)
    mean_rate_read_val = np.sum(read_val, axis=1) / (1e-3 * args.sentence_duration)

    z_s_enc = context_encoded[0]
    z_r_enc = questions_encoded[0]
    z_key = np.concatenate((write_key[0], read_key[0]), axis=0)
    z_value = np.concatenate((write_val[0], read_val[0]), axis=0)

    print("z_s_enc", z_s_enc.shape)
    print("z_r_enc", z_r_enc.shape)
    print("z_key", z_key.shape)
    print("z_value", z_value.shape)

    all_neurons = np.concatenate((
        np.pad(z_s_enc, ((0, args.sentence_duration), (0, 0))),
        np.pad(z_r_enc, ((z_s_enc.shape[0], 0), (0, 0))),
        z_key,
        z_value
    ), axis=1)

    print("z_s_enc", (np.sum(z_s_enc, axis=0) / (1e-3 * (1 + 1) * args.sentence_duration)).mean())
    print("z_r_enc", (np.sum(z_r_enc, axis=0) / (1e-3 * (1 + 1) * args.sentence_duration)).mean())
    print("z_key", (np.sum(z_key, axis=0) / (1e-3 * (1 + 1) * args.sentence_duration)).mean())
    print("z_value", (np.sum(z_value, axis=0) / (1e-3 * (1 + 1) * args.sentence_duration)).mean())
    print("all_neurons", (np.sum(all_neurons, axis=0) / (1e-3 * (1 + 1) * args.sentence_duration)).mean())

    # Make some plots
    # fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all')
    # ax[0].pcolormesh(story[0].T, cmap='binary')
    # ax[0].set_ylabel('story')
    # ax[1].pcolormesh(query.T, cmap='binary')
    # ax[1].set_ylabel('query')
    # plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(context_one_hot[0].T, cmap='binary')
    ax[0, 0].set_ylabel('context')
    ax[0, 1].barh(range(args.embedding_size), mean_rate_story_encoding[0])
    ax[0, 1].set_ylim([0, args.embedding_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(questions_one_hot[0].T, cmap='binary')
    ax[1, 0].set_ylabel('question')
    ax[1, 1].barh(range(args.embedding_size), mean_rate_query_encoding[0])
    ax[1, 1].set_ylim([0, args.embedding_size])
    ax[1, 1].set_yticks([])
    plt.tight_layout()

    # context one-hot encoding
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.set_ylabel('Neuron Index')
    ax.set_xlabel('Time Step')
    ax.pcolormesh(context_one_hot[0].T, cmap='binary')
    plt.tight_layout()

    # question one-hot encoding
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.set_ylabel('Neuron Index')
    ax.set_xlabel('Time Step')
    ax.pcolormesh(questions_one_hot[0].T, cmap='binary')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(context_encoded[0].T, cmap='binary')
    ax[0, 0].set_ylabel('context encoded')
    ax[0, 1].barh(range(args.embedding_size), mean_rate_story_encoding[0])
    ax[0, 1].set_ylim([0, args.embedding_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(questions_encoded[0].T, cmap='binary')
    ax[1, 0].set_ylabel('question encoded')
    ax[1, 1].barh(range(args.embedding_size), mean_rate_query_encoding[0])
    ax[1, 1].set_ylim([0, args.embedding_size])
    ax[1, 1].set_yticks([])
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(write_key[0].T, cmap='binary')
    ax[0, 0].set_ylabel('write keys')
    ax[0, 1].barh(range(args.memory_size), mean_rate_write_key[0])
    ax[0, 1].set_ylim([0, args.memory_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(write_val[0].T, cmap='binary')
    ax[1, 0].set_ylabel('write values')
    ax[1, 1].barh(range(args.memory_size), mean_rate_write_val[0])
    ax[1, 1].set_ylim([0, args.memory_size])
    ax[1, 1].set_yticks([])
    plt.tight_layout()

    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    # ax.matshow(mem[0], cmap='RdBu')
    # plt.tight_layout()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    cax = ax.imshow(mem[0], cmap='viridis')
    # 设置横坐标和纵坐标的刻度及标签
    xticks_interval = 20
    yticks_interval = 20
    ax.set_xticks(np.arange(0, 100, xticks_interval))
    ax.set_yticks(np.arange(0, 100, yticks_interval))
    ax.set_xticklabels(np.arange(0, 100, xticks_interval))
    ax.set_yticklabels(np.arange(0, 100, yticks_interval))
    # 添加颜色条
    fig.colorbar(cax)
    ax.set_title("Synaptic weights between Perception and Response neurons")
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(read_key[0].T, cmap='binary')
    ax[0, 0].set_ylabel('read keys')
    ax[0, 1].barh(range(args.memory_size), mean_rate_read_key[0])
    ax[0, 1].set_ylim([0, args.memory_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(read_val[0].T, cmap='binary')
    ax[1, 0].set_ylabel('read values')
    ax[1, 1].barh(range(args.memory_size), mean_rate_read_val[0])
    ax[1, 1].set_ylim([0, args.memory_size])
    ax[1, 1].set_yticks([])
    plt.tight_layout()

    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    # ax.pcolormesh(outputs[0, None].T, cmap='viridis')
    # ax.set_aspect(0.1)
    # plt.tight_layout()

    def plot_vertical_heatmap(ax, data, title, bar_width=0.8, xticks_interval=20):
        bar_positions = np.arange(len(data[0]))
        bars = ax.bar(bar_positions, data[0], width=bar_width, color='skyblue')

        # 设置 x 轴刻度
        x_ticks = np.arange(0, 100, xticks_interval)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

        # 设置 y 轴刻度
        # y_ticks = np.arange(0, 1.1, 0.2)
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])

        # 不显示 y 轴刻度
        ax.set_yticks([])

        ax.set_title(title)

    # 创建一个包含两个子图的图形
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # 绘制第一个子图 (outputs)
    plot_vertical_heatmap(ax[0], outputs, title='Outputs')

    # 绘制第二个子图 (answers)
    plot_vertical_heatmap(ax[1], answers_array, title='Answers')

    plt.tight_layout()
    plt.show()

    def visualize_word2index(word2index, n):
        words = list(word2index.keys())[:n]
        indices = list(word2index.values())[:n]

        plt.figure(figsize=(16, 8))
        plt.bar(words, indices)
        plt.xlabel('Words')
        plt.ylabel('Indices')
        plt.title(f'Word to Index Mapping')
        plt.xticks(rotation=50, ha='right', fontsize=10)  # Rotate and set font size
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    visualize_word2index(train_word2index_set[args.example], train_vocab_size[args.example])
    #
    # # Convert the dictionary to a pandas DataFrame
    # df = pd.DataFrame(list(train_word2index_set[args.example].items()), columns=['Word', 'Index'])
    #
    # # Plotting the table
    # fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figsize as needed
    # ax.axis('tight')
    # ax.axis('off')
    # ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    plt.show()



if __name__ == '__main__':
    main()
