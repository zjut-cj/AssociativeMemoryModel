"""Plot layer outputs of the model for the cross-modal associations task"""

import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torchvision

import utils.checkpoint
from data.mnist_datasets import MNISTDataset, SequentialMNISTDataset, HeteroAssociativeMNISTDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import BackUp, InhibitoryMemoryModel
from models.neuron_models import IafPscDelta
from utils.utils import salt_pepper_noise, apply_mask, compute_average_ssim, gaussian_perturb_image
# from models.protonet_models import SpikingProtoNet
from models.spiking_model import SpikingProtoNet
from tqdm import tqdm
from models.direct_encoding_model import DirectEncodingModel
from models.latency_encoding_model import LatencyEncodingModel
import utils.meters


def main():
    parser = argparse.ArgumentParser(description='MNIST one shot task plotting')
    parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')
    parser.add_argument('--check_checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')
    parser.add_argument('--check_params', default=1, type=int, choices=[0, 1], metavar='CHECK_PARAMS',
                        help='When loading from a checkpoint check if the model was trained with the same parameters '
                             'as requested now (default: 1)')

    parser.add_argument('--sequence_length', default=3, type=int, metavar='N',
                        help='Number of image per example (default: 3)')
    parser.add_argument('--num_classes', default=5, type=int, metavar='N',
                        help='Number of random classes per sample (default: 5)')
    parser.add_argument('--dataset_size', default=1000, type=int, metavar='DATASET_SIZE',
                        help='Number of examples in the dataset (default: 10000)')
    parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                        help='Number of time steps for each item in the sequence (default: 100)')
    parser.add_argument('--fix_cnn_thresholds', action='store_false',
                        help='Do not adjust firing threshold after conversion (default: will adjust v_th via a bias)')

    parser.add_argument('--embedding_size', default=64, type=int, metavar='N',
                        help='Embedding size (default: 64)')
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
    parser.add_argument('--readout_delay', default=1, type=int, metavar='N',
                        help='Synaptic delay of the feedback-connections from value-neurons to key-neurons in the '
                             'reading layer (default: 1)')

    parser.add_argument('--thr', default=0.05, type=float, metavar='N',
                        help='Spike threshold (default: 0.1)')
    parser.add_argument('--perfect_reset', action='store_true',
                        help='Set the membrane potential to zero after a spike')
    parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                        help='The number of time steps the neuron is refractory (default: 3)')
    parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                        help='Neuron membrane time constant (default: 20.0)')

    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='Seed for initializing (default: none)')
    parser.add_argument('--data_seed', default=None, type=int, metavar='N',
                        help='Seed for the dataset (default: none)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data loading code
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    test_set = MNISTDataset(root='/usr/common/datasets/MNIST', train=False, classes=args.num_classes,
                                      dataset_size=args.dataset_size, image_transform=image_transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=1, prefetch_factor=2)

    # Create ProtoNetSpiking
    image_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                        perfect_reset=args.perfect_reset,
                                                        refractory_time_steps=args.refractory_time_steps,
                                                        tau_mem=args.tau_mem,
                                                        spike_function=SpikeFunction),
                                            # weight_dict=image_protonet_checkpoint['state_dict'],
                                            input_size=784,
                                            output_size=args.embedding_size,
                                            num_time_steps=args.num_time_steps,
                                            refractory_time_steps=args.refractory_time_steps,
                                            use_bias=args.fix_cnn_thresholds)

    # Create the model
    model = InhibitoryMemoryModel(
        output_size=784,
        memory_size=args.memory_size,
        num_time_steps=args.num_time_steps,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
        image_embedding_layer=image_embedding_layer,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg),
        dynamics=IafPscDelta(thr=args.thr,
                             perfect_reset=args.perfect_reset,
                             refractory_time_steps=args.refractory_time_steps,
                             tau_mem=args.tau_mem,
                             spike_function=SpikeFunction))

    # Load checkpoint
    if args.checkpoint_path:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = utils.checkpoint.load_checkpoint(args.checkpoint_path, device)
        best_loss = checkpoint['best_loss']
        epoch = checkpoint['epoch']
        print("Best loss {}".format(best_loss))
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

    ssims = utils.meters.AverageMeter('SSIM', ':.4e')

    for i, sample in enumerate(tqdm(test_loader)):
        image_sequence, labels, image_query, targets = sample
        image_targets = image_query
        # image_query = gaussian_perturb_image(image_query, 0.8)
        image_query = apply_mask(image_query, 0.9)
        outputs, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs = model(image_sequence,
                                                                                             image_query)
        outputs = outputs.view(-1, 1, 28, 28)

        ssim = compute_average_ssim(outputs, image_targets)
        ssims.update(ssim.item(), image_sequence.size(0))

    print("Average SSIM: ", ssims.avg)



if __name__ == '__main__':
    main()
