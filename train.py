#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
import torch
# from trainer.trainer_OT import train
# from trainer.trainer_minibatch_noisy import train

# from trainer.trainer_minibatch import train
# from trainer.trainer_minibatch_Maha import train
from trainer.trainer_minibatch_Orliz import train

from tools.config_loader import get_config


if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='exp_name', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', default='settings', type=str,
                        help='Name of the setting file.')

    args = parser.parse_args()

    config = get_config(args.config)

    config.exp_name = args.exp_name
    train(config)
