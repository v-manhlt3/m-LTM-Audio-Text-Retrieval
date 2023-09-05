#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t_ot, t2a_ot, t2a, a2t, a2t_ot_kernel, t2a_ot_kernel
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent, WeightTriplet, \
                        POTLoss, OTLoss, WassersteinLoss
from models.ASE_model import ASE
# from models.ASE_model_sliceW import ASE
from data_handling.DataLoader import get_dataloader2


def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)

    # set up logger
    exp_name = config.exp_name

    folder_name = '{}_data_{}_noise{}_eps{}_m{}_lr_{}_'.format(exp_name, config.dataset,
                                             config.training.noise_p,
                                             config.training.epsilon,
                                             config.training.m,
                                             config.training.lr)

    log_output_dir = Path('rbf-output', folder_name, 'logging')
    model_output_dir = Path('rbf-output', folder_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_output_dir) + '/tensorboard')

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    model = ASE(config)
    model = model.to(device)

    # set up optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if config.training.loss == 'triplet':
        criterion = TripletLoss(margin=config.training.margin)
    elif config.training.loss == 'ntxent':
        criterion = NTXent(noise_p=config.training.noise_p)
    elif config.training.loss == 'weight':
        criterion = WeightTriplet(margin=config.training.margin)
    elif config.training.loss == 'pot':
        criterion = POTLoss(epsilon=config.training.epsilon, m=config.training.m, use_cosine=config.training.use_cosine)
    elif config.training.loss == 'wloss':
        criterion = WassersteinLoss(epsilon=config.training.epsilon, use_cosine=config.training.use_cosine, reg=config.training.reg)
    elif config.training.loss == 'ot':
        criterion = OTLoss(epsilon=config.training.epsilon, use_cosine=config.training.use_cosine)
    # elif config.training.loss == 'learnW':
    #     criterion = LearnScoreW()
    else:
        criterion = BiDirectionalRankingLoss(margin=config.training.margin)

    # set up data loaders
    train_loader = get_dataloader2('train', config)
    val_loader = get_dataloader2('val', config)
    test_loader = get_dataloader2('test', config)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    ep = 1

    # resume from a checkpoint
    if config.training.resume:
        checkpoint = torch.load(config.path.resume_model)
        model.load_state_dict(checkpoint['model'])
        ep = checkpoint['epoch']

    # training loop
    recall_sum_a2t = []
    recall_sum_t2a = []
    num_test_samples = 958

    # best_checkpoint_a2t = torch.load(str(model_output_dir) + '/t2a_best_model.pth')
    # model.load_state_dict(best_checkpoint_a2t['model'])
    # r_sum_a2t, r_sum_t2a = validate(val_loader, model, device, writer, 1, use_ot=config.training.use_ot, use_cosine=config.training.use_cosine)
    # for epoch in range(ep, config.training.epochs + 1):
    #     main_logger.info(f'Training for epoch [{epoch}]')

    #     epoch_loss = AverageMeter()
    #     start_time = time.time()
    #     model.train()


    #     for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):

    #         audios, captions, audio_ids, _ = batch_data
    #         # if count_train_data < max_count_train_data:
    #         #     train_valid.append(batch_data)
    #         # print(captions)

    #         # move data to GPU
    #         audios = audios.to(device)
    #         audio_ids = audio_ids.to(device)

    #         # # old exp
    #         # audio_embeds, caption_embeds = model(audios, captions)
            
    #         # # new exp
    #         audio_embeds, caption_embeds = model(audios, captions)

    #         # loss = criterion(audio_embeds, caption_embeds, audio_ids)
    #         loss = criterion(audio_embeds, caption_embeds, audio_ids)

    #         optimizer.zero_grad()

    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
    #         optimizer.step()

    #         epoch_loss.update(loss.cpu().item())
    #     writer.add_scalar('train/loss', epoch_loss.avg, epoch)

    #     elapsed_time = time.time() - start_time

    #     main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
    #                      f'\ttime: {elapsed_time:.1f}, lr: {scheduler.get_last_lr()[0]:.6f}.')

    #     # validation loop, validation after each epoch
    #     main_logger.info("Validating...")
    #     r_sum_a2t, r_sum_t2a = validate(val_loader, model, device, writer, epoch, use_ot=config.training.use_ot, use_cosine=config.training.use_cosine)
    #     # r1_t2a, r1_a2t = validate_train_data(train_loader, model, device, epoch, use_ot=config.training.use_ot, use_cosine=config.training.use_cosine)
    #     # # r_sum = r1 + r5 + r10
    #     # writer.add_scalar('train/r1_a2t', r1_a2t, epoch)
    #     # writer.add_scalar('train/r1_t2a', r1_t2a, epoch)

    #     recall_sum_a2t.append(r_sum_a2t)
    #     recall_sum_t2a.append(r_sum_t2a)

    #     # save model
    #     if r_sum_a2t >= max(recall_sum_a2t):
    #         main_logger.info('Model saved.')
    #         torch.save({
    #             'model': model.state_dict(),
    #             'optimizer': model.state_dict(),
    #             'epoch': epoch,
    #         }, str(model_output_dir) + '/a2t_best_model.pth')
    #     if r_sum_t2a >= max(recall_sum_t2a):
    #         main_logger.info('Model saved.')
    #         torch.save({
    #             'model': model.state_dict (),
    #             'optimizer': model.state_dict(),
    #             'epoch': epoch,
    #         }, str(model_output_dir) + '/t2a_best_model.pth')

    #     scheduler.step()

    # Training done, evaluate on evaluation set
    main_logger.info('-'*90)
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint_a2t = torch.load(str(model_output_dir) + '/a2t_best_model.pth')
    model.load_state_dict(best_checkpoint_a2t['model'])
    best_epoch_a2t = best_checkpoint_a2t['epoch']
    main_logger.info(f'Best checkpoint (Audio-to-caption) occurred in {best_epoch_a2t} th epoch.')
    validate_a2t(test_loader, model, device, use_ot=config.training.use_ot,  use_cosine=config.training.use_cosine)

    best_checkpoint_t2a = torch.load(str(model_output_dir) + '/t2a_best_model.pth')
    model.load_state_dict(best_checkpoint_t2a['model'])
    best_epoch_t2a = best_checkpoint_t2a['epoch']
    main_logger.info(f'Best checkpoint (Caption-to-audio) occurred in {best_epoch_t2a} th epoch.')
    validate_t2a(test_loader, model, device, use_ot=config.training.use_ot,  use_cosine=config.training.use_cosine)
    main_logger.info('Evaluation done.')
    writer.close()


def validate(data_loader, model, device, writer, epoch, use_ot=False, use_cosine=True):
    val_logger = logger.bind(indent=1)
    model.eval()
    t2a_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    a2t_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}

    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            audios = audios.to(device)
            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate text to audio retrieval
        # r1, r5, r10, r50, medr, meanr = t2a(audio_embs, cap_embs, False,use_ot, use_cosine)
        r1, r5, r10, r50, medr, meanr, crossentropy_t2a  = t2a_ot(audio_embs, cap_embs, use_ot, use_cosine)
        # r1, r5, r10, r50, medr, meanr, crossentropy_t2a  = t2a_ot_kernel(audio_embs, cap_embs, use_ot, use_cosine)
        r_sum_t2a = r1 +r5 + r10
        t2a_metrics['r1'] += r1
        t2a_metrics['r5'] += r5
        t2a_metrics['r10'] += r10
        t2a_metrics['median'] += medr
        t2a_metrics['mean'] += meanr
        writer.add_scalar("valid/r1_t2a", r1, epoch)

        # evaluate audio to text retrieval
        # r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(audio_embs, cap_embs, False,use_ot, use_cosine)
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, crossentropy_a2t = a2t_ot(audio_embs, cap_embs, use_ot, use_cosine)
        # r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, crossentropy_a2t = a2t_ot_kernel(audio_embs, cap_embs, use_ot, use_cosine)

        r_sum_a2t = r1_a + r5_a + r10_a

        a2t_metrics['r1'] += r1_a
        a2t_metrics['r5'] += r5_a
        a2t_metrics['r10'] += r10_a
        a2t_metrics['median'] += medr_a
        a2t_metrics['mean'] += meanr_a
        writer.add_scalar("valid/r1_at2", r1_a, epoch)

 
        val_logger.info('Audio to caption: r1: {:.4f}, r5: {:.4f}, '
                        'r10: {:.4f}, medr: {:.4f}, meanr: {:.4f}'.format(
                        a2t_metrics['r1'], a2t_metrics['r5'], a2t_metrics['r10'], a2t_metrics['median'], a2t_metrics['mean']))

        val_logger.info('Caption to audio: r1: {:.4f}, r5: {:.4f}, '
                        'r10: {:.4f}, medr: {:.4f}, meanr: {:.4f}'.format(
                         t2a_metrics['r1'], t2a_metrics['r5'], t2a_metrics['r10'], t2a_metrics['median'], t2a_metrics['mean']))
        return r_sum_a2t, r_sum_t2a

def validate_train_data(data_loader, model, device, epoch, use_ot=False, use_cosine=True):
    val_logger = logger.bind(indent=1)
    model.eval()
    t2a_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}
    a2t_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}

    max_count_train_data = 4
    train_valid = []
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None
        audio_list = []
        cap_list = []
        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if i>max_count_train_data:
                break
            audios, captions, audio_ids, indexs = batch_data
            audios = audios.to(device)
            audio_embeds, caption_embeds = model(audios, captions)
            audio_list.append(audio_embeds)
            cap_list.append(caption_embeds)

            
            # if audio_embs is None:
            #     audio_embs = np.zeros((256*4, audio_embeds.size(1)))
            #     cap_embs = np.zeros((256*4, caption_embeds.size(1)))

            # audio_embs[indexs] = audio_embeds.cpu().numpy()
            # cap_embs[indexs] = caption_embeds.cpu().numpy()
        audio_list = torch.vstack(audio_list)
        cap_list = torch.vstack(cap_list)
        # evaluate text to audio retrieval
        r1, r5, r10, r50, medr, meanr, crossentropy_t2a  = t2a_ot(audio_list.cpu().numpy(), cap_list.cpu().numpy(), use_ot, use_cosine, train_data=True)
        r_sum_t2a = r1 +r5 + r10
        t2a_metrics['r1'] += r1
        t2a_metrics['r5'] += r5
        t2a_metrics['r10'] += r10
        t2a_metrics['median'] += medr
        t2a_metrics['mean'] += meanr
        # writer.add_scalar("train/train_validation_t2a", r1, epoch)

        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, crossentropy_a2t = a2t_ot(audio_list.cpu ().numpy(), cap_list.cpu().numpy(), use_ot, use_cosine, train_data=True)
        r_sum_a2t = r1_a + r5_a + r10_a

        a2t_metrics['r1'] += r1_a
        a2t_metrics['r5'] += r5_a
        a2t_metrics['r10'] += r10_a
        a2t_metrics['median'] += medr_a
        a2t_metrics['mean'] += meanr_a
        # writer.add_scalar("train/train_validation_a2t", r1_a, epoch)
        return r1, r1_a

def validate_a2t(data_loader, model, device, use_ot, use_cosine):
    val_logger = logger.bind(indent=1)
    model.eval()
    a2t_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}

    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            # move data to GPU
            audios = audios.to(device)
            # print(captions)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate audio to text retrieval
        # r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(audio_embs, cap_embs, False,use_ot, use_cosine)
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a,_ = a2t_ot(audio_embs, cap_embs,use_ot, use_cosine)
        a2t_metrics['r1'] += r1_a
        a2t_metrics['r5'] += r5_a
        a2t_metrics['r10'] += r10_a
        a2t_metrics['median'] += medr_a
        a2t_metrics['mean'] += meanr_a

        
        val_logger.info('Audio to caption: r1: {:.4f}, r5: {:.4f}, '
                        'r10: {:.4f}, medr: {:.4f}, meanr: {:.4f}'.format(
                        a2t_metrics['r1'], a2t_metrics['r5'], a2t_metrics['r10'], a2t_metrics['median'], a2t_metrics['mean']))
        
def validate_t2a(data_loader, model, device, use_ot, use_cosine):
    val_logger = logger.bind(indent=1)
    model.eval()
    t2a_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}

    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            # move data to GPU
            audios = audios.to(device)
            # print(captions)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate text to audio retrieval
        # r1, r5, r10, r50, medr, meanr = t2a(audio_embs, cap_embs, False,use_ot, use_cosine)
        r1, r5, r10, r50, medr, meanr,_ = t2a_ot(audio_embs, cap_embs,use_ot, use_cosine)
        t2a_metrics['r1'] += r1
        t2a_metrics['r5'] += r5
        t2a_metrics['r10'] += r10
        t2a_metrics['median'] += medr
        t2a_metrics['mean'] += meanr
       
        val_logger.info('Caption to audio: r1: {:.4f}, r5: {:.4f}, '
                        'r10: {:.4f}, medr: {:.4f}, meanr: {:.4f}'.format(
                         t2a_metrics['r1'], t2a_metrics['r5'], t2a_metrics['r10'], t2a_metrics['median'], t2a_metrics['mean']))