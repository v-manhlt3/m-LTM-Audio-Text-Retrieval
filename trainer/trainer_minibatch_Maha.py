#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import platform
import sys
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t_ot, t2a_ot, t2a, a2t
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent, WeightTriplet,  MahalalobisLoss
from models.ASE_model_Maha import ASE
from data_handling.DataLoader import get_dataloader2
from data_handling.Pretrained_dataset import pretrain_dataloader
from models.BERT_Config import MODELS

tokenizer =  MODELS["bert-base-uncased"][1].from_pretrained("bert-base-uncased")

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

    log_output_dir = Path('exp-outputs', folder_name, 'logging')
    model_output_dir = Path('exp-outputs', folder_name, 'models')
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

    if config.training.loss == 'triplet':
        criterion = TripletLoss(margin=config.training.margin)
    elif config.training.loss == 'ntxent':
        criterion = NTXent()
    else:
        criterion = MahalalobisLoss(epsilon=config.training.epsilon, reg=config.training.reg, m=config.training.m, pot=config.training.use_pot)
    model = ASE(config)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        criterion = torch.nn.DataParallel(criterion)

    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # set up data loaders
    train_loader = get_dataloader2('train', config, config.dataset)
    val_loader = get_dataloader2('val', config, config.dataset)
    test_loader = get_dataloader2('test', config, config.dataset)

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

    for epoch in range(ep, config.training.epochs + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):

            audios, captions, audio_ids, _ = batch_data
            # move data to GPU
            audios = audios.to(device)
            audio_ids = audio_ids.to(device)

            tokenized = tokenizer(captions, add_special_tokens=True,padding=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            audio_embeds, caption_embeds = model(audios, input_ids, attention_mask)
            
            M = model.L
            loss = criterion(audio_embeds, caption_embeds, M)
            loss = torch.mean(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            # constraint matrix L#################
            if torch.cuda.device_count()>1:
                M = model.module.L
            else: 
                M = model.L
            M = torch.nan_to_num(M)
            u, s, v =torch.svd(M)
            s = torch.clamp(s, min=0)
            M_constraint = u@torch.diag(s)@v.t()
            model.L.data = M_constraint
            ######################################
            
            epoch_loss.update(loss.cpu().item())
        
        small_eigen = s<1
        num_small_eigen = torch.sum(small_eigen)
        writer.add_scalar('train/num_small_eigen', num_small_eigen.item(), epoch)
        writer.add_scalar('train/loss', epoch_loss.avg, epoch)

        elapsed_time = time.time() - start_time

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {scheduler.get_last_lr()[0]:.6f}.')

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        r_sum_a2t, r_sum_t2a = validate(val_loader, model, device, writer, epoch, model.L)

        recall_sum_a2t.append(r_sum_a2t)
        recall_sum_t2a.append(r_sum_t2a)

        # save model
        if r_sum_a2t >= max(recall_sum_a2t):
            main_logger.info('Model saved.')
            torch.save({
                'model': model.state_dict(),
                'optimizer': model.state_dict(),
                'epoch': epoch,
            }, str(model_output_dir) + '/a2t_best_model.pth')
        if r_sum_t2a >= max(recall_sum_t2a):
            main_logger.info('Model saved.')
            torch.save({
                'model': model.state_dict (),
                'optimizer': model.state_dict(),
                'epoch': epoch,
            }, str(model_output_dir) + '/t2a_best_model.pth')

        scheduler.step()

    # Training done, evaluate on evaluation set
    main_logger.info('-'*90)
    main_logger.info('Training done. Start evaluating.')

    best_checkpoint_t2a = torch.load(str(model_output_dir) + '/t2a_best_model.pth')
    model.load_state_dict(best_checkpoint_t2a['model'])
    best_epoch_t2a = best_checkpoint_t2a['epoch']
    main_logger.info(f'Best checkpoint (Caption-to-audio) occurred in {best_epoch_t2a} th epoch.')
    validate_t2a(test_loader, model, device, L=model.L)
    main_logger.info('Evaluation done.')
    writer.close()

    best_checkpoint_a2t = torch.load(str(model_output_dir) + '/a2t_best_model.pth')
    model.load_state_dict(best_checkpoint_a2t['model'])
    best_epoch_a2t = best_checkpoint_a2t['epoch']
    main_logger.info(f'Best checkpoint (Audio-to-caption) occurred in {best_epoch_a2t} th epoch.')
    validate_a2t(test_loader, model, device, L=model.L)

    


def validate(data_loader, model, device, writer, epoch, M):
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
            # audio_embeds, caption_embeds = model(audios, captions)

            tokenized = tokenizer(captions, add_special_tokens=True,padding=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            audio_embeds, caption_embeds = model(audios, input_ids, attention_mask)
            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()
        
        # M = torch.diag(L)
        # M = L
        # evaluate text to audio retrieval
        # r1, r5, r10, r50, medr, meanr = t2a(audio_embs, cap_embs, False,use_ot, use_cosine)
        # r1, r5, r10, r50, medr, meanr, crossentropy_t2a  = t2a_ot(audio_embs, cap_embs, use_ot, use_cosine, M=M)
        r1, r5, r10, r50, medr, meanr  = t2a_ot(audio_embs, cap_embs, M)
        r_sum_t2a = r1 +r5 + r10
        t2a_metrics['r1'] += r1
        t2a_metrics['r5'] += r5
        t2a_metrics['r10'] += r10
        t2a_metrics['median'] += medr
        t2a_metrics['mean'] += meanr
        writer.add_scalar("valid/r1_t2a", r1, epoch)

        # evaluate audio to text retrieval
        # r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(audio_embs, cap_embs, False,use_ot, use_cosine)
        # r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, crossentropy_a2t = a2t_ot(audio_embs, cap_embs, use_ot, use_cosine)
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t_ot(audio_embs, cap_embs, M)
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

def validate_a2t(data_loader, model, device, L):
    val_logger = logger.bind(indent=1)
    model.eval()
    a2t_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}

    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None
        # M = torch.diag(L)
        M=L

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            audios = audios.to(device)

            tokenized = tokenizer(captions, add_special_tokens=True,padding=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            audio_embeds, caption_embeds = model(audios, input_ids, attention_mask)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t_ot(audio_embs, cap_embs, M)
        a2t_metrics['r1'] += r1_a
        a2t_metrics['r5'] += r5_a
        a2t_metrics['r10'] += r10_a
        a2t_metrics['median'] += medr_a
        a2t_metrics['mean'] += meanr_a

        
        val_logger.info('Audio to caption: r1: {:.4f}, r5: {:.4f}, '
                        'r10: {:.4f}, medr: {:.4f}, meanr: {:.4f}'.format(
                        a2t_metrics['r1'], a2t_metrics['r5'], a2t_metrics['r10'], a2t_metrics['median'], a2t_metrics['mean']))
        
def validate_t2a(data_loader, model, device, L):
    val_logger = logger.bind(indent=1)
    model.eval()
    t2a_metrics = {"r1":0, "r5":0, "r10":0, "mean":0, "median":0}

    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None
        M = L
        pos_eigen = L>0
        print("positive eigen: ", torch.sum(pos_eigen))

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            audios = audios.to(device)

            tokenized = tokenizer(captions, add_special_tokens=True,padding=True, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            audio_embeds, caption_embeds = model(audios, input_ids, attention_mask)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        # evaluate text to audio retrieval
        r1, r5, r10, r50, medr, meanr = t2a_ot(audio_embs, cap_embs, M)
        t2a_metrics['r1'] += r1
        t2a_metrics['r5'] += r5
        t2a_metrics['r10'] += r10
        t2a_metrics['median'] += medr
        t2a_metrics['mean'] += meanr
       
        val_logger.info('Caption to audio: r1: {:.4f}, r5: {:.4f}, '
                        'r10: {:.4f}, medr: {:.4f}, meanr: {:.4f}'.format(
                         t2a_metrics['r1'], t2a_metrics['r5'], t2a_metrics['r10'], t2a_metrics['median'], t2a_metrics['mean']))