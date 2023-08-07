import torch
import webdataset as wds
from models.ASE_model import ASE
import os
import argparse
import json
from tqdm import tqdm
from data_handling.DataLoader import get_dataloader_esc
from tools.config_loader import get_config
from sentence_transformers import util
import numpy as np
import ot

class_index_dict_path = 'data/ESC50_class_labels_indices_space.json'
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

def eval_zeroshot_audio_event(config):

    test_dataloader = get_dataloader_esc(config)
    model = ASE(config)
    model = model.to(torch.device("cuda"))

    best_ckpt_a2t = torch.load(config.path.resume_model + "/a2t_best_model.pth")
    model.load_state_dict(best_ckpt_a2t['model'])
    model.eval()

    # all_text_features = []
    all_audio_features = []
    labels = []
    all_texts = [f"This is a sound of {t}." for t in class_index_dict.keys()]
    # all_texts = torch.tensor(all_texts).to(torch.device("cuda"))

    _, caption_embeds = model(None, all_texts)
    print(caption_embeds.size())
    all_class_labels = [] 
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            audio, caption,_,_ = batch
            audio = audio.to(torch.device("cuda"))
            print(caption)
            text_label = caption[0].split("of")[-1].strip().replace("the ", "")
            label = class_index_dict[text_label]

            audio_embed, _= model(audio, None)
            labels.append(label)
            all_audio_features.append(audio_embed)
            all_class_labels.append(torch.tensor([label]).long())

    all_class_labels = torch.cat(all_class_labels, dim=0)
    all_audio_features = torch.vstack(all_audio_features)
    all_caption_features = caption_embeds

    if config.training.use_ot:
        a = torch.ones(all_audio_features.size(0))/all_audio_features.size(0)
        b = torch.ones(all_caption_features.size(0))/all_caption_features.size(0)
        a = a.to(torch.device("cuda"))
        b = b.to(torch.device("cuda"))

        M_dist = util.cos_sim(all_audio_features, all_caption_features)
        M_dist = 1 - M_dist
        M_dist = M_dist / M_dist.max()
        d = ot.sinkhorn(a,b,M_dist, reg=0.05, numItermax=10).cpu()
    else:
        d = util.cos_sim(all_audio_features, all_caption_features).squeeze(0).detach().cpu()

    ground_truth = all_class_labels.view(-1, 1)
    ranking = torch.argsort(d, descending=True)
    preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
    preds = preds.detach().cpu().numpy()
    print("prediction mean: ", preds.mean())
    # ind = np.argsort(d)[::-1]
    # list_rank = []
    # for i in range(len(labels)):
    #     rank = ind[i,labels[i]]
    #     list_rank.append(rank)
    # # print(list_rank)
    # ranks = np.array(list_rank)

    # med_rank = np.mean(list_rank)
    ranks = preds
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    print("R1: {}| R5: {}| R10: {}| mAP10: {}".format(r1, r5, r10, mAP10))


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Setting.")
    parser.add_argument('-c', '--config', default='settings', type=str)

    args = parser.parse_args()
    config = get_config(args.config)
    eval_zeroshot_audio_event(config)