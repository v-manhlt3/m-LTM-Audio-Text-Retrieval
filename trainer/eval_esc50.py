import torch
import webdataset as wds
from models.ASE_model import ASE
# from models.ASE_model_lowrank import ASE
import os
import argparse
import json
from tqdm import tqdm
from data_handling.DataLoader import get_dataloader_esc
from tools.config_loader import get_config
from sentence_transformers import util
import numpy as np
import ot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class_index_dict_path = 'data/ESC50_class_labels_indices_space.json'
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

def eval_zeroshot_audio_event(config):

    test_dataloader = get_dataloader_esc(config)
    model = ASE(config)
    model = model.to(torch.device("cuda"))

    best_ckpt_a2t = torch.load(config.path.resume_model + "/a2t_best_model.pth")
    model.load_state_dict(best_ckpt_a2t['model'])
    model.eval()
    # L = model.L
    # all_text_features = []
    all_audio_features = []
    labels = []
    all_texts = [f"This is a sound of {t}." for t in class_index_dict.keys()]

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
            # audio_embed = model.encode_audio(audio)
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

        # M = torch.diag(L)
        # M = L
        # pairwise_dist = all_audio_features.unsqueeze(1).repeat(1,all_caption_features.size(0),1) - all_caption_features.unsqueeze(0).repeat(all_audio_features.size(0), 1,1)
        # t_pairwise_dist = pairwise_dist.transpose(1,2)
        # M_dist = torch.einsum("ijk,ikj,kk->ij", pairwise_dist.float(), t_pairwise_dist.float(), M.float())
        # M_dist = torch.sqrt(M_dist)

        M_dist = M_dist / M_dist.max()
        d = ot.sinkhorn(a,b,M_dist, reg=0.05, numItermax=10).cpu()
    else:
        d = util.cos_sim(all_audio_features, all_caption_features).squeeze(0).detach().cpu()

    # pos_eigen = L>0
    # print("positive eigen: ", torch.sum(pos_eigen))
    ground_truth = all_class_labels.view(-1, 1)
    ranking = torch.argsort(d, descending=True)
    preds = torch.where(ranking == ground_truth)[1]  # (yusong) this line is slow because it uses single thread
    preds = preds.detach().cpu().numpy()
    print("prediction mean: ", preds.mean())
    visualize_emb(all_caption_features, all_audio_features)
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

def visualize_emb(text_emb, audio_emb):
    tsne = TSNE(n_iter=1000, perplexity=2)

    mean_text = torch.mean(text_emb, dim=0)
    mean_audio = torch.mean(audio_emb, dim=0)
    gap = torch.pow(mean_text- mean_audio, 2).sum().sqrt()
    print("Modality gap: ", gap)

    text_label = torch.zeros(text_emb.size(0))*5
    audio_label = torch.ones(audio_emb.size(0))
    label = torch.concat((text_label, audio_label)).detach().cpu().numpy()
    label_text = ['red' if i==0 else 'blue' for i in label]

    data = torch.concat((text_emb, audio_emb), dim=0).detach().cpu().numpy()
    tsne_emb = tsne.fit_transform(data)
    max_x1 = tsne_emb[:,0].max()
    min_x1 = tsne_emb[:,0].min()
    max_x2 = tsne_emb[:,1].max()
    min_x2 = tsne_emb[:,1].min()

    tsne_emb[:,0] = (tsne_emb[:, 0]-min_x1)/(max_x1- min_x1)
    tsne_emb[:,1] = (tsne_emb[:, 1]-min_x2)/(max_x2- min_x2)

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,title="The shared embedding space")
    scatter = ax.scatter(
        x = tsne_emb[:,0],
        y = tsne_emb[:,1],
        c = label,
        cmap=plt.cm.get_cmap('rainbow'), 
        alpha=1.0,
    )
    print(scatter.legend_elements()[0])
    legend1 = ax.legend(handles=scatter.legend_elements()[0],labels=['text', 'audio'],
                    loc="lower left", title="Classes")
    plt.savefig("tsne.png")



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Setting.")
    parser.add_argument('-c', '--config', default='settings', type=str)

    args = parser.parse_args()
    config = get_config(args.config)
    eval_zeroshot_audio_event(config)