import h5py
import numpy as np
import csv
import os

def load_caption(h5_file):
    f = h5py.File(h5_file, "r")
    final_caps = []
    captions = [caption for caption in f['caption'][:]]
    for caps in captions:
        for cap in caps:
            final_caps.append(cap.decode())
    audio_names = [id.decode() for id in f['audio_name']]

    caps_dict = {ind: v for ind,v in enumerate(final_caps)}
    audios_dict = {ind:v for ind,v in enumerate(audio_names)}
    return caps_dict, audios_dict

# def audioid_to_name(id_matrix, audio_dict):


def get_t2a_ranking(matrix_d, preds, h5_file, folder):
    caps_dict, audios_dict = load_caption(h5_file)
    r1_inds = preds<1
    r5_d = matrix_d[:, :5]
    caps_ind = np.where(preds<1)[0]
    print("caps_inds shape: ", caps_ind.shape)
    with open(os.path.join(folder, 'rank-5.csv'), "w") as f1:
        writer = csv.writer(f1)
        for i in range(r5_d.shape[0]):
            list_audio = []
            # print()
            for aud in r5_d[i]:
                audio_name = audios_dict[aud]
                list_audio.append(audio_name)
            writer.writerow(list_audio)
    with open(os.path.join(folder, 'r1-retrieval-caps.csv'), "w") as f1:
        writer = csv.writer(f1)
        for i in caps_ind:

            writer.writerow((i, caps_dict[i]))

def filter_caps(file1, file2):
    f1 = open(file1, "r")
    f2 = open(file2, "r")
    dict1 =dict()
    dict2 =dict()
    reader1 = csv.reader(f1)
    reader2 = csv.reader(f2)
    for row in reader1:
        dict1[row[0]]=row[1]
    for row in reader2:
        dict2[row[0]]=row[1]
    file1_caps = dict1.keys()
    results = []
    for id,caps in zip(dict2.keys(),dict2.values()):
        if id not in file1_caps:
            results.append((id, caps))

    with open("filter_caps.csv", "w") as fout:
        writer = csv.writer(fout)
        for ele in results:
            writer.writerow(ele)
    print(len(results))
    f1.close()
    f2.close()

if __name__ =="__main__":
    h5_path = "../data/AudioCaps/hdf5s/test/test.h5"
    # caps, auds = load_caption(h5_path)
    # print(len(caps))
    # print(len(auds))
    file1 = "../tools/NTXent-csv/r1-retrieval-caps.csv"
    file2 = "../tools/Maha-csv/r1-retrieval-caps.csv"
    filter_caps(file1, file2)