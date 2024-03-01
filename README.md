# Deep Audio-Text Retrieval through the Lens of Transportation

## Setup

- Clone the respository
- Create conde environment with dependencies: ``` conda env create -f environment.yaml -n [env-name]&&conda activate [env-name]```
- Create a pretrained folder: ``` mdkir -p pretrained_models/audio_encoder```
- Go to ```pretrained_models/audio_encoder``` and download the pretrained ResNet38 audio encoder model: ``` gdown https://zenodo.org/records/3987831/files/ResNet38_mAP%3D0.434.pth?download=1 -O ResNet38.pth ```
- Download AudioCaps and Clotho datasets. AudioCaps dataset can be downloaded at [link](https://github.com/XinhaoMei/ACT) and Clotho dataset can be downloaded at [link](https://zenodo.org/record/4783391#.YkRHxTx5_kk).
- Unzip datasets and put wavefiles under ```data/AudioCaps/waveforms``` or ```data/Clotho/waveforms```

## Data


## Training
## Zeroshot evaluation

## Cite
```
@inproceedings{
luong2024revisiting,
title={Revisiting Deep Audio-Text Retrieval Through the Lens of Transportation},
author={Manh Luong and Khai Nguyen and Nhat Ho and Reza Haf and Dinh Phung and Lizhen Qu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=l60EM8md3t}
}
```
