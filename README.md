# Deep Audio-Text Retrieval through the Lens of Transportation

## Setup

- Clone the respository
- Create conde environment with dependencies: ``` conda env create -f environment.yaml -n [env-name]&&conda activate [env-name]```
- Create a pretrained folder: ``` mdkir -p pretrained_models/audio_encoder```
- Go to ```pretrained_models/audio_encoder``` and download the pretrained ResNet38 audio encoder model: ``` gdown https://zenodo.org/records/3987831/files/ResNet38_mAP%3D0.434.pth?download=1 -O ResNet38.pth ```
- Download AudioCaps and Clotho datasets. AudioCaps dataset can be downloaded at [link](https://github.com/XinhaoMei/ACT) and Clotho dataset can be downloaded at [link](https://zenodo.org/record/4783391#.YkRHxTx5_kk).
- Unzip datasets and put wavefiles under ```data/AudioCaps/waveforms``` or ```data/Clotho/waveforms```

## Training
- The training config is in the setting folder ```settings/m-ltm-settings.yaml```
- Set value of dataset parameter in the config file to etheir "AudioCaps" or "Clotho" to train model on AudioCaps or Clotho dataset.
- Run experiments: ```python train.py -n [exp_name] -c m-ltm-settings ```


## Zeroshot evaluation
- Download the test data of ESC50 from the [link](https://drive.google.com/file/d/19Nf52bXquC4t1yTZGJrz5HGor8v2CvJy/view?usp=sharing)
- Run the evaluation: ``` python trainer/eval_esc50.py -c m-ltm-settings -p [pretrained model's folder]```

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

## Acknowledgement
- We use the model and training code from [On Metric Learning for Audio-Text Cross-Modal Retrieval](https://github.com/XinhaoMei/audio-text_retrieval) github with some modifications.