# IMCNet


This is a PyTorch implementation of our IMCNet for unsupervised video object segmentation.

**Implicit Motion-Compensated Network for Unsupervised Video Object Segmentation**. 
<br>
[<a href="https://arxiv.org/abs/2204.02791" target="_blank">ArXiv</a>] [<a href="https://ieeexplore.ieee.org/document/9751597" target="_blank">TCSVT</a>]

## Prerequisites

Install <a href="https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch" target="_blank">deformable convolution (DCNv2)</a>. The MCM modele presents a feature alignment process based on deformable convolution.
```bash
bash ./models/libs/make.sh
```
Our MCM uses features from the adjacent frames to dynamically predict offsets of sampling convolution kernels (```./models/libs/DCNv2/dcn_conv.py```).

The training and testing experiments are conducted using PyTorch 1.8.1 with a single NVIDIA TITAN RTX GPU with 24GB Memory.

- python 3.8
- pytorch 1.8.1
- torchvision 0.9.1

Other minor Python modules can be installed by running
```bash
pip install opencv-python tqdm tensorboard 
```

## Datasets

- <a href="https://davischallenge.org/davis2017/code.html#unsupervised" target="_blank">DAVIS dataset</a>: We use all the data in the train and validation subset of DAVIS 2016. However, please download DAVIS 2017 (Unsupervised 480p) to fit the code. <a href="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" target="_blank">Download Link</a>

- <a href="https://youtube-vos.org/dataset/" target="_blank">YouTube-VOS dataset</a>: The training set of YouTube-VOS (2019 version) is used to train our IMCNet. A subset of the training set of YouTube-VOS selected 18K frames, which is obtained by sampling images containing a single object per sequence (```./dataloaders/ytvos_train.txt```). We first pre-train our network for 200K iterations on the subset of YouTube-VOS (see Section III.B).

- <a href="http://saliencydetection.net/duts/" target="_blank">DUTS dataset</a>: <a href="http://saliencydetection.net/duts/download/DUTS-TR.zip" target="_blank">DUTS-TR</a> which is the training set of DUTS was used to train our IMCNet with our joint training strategy (see Section II.E in our paper).

- Path configuration: Dataset path settings is in ```./conf/global_settings.py```.
```python
DATASET_CONF = {
    'davis2016': {
        ...,
        db_root_dir = 'path to dataset',
        ...
    },
    'youtubevos2019': {
        ...,
        db_root_dir = 'path to dataset',
        ...
    },
    ...
}
```
In datasets folder:
```text
|--datasets
    |--DAVIS2017
        |--Annotations_unsupervised
            |--480p
        |--ImageSets
            |--2016
        |--JPEGImages
            |--480p
    |--YouTubeVOS
        |--2019
            |--train
                |--Annotations
                |--JPEGImages
    |--DUTS
        |--DUTS-TR
            |--DUTS-TR-Image
            |--DITS-TR-Mask
```

## Train
1. Download the pretrained backbone (ResNet101) from <a href="https://drive.google.com/drive/folders/1-9J_tYTr-8zIvp-wWssQAVmUTtVgxIqo?usp=sharing" target="_blank">Google Drive</a> into ```./checkpoints/pre``` folder.
2. The training process is divided into two stages. **Stage 1**: we first pre-train our network for 200K iterations on a subset of YouTube-VOS. **Stage 2**: we fine-tune the entire network on the training set of DAVIS 2016 and DUTS with our joint training strategy.

- Stage 1:
```bash
bash ./scripts/train_s1.sh
```
- Stage 2:
```bash
bash ./scripts/train_s2.sh
```
## Test
1. Run ```infer.py``` to obtain binary segmentation results.
```bash
bash ./scripts/infer_davis.sh  # DAVIS 2016
bash ./scripts/infer_davis_multi  # DAVIS 2016 with multi-scale inference
bash ./scripts/infer_ytboj.sh  # YouTube-Objects
bash ./scripts/infer_ytboj_multi.sh  # YouTube-Objects with multi-scale inference
```
2. Run post <a href="https://github.com/lucasb-eyer/pydensecrf" target="_blank">CRF processing</a> for results without multi-scale inference.

## Segmentation Results
1. The segmentation result on DAVIS 2016 val can be downloaded from <a href="https://drive.google.com/drive/folders/1-UwVIOezp3wHLD9kSaFmfZXR_5KCgknb?usp=sharing" target="_blank">Google Drive</a>, and multi-scale inference can be downloaded from <a href="https://drive.google.com/drive/folders/1-TZXqlAet8-KHZP701yl60SRTpUkOt94?usp=sharing" target="_blank">Google Drive</a>.
2. The segmentation result on Youtube-Objects can be downloaded from <a href="https://drive.google.com/drive/folders/1-KlmQXNexF6c7wCLOI9V5GBZj4s--p34?usp=sharing" target="_blank">Google Drive</a>, and multi-scale inference can be downloaded from <a href="https://drive.google.com/drive/folders/1-KiRShE5tk08IE9UIA6arp0MAl-vd8Xw?usp=sharing" target="_blank">Google Drive</a>.

## Citation
* Lin Xi, W. Chen, X. Wu, Z. Liu and Z. Li, "Implicit Motion-Compensated Network for Unsupervised Video Object Segmentation," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 9, pp. 6279-6292, Sept. 2022.
```text
@ARTICLE{9751597,
    author={Xi, Lin and Chen, Weihai and Wu, Xingming and Liu, Zhong and Li, Zhengguo},
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={Implicit Motion-Compensated Network for Unsupervised Video Object Segmentation}, 
    year={2022},
    volume={32},
    number={9},
    pages={6279-6292}
}
```
