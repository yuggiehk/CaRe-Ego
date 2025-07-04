# CaRe-Ego
[CaRe-Ego: Contact-aware Relationship Modeling for Egocentric Interactive Hand-object Segmentation](https://arxiv.org/abs/2407.05576)

Yuejiao Su, Yi Wang, and Lap-Pui Chau

[![](https://raw.githubusercontent.com/yuggiehk/CaRe-Ego/d35a2d3d306f7297090a80c52d48f9f655b23c7f/assets/arxiv.svg)](https://arxiv.org/abs/2407.05576)
[![](https://raw.githubusercontent.com/yuggiehk/CaRe-Ego/d35a2d3d306f7297090a80c52d48f9f655b23c7f/assets/projectpage.svg)](https://yuggiehk.github.io/CaRe-Ego/)




## Abstract
Egocentric Interactive hand-object segmentation (EgoIHOS) requires segmenting hands and interacting objects in egocentric images, which is crucial for understanding human behaviors in assistive systems. Current methods often overlook the essential interactive relationships between hands and objects, or merely establish coarse hand-object associations to recognize targets, leading to suboptimal accuracy. To address this issue, we propose a novel CaRe-Ego method that achieves state-of-the-art performance by emphasizing contact between hands and objects from two aspects. First, to explicitly model hand-object interactive relationships, we introduce a Hand-guided Object Feature Enhancer (HOFE), which utilizes hand features as prior knowledge to extract more contact-relevant and distinguishing object features. Second, to promote the network concentrating on hand-object interactions, we design a Contact-Centric Object Decoupling Strategy (CODS) to reduce interference during training by disentangling the overlapping attributes of the segmentation targets, allowing the model to capture specific contact- aware features associated with each hand. Experiments on various in-domain and out-of-domain test sets show that Care-Ego significantly outperforms existing methods while exhibiting robust generalization capability.

## Method
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/fig_2.png?raw=true" alt="My Image"/>
</div>

## Video Demonstrations
Although the CaRe-Ego is performed on Egocentric images, we can validate it on **out-of-distribution videos** frame-by-frame. We validate the effectiveness of CaRe-Ego on several out-of-distribution videos from the [THU-READ dataset](https://ivg.au.tsinghua.edu.cn/dataset/THU_READ.php).

<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/video1.gif" alt="My Image" />
</div>

<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/video2.gif" alt="My Image" />
</div>

<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/video3.gif" alt="My Image" />
</div>

<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/video4.gif" alt="My Image" />
</div>

## Qualitative Results
Comparison results on the EgoHOS **in-domain test set**.
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/1.png" style="max-width: 100%; height: auto;">
</div>

Comparison results on the EgoHOS **out-of-domain test set** (left) and **out-of-distribution mini-HOI4D dataset** (right). The mini-HOI4D dataset is derived from the [HOI4D dataset](https://hoi4d.github.io/). You can download mini-HOI4D [here](https://drive.google.com/file/d/19byWlLpmm_TrwABlFcbUlQlwlbUX-CZc/view?usp=drive_link).
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/51.png?raw=true" style="max-width: 100%; height: auto;">
</div>


## Setup

### Dataset preparation

The training data is from the [EgoHOS](https://github.com/owenzlz/EgoHOS) dataset. The test dataset consists of two types: EgoHOS dataset and mini-HOI4D dataset. You can download these two datasets through this [link](https://drive.google.com/file/d/19A47SlqjOLw7lJJLhTxWuehXfWrAt9Sw/view?usp=drive_link).

After downloading and unziping the file, the structure of the data folder should be organized as follows,
```
- train
	|- image
	|- label
	|- label_hand
	|- lbl_obj_left
	|- lbl_obj_right
	|- lbl_obj_two
	|- label_contact_first
- test_indomain
	|- image
	|- label
	|- label_hand
	|- lbl_obj_left
	|- lbl_obj_right
	|- lbl_obj_two
	|- label_contact_first
- test_outdomain
	|- image
	|- label
	|- label_hand
	|- lbl_obj_left
	|- lbl_obj_right
	|- lbl_obj_two
	|- label_contact_first
- minihoi4d
	|- image
	|- label
	|- label_hand
	|- lbl_obj_left
	|- lbl_obj_right
	|- lbl_obj_two
	|- label_contact_first
```
### Setup

Create the environment by:
```
conda env create -f mmseg.yml
conda activate mmseg
```
Install the MMSegmentation framework first using their [guidance](https://github.com/open-mmlab/mmsegmentation).
```
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

Download this repo:
```
cd ..
git clone https://github.com/yuggiehk/CaRe-Ego.git
```
Then copy each file in Care-Ego to corresponding mmsegmentation folder (file in CaRe-Ego            --->   MMsegmentation folder):
```
CaRe-Ego/configs/CaRego.py  --->  mmsegmentation/configs/
```
```
CaRe-Ego/datasets/EgoHOS_with_ORD.py ---> mmsegmentation/mmseg/datasets/

# add following command in mmsegmentation/mmseg/datasets/__init__.py
from .EgoHOS_with_ORD import SeperateObjectEgohos
```
```
CaRe-Ego/models/add_data_preprocess.py —> mmsegmentation/mmseg/models/

# add following command in mmsegmentation/mmseg/models/__init__.py
from .add_data_preprocess import SeperateTwoObjDataPreProcessor
```
```
CaRe-Ego/models/segmentors/segmentor.py —> mmsegmentation/mmseg/models/segmentors/

# add following command in mmsegmentation/mmseg/models/segmentors/__init__.py
from .segmentor import CaregoSegmentor
```
```
CaRe-Ego/models/decoder_heads/add_Unet_decoder_output.py ---> mmsegmentation/mmseg/models/decode_heads/
CaRe-Ego/models/decoder_heads/add_Unet_decoder_input.py ---> mmsegmentation/mmseg/models/decode_heads/
CaRe-Ego/models/decoder_heads/add_Unet_decoder_with_seperate_heads_obj.py ---> mmsegmentation/mmseg/models/decode_heads/

# add following command in mmsegmentation/mmseg/models/decode_heads/__init__.py
from .add_Unet_deocder_output import CaregoDecoder
from .add_unet_deocder_input import CaregoDecoder2
from .add_Unet_decoder_with_seperate_heads_obj import CaregoDecoder3
```
```
CaRe-Ego/datasets/transforms/add_transform_with_ORD.py
--> mmsegmentation/mmseg/datasets/transforms/
CaRe-Ego/datasets/transforms/add_transforms_egohos.py --->
mmsegmentation/mmseg/datasets/transforms/

# add following command in mmsegmentation/mmseg/datasets/transforms/__init__.py
from .add_transforms_egohos import LoadMultiLabelImageFromFile
from .add_transform_with_ORD import LoadSeperateTwoObjAnnotation, LabelResizeSeperateTwoObj,RandomSeperateObjectCrop,PackSeperateTwoObjLabelSegInputs, ThreeLabelResizeSeperateTwoobj
```
```
CaRe-Ego/metrics/add_new_seperate_iou.py ---> mmsegmentation/mmseg/evaluation/metrics/add_new_seperate_iou.py
# add following command in mmsegmentation/mmseg/evaluation/metrics/__init__.py
from .add_new_seperate_iou import NewSeperateIou
```
Then run:
```
python setup.py install
```

Download the pretrained model [here](https://drive.google.com/file/d/1e8Te2B_iPB-2tDP445J_MDaMaEcuU0uP/view?usp=drive_link). And replace the 'pretrained' in 'model' in CaRego.py config file with this path.

### Training
Replace the dataset root in config file to your own path and then you can train our model.

If you use one GPU, run:
```
python tools/train.py configs/CaRego.py 
```
If you use multiple GPUs, run:
```
# bash tools/dist_train.sh  config_file number_of_gpus
bash tools/dist_train.sh  configs/CaRego.py 4
```
### Inference
We save our best ckpt in mIoU [here](https://drive.google.com/file/d/1F8QyhSeHaJfS7QLRfMKdgTVKa69JRp-i/view?usp=drive_link).

Download the best ckpt, and perform:
```
# for one GPU
# python tools/test.py config_file checkpoint_path 
python tools/test.py configs/CaRego.py ./best_mIoU_ckpt.pth
```
## Acknowledgements
The research work was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust.

The code of the CaRe-Ego is built upon the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) codebase, thanks for their work.








