# CaRe-Ego
[CaRe-Ego: Contact-aware Relationship Modeling for Egocentric Interactive Hand-object Segmentation](https://arxiv.org/abs/2407.05576)

Yuejiao Su, Yi Wang, and Lap-Pui Chau

[![外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传](https://img-home.csdnimg.cn/images/20230724024159.png?origin_url=https%3A%2F%2Fraw.githubusercontent.com%2Fyuggiehk%2FCaRe-Ego%2F17be80be91e4c5cce0a2c2a05fca1510d5722276%2Fassets%2Farxiv.svg&pos_id=img-QRZ8Jfd9-1751439735282)](https://arxiv.org/abs/2407.05576)
[![外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传](https://img-home.csdnimg.cn/images/20230724024159.png?origin_url=https%3A%2F%2Fraw.githubusercontent.com%2Fyuggiehk%2FCaRe-Ego%2F58a2692e8eaf8931b3556d0c56dd86454066c83e%2Fassets%2Fprojectpage.svg&pos_id=img-71KxifUS-1751439735283)](https://yuggiehk.github.io/CaRe-Ego/)




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

Comparison results on the EgoHOS **out-of-domain test set** (left) and **out-of-distribution mini-HOI4D dataset** (right). The mini-HOI4D dataset is derived from the [HOI4D dataset](https://hoi4d.github.io/). You can download mini-HOI4D [here](https://drive.google.com/drive/folders/1HhnGHS67YFAklArkuFd8FjMIWZGsWCog?usp=drive_link).
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/51.png?raw=true" style="max-width: 100%; height: auto;">
</div>


## Setup

### Dataset preparation

#### Training data preparation

The training data is from the [EgoHOS](https://github.com/owenzlz/EgoHOS) dataset. Download it by:
```
gdown --fuzzy https://drive.google.com/file/d/1sk0TVEhZESNF67OW3fz9D5coqpIWkwuK/view?usp=sharing
unzip data.zip
rm -rf data.zip
```
And the EgoHOS dataset will be organized as follows,
```
- [egohos dataset root]
    |- train
        |- image
        |- label
    |- val 
        |- image
        |- label
    |- test_indomain
        |- image
        |- label
    |- test_outdomain
        |- image
        |- label
```
Pre-process the EgoHOS dataset to 1. generate the contact boundary label for each egocentric image, and 2. seperate the hand, left-hand object, right-object, and two-hand object labels. Replace the dataset folder path and then run:
```
# generate the contact boundary
python generate_cb.py
# seperate the hand, left-hand object, right-object, and two-hand object labels
python seperate_label.py
```
Then the EgoHOS dataset will be like:
```
- [egohos dataset root]
    |- train
        |- image
        |- label
        |- label_contact
        |- lbl_obj_left
        |- lbl_obj_right
        |- lbl_obj_two
    |- val 
        |- image
        |- label
        |- label_contact
    |- test_indomain
        |- image
        |- label
        |- label_contact
    |- test_outdomain
        |- image
        |- label
        |- label_contact
```
Note: we only need to seperate the object labels in training set. The final predicted masks can be generated by ORD strategy when inference. So the labels of the val, in-domain test, and out-of-domain test sets are not seperated.
#### Inference data preparation
The inference data contains EgoHOS in-domain, out-of-domain test sets. It also consists mini-HOI4D dataset, which is derived from HOI4D dataset. Download mini-HOI4D dataset [here](https://drive.google.com/drive/folders/1HhnGHS67YFAklArkuFd8FjMIWZGsWCog) and organize it as follows,
```
- [mini-hoi4d dataset root]
     |- image
     |- mask
     |- visualization
```
### Setup

Install the MMSegmentation framework first using their [guidance](https://github.com/open-mmlab/mmsegmentation).

Download this repo:
```
git clone https://github.com/yuggiehk/CaRe-Ego.git
```



## Acknowledgements
The research work was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust.

The code of the CaRe-Ego is built upon the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) codebase, thanks for their work.








