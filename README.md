# CaRe-Ego
[CaRe-Ego: Contact-aware Relationship Modeling for Egocentric Interactive Hand-object Segmentation](https://arxiv.org/abs/2407.05576)

[![Logo](https://raw.githubusercontent.com/yuggiehk/CaRe-Ego/17be80be91e4c5cce0a2c2a05fca1510d5722276/assets/arxiv.svg)](https://arxiv.org/abs/2407.05576)

Yuejiao Su, Yi Wang, and Lap-Pui Chau

## Abstract
Egocentric Interactive hand-object segmentation (EgoIHOS) requires segmenting hands and interacting objects in egocentric images, which is crucial for understanding human behaviors in assistive systems. Current methods often overlook the essential interactive relationships between hands and objects, or merely establish coarse hand-object associations to recognize targets, leading to suboptimal accuracy. To address this issue, we propose a novel CaRe-Ego method that achieves state-of-the-art performance by emphasizing contact between hands and objects from two aspects. First, to explicitly model hand-object interactive relationships, we introduce a Hand-guided Object Feature Enhancer (HOFE), which utilizes hand features as prior knowledge to extract more contact-relevant and distinguishing object features. Second, to promote the network concentrating on hand-object interactions, we design a Contact-Centric Object Decoupling Strategy (CODS) to reduce interference during training by disentangling the overlapping attributes of the segmentation targets, allowing the model to capture specific contact- aware features associated with each hand. Experiments on various in-domain and out-of-domain test sets show that Care-Ego significantly outperforms existing methods while exhibiting robust generalization capability.

## Method
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/fig_2.png?raw=true" alt="My Image"/>
</div>


## Qualitative Results
Comparison results on the EgoHOS **in-domain test set**.
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/1.png" alt="My Image" height="400"/>
</div>

Comparison results on the EgoHOS **out-of-domain test set**.
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/2.png" alt="My Image" height="400"/>
</div>

Comparison results on the **out-of-distribution** mini-HOI4D dataset. The dataset of mini-HOI4D will be released soon.
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/3.png" alt="My Image" height="400"/>
</div>

## Quantitative Results
Comparison results on the EgoHOS **in-domain test set** measured by IoU/Acc and mIoU/mAcc. 
![1](https://github.com/user-attachments/assets/ff38b294-11af-4046-991c-91110f5b406a)

Comparison results on the EgoHOS **out-of-domain test set** measured by IoU/Acc and mIoU/mAcc. 
![图片_20240718105915](https://github.com/user-attachments/assets/e05bf7e3-5f61-49d4-b4ce-a2038e265d6b)

Comparison results on the **out-of-distribution** mini-HOI4D test set measured by IoU/Acc and mIoU/mAcc. 
![图片_20240718105954](https://github.com/user-attachments/assets/d831c34b-568c-435e-9f1b-7264f13b35a2)

## Video Demonstrations
Although the CaRe-Ego is performed on Egocentric images, we can validate it on **out-of-distribution videos** frame-by-frame.

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



## Acknowledgements
The research work was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust.

The code of the CaRe-Ego is built upon the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) codebase, thanks for their work.








