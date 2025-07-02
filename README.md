# CaRe-Ego
[CaRe-Ego: Contact-aware Relationship Modeling for Egocentric Interactive Hand-object Segmentation](https://arxiv.org/abs/2407.05576)

Yuejiao Su, Yi Wang, and Lap-Pui Chau

[![Logo](https://raw.githubusercontent.com/yuggiehk/CaRe-Ego/17be80be91e4c5cce0a2c2a05fca1510d5722276/assets/arxiv.svg)](https://arxiv.org/abs/2407.05576)
[![Logo](https://raw.githubusercontent.com/yuggiehk/CaRe-Ego/58a2692e8eaf8931b3556d0c56dd86454066c83e/assets/projectpage.svg)](https://yuggiehk.github.io/CaRe-Ego/)




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
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/1.png" alt="My Image" height="400"/>
</div>

Comparison results on the EgoHOS **out-of-domain test set** (left) and **out-of-distribution mini-HOI4D dataset** (right). The mini-HOI4D dataset is derived from the [HOI4D dataset](https://hoi4d.github.io/). You can download mini-HOI4D [here]().
<div align="center">
    <img src="https://github.com/yuggiehk/CaRe-Ego/blob/main/imgs/51.png?raw=true" alt="My Image" height="400"/>
</div>



## Acknowledgements
The research work was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust.

The code of the CaRe-Ego is built upon the [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) codebase, thanks for their work.








