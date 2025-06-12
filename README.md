# It’s Not the Target, It’s the Background: Rethinking Infrared Small Target Detection via Deep Patch-Free Low-Rank Representations

This is the official repository for our paper, and we will open-source the code once the paper is accepted.

![Comparison of the proposed LRRNet with existing data-driven methods.](https://github.com/HaLongbao/LRRNet/blob/main/teaser.png)
Comparison of the proposed LRRNet with existing data-driven methods.

This design is based on **a key insight**: the low-rank nature of infrared images implies an inherent compressibility---when the rank is sufficiently small, the image information can be sparsely encoded while being nearly ''losslessly'' preserved. 

To the best of our knowledge, this is the **first** work to directly learn low-rank background structures using deep neural networks in an end-to-end manner.

Extensive experiments on multiple public datasets demonstrate that LRRNet outperforms 38 state-of-the-art methods in terms of detection accuracy, robustness, and computational efficiency. Remarkably, it achieves real-time performance with an average speed of 82.34 FPS. Evaluations on the challenging NoisySIRST dataset further confirm the model’s resilience to sensor noise.

![](https://github.com/HaLongbao/LRRNet/blob/main/visual-v1.png)
Visualization of the LRRNet results on the NoisySIRST dataset under Gaussian white noise interference with a variance of 30.

