# ManipDreamer

## Overview
![overview](./assets/method.png)
Fourier Neural Operators (FNO) have emerged as promising solutions for efficiently solving partial differential equations (PDEs) by learning infinite-dimensional function mappings through frequency domain transformations. 
However, the sparsity of high-frequency signals limits computational efficiency for high-dimensional inputs, 
and fixed-pattern truncation often causes high-frequency signal loss, reducing performance in scenarios such as high-resolution inputs or long-term predictions. 
To address these challenges, we propose FreqMoE, an efficient and progressive training framework that exploits the dependency of high-frequency signals on low-frequency components. 
The model first learns low-frequency weights and then applies a sparse upward-cycling strategy to construct a mixture of experts (MoE) in the frequency domain, 
effectively extending the learned weights to high-frequency regions. 
Experiments on both regular and irregular grid PDEs demonstrate that FreqMoE achieves up to <b>16.6%</b> accuracy improvement while using merely <b>2.1%</b> parameters (<b>47.32x</b> reduction) compared to dense FNO. 
Furthermore, the approach demonstrates remarkable stability in long-term predictions and generalizes seamlessly to various FNO variants and grid structures, 
establishing a new "<b>L</b>ow frequency <b>P</b>retraining, <b>H</b>igh frequency <b>F</b>ine-tuning" paradigm for solving PDEs.