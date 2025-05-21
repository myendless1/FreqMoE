# FreqMoE: Dynamic Frequency Enhancement for Neural PDE Solvers
The official repository of paper "FreqMoE: Dynamic Frequency Enhancement for Neural PDE Solvers" 
<div align="left">
 <a href='https://arxiv.org/pdf/2505.06858'><img src='https://img.shields.io/badge/arXiv-2505.06858-b31b1b.svg'></a> &nbsp;
 <a href='https://tarpelite.github.io/FreqMoE/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
</div>

## News
+ [2025-04-29]</b> 🎉 Our paper has been accepted by IJCAI 2025!
+ [2025-05-21]</b> 🎉 Training code for Regular Grid PDEs (2DCFD) are open-sourced.


## Introduction

![method](./assets/method.png)
Fourier Neural Operators (FNO) have emerged as promising solutions for efficiently solving partial differential equations (PDEs) by learning infinite-dimensional function mappings through frequency domain transformations. 
However, the sparsity of high-frequency signals limits computational efficiency for high-dimensional inputs, 
and fixed-pattern truncation often causes high-frequency signal loss, reducing performance in scenarios such as high-resolution inputs or long-term predictions. 
To address these challenges, we propose FreqMoE, an efficient and progressive training framework that exploits the dependency of high-frequency signals on low-frequency components. 
The model first learns low-frequency weights and then applies a sparse upward-cycling strategy to construct a mixture of experts (MoE) in the frequency domain, 
effectively extending the learned weights to high-frequency regions. 
Experiments on both regular and irregular grid PDEs demonstrate that FreqMoE achieves up to <b>16.6%</b> accuracy improvement while using merely <b>2.1%</b> parameters (<b>47.32x</b> reduction) compared to dense FNO. 
Furthermore, the approach demonstrates remarkable stability in long-term predictions and generalizes seamlessly to various FNO variants and grid structures, 
establishing a new "<b>L</b>ow frequency <b>P</b>retraining, <b>H</b>igh frequency <b>F</b>ine-tuning" paradigm for solving PDEs.

## Training

This example opensource code uses the [PDEBench](https://github.com/ArashMehrjou/PDEBench) 2DCFD dataset. The original data is in HDF5 format and contains multi-variable fields for fluid dynamics (e.g., velocity, density, pressure).
### Dataset Preparation
#### Data Caching with LMDB

To accelerate training, the dataset is cached into an LMDB database. On the first run, the code automatically converts the HDF5 dataset to LMDB format and caches it. Subsequent runs load data directly from LMDB, greatly improving data loading speed. The cache directory should be set as:

```
/path/to/your/cache/directory
```

No manual operation is needed; the scripts will check and create the cache as required, however, it may consume like 10 mins.

#### Statistics (std/mean) Mechanism

Before training, the mean and standard deviation (std) for each variable are computed for normalization. If the specified stats file does not exist, the code will automatically scan the data and generate it. Precomputed stats files are provided in:

```
stats/std_mean_values_2dcfd_rand_m0.1_0.01.txt
stats/std_mean_values_2dcfd_rand_m0.1_1e-08.txt
stats/std_mean_values_2dcfd_turb_m0.1_1e-08.txt
```

If you use a custom dataset, simply run the training once to generate the stats file automatically.

---

### Two-Stage Training Paradigm

### 1. Low-Frequency Pretraining

First, train the backbone network with low-frequency modes. Use the following core command (replace all paths and task names as needed):

```bash
python train_fno_2d.py \
    --train-path /path/to/your/2D_CFD_data.hdf5 \
    --test-path /path/to/your/2D_CFD_data.hdf5 \
    --stats-path /path/to/your/stats/stats.txt \
    --cache-dir /path/to/your/cache/ \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 1e-3 \
    --modes 4 \
    --width 64 \
    --project-name your-project-name \
    --task-name your-task-name
```

**Key arguments:**
- `--train-path`, `--test-path`: Path to your HDF5 dataset.
- `--stats-path`: Path to the stats file.
- `--cache-dir`: Path to the LMDB cache directory.
- `--modes`: Number of low-frequency modes.
- `--width`: Network width.
- `--task-name`: Custom task name for logging and checkpoints.

### 2. High-Frequency Finetuning

After pretraining, finetune the model with higher-frequency modes, loading the pretrained weights. Use the following core command (replace all paths and task names as needed):

```bash
python train_gated_freq_MoE_from_dense.py \
    --train-path /path/to/your/2D_CFD_data.hdf5 \
    --test-path /path/to/your/2D_CFD_data.hdf5 \
    --stats-path /path/to/your/stats/stats.txt \
    --cache-dir /path/to/your/cache/ \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --modes 4 \
    --width 64 \
    --num-experts 64 \
    --sparsity-weight 0.01 \
    --project-name your-project-name \
    --task-name your-task-name \
    --experts-active-ratio-test 1.0 \
    --ckpt-path /path/to/your/ckpts/your_pretrained_model.pt
```

**Key arguments:**
- `--modes`: Number of low-frequency modes used in pretraining.
- `--num-experts`: Number of experts (typically `(target_modes / modes) ** 2`).
- `--experts-active-ratio-test`: Active expert ratio (e.g., 1.0).
- `--ckpt-path`: Path to the pretrained model checkpoint.
- Other arguments are similar to the pretraining stage.

---

## Additional Notes

- Replace all `/path/to/your/xxx` with your actual paths.
- Training logs and model checkpoints will be saved automatically.
- It is recommended to always perform low-frequency pretraining before high-frequency finetuning for best results.

