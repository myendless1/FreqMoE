#!/bin/bash

## example usage:
# bash scripts/train_2d_modesXexpertsY.sh 0.333 False rand_128 0 1e-3 4 63 100 32 True

export activate_rate=${1:-1.0}
export data_name=${2:-rand_128}
export device_id=${3:-0}
export lr=${4:-1e-4}
export modes=${5:-4}
export target_modes=${6:-32}
export num_epochs=${8:-100}
export width=${9:-64}
export upcycle=${10:-True}

# calculate num_experts as (target_modes / modes) ** 2
if [ $(echo "$target_modes % $modes" | bc) -ne 0 ]; then
    echo "final modes should be divisible by modes"
fi
export num_experts=$(python -c "import math; print(int(math.pow(${target_modes} / ${modes}, 2)))")

echo "modes: ${modes}"
echo "target modes: ${target_modes}"
echo "width: ${width}"
echo "data: ${data_name}"
echo "learning rate: ${lr}"
echo "device id: ${device_id}"
echo "activate rate: ${activate_rate}"
echo "num experts: ${num_experts}"
echo "num epochs: ${num_epochs}"
echo "upcycle: ${upcycle}"

export data_dir="/path/to/your/data/dir"
export work_dir="/path/to/your/moe-fno/"
if [ ! -d "${data_dir}" ]; then
    mkdir -p ${data_dir}
fi

if [ $upcycle = "True" ]; then
    export ckpt_str="--ckpt-path ${work_dir}ckpts/fno_2d_cfd2d_${data_name}_ep${num_epochs}_modes_${modes}_lr${lr}_width${width}/best_model.pt"
    export do_upcycle="_upcycle"
else
    export ckpt_str=""
    export do_upcycle=""
fi

if [ $data_name = "rand_128" ]; then
    export data_path="${data_dir}/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    export stats_path="${work_dir}stats/std_mean_values_2dcfd_rand_m0.1_0.01.txt"
    export cache_dir="${data_dir}_rand_128"
    export task_name="fno_2d_cfd2d_rand_128_ep100_modes_4to32_${activate_rate}"
elif [ $data_name = "rand_512" ]; then
    export data_path="${data_dir}/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    export stats_path="${work_dir}stats/std_mean_values_2dcfd_rand_m0.1_1e-08.txt"
    export cache_dir="${data_dir}_rand_512"
    export task_name="fno_2d_cfd2d_rand_512_ep100_modes_4to32_${activate_rate}"
elif [ $data_name = "turb_512" ]; then
    export data_path="${data_dir}/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    export stats_path="${work_dir}stats/std_mean_values_2dcfd_turb_m0.1_1e-08.txt"
    export cache_dir="${data_dir}_turb_512"
    export task_name="fno_2d_cfd2d_turb_512_ep100_modes_4to32_${activate_rate}"
else
    echo "data_name should be in ['rand_128', 'rand_512', 'turb_512']"
    exit 1
fi
export task_name="fno_2d_${data_name}_ep${num_epochs}_modes_${modes}to_modes_${target_modes}"
export cache_dir="${data_dir}_${data_name}"

if [ ! -d "${data_dir}_${data_name}" ]; then
    cp -r ~/workspace/2025RS/ICML2025/tmp_data/data_${data_name} /workspace/moe-fno/
else
    echo "data already exists: ${data_dir}_${data_name}"
fi

echo "starting tmux session: train_modes${modes}to${target_modes}_${data_name}_${activate_rate}_lr${lr}_width${width}${do_upcycle}"
echo "creating task: fno_2d_cfd2d_${data_name}_ep100_modes_${modes}to${target_modes}_${activate_rate}_lr${lr}_width${width}${do_upcycle}"

export cmd="CUDA_VISIBLE_DEVICES=${device_id} accelerate launch train_gated_freq_MoE_from_dense.py \
    --train-path ${data_path} \
    --test-path ${data_path} \
    --stats-path ${stats_path} \
    --cache-dir ${cache_dir} \
    --batch-size 32 \
    --num-epochs ${num_epochs} \
    --learning-rate ${lr} \
    --modes ${modes} \
    --target-modes ${target_modes}
    --width ${width} \
    --sparsity-weight 0.01 \
    --project-name FreqMoE-20250119 \
    --task-name fno_2d_cfd2d_${data_name}_ep100_modes_${modes}to${target_modes}_${activate_rate}_lr${lr}_width${width}${do_upcycle}\
    --experts-active-ratio-test $activate_rate  ${ckpt_str}"
echo $cmd
tmux new-session -d -s train_modes${modes}to${target_modes}_${data_name}_${activate_rate}_lr${lr}_width${width}${do_upcycle} "$cmd"