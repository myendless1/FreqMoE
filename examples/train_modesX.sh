#!/bin/bash
#usage: bash scripts/train_2d_modesX.sh 4 rand_128 1e-3 0 100 64


export modes=${1-4}
export data_name=${2:-rand_128}
export lr=${3:-1e-4}
export device_id=${4:-0}
export num_epochs=${5:-100}
export width=${6:-64}

echo "modes: ${modes}"
echo "width: ${width}"
echo "data: ${data_name}"
echo "learning rate: ${lr}"
echo "device id: ${device_id}"
echo "num epochs: ${num_epochs}"

export data_dir="/path/to/your/data/dir"
export work_dir="/path/to/your/moe-fno/"
if [ ! -d "${data_dir}" ]; then
    mkdir -p ${data_dir}
fi

if [ $data_name = "rand_128" ]; then
    export data_path="${data_dir}/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
    export stats_path="${work_dir}stats/std_mean_values_2dcfd_rand_m0.1_0.01.txt"
elif [ $data_name = "rand_512" ]; then
    export data_path="${data_dir}/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    export stats_path="${work_dir}stats/std_mean_values_2dcfd_rand_m0.1_1e-08.txt"
elif [ $data_name = "turb_512" ]; then
    export data_path="${data_dir}/2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5"
    export stats_path="${work_dir}stats/std_mean_values_2dcfd_turb_m0.1_1e-08.txt"
else
    echo "data_name should be in ['rand_128', 'rand_512', 'turb_512']"
    exit 1
fi
export task_name="fno_2d_cfd2d_${data_name}_ep${num_epochs}_modes_${modes}_lr${lr}_width${width}"
export cache_dir="${data_dir}_${data_name}"

if [ ! -d "${data_dir}_${data_name}" ]; then
    cp -r ~/workspace/2025RS/ICML2025/tmp_data/data_${data_name} /workspace/moe-fno/
else
    echo "data already exists: ${data_dir}_${data_name}"
fi

echo "starting tmux session: train_modes${modes}_${data_name}_lr${lr}_width${width}"
echo "creating task: fno_2d_cfd2d_${data_name}_ep${num_epochs}_modes_${modes}_lr${lr}_width${width}"

export cmd="CUDA_VISIBLE_DEVICES=${device_id} accelerate launch train_fno_2d.py \
    --train-path ${data_path} \
    --test-path ${data_path} \
    --stats-path ${stats_path} \
    --cache-dir ${cache_dir} \
    --batch-size 32 \
    --num-epochs ${num_epochs} \
    --learning-rate ${lr} \
    --modes ${modes} \
    --width ${width} \
    --project-name FreqMoE-20250119 \
    --task-name fno_2d_cfd2d_${data_name}_ep${num_epochs}_modes_${modes}_lr${lr}_width${width}"
echo ""
echo $cmd
tmux new-session -d -s train_modes${modes}_${data_name}_lr${lr}_width${width} "$cmd"