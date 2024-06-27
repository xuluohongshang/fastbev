
#! /bin/bash

export PATH="/mnt/ve_parking/sunlibo/bins/sunlibo/bin/:${PATH}"

# 修改配置参数尽量在配置文件中修改！
TRAIN_CONFIG=configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py
bash tools/ddp_train.sh configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
        "work_dir=./work_dirs/train_debug
        log_config.interval=2
        data.samples_per_gpu=1
        data.workers_per_gpu=1
        data.train.dataset.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_train_4d_interval3_max60.pkl
        data.val.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_val_4d_interval3_max60.pkl
        data.test.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_val_4d_interval3_max60.pkl
        optimizer.type=AdamW"