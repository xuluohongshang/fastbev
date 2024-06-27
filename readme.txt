

mkdir -p data/nuscenes
ln -s /tos://haomo-public/lucas-generation/public_datasets/nuScenes/gts data/nuscenes/gts
ln -s /tos://haomo-public/lucas-generation/public_datasets/nuScenes/maps data/nuscenes/maps
ln -s /tos://haomo-public/lucas-generation/public_datasets/nuScenes/samples data/nuscenes/samples
ln -s /tos://haomo-public/lucas-generation/public_datasets/nuScenes/sweeps data/nuscenes/sweeps
ln -s /tos://haomo-public/lucas-generation/public_datasets/nuScenes/v1.0-trainval data/nuscenes/v1.0-trainval

pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13


创建可用于训练的mini数据集：https://blog.csdn.net/weixin_42108183/article/details/129190315
    1.利用BEVFormer项目生成 nuscenes_infos_temporal_train.pkl 、…test.pkl、 …vak.pkl: 
        python tools/create_data.py nuscenes --root-path ./data/nuscenes_data/nuscenes \
        --out-dir ./data/nuscenes_data/nuscenes_v1.0-mini_pkl \
        --extra-tag nuscenes --version v1.0-mini --canbus ./data/nuscenes_data/ 
    2.BEVFormer下的数据集剪切到FastBEV/data/nuscenes_v1.0-mini_pkl/，并l重命名为nuscenes_infos_train.pkl、nuscenes_infos_val.pkl、nuscenes_infos_test.pkl
    3.运行tools/haomo/nuscenes_seq_converter_mini.py生成 训练所需要的nuscenes_infos_train_4d_interval3_max60.pkl、…val_4d_interval3_max60.pkl…文件:
        python tools/haomo/nuscenes_seq_converter_mini.py

训练调试：
    如果config中有使用SyncBN,则需要改为BN,或通过--cfg-options修改config:
    CUDA_VISIBLE_DEVICES="0" python tools/train.py configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4copy.py --work-dir /mnt/share_disk/sunlibo/debug\
     --cfg-options norm_cfg.type=BN model.backbone_pinhole.norm_cfg.type=BN model.backbone_fisheye.norm_cfg.type=BN model.neck.norm_cfg.type=BN
    
    CUDA_VISIBLE_DEVICES="1" bash tools/ddp_train.sh configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4copy.py \
        "work_dir=./work_dirs/debug runner.max_epochs=6 log_config.interval=2 log_config.interval=10 data.samples_per_gpu=16 data.workers_per_gpu=2 "
    
    ./tools/dist_train.sh configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4copy.py 2

    CUDA_VISIBLE_DEVICES="0,1" ./tools/ddp_train.sh configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4copy.py \
        "work_dir=./work_dirs/debug
        log_config.interval=2
        data.samples_per_gpu=1
        data.workers_per_gpu=1 optimizer.type=AdamW"
    
    CUDA_VISIBLE_DEVICES="0,1" ./tools/ddp_train.sh configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
        "work_dir=./work_dirs/debug
        log_config.interval=2
        data.samples_per_gpu=1
        data.workers_per_gpu=1
        data.train.dataset.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_train_4d_interval3_max60.pkl
        data.val.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_val_4d_interval3_max60.pkl
        data.test.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_val_4d_interval3_max60.pkl
        optimizer.type=AdamW"

评测：
    ./tools/dist_test.sh configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
        work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth 1
    
    仅推理:
    CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --nproc_per_node=2 \
        --master_port=29509 \
        tools/test.py \
        configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
        work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth \
        --launcher pytorch  --eval bbox  --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4


测试软连接： ls data/nuscenes/sweeps/CAM_BACK/n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243137570.jpg

数据集：
/mnt/ve_parking/zhenghaowen/mmdetection3d/data/nuscenes -> /oss://haomo-algorithms/release/algorithms/manual_created_cards/628c5f667844160dda20fcc8/



lucas提交任务进行本地训练：
cd /mnt/ve_parking/sunlibo/projects/fastbev
/mnt/ve_parking/sunlibo/bins/sunlibo/bin/python3 -m pip install -e . 进行本地安装，检查安装可以使用上述方式查看安装包指向
/mnt/ve_parking/sunlibo/bins/sunlibo/bin/python3 -m pip list