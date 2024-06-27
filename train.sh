#! /bin/bash

set -x
set +e

# 假设本地训练方式使用的训练代码（从云效平台clone）为：/mnt/ve_parking/sunlibo/psd/mmdetection3d/
# 加载本地独立的python环境训练，备份的python环境包来自：/mnt/ve_parking/kunkka/bins/mmdet3d.tgz python3.8
# 可以解压上面压缩包mmdet3d.tgz到/mnt/ve_parking/sunlibo/bins/，然后执行：tar -xzf mmdet3d.tgz
# 执行：/mnt/ve_parking/sunlibo/bins/mmdet3d/bin/python3 -m pip list |grep mmdet 可以查看mmdet3d本地包指向的本地环境，
# 如果不对，需要cd /mnt/ve_parking/sunlibo/psd/mmdetection3d/ 使用来重新安装mmdet3d，安装命令为：
# /mnt/ve_parking/sunlibo/bins/mmdet3d/bin/python3 -m pip install -e . 进行本地安装，检查安装可以使用上述方式查看安装包指向
# 运行参数说明：
# $1 代表挂载盘下的用户目录，如sunlibo
# $2 训练加载的离线的mmdetection3d训练框架所在目录
# $3 训练加载的config文件
# $4 训练保存的模型及日志目录


# bash pipeline_preload.sh
# bash pipeline_train.sh


# #不传入train.sh参数时，常规训练
# if [ -z $1 ];
# then
#     # Train for others, comment above entrance.
#     bash pipeline_preload.sh
#     bash pipeline_train.sh
# else
#     # Train for loop only
#     # bash train.sh sunlibo psd hm_psd_vpformer.py
#     bash pipeline_loop.sh $1 $2 $3 loop_psd_bev_ipm2grid240b32_e32_$(date +%s)
# fi



#################################################### multi-run, slb add，一次提交多次训练
set -x
set +e

mkdir -p /training_logs
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
# pip install shapely
# pip install rtree

# # 安装Engine
# sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# apt install curl
# curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# apt install ros-melodic-ros-base
# echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# source ~/.bashrc

# if [ -z "${HM_WORK_HOME}" ];then
#   HM_WORK_HOME=${PWD}
# fi

# mkdir ~/cloud_path
# chmod 700 ~/cloud_path
# cd ${HM_WORK_HOME}
# cp tools/ssh_keys/* ~/cloud_path
# cat ~/cloud_path/id_rsa.pub >> ~/cloud_path/authorized_keys
# chmod 600 ~/cloud_path/authorized_keys
# chmod 600 ~/cloud_path/id_rsa
# touch ~/cloud_path/known_hosts
# chmod 644 ~/cloud_path/known_hosts

# cd ${HM_WORK_HOME}

# echo "pipeline init done"

cd /share/sunlibo/projects/fastbev

mkdir -p /mnt && ln -s /share /mnt/ve_parking
sleep 1s
if [ -z "${HM_WORK_HOME}" ];then
  HM_WORK_HOME=${PWD}
fi
cd ${HM_WORK_HOME}

# python环境及mmdet3d本地包依赖：/mnt/ve_parking/sunlibo/bins/mmdet3d/bin/python3
export PATH="/mnt/ve_parking/sunlibo/bins/sunlibo/bin/:${PATH}"

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

# 修改配置参数尽量在配置文件中修改！
TRAIN_CONFIG=configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py
bash tools/ddp_train.sh ${HM_WORK_HOME}/$TRAIN_CONFIG \
        "work_dir=./work_dirs/train_debug
        log_config.interval=2
        data.samples_per_gpu=1
        data.workers_per_gpu=1
        data.train.dataset.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_train_4d_interval3_max60.pkl
        data.val.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_val_4d_interval3_max60.pkl
        data.test.ann_file=data/nuscenes_v1.0-mini_pkl/nuscenes_infos_val_4d_interval3_max60.pkl
        optimizer.type=AdamW"


# # loop:
# bash pipeline_loop_train.sh sunlibo psd hm_psd_vpformer_grid150.py loop_psd_bev_v2_train0908tos_o12w_h6w_ipm2grid150_8gpu_b32_e32
# bash pipeline_loop_train.sh sunlibo psd hm_psd_vpformer_grid208.py loop_psd_bev_v2_train0908tos_o12w_h6w_ipm2grid208_8gpu_b32_e32
# bash pipeline_loop_train.sh sunlibo psd hm_psd_vpformer_grid240.py loop_psd_bev_v2_train0908tos_o12w_h6w_ipm2grid240_8gpu_b32_e32
