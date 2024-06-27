import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import json
datasetroot = './data/nuscenes_data/nuscenes'

def get_lidar_from_nuscenes(nuscenes):
    max_f = 0
    min_f = 0
    
    my_scene = nuscenes.scene[0]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nuscenes.get('sample', first_sample_token)
    sensor = 'LIDAR_TOP'
    lidar_top_data = nuscenes.get('sample_data', my_sample['data'][sensor])
    # 获取lidar数据路径
    lidar_file_path = os.path.join(datasetroot,lidar_top_data['filename'])
    # 从lidar的bin文件中读取数据
    pc = LidarPointCloud.from_file(lidar_file_path)
    print(pc.points[:, :])
    print(np.transpose(pc.points[:, :]))
    points = np.transpose(pc.points[:, :])
    for p in points:
        f = p[-1] # 检验I的范围
        if f > max_f:
            max_f = f
        elif f < min_f:
            min_f = f  
    print(f"max_f: {max_f}, min_f: {min_f}")
    
    # 渲染传感器数据可视化图
    sensor_radar = 'RADAR_FRONT'  
    sensor_lidar = 'LIDAR_TOP'
    radar_front_data = nusc.get('sample_data',my_sample['data'][sensor_radar])  
    lidar_front_data = nusc.get('sample_data',my_sample['data'][sensor_lidar])  
    
    nusc.render_sample_data(lidar_front_data['token'],out_path="./data/render_sample_data.png")



def test_nccl_ops():
    num_gpu = 2

    import torch.multiprocessing as mp
    dist_url = "file:///tmp/nccl_tmp_file"
    mp.spawn(_test_nccl_worker, nprocs=num_gpu, args=(num_gpu, dist_url), daemon=False)
    print("NCCL init succeeded.")


def _test_nccl_worker(rank, num_gpu, dist_url):
    import torch.distributed as dist

    dist.init_process_group(backend="NCCL", init_method=dist_url, rank=rank, world_size=num_gpu)
    dist.barrier()
    print("Worker after barrier")


if __name__ == '__main__':
    
    # nusc = NuScenes(version='v1.0-mini', dataroot=datasetroot, verbose=True)
    # get_lidar_from_nuscenes(nusc)
    
    nusc = NuScenes(version='v1.0-mini', dataroot=datasetroot, verbose=False)
    my_scene = nusc.scene[0]
    last_sample_token = my_scene['last_sample_token']  

    sensor_lidar = 'LIDAR_TOP'  
    my_sample = nusc.get('sample', last_sample_token)
    sensor_lidar_data = nusc.get('sample_data',my_sample['data'][sensor_lidar])  
    print(json.dumps(sensor_lidar_data, indent=4))
    # # 获取非关键帧
    # sensor_lidar_last = nusc.get('sample_data',sensor_lidar_data['prev'])  
    # print(json.dumps(sensor_lidar_last, indent=4))

    # test_nccl_ops()