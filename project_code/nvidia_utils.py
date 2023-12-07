import os
import pynvml
import time
from collections import Counter

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def set_cuda_visible_device(criteria="mem+util", repeat_num=5, manual=None):
    """
    criteria: 'mem', 'util', 'mem+util'
    """
    if type(manual) == int:
        use_gpu_id = manual
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu_id)
        print(f"env variable CUDA_VISIBLE_DEVICES is set as {use_gpu_id}")
        
    decided_gpu_id_list = []
    for i in range(repeat_num):
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info_dict = {}
        for device_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # power = pynvml.nvmlDeviceGetPowerUsage(handle)
            mem = int(meminfo.used/meminfo.total * 100)
            gpu_info_dict[device_id] = {'mem': mem, 'util_mem': util.memory, 'util': util.gpu, 'mem+util': mem+util.gpu}
            
        pynvml.nvmlShutdown()
        # print(f"{i=} {gpu_info_dict=}")
        
        # sort the GPU info list by memory usage
        def get_sort_func(criteria):
            return lambda kv: kv[1][criteria]
        least_used_gpu_id = sorted(gpu_info_dict.items(), key=get_sort_func(criteria))[0][0]
        decided_gpu_id_list.append(least_used_gpu_id)
        time.sleep(0.5)
        
    use_gpu_id = most_common(decided_gpu_id_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu_id)
    print(f"env variable CUDA_VISIBLE_DEVICES is set as {use_gpu_id}")
    
if __name__ == "__main__":
    set_cuda_visible_device('mem+util')
    print()
