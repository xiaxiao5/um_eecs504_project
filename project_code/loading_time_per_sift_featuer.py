import matplotlib.pyplot as plt
import json
import cv2
import cProfile
from timeit import timeit
import glob
import os
import numpy as np
from functools import partial
import time
from PIL import Image

def _sift_load(local_path):
    start_time = time.time()
    with open(local_path, 'r') as f:
        # Read the JSON object from the file
        json_data = f.read()
    # Convert the JSON object back to the keypoints and descriptors
    keypoints_dict, descriptors = json.loads(json_data)
    # Convert the keypoints back to cv2.KeyPoint objects
    keypoints = tuple([cv2.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'], kp['response'], kp['octave'], kp['class_id'])
                 for kp in keypoints_dict])
    descriptors = np.array(descriptors).astype(np.float32)    
    time_cost = time.time()-start_time
    # print(f"{local_path=}, {len(descriptors)=}, loading_time={time_cost}")
    return keypoints, descriptors, time_cost
    
def _sift_compute(filepath, sifter=cv2.SIFT_create()):
    start_time = time.time()
    with Image.open(filepath) as img:
        frame = np.array(img)
    keypoints, descriptors = sifter.detectAndCompute(frame, None)
    time_cost = time.time()-start_time
    # print(f"{filepath=}, {len(descriptors)=}, loading_time={time_cost}")
    return keypoints, descriptors, time_cost
    
info_dict = {}
total_num_features = 0
total_time = 0
all_frame_folder_path = glob.glob("/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/*/*")
try:
    for frame_folder_path in all_frame_folder_path:
        if ".MP4" in frame_folder_path: 
            continue
        print(frame_folder_path)
        info_dict[frame_folder_path] = {}
        all_feature_pattern = lambda frame_no: os.path.join(frame_folder_path, f"{frame_no:010d}", "sift_feature.json")
        all_file_pattern = lambda frame_no: os.path.join(frame_folder_path, f"{frame_no:010d}", "full_scale.jpg")
        vid_num_frames = len(os.listdir(frame_folder_path))
        vid_total_time = 0
        valid_frame_num = 0
        vid_feature_num = 0
        start_frame_no = vid_num_frames//3 * 1
        end_frame_no = vid_num_frames//3 * 2
        for frame_no in range(start_frame_no, end_frame_no):# :#vid_num_frames+1):
            
            feature_path = all_feature_pattern(frame_no)
            if not os.path.exists(feature_path): 
                continue
            # print(feature_path)
            
    #         def timeit_fun():
    #             _sift_load(feature_path)
            info_dict[frame_folder_path][feature_path] = {}
            for mode in ['load', 'compute']:
                # print(f"===================={mode}====================")
                if mode == 'load':
                    timeit_fun = partial(_sift_load, local_path=feature_path)
                    
                elif mode == 'compute':
                    file_path = all_file_pattern(frame_no)
                    timeit_fun = partial(_sift_compute, filepath=file_path)
                
                #     
                valid_frame_num += 1
                _, ds, time_per_frame = timeit_fun() # _sift_load(feature_path)
                num_features_per_frame = len(ds)
                total_num_features += num_features_per_frame
                vid_feature_num += num_features_per_frame
                
        #         time_per_frame = timeit(timeit_fun, number=1)
                total_time += time_per_frame
                vid_total_time += time_per_frame
                
                info_dict[frame_folder_path][feature_path]["feature_num"] = num_features_per_frame
                info_dict[frame_folder_path][feature_path][f"{mode}_time"] = time_per_frame
                
    #     info_dict[frame_folder_path][mode]["avg_time_by_frame"] = vid_total_time/valid_frame_num if valid_frame_num>0 else None
    #     info_dict[frame_folder_path][mode]["avg_feature_frame"] = vid_feature_num/valid_frame_num if valid_frame_num>0 else None
    #     print(frame_folder_path, info_dict[frame_folder_path][mode]["avg_feature_frame"])
    #     info_dict[frame_folder_path][mode]["avg_time_by_feature"] = vid_total_time/vid_feature_num if vid_feature_num>0 else None
except Exception as e:
    print(e)
finally:    
    # save dict
    with open('/tmp/info_dict.json', 'w') as f:
        json.dump(info_dict, f)
        
# flatten dict
info_list = []
for video in info_dict.values():
    for frame in video.values():
        feature_num = frame['feature_num']
        compute_time = frame['compute_time']
        load_time = frame['load_time']
        info_list.append((feature_num, compute_time, load_time))
info_list.sort(key=lambda x: x[0])

# # darw
# # Unpack the sorted data into separate lists
# feature_nums, compute_times, load_times = zip(*info_list)
# fig, ax1 = plt.subplots()
# # Plot the compute times and load times against the feature numbers
# ax1.plot(feature_nums, compute_times, label='Compute Time')
# ax1.plot(feature_nums, load_times, label='Load Time')
# # Add a legend, labels, and a title
# ax1.legend()
# ax1.set_xlabel('Feature Number')
# ax1.set_ylabel('Time (s)')
# plt.title('Compute and Load Times vs Feature Number')
# # twin object for two different y-axis on the sample plot
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.hist(feature_nums, bins=30, alpha=0.3, color='g')
# ax2.set_ylabel('Frequency')  # we already handled the x-label with ax1
# # Save the figure to disk
# plt.savefig('/tmp/times_vs_feature_num.png')
# # Show the plot
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Sort the data by feature number
info_list.sort(key=lambda x: x[0])

# Unpack the sorted data into separate lists
feature_nums, compute_times, load_times = zip(*info_list)

fig, ax1 = plt.subplots()

# Plot the compute times and load times against the feature numbers
ax1.plot(feature_nums, compute_times, label='Compute Time')
ax1.plot(feature_nums, load_times, label='Load Time')

# Add markers for max, min, mean points
compute_times_np = np.array(compute_times)
load_times_np = np.array(load_times)
feature_nums_np = np.array(feature_nums)

# For compute times
compute_max_idx = np.argmax(compute_times_np)
compute_min_idx = np.argmin(compute_times_np)
compute_mean = np.mean(compute_times_np)

ax1.plot(feature_nums_np[compute_max_idx], compute_times_np[compute_max_idx], 'ro')
ax1.plot(feature_nums_np[compute_min_idx], compute_times_np[compute_min_idx], 'go')
ax1.plot(feature_nums, [compute_mean]*len(feature_nums), 'r--')

# For load times
load_max_idx = np.argmax(load_times_np)
load_min_idx = np.argmin(load_times_np)
load_mean = np.mean(load_times_np)

ax1.plot(feature_nums_np[load_max_idx], load_times_np[load_max_idx], 'bo')
ax1.plot(feature_nums_np[load_min_idx], load_times_np[load_min_idx], 'yo')
ax1.plot(feature_nums, [load_mean]*len(feature_nums), 'b--')

# Set finer granularity for time (s) y-axis
ax1.yaxis.set_major_locator(plt.MaxNLocator(20))

# Set finer granularity for feature number x-axis
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))

# Add a legend, labels, and a title
ax1.legend()
ax1.set_xlabel('Feature Number')
ax1.set_ylabel('Time (s)')
plt.title('Compute and Load Times vs Feature Number')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.hist(feature_nums, bins=30, alpha=0.3, color='g')
ax2.set_ylabel('Frequency')  # we already handled the x-label with ax1

# Save the figure to disk
fig.savefig('/tmp/times_vs_feature_num.png')

# Show the plot
plt.show()

    
print()


