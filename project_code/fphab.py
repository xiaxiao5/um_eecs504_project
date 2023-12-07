import json
import os
import glob
import numpy as np
import pose as P
import fiftyone as fo
from fiftyone import ViewField as FF
import fiftyonize_utils as fu
import socket
import time
import shutil
from moviepy.editor import ImageSequenceClip

def make_fiftyone_dataset(DATASET_FO_VERSION, dataset_local_dir, dataset_prefix, use_media_onz, force_reload):
    """
    args:
        dataset_media_name: full name of dataset media. e.g., "MILLYCookbook_media_v000" instead of "000"
        completely_import: boolean. If True: completely load all media and all frame level annotation. It's time-comsuming. If False: load couple of samples with couple of frame level annotations as an example dataset.
        use_media_onz: In the case of have access to /z, if True, no matter what media_version is specified, it will use mdeia_v000 on /z. if False, transfer the speficied media to local and use it. Usually True for developing on zoom while False for computing on lgns
    """
    # locate raw media path
    dataset_name=f"{dataset_prefix}_FO_v{DATASET_FO_VERSION}"
    print(f"{dataset_name=}")
    print(f"==locating raw media==")
    start_time = time.time()
    dataset_media_name = f"{dataset_prefix}_media_v{version_fo2media_map[DATASET_FO_VERSION.split('.')[0]]}" 
    raw_media_path = prepare_raw_media(dataset_local_dir, dataset_media_name, use_media_onz, dataset_prefix)
    print(f"==locating raw media DONE in {time.time()-start_time:.2f}s. {raw_media_path=}==")
    
    # prepare fiftyone dataset
    # print(f"==build FO index==")
    
    if not fo.dataset_exists(dataset_name) or force_reload:
        print(f"==building FO index==")
        start_time = time.time()
        if fo.dataset_exists(dataset_name):
            decision = input(f"==DO YOU REALLY WANT TO DELETE {dataset_name}?==\n Enter to confirm while Ctrl+C to cancel\n")
            fo.delete_dataset(dataset_name)
            
        # make dataset_raw -- all video dataset
        folder_patt = os.path.join(raw_media_path, "Video_files", "*", "*", "*", "color")
        video_list = [path_+".mp4" for path_ in glob.glob(folder_patt)]
##      
        for remove_name in ['Subject_2/close_milk/4', 'Subject_2/put_tea_bag/2', 'Subject_4/flip_sponge/2']:
            remove_path = os.path.join(raw_media_path, "Video_files", remove_name, "color.mp4")
            if remove_path in video_list:
                video_list.remove(remove_path)
                
        dataset_raw = fo.Dataset.from_videos(video_list, name=dataset_name)
        fu.load_dataset_info(dataset_raw, raw_media_path, DATASET_FO_VERSION, version_fo2media_map)
        try:
            with open(os.path.join(dataset_raw.info['dataset_dir'], 'LGPF', 'open_vocab_phrase.json'),'r') as f:
                dataset_raw.info['open_vocab_phrase_dict'] = json.load(f)
        except FileNotFoundError:
            dataset_raw.info['open_vocab_phrase_dict'] = None

        # dataset_videos
        dataset_videos = dataset_raw.skip(0)
        fu.check_and_save_view(dataset_raw, dataset_videos, 'videos');
        
        # split
        fu.make_split(dataset_videos)
        
        # load video level annotations
        build_video_ann(dataset_videos)
        
        # frame level: initialization, make frame info dir, narr
        build_frames(dataset_videos, check_filepath=True)
        
##        
        all_og_filepath, all_filepath = dataset_videos.values(["frames.og_filepath", "frames.filepath"], unwind=True)
        for f_i in range(len(all_og_filepath)):
            if os.path.exists(all_filepath[f_i]) == False:
                shutil.copy(all_og_filepath[f_i], all_filepath[f_i])
            
##     
        # make videos
        make_videos(dataset_videos)
        
        # load step by step
        # frame level: 3D pose
        dataset_clips = dataset_videos.to_clips("narration")
        load_pose_ann(dataset_clips, return_type='path')
        function_kwargs = {
#            "extract_frames": {"sample_frames": True,
#                               "frames_patt": "%010d/full_scale.jpg",
#                               "verbose": True,
#                               "force_sample": False},
#             "get_pose": {"return_type": "path", "write_fo": False, "force_generate": False, "source": "groundtruth"},
#            "check_completion": {"read_type": "path", "return_type": "path", "write_fo": False, "registration_method": "copy"},
            "forecasting_groundtruth": {"read_type": "path", "return_type": "path", "write_fo": True, "force_generate":False, "skip_generate": False, "narration_field": "narration", "registration_method": "copy"}
        }
        for function, fargs in function_kwargs.items():
            fargs["dataset"] = dataset_clips
            fargs["num_workers"] = os.cpu_count()
            fargs["function"] = function
            print(f"{function=} start")
            start_time = time.time()
            process = fu.MyProcess(**fargs)
            process.start_and_join()
            print(f"{function=} done in {time.time()-start_time:.0f}s")
        print(f"==frame level loading DONE in {time.time()-start_time:.2f}s==")
        
        # make clips: check all completion
        match_num_frames_not_zeros = FF("frames").length()>0
        dataset_clips = dataset_videos.to_clips("narration").match(match_num_frames_not_zeros)
        
        # filter: the clips that have all frames with valid forecasting annotation saved on disk
        if function_kwargs["forecasting_groundtruth"]["return_type"] == "path":
            match_clips_all_frames_has_ann = FF("frames").filter(FF("forecasting_groundtruth_path")!=None).length() == FF("frames").length()
        else:
            match_clips_all_frames_has_ann = FF("frames").filter(FF("forecasting_groundtruth")!=None).length() == FF("frames").length()
        clip_ids = dataset_clips.match(match_clips_all_frames_has_ann).values("id")
        dataset_clips = dataset_clips.select(clip_ids)
        if "clips" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("clips")
        dataset_raw.save_view("clips", dataset_clips)
        
        print(f"{len(dataset_clips)=}\n{dataset_clips.count('frames')=}")
        
        # make map mapping from a global frame number to a clip id and frame number in the clip
        print(f"preparing global frame number to (clip, frame number) mapping")
        dataset_dir = dataset_raw.info['dataset_dir']
        for split in ['train', 'val']:
            dataset_raw.info[f'global_frame_no_to_clip_frame_no_ordered_list_{split}'] = None
            dataset_raw.info[f'global_frame_no_to_clip_frame_no_ordered_list_{split}_path'] = fu.get_global_frame_no_to_clip_frame_no_ordered_list(dataset_clips.match_tags(split),
                                                                                                                                                   return_path=f"{os.path.join(dataset_dir, 'fiftyone', dataset_name, f'get_global_frame_no_to_clip_frame_no_ordered_list_{split}_{socket.gethostname()}.json')}")
            dataset_raw.save()
                
        
    else:
        print(f"==reading existing FO index==")
        dataset_raw = fo.load_dataset(dataset_name)
        
        with open(os.path.join(dataset_raw.info['dataset_dir'], 'LGPF', 'open_vocab_phrase.json'),'r') as f:
            dataset_raw.info['open_vocab_phrase_dict'] = json.load(f)
                
        
        dataset_videos = dataset_raw.load_saved_view("videos")
                
        # dataset_blobs = dataset_raw.load_saved_view("blobs")
        
        dataset_clips = dataset_raw.load_saved_view("clips")


            
    return dataset_raw, dataset_clips
    
def build_video_ann(dataset):
    """
    build video level annotations
    args:
        dataset: video dataset
    """
    print(f"loading video level annotations")
    start_time = time.time()
    for video_sample in dataset.iter_samples(progress=True, autosave=True):
        video_sample['frame_par_dir'] = os.path.splitext(video_sample.filepath)[0]
##        
        video_sample.metadata = fo.core.metadata.VideoMetadata(total_frame_count=len(glob.glob(os.path.join(video_sample.filepath[:-4], "*.jpeg"))))
        # video_sample.metadata = fo.core.metadata.VideoMetadata(total_frame_count=len([name for name in os.listdir(folder) if os.path.isdir(os.path.join(video_sample['frame_par_dir'], name))]))      
        
        video_sample["narration"] = fo.TemporalDetections()
        narration_text = video_sample.filepath.split("/")[-3]
        video_sample["narration"].detections.append(fo.TemporalDetection(label=narration_text, support=[1, video_sample.metadata.total_frame_count]))
        
        video_sample['subject'] = video_sample.filepath.split("/")[-4]
        video_sample['seq_idx'] = video_sample.filepath.split("/")[-2]
        if video_sample.filepath.split("/")[-1] == "color.mp4":
            video_sample['sensor'] = 'rgb'
        elif video_sample.filepath.split("/")[-1] == "depth.mp4":
            video_sample['sensor'] = 'depth'
            
    print(f"==video level loading DONE in {time.time()-start_time:.2f}s==")
    
def build_frames(dataset, make_frame_info_dir):
    # build frames: init fo.Frame() with filepath/frame dirpath
    # dataset: video
    start_time = time.time()
    print(f"==building/initializing frames==")
    
    frame_info_dir_list = [] # #vid->#frames
    filepath_list = []
##    
    og_filepath_list = []
    # video_frames_list = []
#     with fo.ProgressBar(iters_str="video") as pb:
#         for (v_filepath, total_frame_count) in pb(zip(*dataset.values(["filepath", "metadata.total_frame_count"]))):
    for video_sample in dataset.iter_samples(progress=True): # autosave and batch_size doesn't work (save)
        frame_par_dir = video_sample.frame_par_dir # vid path as frame info dir -- the same as default to_frames setting
        total_frame_count = video_sample.metadata.total_frame_count
##        
        if video_sample['sensor'] == 'rgb':
            sensor_prefix = 'color'
        elif video_sample['sensor'] == 'depth':
            sensor_prefix = 'depth'
        # og_filepath_list.append([os.path.join(frame_par_dir, f"{sensor_prefix}_{frame_no:04d}.jpeg") for frame_no in range(0, total_frame_count)])
        
        # video_sample.frames = {frame_no: fo.Frame() for frame_no in range(1, total_frame_count+1)}
        # video_sample.frames = {frame_no: fo.Frame(filepath=os.path.join(frame_par_dir, f"{frame_no:010d}", "full_scale.jpg"), frame_info_dir=os.path.join(frame_par_dir, f"{frame_no:010d}"), og_filepath=os.path.join(frame_par_dir, f"{sensor_prefix}_{frame_no:04d}.jpeg")) for frame_no in range(1, total_frame_count+1)}
        for frame_no in range(1, total_frame_count+1):
            video_sample.frames[frame_no]['frame_info_dir'] = os.path.join(frame_par_dir, f"{frame_no:010d}")
            video_sample.frames[frame_no]['filepath'] = os.path.join(frame_par_dir, f"{frame_no:010d}", "full_scale.jpg")
##            
            video_sample.frames[frame_no]['og_filepath'] = os.path.join(frame_par_dir, f"{sensor_prefix}_{frame_no-1:04d}.jpeg")
            
        video_sample.save()
        
        # frame_info_dir_list.append([os.path.join(frame_par_dir, f"{frame_no:010d}") for frame_no in range(1, total_frame_count+1)])
        # filepath_list.append([os.path.join(frame_info_dir, "full_scale.jpg") for frame_info_dir in frame_info_dir_list[-1]])
        
    # print(f"==scanned all in {time.time()-start_time:.2f}s==")
    # print(f"setting values...")
    # dataset.set_values("frames.frame_info_dir", frame_info_dir_list)
    # dataset.set_values("frames.filepath", filepath_list)
##      
 #    # dataset.set_values("frames.og_filepath", og_filepath_list)
    # print(f"making frame_info_dirs")
    
    if make_frame_info_dir:
       frame_info_dir_list = np.unique(dataset.values("frames.frame_info_dir", unwind=True)).tolist()
       for frame_info_dir in frame_info_dir_list:
           os.makedirs(frame_info_dir, exist_ok=True)    
    print(f"==DONE building/initializing frames in {time.time()-start_time:.2f}s==")
    
def make_videos(dataset):
    frame_path_dataset = dataset.values("frames.filepath")
    for v_i, video_sample in enumerate(dataset.iter_samples(progress=True)):
        # draw video by sewing all frames by ffmpeg
        if not os.path.exists(video_sample.filepath):
            clip = ImageSequenceClip(frame_path_dataset[v_i], fps=30)
            clip.write_videofile(video_sample.filepath)
        
from matplotlib import pyplot as plt
from PIL import Image
# Display utilities
def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']
    
    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)

# frame_no = 1
# img_path = clip_sample.frames[frame_no].filepath
# skel_proj = skeleton_video[frame_no]
# # Plot everything
# fig = plt.figure()
# # Load image and display
# ax = fig.add_subplot(221)
# img = Image.open(img_path)
# ax.imshow(img)
# visualize_joints_2d(ax, skel_proj, joint_idxs=False)


def load_pose_ann(dataset, return_type):
    """
    dataset: has to be ClipsView
    """
    print(f"==prepare pose info==")
    start_time = time.time()
    # psoe 
    for clip_sample in dataset:
##
        pose_file = os.path.join('/'.join(os.path.splitext(clip_sample.filepath)[0].replace("Video_files", "Hand_pose_annotation_v1").split("/")[:-1]), "skeleton.txt")
        skeleton_video = read_skeleton_file(pose_file, target_joint_order=dataset.default_skeleton.labels, normalize=True) # (#frames, 21 joints, 2D-x, y). joints are ordered for my protocal
            
        for frame_no, frame_sample in clip_sample.frames.items():
            frame_sample["pose_path"] = frame_sample.filepath.replace("full_scale.jpg", P.pose_file_name)
##
            if os.path.exists(frame_sample["pose_path"]) == False:
                skeleton_frame = skeleton_video[frame_no-1] # (21 joints, 3D)
                skeleton_frame_Keypoints = fo.Keypoints(keypoints=[fo.Keypoint(points=skeleton_frame.tolist(), label='Right', hand_confidence=1.0, index=0)])
                P.write_Keypoints(skeleton_frame_Keypoints, frame_sample["pose_path"])
            if return_type == "instance":
                frame_sample["pose"] = P.read_Keypoints(frame_sample["pose_path"])
        clip_sample.save()
    print(f"==prepare pose info DONE in {time.time() - start_time:0f}s==")
    return dataset 


cam_extr = np.array(
    [[0.999988496304, -0.00468848412856, 0.000982563360594,
      25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
     [-0.000969709653873, 0.00274303671904, 0.99999576807,
      3.902], [0, 0, 0, 1]]) # 4x4
cam_intr = np.array([[1395.749023, 0, 935.732544],
                     [0, 1395.749268, 540.681030], [0, 0, 1]])
# fphab_joint_order = ["Wrist", "TMCP", "IMCP", "MMCP", "RMCP", "PMCP", "TPIP", "TDIP", "TTIP", "IPIP", "IDIP", "ITIP", "MPIP", "MDIP", "MTIP", "RPIP", "RDIP", "RTIP", "PPIP", "PDIP", "PTIP"]
def read_skeleton_file(skeleton_path, target_joint_order, normalize):
    """
    return project 2D skeletons for all frames (#frames, 21 joints, 2D)
    """
    # skeleton_path = os.path.join(skel_root, sample['subject'], sample['action_name'], sample['seq_idx'], 'skeleton.txt')
    # print('Loading skeleton from {}'.format(skeleton_path))
    skeleton_vals = np.loadtxt(skeleton_path) # (#frames, 21 joints * 3D + 1 frame_idx)
    all_skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21, -1) # (#frames, 21 joints, 3D)
    
    # change joint order to match the one in the dataset
    indices = [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]  # indices of fphab joints in the order of mediapipe joints order
    ordered_all_skeleton = all_skeleton[:, indices, :] # (#frames, 21 joints, 3D)
    
    # project to 2D
    # Apply camera extrinsic to hand skeleton
    skel_hom = np.concatenate([ordered_all_skeleton, np.ones([ordered_all_skeleton.shape[0], ordered_all_skeleton.shape[1], 1])], -1) # (#frames, 21 joints, 4D)
    # skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
    skel_camcoords = np.einsum('ijk,lk->ilj', skel_hom, cam_extr).transpose(0,2,1)[:, :, :3].astype(np.float32) # (#frames, 21 joints, 3D)
    skel_hom2d = np.einsum('ijk,lk->ilj', skel_camcoords, cam_intr).transpose(0,2,1) # (#frames, 21 joints, 3D)
    skel_proj = (skel_hom2d / skel_hom2d[:, :, 2:])[:, :, :2] # (#frames, 21 joints, 2D - x, y)
    if normalize:
        skel_proj[:, :, 0] /= 1920
        skel_proj[:, :, 1] /= 1080
    return skel_proj

def prepare_raw_media(dataset_local_dir, dataset_media_name, use_media_onz, dataset_prefix):
    """
    args:
        dataset_media_name: e.g., MILLYCookbook_media_v000
    """
    v000_path = f"/z/dat/{dataset_prefix}/{dataset_prefix}_media_v000"
    if use_media_onz and os.path.exists(v000_path):
        raw_media_path = v000_path
    else:
        # If has access but don't want ot use (computing node), rsync v000 from /z to local
        # If no access to /z, download the specified version to loal
        # raw_media_path = download_dataset(dataset_media_name, dataset_local_dir)
        raw_media_path = dataset_local_dir
    print(f"=={raw_media_path=}==")
    return raw_media_path

version_fo2media_map = {
    "001": "001",    
    "003": "001",
    "004": "001",
    "005": "001",
    "006": "001",
    '009': '001'
}
    
if __name__ == "__main__": 
    print(f"EOF")   
