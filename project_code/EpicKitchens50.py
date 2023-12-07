# import torch.backends.cudnn as cudnn
# cudnn.benchmark = False
import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.types as fot
import fiftyone.core.metadata as fm
import fiftyone.utils.torch as fout
import fiftyone.utils.video as fuv
import fiftyone.core.metadata as fom
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from fiftyone import ViewField as FF
import torch.nn as nn

import sys
import torch
import fiftyone as fo
import os
import glob
import json
from fiftyone import ViewField as FF
from PIL import Image
import mediapipe
import numpy as np
import json
import cv2
import time
import utils as U
import multiprocessing
import pose as P
import pandas as pd
import fiftyonize_utils as fu
import socket

match_num_frames_that_have_forecasting_groundtruth_keypoints = FF("frames").filter(FF("forecasting_groundtruth.keypoints").length()>0).length()
match_num_all_frames = FF("frames").length()
match_num_frames_that_have_narration_label = FF("frames").filter(FF("narration.label")!=None).length()

dependency_map = {"006": "003"} 


version_fo2media_map = {
    "001": "001",    
    "003": "001",
    "004": "001",
    "005": "001",
    "006": "001",
    '009': '001'
}

def load_frame_annotations(dataset): 
    # has to be videos view since the clips view to narration field could overlap and hard to organize them at clips level.
    # del:has to be clips view since i) no need to care non annotated segs; ii) no need to deal with gap when set_values()
    print(f"start loading frame narration")
    start_time = time.time()
    frame_narration = []
    for video_sample in dataset.iter_samples(progress=True):
        frame_narration_vid = {frame_no: fo.Classifications() for frame_no in range(1, len(video_sample.frames)+1)}
        detections = sorted(video_sample["narration"].detections, key=lambda x: x.support[0])
        prev_support = [0,0]
        for TD in detections:
            if TD.support[0] <= prev_support[1]:
                # overlap
                for frame_i in range(TD.support[0], prev_support[1]+1):
                    frame_narration_vid[frame_i].classifications.append(fo.Classification(label=TD.label))
                frame_narration_vid.update(dict(zip(range(prev_support[1]+1, TD.support[1]+1), [fo.Classifications(classifications=[fo.Classification(label=TD.label)])] * (TD.support[1] - (prev_support[1]+1) + 1))))
            else:
                # no overlap with prev clip
                frame_narration_vid.update(dict(zip(range(TD.support[0], TD.support[1]+1), [fo.Classifications(classifications=[fo.Classification(label=TD.label)]) for _ in range(TD.support[1] - TD.support[0] + 1)]))) # if use [] * len, all elements have sharable data id in memory so will be changed synchronized
            prev_support = TD.support
        frame_narration.append(list(frame_narration_vid.values()))
    print(f"prepared frame narration in memory in {time.time()-start_time:.0f}s")
    dataset.set_values("frames.narration", frame_narration)
    print(f"DONE loading frame narration to Mongo in {time.time()-start_time:.0f}s")
    return 
    
    # narration
    narration_list = []
    for clip_sample in dataset.iter_samples(progress=True):
        clip_frame_count = clip_sample.support[1] - clip_sample.support[0] + 1
        narration_text = clip_sample.narration.label
        narration_list += [fo.Classification(label=narration_text)] * clip_frame_count
    dataset.set_values("frames.narration", narration_list)
        
    with fo.ProgressBar(iters_str="video") as pb:
        for total_frame_count, narration_tempdet_one_vid in pb(zip(*dataset.values(["metadata.total_frame_count", "narration.detections"]))):
            narr_one_vid = []
            narration_tempdet_one_vid = sorted(narration_tempdet_one_vid, key=lambda x: x.support[0])
            # assumption: no gap in TemporalDetections and sorted 
            for narration in narration_tempdet_one_vid:
                narr_one_vid += [fo.Classification(label=narration["label"])] * (narration["support"][1] - narration["support"][0] + 1)
            if len(narr_one_vid) < total_frame_count:
                narr_one_vid += [fo.Classification()] * (total_frame_count-len(narr_one_vid))
            narration_list.append(narr_one_vid)
    dataset.set_values("frames.narration", narration_list)
    
def build_frames(dataset, make_frame_info_dir):
    # build frames: init fo.Frame() with filepath/frame dirpath
    # dataset: video
    start_time = time.time()
    print(f"==building/initializing frames==")
    
    frame_info_dir_list = [] # #vid->#frames
    filepath_list = []
    # video_frames_list = []
#     with fo.ProgressBar(iters_str="video") as pb:
#         for (v_filepath, total_frame_count) in pb(zip(*dataset.values(["filepath", "metadata.total_frame_count"]))):
    for video_sample in dataset.iter_samples(progress=True): # autosave and batch_size doesn't work/save
        frame_par_dir = video_sample.filepath[:-4] # vid path as frame info dir -- the same as default to_frames setting
        total_frame_count = video_sample.metadata.total_frame_count
        
        # video_sample = dataset[v_filepath]
        video_sample.frames = {frame_no: fo.Frame() for frame_no in range(1, total_frame_count+1)} # for 29152 frames, if access DB one time per frame, 55s;  if access DB once per video, 1(init dict)+9(DB)s
        video_sample.save()
        
        frame_info_dir_list.append([os.path.join(frame_par_dir, f"{frame_no:010d}") for frame_no in range(1, total_frame_count+1)])
        filepath_list.append([os.path.join(frame_info_dir, "full_scale.jpg") for frame_info_dir in frame_info_dir_list[-1]])
    print(f"==scanned all in {time.time()-start_time:.2f}s==")
    print(f"setting values...")
    dataset.set_values("frames.frame_info_dir", frame_info_dir_list)
    dataset.set_values("frames.filepath", filepath_list)
    print(f"making frame_info_dirs")
    if make_frame_info_dir:
       frame_info_dir_list = np.unique(dataset.values("frames.frame_info_dir", unwind=True)).tolist()
       for frame_info_dir in frame_info_dir_list:
           os.makedirs(frame_info_dir, exist_ok=True)    
    print(f"==DONE building/initializing frames in {time.time()-start_time:.2f}s==")
    
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

def build_video_ann(dataset):
    dataset_dir = dataset.info["dataset_dir"]
    
    splits = ["train", "validation"]
    annotation_dir_path = os.path.join(dataset_dir, "epic-kitchens-100-annotations")
    annotation_files = []
    for split in splits:
        annotation_file = pd.read_csv(os.path.join(annotation_dir_path, f"EPIC_100_{split}.csv"))
        annotation_file["split"] = split
        annotation_files.append(annotation_file)
    annotation = pd.concat(annotation_files)
    # filepath as key of each row
    annotation["video_path"] = dataset_dir + "/" + "videos" + "/" + "train" + "/" + annotation["participant_id"] + "/" + annotation["video_id"]+".MP4"
        
    # for videos, each video one TemporalDetections
    for video_sample in dataset.iter_samples(progress=True, autosave=True):
        video_sample["narration"] = fo.TemporalDetections()
        # for split annotations to find ann for this vid
#        for split in splits:
#             for i, narr_ann in annotation[split].iterrows():
#                 # per rwo / narr                
#                 ann_video_path = os.path.join(dataset_dir, "videos", "train", narr_ann["participant_id"], narr_ann["video_id"]+".MP4")
#                 if ann_video_path != video_sample.filepath:
#                     continue
#                else:
        for i, narr_ann in annotation[annotation.video_path==video_sample.filepath].iterrows():
            support = [narr_ann["start_frame"], narr_ann["stop_frame"]]
            narration_text = narr_ann["narration"]
            video_sample["narration"].detections.append(fo.TemporalDetection(label=narration_text, support=support, split=split))
        video_sample["narration"].detections = sorted(video_sample["narration"].detections, key=lambda x: x.support[0])
    return

def make_frame_dataset_from_clips(dataset_clips, frames_dataset):
    """
    This function is for making a frame dataset from a video dataset with clips. The frames will inherit the narration from the clip. One frame sample has one narration only as Classification. Frames covered by multiple clips will be duplicated with different narrations from the clips.
    args:
        dataset_clips: og dataset's clip view
        frames_dataset: A new Frame dataset to be returned after loading all frames from clips
    
    return: a frame dataset (a new Dataset object, not a view of the original dataset)
    """
    
    for clip_sample_no in range(len(dataset_clips)):
        one_clip_dataset = dataset_clips.skip(clip_sample_no).limit(1)
        clip_sample = one_clip_dataset.first()
        
        # to_frames
        frames_in_clip_D = one_clip_dataset.to_frames().clone()
        
        # loade the fields to_frames lost
        clip_text = clip_sample['narration'].label
        frames_in_clip_D.set_values('narration', [fo.Classification(label=clip_text) for i in range(len(frames_in_clip_D))])
        
        # add to frames_dataset
        frames_dataset.add_samples(frames_in_clip_D)
        
    return frames_dataset

def build_blobs(dataset_videos):
    """
    build a fo.Detection field that representing blobs in each video. Blos means a group of frames that are adjacent to each other but not necessarily have the same label. There are always gaps between two blobs.
    This is for easier processing of video frames, e.g., for registration
    """
    print(f"start building blobs")
    start_time = time.time()
    for video_sample in dataset_videos.iter_samples(progress=True, autosave=True):
        range_field_one_video = video_sample["narration"]
        if range_field_one_video == None or len(range_field_one_video.detections)==0:
            continue
        detections = []
        # prev_seg = []
        blob_start = None
        blob_end = None
        for seg_i, one_segment in enumerate(sorted(range_field_one_video.detections, key=lambda x: x.support[0])):
            if blob_start==None: # prev_seg == []:
                # prev_seg = one_segment.support
                blob_start, blob_end = one_segment.support
            elif blob_end >= one_segment.support[0]: # prev_seg[1] >= one_segment.support[0]:
                blob_end = max(blob_end, one_segment.support[1])
            else:
                detections.append(fo.TemporalDetection(support=[blob_start, blob_end]))
                blob_start, blob_end = one_segment.support
                
        # prev_seg = one_segment.support
            
            if seg_i == len(range_field_one_video.detections)-1:                    
                detections.append(fo.TemporalDetection(support=[blob_start, blob_end])) # add the last blob
        video_sample["narration_blob"] = fo.TemporalDetections(detections=detections)
    dataset_blobs = dataset_videos.to_clips("narration_blob")
    print(f"DONE building blobs in {time.time()-start_time:.0f}s")
    return dataset_blobs
                
def make_fiftyone_dataset(DATASET_FO_VERSION, dataset_local_dir, window_size, grid_len, hand_side, landmark_name, scope_dict, visualize=False, use_media_onz=True, force_reload=False, completely_import_vid=True, load_frame_step_importer=False, load_frame_segbox_importer=False, load_frame_info_importer=False, load_from_prev_version=None, load_frame_samples=True, light_mode=True, dataset_prefix="Ego4D"):
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
            
        if DATASET_FO_VERSION == "009":
            dataset_raw = fo.load_dataset("EpicKitchens50_FO_v006").clone('EpicKitchens50_FO_v009', persistent=True)
            dataset_clips = dataset_raw.to_clips("narration")
            fu.set_FGT_indicator(dataset_clips, from_path=True)
            return dataset_raw, dataset_clips, None
        
        # load raw media
        videos_patt = os.path.join(raw_media_path, 'videos', 'train', '*', f"*.MP4")
        dataset_raw = fo.Dataset.from_videos_patt(videos_patt, dataset_name)
        dataset_raw.persistent = True
        dataset_raw.info["dataset_dir"] = raw_media_path
        dataset_raw.info["fo_version"] = DATASET_FO_VERSION
        dataset_raw.info["media_version"] = version_fo2media_map[DATASET_FO_VERSION]
        # dataset level settings
        # dataset_raw.skeletons = {"pose": fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]]), "forecasting_groundtruth": fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])}
        dataset_raw.default_skeleton = fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])
        dataset_raw.save()
        
        # dataset_videos
        dataset_videos = dataset_raw.skip(0) # has to be view to be saved; skip(200).limit(35) # limit(len(dataset_raw))
        # dataset_videos = dataset_raw.select_by("filepath", ["/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_05.MP4"])
        dataset_raw.save_view("videos", dataset_videos)
        if "videos" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("videos")
        dataset_raw.save_view("videos", dataset_videos)
        print(f"{dataset_videos=}")
        dataset_videos.compute_metadata()
        
        # split
        dataset_videos.untag_samples("val")
        dataset_videos.untag_samples("train")
        ratio_val = 0.1
        dataset_videos.take(int(ratio_val * len(dataset_videos)), seed=51).tag_samples("val")
        dataset_videos.match_tags("val", bool=False).tag_samples("train")
        dataset_videos.save()
        
        # load video level annotations
        print(f"loading video level annotations")
        build_video_ann(dataset_videos)
        print(f"==video level loading DONE in {time.time()-start_time:.2f}s==")        
        
        # frame level: initialization, make frame info dir, narr
        print(f"frame level init and narrations")
        build_frames(dataset_videos, make_frame_info_dir=False)
        # load_frame_annotations(dataset_videos) # don't need this for training anymore. but if need it for visualization, debug it. it seems wrong
        
        # dataset_blobs: narrow down.  # dataset_videos = dataset_videos
        print(f"building blobs")
        # dataset_blobs = build_blobs(dataset_videos)
        dataset_videos.set_values('video_support', [fo.TemporalDetection(support=[1, len(vs.frames)]) for vs in dataset_videos.iter_samples('frames')])
        dataset_blobs = dataset_videos.to_clips("video_support")
        print(f"{dataset_blobs=}")
        if "blobs" in dataset_raw.list_saved_views():
            dataset_raw.delete_saved_view("blobs")
        dataset_raw.save_view("blobs", dataset_blobs)    
        
        # load step by step
        print(f"==load frame level samples==")
        function_kwargs = {
#            "extract_frames": {"sample_frames": True,
#                               "frames_patt": "%010d/full_scale.jpg",
#                               "verbose": True,
#                               "force_sample": False},
            "get_pose": {"return_type": "path", "write_fo": False, "force_generate": True},
            "check_completion": {"read_type": "path", "return_type": "path", "write_fo": False, "registration_method": "copy"},
#             "forecasting_groundtruth": {"read_type": "path", "return_type": "path", "write_fo": True, "force_generate":True, "skip_generate": False, "narration_field": "narration", "registration_method": "copy"}
        }
        for function, fargs in function_kwargs.items():
            fargs["dataset"] = dataset_blobs
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
        match_clips_all_frames_has_ann = FF("frames").filter(FF("forecasting_groundtruth_path")!=None).length() == FF("frames").length()
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
            dataset_raw.info[f'global_frame_no_to_clip_frame_no_ordered_list_{split}_path'] = fu.get_global_frame_no_to_clip_frame_no_ordered_list(dataset_clips.match_tags(split), return_path=f"{os.path.join(dataset_dir, 'fiftyone', dataset_name, f'get_global_frame_no_to_clip_frame_no_ordered_list_{split}_{socket.gethostname()}.json')}")
            dataset_raw.save()
                
        
    else:
        print(f"==reading existing FO index==")
        dataset_raw = fo.load_dataset(dataset_name)
        
        dataset_videos = dataset_raw.load_saved_view("videos")
        
        dataset_blobs = dataset_raw.load_saved_view("blobs")
        
        dataset_clips = dataset_raw.load_saved_view("clips")
        
        # # to frames
        # dataset_frames = dataset_clips.to_frames(sample_frames=False,
        #                                           frames_patt="%010d/full_scale.jpg",
        #                                           verbose=True,
        #                                           force_sample=False)
        # if "frames" in dataset_raw.list_saved_views():
        #     dataset_raw.delete_saved_view("frames")
        # dataset_raw.save_view("frames", dataset_frames)
        # print(f"{dataset_frames=}")
                
        
    # visualization
    session = None
    if visualize:
        app_config = fo.app_config.copy()
        app_config.color_by = "label"
        app_config.multicolor_keypoints = True
        app_config.use_frame_number  = True
        session = fo.launch_app(config=app_config)
            
    return dataset_raw, dataset_clips, session



if __name__ == "__main__": 
    dataset_raw, dataset_clips, session = make_fiftyone_dataset(DATASET_FO_VERSION=sys.argv[1], dataset_local_dir=sys.argv[2], window_size=30, grid_len=16, hand_side='right', landmark_name="MEAN_ALL", scope_dict={"recipe_scope": [0], "sensor_scope": ["pv"], "step_scope": {0: [2]}}, force_reload=False, completely_import_vid=True, load_frame_step_importer=False, load_frame_segbox_importer=False, load_from_prev_version=None, load_frame_info_importer=False, load_frame_samples=True, use_media_onz=True, visualize=False, light_mode=True, dataset_prefix="EpicKitchens50")
    if session != None:
        print(f"pending by FO Application...")
        session.wait()
    print(f"EOF")   
    
    
    
    
