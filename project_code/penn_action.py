import json
import scipy
import sys
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
import utils as U

skeleton_info = {'labels': ["nose", "right shoulder", "left shoulder", "right elbow", "left elbow", "right wrist", "left wrist", "right hip", "left hip", "right knee", "left knee", "right ankle", "left ankle"], 
                 'edges': [[5, 3, 1, 2, 4, 6], [11, 9, 7, 8, 10, 12]]}
def make_fiftyone_dataset(DATASET_FO_VERSION, dataset_local_dir, dataset_prefix, use_media_onz, force_reload, check_videos_flag):
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
    
    if not fo.dataset_exists(dataset_name) or force_reload:
        print(f"==building FO index==")
        start_time = time.time()
        if fo.dataset_exists(dataset_name):
            input(f"==DO YOU REALLY WANT TO DELETE {dataset_name}?==\n Enter to confirm while Ctrl+C to cancel\n")
            fo.delete_dataset(dataset_name)
            
        # make dataset_raw -- all video dataset
        video_list = sorted([path_+".mp4" for path_ in U.find_folders(os.path.join(raw_media_path, 'frames'), r'\d{4}')])
                
        dataset_raw = fo.Dataset.from_videos(video_list, name=dataset_name)
        fu.load_dataset_info(dataset_raw, raw_media_path, DATASET_FO_VERSION, version_fo2media_map)
        with open(os.path.join(dataset_raw.info['dataset_dir'], 'LGPF', 'open_vocab_phrase.json'),'r') as f:
            dataset_raw.info['open_vocab_phrase_dict'] = json.load(f)
                
        # dataset_videos
        dataset_videos = dataset_raw.skip(0)
        fu.check_and_save_view(dataset_raw, dataset_videos, 'videos');
        
        # split
        fu.make_split(dataset_videos)
        
        # read annotation files
        raw_annotation = read_annotation_files(raw_media_path)

        # load video level annotations
        build_video_ann(dataset_videos, raw_annotation)
        
        # frame level: initialization, make frame info dir, narr
        build_frames(dataset_videos, raw_annotation, check_filepath=True)

        # make videos
        if check_videos_flag:
            make_videos(dataset_videos)
        
        # load step by step
        # frame level: 3D pose
        dataset_clips = dataset_videos.to_clips("narration")
        load_pose_ann(dataset_clips, raw_annotation, return_type='path')
        function_kwargs = {
#            "extract_frames": {"sample_frames": True,
#                               "frames_patt": "%010d/full_scale.jpg",
#                               "verbose": True,
#                               "force_sample": False},
#             "get_pose": {"return_type": "path", "write_fo": False, "force_generate": False, "source": "groundtruth"},
#            "check_completion": {"read_type": "path", "return_type": "path", "write_fo": False, "registration_method": "copy"},
            "forecasting_groundtruth": {"read_type": "path", "return_type": "path", "write_fo": True, "force_generate":True, "skip_generate": False, "narration_field": "narration", "registration_method": "copy"}
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
                                                                                                                                                   return_path=f"{os.path.join(dataset_dir, 'fiftyone', dataset_name, f'get_global_frame_no_to_clip_frame_no_ordered_list_{split}_{socket.gethostname()}.json')}",
                                                                                                                                                   force_generate=True)
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
    
def build_video_ann(dataset, raw_annotation):
    """
    build video level annotations
    args:
        dataset: video dataset
    """
    print(f"loading video level annotations")
    start_time = time.time()
    for video_sample in dataset.iter_samples(progress=True, autosave=True):
        # raw annotation for cur video
        # video_name = f"{int(os.path.splitext(video_sample['filepath'].split('/')[-1])[0]):4d}"
        video_name = os.path.splitext(video_sample['filepath'].split('/')[-1])[0]
        raw_ann_vid = raw_annotation[video_name]

        video_sample['frame_par_dir'] = os.path.splitext(video_sample.filepath)[0]

        # total_frame_count = len(os.path.join(video_sample['frame_par_dir'], '*.jpg'))
        total_frame_count = raw_ann_vid['total_frame_count']
        video_sample.metadata = fo.core.metadata.VideoMetadata(total_frame_count=total_frame_count, frame_width=raw_ann_vid['video_dimensions'][1], frame_height=raw_ann_vid['video_dimensions'][0], frame_rate=30)
        
        video_sample["narration"] = fo.TemporalDetections()
        narration_text = raw_ann_vid['raw_narration']
        video_sample["narration"].detections.append(fo.TemporalDetection(label=narration_text, support=[1, video_sample.metadata.total_frame_count]))
        
        video_sample['pose'] = raw_ann_vid['pose']
        video_sample['video_dimensions'] = raw_ann_vid['video_dimensions']
            
    print(f"==video level loading DONE in {time.time()-start_time:.2f}s==")
    
def build_frames(dataset, raw_annotation, check_filepath):
    # build frames: init fo.Frame() with filepath/frame dirpath
    # dataset: video
    start_time = time.time()
    print(f"==building/initializing frames==")

    for video_sample in dataset.iter_samples(progress=True): # autosave and batch_size doesn't work (save)
        frame_par_dir = video_sample.frame_par_dir # vid path as frame info dir -- the same as default to_frames setting
        total_frame_count = video_sample.metadata.total_frame_count

        # video_name = f"{int(os.path.splitext(video_sample['filepath'].split('/')[-1])[0]):4d}"
        video_name = os.path.splitext(video_sample['filepath'].split('/')[-1])[0]
        raw_ann_vid = raw_annotation[video_name]
        for frame_no in range(1, total_frame_count+1):
            video_sample.frames[frame_no]['frame_info_dir'] = os.path.join(frame_par_dir, f"{frame_no:010d}")
            video_sample.frames[frame_no]['og_filepath'] = os.path.join(frame_par_dir, f"{frame_no:06d}.jpg")
            video_sample.frames[frame_no]['filepath'] = os.path.join(frame_par_dir, f"{frame_no:010d}", "full_scale.jpg")
            video_sample.frames[frame_no]['narration'] = raw_ann_vid['raw_narration']
            video_sample.frames[frame_no]['pose_view'] = raw_ann_vid['pose']

            if check_filepath and os.path.exists(video_sample.frames[frame_no]['filepath']) == False:
                os.makedirs(video_sample.frames[frame_no]['frame_info_dir'], exist_ok=True)
                shutil.copy(video_sample.frames[frame_no]['og_filepath'], video_sample.frames[frame_no]['filepath'])
        video_sample.save()

    print(f"==DONE building/initializing frames in {time.time()-start_time:.2f}s==")
    
def make_videos(dataset):
    frame_path_dataset = dataset.values("frames.filepath")
    for v_i, video_sample in enumerate(dataset.iter_samples(progress=True)):
        # draw video by sewing all frames by ffmpeg
        if not os.path.exists(video_sample.filepath):
            clip = ImageSequenceClip(frame_path_dataset[v_i], fps=30)
            clip.write_videofile(video_sample.filepath)

def load_pose_ann(dataset, raw_annotation, return_type):
    """
    dataset: has to be ClipsView
    """
    print(f"==prepare pose info==")
    start_time = time.time()

    # psoe
    for clip_sample in dataset:
        video_name = os.path.splitext(clip_sample['filepath'].split('/')[-1])[0]
        raw_ann_vid = raw_annotation[video_name]
        for frame_number, frame_sample in clip_sample.frames.items():
            frame_info_dict = raw_ann_vid['frames'][str(frame_number)]

            frame_sample["pose_path"] = frame_sample.filepath.replace("full_scale.jpg", P.pose_file_name)
            if True or os.path.exists(frame_sample["pose_path"]) == False:
                skeleton_frame = frame_info_dict['joint_info']['coord'] # (13 joints, 2D(x_width,y_height))
                norm_skeleton_frame = np.array(skeleton_frame) / np.array([clip_sample.metadata.frame_width, clip_sample.metadata.frame_height])

                skeleton_frame_Keypoints = fo.Keypoints(keypoints=[fo.Keypoint(points=norm_skeleton_frame, label='Human_01', human_confidence=1.0, index=0, visibility=frame_info_dict['joint_info']['visibility'])])
                P.write_Keypoints(skeleton_frame_Keypoints, frame_sample["pose_path"])
            if return_type == "instance":
                frame_sample["pose"] = P.read_Keypoints(frame_sample["pose_path"])
        clip_sample.save()

    print(f"==prepare pose info DONE in {time.time() - start_time:0f}s==")
    return dataset 

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
}

def read_annotation_files(base_dir):
    annotation_parsed = {}

    from scipy.io import loadmat
    annotation_path_list = sorted(glob.glob(os.path.join(base_dir, 'labels', '*.mat')))
    for video_i in range(len(annotation_path_list)):
        # load the i_th annotation
        annotation_path = annotation_path_list[video_i]
        annotation = loadmat(annotation_path)

        video_name = os.path.splitext(annotation_path.split('/')[-1])[0] # e.g., 0001
        annotation_parsed[video_name] = {'total_frame_count': int(annotation['nframes'][0][0]),
                                         'raw_narration': annotation['action'][0], 
                                         'pose': annotation['pose'][0],
                                         'video_dimensions': annotation['dimensions'][0].tolist(), # [H, W, T]
                                         'frames': {},
                                         }
        for frame_i in range(annotation['nframes'][0][0]):
            # annotation_parsed[video_name]['frames'][str(frame_number)] = {}
            frame_info_dict = {}

            # path
            frame_path = os.path.join(base_dir, 'frames', f'{video_i+1:04}', f'{frame_i+1:06}.jpg')
            frame_info_dict['path'] = frame_path

            # joint coords
            frame_info_dict['joint_info'] = {'coord': [], 'visibility': []}
            for j_i in range(len(annotation['visibility'][frame_i])):
                coord = (annotation['x'][frame_i][j_i], annotation['y'][frame_i][j_i])
                visibility = annotation['visibility'][frame_i][j_i]
                frame_info_dict['joint_info']['coord'].append(coord)
                frame_info_dict['joint_info']['visibility'].append(visibility)
            
            # log
            annotation_parsed[video_name]['frames'][str(frame_i+1)] = frame_info_dict
    return annotation_parsed

if __name__ == "__main__": 
    mat = scipy.io.loadmat("/z/dat/PennAction/Penn_Action_media_v000/labels/0001.mat")
    print(f"EOF")
