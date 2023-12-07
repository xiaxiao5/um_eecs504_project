import fiftyone as fo
import time
import multiprocessing
import tqdm
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
import fiftyone.utils.annotations as foua
import wandb
from moviepy.editor import ImageSequenceClip

def build_blobs(dataset_videos, src_field_name):
    """
    build a fo.Detection field that representing blobs in each video. Blos means a group of frames that are adjacent to each other but not necessarily have the same label. There are always gaps between two blobs.
    This is for easier processing of video frames, e.g., for registration
    """
    print(f"start building blobs")
    start_time = time.time()
    for video_sample in dataset_videos.iter_samples(progress=True, autosave=True):
        range_field_one_video = video_sample[src_field_name]
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
        video_sample[f"{src_field_name}_blob"] = fo.TemporalDetections(detections=detections)
    dataset_blobs = dataset_videos.to_clips(f"{src_field_name}_blob")
    print(f"DONE building blobs in {time.time()-start_time:.0f}s")
    return dataset_blobs


class MyProcess(multiprocessing.Process):
    def __init__(self, dataset, num_workers, function, **kwargs):
        super().__init__()
        
        
        self.num_workers = num_workers
        self.dataset = dataset
        self.num_samples = len(self.dataset) # don't touch self.dataset out of lock
        self.function = function
        self.kwargs = kwargs
        field_name = ""
        self.return_type = ""
        # self.end = None
        self.result_list = []
        if self.function == "get_pose":
            field_name = "frames.pose"
            if field_name.split(".")[-1] not in self.dataset.first().frames[1].field_names:
                vs = self.dataset.first()
                vs.frames[1][field_name.split(".")[-1]] = fo.Keypoints()
                vs.save()
            self.worker_function = self.get_pose
            self.return_type = kwargs["return_type"]
            self.write_fo = kwargs["write_fo"]
        elif self.function == "check_completion":
            field_name = "frames.complete_pose"
            if field_name.split(".")[-1] not in self.dataset.first().frames[1].field_names:
                vs = self.dataset.first()
                vs.frames[1][field_name.split(".")[-1]] = fo.Keypoints()
                vs.save()            
            self.worker_function = self.check_completion
            self.registeror = P.SIFT_RANSC()
            self.return_type = kwargs["return_type"]
            self.write_fo = kwargs["write_fo"]
            self.read_type = kwargs["read_type"]
        elif self.function == "forecasting_groundtruth":
            field_name = "frames.forecasting_groundtruth"
            for init_frame_field in ["pose", "forecasting_groundtruth"]:
                if init_frame_field not in self.dataset.first().frames[1].field_names:
                    vs = self.dataset.first()
                    vs.frames[vs.support[0]][init_frame_field] = fo.Keypoints()
                    vs.save()
            self.worker_function = self.make_forecasting_gt
            self.forecasting_steps = 30
            self.registeror = P.SIFT_RANSC()
            self.return_type = kwargs["return_type"]
            self.write_fo = kwargs["write_fo"]
            self.read_type = kwargs["read_type"]
        elif self.function == "extract_frames":
            self.worker_function = self.extract_frames
        elif self.function == "load_TDs_to_frames":
            self.worker_function = self.load_TDs_to_frames
            self.field_name = self.kwargs["field_name"]
        elif self.function == "initialize_frames":
            self.worker_function = self.initialize_frames
        else:
            NotImplementedError(f"{self.function=} not implemented")
            
         
        if self.return_type == "path":
            self.field_name = field_name+"_path"
        elif self.return_type == "instance":
            self.field_name = field_name
        
        # lock to make sure self.dataset is always valid
        self.lock = multiprocessing.Lock()
        
    def start_and_join(self):
        start_time = time.time()
        print(f"Process {self.function} start")
        
        if self.num_workers == 0:
            queue = multiprocessing.Queue()
            
            # Create a list of workers
            for sample_index in range(len(self.dataset)):
                try:
                    self.worker_function(sample_index, queue)
                except Exception as e:
                    print(f"FAILED {sample_index=} got {e=}")
            print(f"processed")    
            # process and get results
            self.result_dict = {idx: [] for idx in range(len(self.dataset))}
            self.finish_flags = []
            self.running_processes = 0
            while len(self.finish_flags) < len(self.dataset):
                self._queue_get(queue, sleep_time=0)
            print(f"===all process done===")
            
        else:
            self.start()
            self.join()
        print(f"Process {self.function} DONE in {time.time()-start_time:.0f}s")
                
    def run(self):
        """A process is finished only when the its finish flag is captured. In this way, we are not only control cpu workload but also queue's.            """
        queue = multiprocessing.Queue()
        
        # process and get results
        num_iters = len(self.dataset)
        self.result_dict = {idx: [] for idx in range(num_iters)}
        self.finish_flags = []
        self.running_processes = 0
        for sample_index in range(len(self.dataset)):
            worker = multiprocessing.Process(target=self.worker_function, args=(sample_index, queue))
            worker.start()
            self.running_processes += 1
            while self.running_processes >= self.num_workers:
                self._queue_get(queue, sleep_time=0)
        print(f"all workers started")
        
        while len(self.finish_flags) < num_iters:
            self._queue_get(queue, sleep_time=0)
        print(f"===all process done===")
        
    def _queue_get(self, queue, sleep_time=0):
        if not queue.empty():
            result = queue.get()
            if type(result) == tuple:
                self.result_dict[result[0]].append(result[1:])
            elif type(result) == int:
                self.finish_flags.append(result)
                self.running_processes -= 1
                print(f"FINISHED sample index {result}/{self.num_samples} and num {len(self.finish_flags)}/{self.num_samples}")
            else:
                raise NotADirectoryError("unexpected result type got")
        time.sleep(sleep_time)
        
    def initialize_frames(self, sample_index, queue):
        """self.dataset is blob: no frame overlap"""
        dataset_one_sample = self.dataset.skip(sample_index).limit(1)
        sample = dataset_one_sample.first()
        sample_frame_num = sample.support[1]-sample.support[0]+1
        frame_par_dir = sample.filepath[:-4] # the video path w/o postfix
        print(f"start {dataset_one_sample.first().id=} {sample_index=} {sample_frame_num=}")
        
        # make frame instance
        sample.frames = {frame_no: fo.Frame() for frame_no in range(sample.support[0], sample.support[1]+1)}
        # frame info dir 
        frame_info_dir_list = [os.path.join(frame_par_dir, f"{sample.support[0]+frame_idx:010d}") for frame_idx in range(sample_frame_num)]
        dataset_one_sample.set_values("frames.frame_info_dir", [frame_info_dir_list], skip_none=True)
        for frame_info_dir in frame_info_dir_list:
            os.makedirs(frame_info_dir, exist_ok=True)
        # filepath
        filepath_list = [os.path.join(frame_info_dir, "full_scale.jpg") for frame_info_dir in frame_info_dir_list]
        dataset_one_sample.set_values("frames.filepath", [filepath_list], skip_none=True)
        
        queue.put(sample_index)
        print(f"done {sample_index=}")
        
    def load_TDs_to_frames_discard(self, sample_index, queue):
        dataset_one_sample = self.dataset.skip(sample_index).limit(1)
        sample = dataset_one_sample.first()
        print(f"start {dataset_one_sample.first().id=} {sample_index=} {len(sample.narration.detections)=}")
        
        list_one_sample = []        
        seg_frame_count = len(sample.frames)
        Tds = sorted(sample[f"{self.field_name}.detections"], key=lambda x: x.support[0])
        for TD in Tds:
            list_one_sample += [fo.Classification(label=TD["label"])] * (TD["support"][1] - TD["support"][0] + 1)
        if len(list_one_sample) < seg_frame_count:
            print(f"this shouldn't be executed if the input dataset is blobed by the targe field name down well")
            list_one_sample += [fo.Classification()] * (seg_frame_count-len(narr_one_vid))
            
        dataset_one_sample.set_values(self.field_name, [list_one_sample], skip_none=True)
        queue.put(sample_index)
        print(f"done {sample_index=}")
        
        
    def extract_frames(self, sample_index, queue):
        self.lock.acquire()
        try:
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            self.lock.release()
        print(f"start {sample_index=} / {self.num_samples=}")
        dataset_one_sample.to_frames(**self.kwargs)
        
        queue.put(sample_index)
        print(f"done {sample_index=}")
    def get_pose(self, sample_index, queue):        
        self.lock.acquire()
        try:
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
            filepath_list, frame_info_dir_list = dataset_one_sample.values(["frames.filepath", "frames.frame_info_dir"], unwind=True)
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            self.lock.release()
            
        print(f"start {len(filepath_list)=} {sample_index=} / {self.num_samples=}")
        pose_list_one_vid = []        
        pose_detector = P.get_pose_detector()
        for frame_index, (filepath, frame_info_dir) in enumerate(zip(filepath_list, frame_info_dir_list)):
            pose_list_one_vid.append(P.get_pose(filepath, frame_info_dir, pose_detector, return_type=self.return_type, force_generate=self.kwargs["force_generate"]))
        if self.write_fo:
            self.lock.acquire()
            try:
                dataset_one_sample = self.check_dataset_view(sample_index, dataset_one_sample)
                dataset_one_sample.set_values(self.field_name, [pose_list_one_vid])
            except Exception as e:
                print(f"error when set_values {sample_index=} {e=}")
            finally:
                self.lock.release()
        pose_detector.close()
        queue.put(sample_index)
        print(f"done {sample_index=}")
        # {self.dataset.summary().split('View stages:')[-1]}
        
    def check_completion(self, sample_index, queue):
        print(f"{sample_index=} acquiring lock")
        self.lock.acquire()
        try:
            print(f"{sample_index=} got lock")
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
#             self.check_dataset(sample_index)
#             dataset_one_sample = self.dataset.skip(sample_index).limit(1)
            filepath_list, frame_info_dir_list, frame_pose_list, frame_number_list = dataset_one_sample.values(["frames.filepath", "frames.frame_info_dir", "frames.pose", "frames.frame_number"], unwind=True)
            sample = dataset_one_sample.first()
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            print(f"{sample_index=} released lock")
            self.lock.release()
            
        print(f"start {len(filepath_list)=} {sample_index=} / {self.num_samples=}")
        if self.read_type=="path":
            # frame_pose_list = [P.read_Keypoints(P.get_local_path(frame_info_dir, P.pose_file_name)) for frame_info_dir in frame_info_dir_list]
            frame_pose_list = [] 
            for ffi, frame_info_dir in enumerate(frame_info_dir_list):
                try:
                    frame_pose_list.append(P.read_Keypoints(P.get_local_path(frame_info_dir, P.pose_file_name)))
                except Exception as e:
                    pose_detector = P.get_pose_detector()                    
                    frame_pose_list.append(P.get_pose(filepath_list[ffi], frame_info_dir_list[ffi], pose_detector, force_generate=True, return_type="instance"))
                    pose_detector.close()
                    print(f"reading local pose error {e=}. redetected")
        elif self.read_type=="instance":
            # frames.pose has the instance
            pass
        
        shape = (sample.metadata.frame_height, sample.metadata.frame_width)
        return_list_one_sample = []
        
        # check first
        # return_list_one_sample.append(P.check_first_frame(frame_pose_list[0]))
        for filepath_idx in range(len(filepath_list)-1):
            # first frame
            if filepath_idx == 0:
                idx_0_check_return = P.check_first_frame(frame_pose_list[0], frame_info_dir_list[filepath_idx], self.return_type)
                if idx_0_check_return != None: # updated
                    if self.return_type=="path":
                        frame_pose_list[filepath_idx] = P.read_Keypoints(P.get_local_path(frame_info_dir_list[filepath_idx], P.complete_pose_file_name))
                    elif self.return_type=="instance":
                        frame_pose_list[filepath_idx] = idx_0_check_return
                    return_list_one_sample.append(idx_0_check_return)
                elif self.return_type=="instance":
                    return_list_one_sample.append(frame_pose_list[filepath_idx+1])
                elif self.return_type=="path":
                    return_list_one_sample.append(filepath_list[filepath_idx+1])
            # rest frames
            check_return = P.check_rest_frames(Keypoints_down=frame_pose_list[filepath_idx+1],
                                                     filepath_up_list=[filepath_list[filepath_idx]],
                                                     filepath_down=filepath_list[filepath_idx+1],
                                                     frame_info_dir_down=frame_info_dir_list[filepath_idx+1],
                                                     Keypoints_up_list=[frame_pose_list[filepath_idx]],
                                                     shape=shape,
                                                     registeror = self.registeror,
                                                     return_type=self.return_type,
                                               registration_method=self.kwargs["registration_method"])
            
            if check_return != None:
                if self.return_type=="path":
                    frame_pose_list[filepath_idx+1] = P.read_Keypoints(P.get_local_path(frame_info_dir_list[filepath_idx+1], P.complete_pose_file_name))
                elif self.return_type=="instance":
                    frame_pose_list[filepath_idx+1] = check_return
                return_list_one_sample.append(check_return)
            elif self.return_type=="instance":
                return_list_one_sample.append(frame_pose_list[filepath_idx+1])
            elif self.return_type=="path":
                return_list_one_sample.append(filepath_list[filepath_idx+1])
        if self.write_fo:
            self.lock.acquire()
            try:
                dataset_one_sample = self.check_dataset_view(sample_index, dataset_one_sample)
                dataset_one_sample.set_values(self.field_name, [return_list_one_sample])
            except Exception as e:
                print(f"error when set_values {sample_index=} {e=}")
            finally:
                self.lock.release()
        queue.put(sample_index)
        print(f"done {sample_index=}")
        
    def make_forecasting_gt(self, sample_index, queue):
        print(f"{sample_index=} acquiring lock")
        self.lock.acquire()
        try:
            print(f"{sample_index=} got lock")
            dataset_one_sample = self.check_dataset_view(sample_index, self.dataset.skip(sample_index).limit(1))
            filepath_list, frame_info_dir_list, frame_Keypoints_list = dataset_one_sample.values(["frames.filepath", "frames.frame_info_dir", "frames.pose"], unwind=True)
            sample = dataset_one_sample.first()
            narration = sample[self.kwargs["narration_field"]].label
            support = sample.support
        except Exception as e:
            print(f"when {sample_index=} working with self.dataset got error {e=}")
        finally:
            print(f"{sample_index=} released lock")
            self.lock.release()
            
        print(f"start {len(filepath_list)=} {sample_index=} / {self.num_samples=}")
        if self.read_type=="path":
            # load pose Keypoints from disk
            frame_Keypoints_list = [] 
            for ffi, frame_info_dir in enumerate(frame_info_dir_list):
                try:
                    frame_Keypoints_list.append(P.read_Keypoints(P.get_local_path(frame_info_dir, P.pose_file_name)))
                except Exception as e:
                    pose_detector = P.get_pose_detector()
                    frame_Keypoints_list.append(P.get_pose(filepath_list[ffi], frame_info_dir_list[ffi], pose_detector, force_generate=True, return_type="instance"))
                    pose_detector.close()
                    print(f"reading local pose error {e=}. redetected")
        elif self.read_type=="instance":
            # frames.pose has the instance loaded in memory
            pass
        
        shape = (sample.metadata.frame_height, sample.metadata.frame_width)
        result_list_one_sample = []
        
        for filepath_idx in range(len(filepath_list)):
            try:
                if filepath_idx % (len(filepath_list)//5) == 0:
                    print(f"frame progress: {sample_index=} {filepath_idx}/{len(filepath_list)}")
            except Exception as e:
                print(f"{e=}")
                print(f"{filepath_idx+1} / {len(filepath_list)}. {filepath_list[0]=}")
            filepath_list_loop = filepath_list[filepath_idx:filepath_idx+self.forecasting_steps+1]
            Keypoints_list_loop = frame_Keypoints_list[filepath_idx:filepath_idx+self.forecasting_steps+1]
            if len(filepath_list_loop) < (self.forecasting_steps+1):
                filepath_list_loop += [filepath_list_loop[-1]] * ((self.forecasting_steps+1)-len(filepath_list_loop)) # note they share the same data in memory. One changed, all changed
                Keypoints_list_loop += [Keypoints_list_loop[-1]] * ((self.forecasting_steps+1)-len(Keypoints_list_loop)) # note they share the same data in memory
            try:
                forecasting_return = P.make_forecasting_groundtruth(
                    filepath_list=filepath_list_loop,
                    Keypoints_list=Keypoints_list_loop,
                    shape=shape,
                    registeror=self.registeror,
                    frame_info_dir = frame_info_dir_list[filepath_idx],
                    force_generate=self.kwargs["force_generate"],
                    return_type=self.return_type,
                    skip_generate=self.kwargs["skip_generate"],
                    narration=narration,
                    support=support,
                    registration_method=self.kwargs["registration_method"],
                )
                result_list_one_sample.append(forecasting_return)
            except Exception as e:
                print(e)
            
        if self.write_fo:
            self.lock.acquire()
            try:
                dataset_one_sample = self.check_dataset_view(sample_index, dataset_one_sample)
                dataset_one_sample.set_values(self.field_name, [result_list_one_sample])
            except Exception as e:
                print(f"error when set_values {sample_index=} {e=}")
            finally:
                self.lock.release()
                
#         dataset_one_sample.set_values(self.field_name, [result_list_one_sample], skip_none=True)
        queue.put(sample_index)
        print(f"done {sample_index=}")
    
    def check_dataset_view(self, sample_index, dataset_view):
        if len(dataset_view) == 0:
            dataset_view.reload()
            print(f"dataset_view reloaded since it was empty view. {sample_index=}")
        return dataset_view

def visualize_forecasting_annotation(dataset_clips, output_dir, narration_field, vis_post_fix, frame_scope, check_registration=None, prediction_path_list=None, num_indicators=2, num_joints=21, num_coords=2):
    """
    args:
        dataset_clips: FO dataset clip view. all clips will be visualized with it's forecasting groundtruth annotation with path in the field "forecasting_groundtruth_path"
        output_dir: str. the directory visualized media media will be placed. The visualized media path will be os.path.join(output_dir, media_filepath). e.g., when output_dir="/z/home/yayuanli/web/tmp/forecasting_GT_v1" and media_path="/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_01/0000003210/full_scale.jpg", the visualized media will be placed at /z/home/yayuanli/web/tmp/forecasting_GT_v1/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_01/0000003210/full_scale.jpg
        narration_field: str. the name of field that saves narration. The field should be fo.Classification type.e.g., "narration", "step". 
        frame_scope: None or a list of str. If None, process all frames. If a list (length must > 0), each str is a all filepath of a frame that will be checked with frame_sample.filepath
        vis_post_fix: str. the post fix of the new names will be used in this function. e.g., new clip dataset name, output video name. It should follow this format: e.g., "gtv1", "gtv2".
    return:
        video_draw_list: a list of video path that are drew
    """
    num_indicators = wandb.config['num_indicators']
    num_joints = wandb.config['num_joints']
    video_draw_list = [] 
    # different index or label or field (name) can lead to different colors; only when index!=None, the name can be shown
    draw_config = foua.DrawConfig(        
        {
            "font_size": 10,
            "per_object_label_colors": True,
            "keypoints_skeleton": {
                "labels": ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], 
                "edges": [[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]]
            },
            "per_keypoints_name_colors": True,
            "per_keypoints_label_colors": True,
            #         "text_color": "#FF0000",
            "show_all_names": True,
            "show_keypoints_names": True,
            "keypoints_edge_linewidth": 1,
            "keypoints_edge_alpha": 1,
            "keypoints_size": 3,
        }
    )    

## get FO Frame Dataset for prediction list
    if prediction_path_list != None:
        print(f"making prediction_path_list as FO Dataset (loaded w/ forecasting Keypoints)")
        prediction_dataset_list = [] 
        for prediction_path in prediction_path_list:
            prediction_dataset = visualize_predictions(prediction_path, to_disk=False, output_dir=None, fps=None)
            prediction_dataset_list.append(prediction_dataset)
        print(f"making... DONE")
        
    # visualize all clips. create one dataset for each clip to avoid memory issue since forecasting_annotation is too large
    get_clip_dataset_name = lambda one_clip_dataset: f"{one_clip_dataset.dataset_name}_clip:{one_clip_dataset.first().id}_{vis_post_fix}"
    
    for clip_no in range(len(dataset_clips)):
        # get one clip dataset from dataset_clips given clip_no
        original_one_clip_dataset = dataset_clips.skip(clip_no).limit(1)
        new_dataset_name = get_clip_dataset_name(original_one_clip_dataset)[-100:]
        one_clip_dataset = original_one_clip_dataset.clone(new_dataset_name)
        # set default_skeleton for pose 
        one_clip_dataset.default_skeleton = fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])
        one_clip_dataset.save()
        # save "clip" view of to one_clip_dataset for easier visualization in browser
        one_clip_dataset.save_view("clips", one_clip_dataset.to_clips("support"))
        
## merge Frame samples from predictions
        if prediction_path_list != None:
            one_clip_dataset_frames = one_clip_dataset.to_frames().clone()
            if len(one_clip_dataset_frames) == 0:
                print(f"NO FRAMES DETECTED IN THIS dataset_clips")
                return []
            for prediction_dataset in prediction_dataset_list:
                one_clip_dataset_frames.merge_samples(prediction_dataset, insert_new=False)
            
        # get clip_sample from one_clip_dataset
        clip_sample = one_clip_dataset.first()
        
        # special treatment for MC dataset due to some naming issue when fiftynizing it
        if "MILLY" in one_clip_dataset.name:
            clip_sample.filepath = clip_sample.filepath.replace("pv.mp4", "pv_fo.mp4")
            clip_sample.save()
        
        # get clip_sample's gt_narration
        gt_narration = clip_sample[narration_field].label
        # if wandb.config['narration_field_name_inf'] in one_clip_dataset_frames.first().field_names and one_clip_dataset_frames.first()[wandb.config['narration_field_name_inf']] != None:
        #     # there is frame level narration field with the name indicated in wandb.config['narration_field_name_inf']
        #     inf_narration = one_clip_dataset_frames.first()[wandb.config['narration_field_name_inf']].label
# else:
        #     # there is no frame level narration field so use the ground truth clip level narration field
        #     inf_narration = gt_narration
        inf_narration = clip_sample[wandb.config['narration_field_name_inf']].label
        clip_post_fix = f"{gt_narration.replace('.', '').replace(' ', '')}_{clip_sample.support[0]}_{clip_sample.support[1]}_{vis_post_fix}_{inf_narration.replace('.','').replace(' ','')}_{clip_sample.tags[0]}"
        
        # check registration correctness:
        if check_registration == True:
            # initialize registration_checker_dict: frame_no->registration_checker
            registration_checker_dict = {}
            timestamp_scope = 30
            
        # load annotation on frame by frame
        print("drawing...")    
        all_frame_draw_path = []        
        for frame_no in clip_sample.frames:
            ##frame_sample
            frame_sample = clip_sample.frames[frame_no]
## pick merged Frame sample
            if prediction_path_list != None:
                frame_sample = one_clip_dataset_frames.match(FF("filepath")==frame_sample.filepath).first()
                
            # if current frame is not the ones I want to visualize, skip
            if frame_scope!= None and frame_sample.filepath not in frame_scope:
                continue
            
            # load forecasting annotation into frame_sample
            gt_fo = P.read_Keypoints(frame_sample["forecasting_groundtruth_path"])
            frame_sample["forecasting_groundtruth"] = set_Keypoints_fields_values(gt_fo, fields_values={'index': None, 'label': "forecasting_groundtruth"}) 
            
            frame_sample["frame_no"] = fo.Classification(label=str(frame_no))
            frame_sample["narration"] = fo.Classification(label=gt_narration)


            # fpred_tensor = P.forecasting_fo_to_torch(frame_sample['forecasting_pred'], num_indicators, num_joints).unsqueeze(0)
            # fgt_tensor = P.forecasting_fo_to_torch(frame_sample["forecasting_groundtruth"], num_indicators, num_joints)[1:].unsqueeze(0)
            # rmse = torch.sqrt(nn.functional.mse_loss(fpred_tensor, fgt_tensor))
            # frame_sample['rmse'] = fo.Classification(label=rmse.item())

            # draw frame
            frame_draw_path = get_vis_output_path(output_dir, original_path=frame_sample.filepath, post_fix=clip_post_fix)
            P.draw_frame(frame_sample, frame_draw_path, draw_config=draw_config)
            # print(f"drew {frame_draw_path}")
            
            # collect frames that are processed for making video 
            all_frame_draw_path.append(frame_draw_path)
            
            # check registration correctness:
            if check_registration == True:
                # initialize checker for current frame
                registration_checker_dict[frame_no] = RegistrationChecker(filepath=frame_sample.filepath, Keypoints_K0=gt_fo)
                
                # check registration correctness for previous frames
                for timestamp in range(timestamp_scope, 0, -1):
                    trace_back_frame_no = frame_no-timestamp
                    if trace_back_frame_no in registration_checker_dict:
                        check_result = registration_checker_dict[trace_back_frame_no].check(timestamp, gt_fo)
                        print(f"{trace_back_frame_no=}; {timestamp=}; {check_result=}")
            
        # draw video by sewing all frames by ffmpeg
        video_draw_path = get_vis_output_path(output_dir, clip_sample.filepath, post_fix=clip_post_fix)
        clip = ImageSequenceClip(all_frame_draw_path, fps=15)
        clip.write_videofile(video_draw_path)
        video_draw_list.append(video_draw_path)
        
        # delete one_clip_dataset to avoid memory issue
        fo.delete_dataset(one_clip_dataset.name)
        
    print(f"DONE")
    return video_draw_list
    
def get_vis_output_path(output_dir, original_path, post_fix):
    """
    1. add output_dir as prefix of original_path
    2. add post_fix before file extension in original_path
    """
    if ".mp4" in original_path:
        type_post_fix = "mp4"
    elif ".MP4" in original_path:
        type_post_fix = "MP4"
    elif ".jpg" in original_path:
        type_post_fix = "jpg"
    new_path = original_path.replace(f".{type_post_fix}", f"_{post_fix}.{type_post_fix}")[1:]
    draw_frame_path = f"{os.path.join(output_dir, new_path)}"
    
    assert draw_frame_path!=original_path, f"draw_frame will overwrite the original frame! {original_path=}"
    os.makedirs(os.path.dirname(draw_frame_path), exist_ok=True)
    if False and os.path.exists(draw_frame_path):
        user_input = int(input(f"path exist: {draw_frame_path}.\n type 1 to overwrite it while 0 to exit program: "))
        if user_input == 1:
            return draw_frame_path
        elif user_input==0:
            exit()
    else:
        return draw_frame_path
    
def set_Keypoints_fields_values(Keypoints, fields_values):
    """
    set the fields in Keypoints to something. This is useful when draw Keypoints since difference in fields index and label will lead to different colors
    fields_values: dict: field name -> value
    """
    for Keypoint in Keypoints.keypoints:
        for field, value in fields_values.items():
            if field in Keypoint.field_names:
                Keypoint[field] = value 
    return Keypoints

    
def nonize_Keypoints_fields(Keypoints, fields):
    """
    set the fields in Keypoints to None. This is useful when draw Keypoints since difference in fields index and label will lead to different colors
    """
    for Keypoint in Keypoints.keypoints:
        for field in fields:
            if field in Keypoint.field_names:
                Keypoint[field] = None
    return Keypoints
    
class RegistrationChecker():
    """
    fucntion: check if the registration is valid: if registered poses are actually registered (so that the head movement is eliminated) instead of just copied from the future poses
    method: compare the registered poses (annotatio) in frame K with the first (`current`) pose in frame K+1, K+2, ... K+T.
    expected behavior: thye should be different
    """
    def __init__(self, filepath, Keypoints_K0):
        self.num_indicators = 2
        self.filepath = filepath
        self.Keypoints_K0_torch = P.forecasting_fo_to_torch(Keypoints_K0, num_indicators=self.num_indicators, num_joints=21)
        
    def check(self, timestamp, Keypoints_Ktimestamp):
        all_coords_registered = self.Keypoints_K0_torch[timestamp]
        all_coords_in_future = P.forecasting_fo_to_torch(Keypoints_Ktimestamp, num_indicators=self.num_indicators, num_joints=21)[0]
        
        return torch.norm(all_coords_registered-all_coords_in_future).item()


def visualize_predictions(prediction_path, output_dir, fps, to_disk, label_path=None, vis_post_fix=None, tags_in_name=None):
    """
    given prediction dict path, darw all frames in there to disk and to a Dataset (persistent=False).
    args:
        prediction_path: (frame_path, prompt) -> tensor with shape (#forecasting step 30,#num_coords 84)
        output_dir: prefix dir to place drew frames and videos
        to_disk: bool. If False, only return the FO.Dataset and not write the frames and video to disk and output_dir, fps won't work.
    output:
        draw frames to disk and make one video from all them
    return:
        video_draw_path: path of the video of all frames have been processed
        dataset_raw: A Frame Dataset with Frame sample loaded predictions
    """
    # different index or label or field (name) can lead to different colors; only when index!=None, the name can be shown
    draw_config = foua.DrawConfig(        
        {
            "font_size": 10,
            "per_object_label_colors": True,
            "keypoints_skeleton": {
                "labels": ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], 
                "edges": [[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]]
            },
            "per_keypoints_name_colors": True,
            "per_keypoints_label_colors": True,
            #         "text_color": "#FF0000",
            "show_all_names": True,
            "show_keypoints_names": True,
            "keypoints_edge_linewidth": 1,
            "keypoints_edge_alpha": 1,
            "keypoints_size": 3,
        }
    )
    prediction_dict = torch.load(prediction_path) # (frame_path, text) -> prediction tensor (forecasting_steps 30, #all_coords 84)
    if label_path is not None:
        label_dict = torch.load(label_path)
        
    # big_exp_name = wandb.config["exp_path"].split("/")[-4] # the exp name in readme
    # small_exp_name = wandb.config["exp_path"].split("/")[-1] # the exp name in outputs folder
    big_exp_name = prediction_path.split("/")[-6] # the exp name in readme
    small_exp_name = prediction_path.split("/")[-3] # the exp name in outputs folder
    # dataset_name = f"{wandb.config['dataset_prefix']}_FO_{big_exp_name}_{small_exp_name}"
    # dataset_raw = fo.Dataset(dataset_name, persistent=False)
    dataset_raw = fo.Dataset()
    
    # load frame by frame
    all_frame_draw_path = []
    for kv in prediction_dict.items():
        # prepare frame sample
        
        # input pair
        try:
            frame_path, narration = kv[0]
        except Exception as e:
            # print(f"error when {kv[0]=} {e=}. SOLVED by set narration='CHECK ANNOTATION'")
            frame_path = kv[0]
            narration = "NARRATION IS NOT SAVED FOR THIS FRAME BUT IT'S THE SAME AS ANNOTATION"
        # poses
        try:
            cur_pose = kv[1]['pose_input']
            forecasted_poses = kv[1]['pose_pred_score']
            rmse = kv[1]['rmse']
            cur_pose_fo = set_Keypoints_fields_values(P.pose_torch_to_fo(cur_pose), fields_values={'index': None, 'label': "cur_pose"})
            print(f"for below frame: {rmse=}")
        except Exception as e:
            cur_pose_fo = fo.Keypoints()
            forecasted_poses = kv[1]            
        if label_path is not None:
            label_narration = narration # TODO: should be another key in label_dict.values
            label_poses = label_dict[kv[0]]['pose_label']
            label_poses_fo, label_trace_fo = P.forecasting_torch_to_fo(label_poses, label_narration, include_cur_pose=False)
            label_poses_fo = set_Keypoints_fields_values(label_poses_fo, fields_values={'index': None, 'label': 'forecasting_groundtruth'})
            label_trace_fo = set_Keypoints_fields_values(label_trace_fo, fields_values={'index': 0, 'label': None})

        # forecasting_field_name = f"forecasting_pred_{big_exp_name}_{small_exp_name}"
        forecasting_field_name = f"forecasting_pred"
        forecasting_trace_field_name = f"forecasting_trace"
        forecasting_text_field_name = f"forecasting_text"
        forecasted_poses_fo, forecasted_trace_fo = P.forecasting_torch_to_fo(forecasted_poses, narration, include_cur_pose=False)
        forecasted_poses_fo = set_Keypoints_fields_values(forecasted_poses_fo, fields_values={'index': None, 'label': forecasting_field_name})
        forecasted_trace_fo = set_Keypoints_fields_values(forecasted_trace_fo, fields_values={'index': 0, 'label': None})
        # frame_no = fdrame_path.split("/")[-2]
        frame_sample = fo.Sample(filepath=frame_path,
                                 file_path=fo.Classification(label=frame_path),
                                 current_pose=cur_pose_fo,
                                 rmse=fo.Classification(label=str(rmse)))
        # frame_sample[f"forecasting_text_{big_exp_name}_{small_exp_name}"] = fo.Classification(label=narration)
        frame_sample[forecasting_text_field_name] = fo.Classification(label=narration)
        frame_sample[forecasting_field_name] = forecasted_poses_fo
        # frame_sample[f"forecasting_trace_{big_exp_name}_{small_exp_name}"] = forecasted_trace_fo
        frame_sample[forecasting_trace_field_name] = forecasted_trace_fo
        
        if label_poses_fo is not None:
            frame_sample["groundtruth_text"] = fo.Classification(label=label_narration)
            frame_sample['groundtruth_forecasting'] = label_poses_fo
            # frame_sample[f"forecasting_trace_{big_exp_name}_{small_exp_name}"] = forecasted_trace_fo
            frame_sample["groundtruth_trace"] = label_trace_fo
        # add frame sample to Dataset
        dataset_raw.add_sample(frame_sample)
        
        if to_disk:
            # draw frame on disk
            post_fix = f"{label_narration.replace('.', '').replace(' ', '')}_{vis_post_fix}_{narration.replace('.','').replace(' ','')}_{tags_in_name}"
            frame_draw_path = get_vis_output_path(output_dir, original_path=frame_sample.filepath, post_fix=post_fix)
            P.draw_frame(frame_sample, frame_draw_path, draw_config=draw_config)
            print(f"{frame_draw_path=}")
            
            # collect frames that are processed for making video
            all_frame_draw_path.append(frame_draw_path)
    if to_disk:
        # draw video by sewing all frames by ffmpeg
        video_draw_path = get_vis_output_path(output_dir, prediction_path+".mp4", post_fix=f"")
        clip = ImageSequenceClip(all_frame_draw_path, fps=fps)
        clip.write_videofile(video_draw_path, fps=fps)
        
        return all_frame_draw_path, video_draw_path, dataset_raw
    else:
        return dataset_raw
    

def get_global_frame_no_to_clip_frame_no_ordered_list(dataset_videoish, return_path=None):
    """
    make the map from global frame no to clip frame no for given whole datset_clips.
    Note that: some clips in EK doesn't even have one frame available so the number of clips in FO dataset may not equal to the number of clips in the returned list.
    args:
        dataset_videoish: fo.Dataset Clip view or video Dataset
    """
    if ((return_path is not None) and (not os.path.exists(return_path))) or return_path == None:
        global_frame_no_to_clip_frame_no_ordered_list = []
        next_global_frame_no = 0 # torch dataloader counts from 0
        for clip_sample in dataset_videoish.iter_samples():
            num_frames = len(clip_sample.frames)
            if 'support'  in clip_sample.field_names:
                support0 = clip_sample.support[0]
            else:
                support0 = 1
            clip_frame_no = list(zip([clip_sample.id]*num_frames, np.arange(support0, support0 + num_frames).tolist()))
            # global_frame_no_to_clip_frame_no_map.update(dict(zip(np.arange(next_global_frame_no, next_global_frame_no + num_frames).tolist(), clip_frame_no)))
            global_frame_no_to_clip_frame_no_ordered_list += clip_frame_no
            next_global_frame_no += num_frames
        
    if return_path is not None:
        if not os.path.exists(return_path):
            # just generated
            os.makedirs(os.path.dirname(return_path), exist_ok=True)
            with open(return_path, "w") as f:
                json.dump(global_frame_no_to_clip_frame_no_ordered_list, f)
        return return_path
    else:
        return global_frame_no_to_clip_frame_no_ordered_list

def read_FGT(dataset, return_fields):
    """
    dataset: a fo.Dataset Clip view or video Dataset with field "frames.forecasting_groundtruth_path"
    returns: a list of clips -> a list of frames -> a list of values (e.g., hand_confidence, Keypoint, etc...)
    """
    print(f"start read_FGT")
    start_time = time.time()
    all_fgt_path = dataset.values("frames.forecasting_groundtruth_path")
    return_fgt_info = []
    for clip_no, fgt_path_a_clip in enumerate(all_fgt_path): # fgt_path_a_clip: list of fgt path (for all frames in a clip)
        if (clip_no+1) % (len(all_fgt_path)//10) == 0:
            print(f"{clip_no=}/{len(all_fgt_path)=}")
        return_fgt_info.append([])
        for fgt_path in fgt_path_a_clip: # fgt_path: a fgt path (for a frame)
            FGT_Keypoints = P.read_Keypoints(fgt_path)
            return_a_frame = [] # return values for one frame. e.g., a list of floats for hand_confidence or a list of Keypoint
            if return_fields == ['keypoints.hand_confidence']:
                for Keypoint in FGT_Keypoints.keypoints:
                    return_a_frame.append(Keypoint.hand_confidence)
            else:
                raise NotImplementedError("")
                
            return_fgt_info[clip_no].append(return_a_frame)
    print(f"done read_FGT in {time.time()-start_time:.0f}s")
    return return_fgt_info


def set_FGT_indicator(dataset, from_path):
    """
    set a field at frame level to indicate if it has confident forecasting groundtruth (FGT) or no. confident GFT means all poses in the forecasting_groundtruth field are initially detected by pose detector (not copied from previou frame or set as default placeholder pose)
    args:
        dataset: a clip dataset (narration clips) has frame level field "frames.forecasting_groundtruth.keypoints.hand_confidence" or "frames.forecasting_groundtruth_path"
    return:
        dataset: a clip dataset (narration clips) being added frame level field "frames.forecasting_groundtruth_indicator"
    """
    
    if from_path:
        all_conf = read_FGT(dataset, return_fields=["keypoints.hand_confidence"])
    else:
        all_conf = dataset.values("frames.forecasting_groundtruth.keypoints.hand_confidence") # dataset.skip(1).limit(1).to_clips("narration")
    print(f"got {len(all_conf)} clips. selecting frames that have good annotation")
    
    forecasting_groundtruth_indicator_frame = []
    forecasting_groundtruth_indicator_clip = []
    all_frames_count = 0
    good_frames_count = 0
    all_clips_count = len(all_conf)
    good_clips_count = 0
    for clip_no, conf_a_clip in enumerate(all_conf):
        forecasting_groundtruth_indicator_frame.append([])
        for frame_no, conf_a_frame in enumerate(conf_a_clip):
            if 0 not in conf_a_frame:
                forecasting_groundtruth_indicator_frame[clip_no].append(1)
                good_frames_count += 1
            else:
                forecasting_groundtruth_indicator_frame[clip_no].append(0)
            all_frames_count += 1
        if 1 in forecasting_groundtruth_indicator_frame[clip_no]:
            good_clips_count += 1
            forecasting_groundtruth_indicator_clip.append(1)
            print(f"{clip_no=} has at least one frame of good annotation")
        else:
            forecasting_groundtruth_indicator_clip.append(0)
    print(f"{good_clips_count/all_clips_count}, {all_clips_count=}")
    print(f"{good_frames_count/all_frames_count}, {all_frames_count=}")    
    
    dataset.set_values("frames.forecasting_groundtruth_indicator", forecasting_groundtruth_indicator_frame)
    dataset.set_values("forecasting_groundtruth_indicator", forecasting_groundtruth_indicator_clip)
    
    return None

def load_dataset_info(dataset, raw_media_path, DATASET_FO_VERSION, version_fo2media_map):
    """
    args:
        dataset: a fo.Dataset or a Dataset view
    """
    dataset.persistent = True
    dataset.info["dataset_dir"] = raw_media_path
    dataset.info["fo_version"] = DATASET_FO_VERSION
    dataset.info["media_version"] = version_fo2media_map[DATASET_FO_VERSION]
    # dataset level settings
    dataset.default_skeleton = fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])
    dataset.compute_metadata()
    dataset.save()
    
def check_and_save_view(dataset_raw, dataset_view, view_name):
    """
    args:
        dataset_raw: a fo.Dataset
    """
    if view_name in dataset_raw.list_saved_views():
        dataset_raw.delete_saved_viwew(view_name)
    dataset_raw.save_view(view_name, dataset_view)
    print(f"{view_name=} {len(dataset_view)=}")
    return dataset_view

def make_split(dataset, ratio_val=0.1):
    """
    args:
        dataset: a fo.Dataset
    """
    dataset.untag_samples("val")
    dataset.untag_samples("train")
    dataset.take(int(ratio_val * len(dataset)), seed=51).tag_samples("val")
    dataset.match_tags("val", bool=False).tag_samples("train")
    dataset.save()


def make_sample_ids(clip_dataset):
    """
    This function is used to make sample_ids for a group dataset. The sample_ids are used to make a dataloader for the group dataset.
    return:
        sample_ids: list of (clip_id, frame_no) for all frames in the subset group dataset
    """
    sample_ids = [] 
    for clip_ids, clip_support, frame_ids in zip(*clip_dataset.values(['id', 'support', 'frames.id'])):
        start_frame_number = clip_support[0]
        sample_ids_ = [(clip_ids, start_frame_number+frame_i) for frame_i in range(len(frame_ids))]
        sample_ids += sample_ids_
    
    return sample_ids

if __name__ == "__main__":
    # gtv1 on EK_FO_v003
    # dataset_clips = fo.load_dataset("EpicKitchens50_FO_v003").match(FF("filepath")=="/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_05.MP4").to_clips("narration").match(FF("narration.label")=="take knife").skip(0).limit(10)
    # visualize_forecasting_annotation(dataset_clips=dataset_clips, output_dir="/z/home/yayuanli/web/tmp/forecasting_GT_v1", narration_field="narration", vis_post_fix="gtv1", frame_scope=None, check_registration=True)
    
    # gtv2 on EK_FO_v004
    # dataset_clips = fo.load_dataset("EpicKitchens50_FO_v004").match(FF("filepath")=="/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_05.MP4").to_clips("narration").match(FF("narration.label")=="take knife").skip(0).limit(10)
    # visualize_forecasting_annotation(dataset_clips=dataset_clips, output_dir="/z/home/yayuanli/web/tmp/forecasting_GT_v1", narration_field="narration", vis_post_fix="gtv2", frame_scope=None, check_registration=True)
    
    # gtv3 on EK_FO_v005
    # dataset_clips = fo.load_dataset("EpicKitchens50_FO_v005").match(FF("filepath")=="/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_05.MP4").to_clips("narration").match_frames(FF("forecasting_groundtruth_path")!=None)
    # vis_post_fix = "gtv3"
    # output_dir = "/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/lgpf_visualization/forecasting_GT_v3"
    
    # visualize_forecasting_annotation(dataset_clips=dataset_clips, output_dir=output_dir, narration_field="narration", vis_post_fix=vis_post_fix, frame_scope=None, check_registration=True)
    
    dataset = fo.load_dataset("EpicKitchens50_FO_v006").clone('EpicKitchens50_FO_v009')
    dataset_test = dataset.to_clips("narration") # .skip(2).limit(1).to_clips("narration")
    set_FGT_indicator(dataset_test, from_path=True)
    print("DONE")
    
    

