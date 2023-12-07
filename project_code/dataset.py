# Prepare PyTorch DataLoaders
# import MILLYCookbook as mc
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import torchvision.transforms as T
import torchvision
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image
from fiftyone import ViewField as FF
import numpy as np
import pose as P
import json
import fiftyonize_utils as fu

def get_iterators(inference_scope=None):
    """
    The FO dataset in the iterator should be a cloned 
    """
    dataset_raw, dataset_fo, _ = get_FO_dataset()
    if wandb.config['function'] == 'save_BLIP_ek50':
        dataset_fo = dataset_raw.to_frames()
    
    splits = ["train", "val"]
    loaders = {}
    for split in splits:
        print(f"{split=}")
        dataset_split_fo = dataset_fo.match_tags(split, bool=True)
        if len(dataset_split_fo) <= 0:
            dataset_split_fo = loaders["train"].dataset.dataset
        # print(f"#frames: {len(dataset_split_fo)}")
        dataset = get_torch_dataset(dataset_split_fo, split, inference_scope=inference_scope)
        if not wandb.config["debug_mode"]:
            data_loader = DataLoader(dataset, wandb.config["batch_size"], num_workers=wandb.config["num_workers"], prefetch_factor=wandb.config["prefetch_factor"], shuffle=wandb.config["train_shuffle"] if split == "train" else False)
        else:
            data_loader = DataLoader(dataset, wandb.config["batch_size"], num_workers=0, prefetch_factor=None, shuffle=False)
        loaders[split] = data_loader
        
    return loaders

def get_FO_dataset():
    """
    wandb.confg:
        EK: dataset_prefix, DATASET_FO_VESION, dataset_local_dir, force_reload, visualize, window_size, grid_len, hand_side, landmark_name, scope_dict, debug_mode, 
        MC: scope_dict, step_scope
    """
    
    if wandb.config["dataset_public_name"] == "EpicKitchens50":
        import EpicKitchens50 as mydata
        dataset_raw, dataset_fo, session = mydata.make_fiftyone_dataset(DATASET_FO_VERSION=wandb.config["DATASET_FO_VERSION"], dataset_local_dir=wandb.config["dataset_local_dir"], window_size=wandb.config["window_size"], grid_len=wandb.config["grid_len"], hand_side=wandb.config["hand_side"], landmark_name=wandb.config["landmark_name"], scope_dict=wandb.config["scope_dict"], light_mode=wandb.config["debug_mode"], force_reload=wandb.config["force_reload"], visualize=wandb.config["visualize"], dataset_prefix=wandb.config["dataset_prefix"])
    elif wandb.config["dataset_public_name"] == "MILLYCookbook":
        import MILLYCookbook as mydata
        dataset_raw, dataset_fo, session = mydata.return_forecasting_dataset(step_scope=wandb.config['scope_dict']['step_scope']['0'])
    elif wandb.config["dataset_public_name"] == "F-PHAB":
        import fphab as mydata
        dataset_raw, dataset_fo = mydata.make_fiftyone_dataset(DATASET_FO_VERSION=wandb.config["DATASET_FO_VERSION"], dataset_local_dir=wandb.config["dataset_local_dir"], dataset_prefix=wandb.config["dataset_prefix"], use_media_onz=False, force_reload=False)
        session = None
    elif wandb.config['dataset_public_name'] == "PennAction":
        import penn_action as mydata
        dataset_raw, dataset_fo = mydata.make_fiftyone_dataset(DATASET_FO_VERSION=wandb.config["DATASET_FO_VERSION"], dataset_local_dir=wandb.config["dataset_local_dir"], dataset_prefix=wandb.config["dataset_prefix"], use_media_onz=False, force_reload=False, check_videos_flag=wandb.config['check_videos_flag'])
        session = None

    return dataset_raw, dataset_fo, session

def get_torch_dataset(dataset, split, inference_scope=None):
    torch_d_name = f"MyDataset_v{wandb.config['DATASET_TORCH_VERSION']}" # 001, 002...
    if wandb.config["dataset_loading_by"] == "sample":
        return globals()[torch_d_name](dataset, split, inference_scope=inference_scope)
    elif wandb.config["dataset_loading_by"] == "frame":
        return globals()[torch_d_name](dataset, split, loading_by="frame", loading_by_split_name=split, inference_scope=inference_scope)
    
# def get_torch_dataset(split, dataset_fo, dataset_og):
#     torch_d_name = f"MyDataset_v{wandb.config['DATASET_TORCH_VERSION']}" # 001, 002...
#     return globals()[torch_d_name](split, dataset_fo, dataset_og, wandb.config["window_size"], wandb.config["hand_side"], wandb.config["landmark_name"], wandb.config["grid_len"])

def set_preprocess(data_loaders, net):
    if wandb.config['network_class_name'] == "MyNetwork":
        for split, loader in data_loaders.items():        
            data_loaders[split].dataset.encoder_image_preprocess = net.encoder_image_preprocess
            data_loaders[split].dataset.encoder_text_preprocess = net.encoder_text_preprocess
            data_loaders[split].dataset.decoder_pose_preprocess = net.decoder_pose_preprocess
            
    # elif wandb.config['network_class_name'] == "MyNetworkV002":
    #     for split, loader in data_loaders.items():
    #         if loader==None: continue
    #         data_loaders[split].dataset.encoder_image_preprocess = net.encoder_image_preprocess_train_val['eval']
    #         data_loaders[split].dataset.encoder_text_preprocess = net.encoder_text_preprocess_train_val['eval']
    #         data_loaders[split].dataset.decoder_pose_preprocess = net.decoder_pose_preprocess
    # 
    # if 'gradcam' in wandb.config['use_embeddings']:
    #     for split, loader in data_loaders.items():
    #         if loader==None: continue
    #         data_loaders[split].dataset.gradcam_vis_processors = net.gradcam_vis_processors['eval']
    #         data_loaders[split].dataset.gradcam_text_processors = net.gradcam_text_processors['eval']

            
    elif wandb.config['network_class_name'] == "MyNetworkV002" or wandb.config['network_class_name'] == "Duplicate":
        for split, loader in data_loaders.items():
            if loader==None: continue
            data_loaders[split].dataset.encoder_image_preprocess = net.module.encoder_image_preprocess_train_val['eval'] if isinstance(net, nn.DataParallel) else net.encoder_image_preprocess_train_val['eval']
            data_loaders[split].dataset.encoder_text_preprocess = net.module.encoder_text_preprocess_train_val['eval'] if isinstance(net, nn.DataParallel) else net.encoder_text_preprocess_train_val['eval']
            data_loaders[split].dataset.decoder_pose_preprocess = net.module.decoder_pose_preprocess if isinstance(net, nn.DataParallel) else net.decoder_pose_preprocess
    
    if 'gradcam' in wandb.config['use_embeddings']:
        for split, loader in data_loaders.items():
            if loader==None: continue
            data_loaders[split].dataset.gradcam_vis_processors = net.module.gradcam_vis_processors['eval'] if isinstance(net, nn.DataParallel) else net.gradcam_vis_processors['eval']
            data_loaders[split].dataset.gradcam_text_processors = net.module.gradcam_text_processors['eval'] if isinstance(net, nn.DataParallel) else net.gradcam_text_processors['eval']
    

class MyDataset_v002(Dataset):
    def __init__(self, dataset, split, loading_by="sample", loading_by_split_name=None, inference_scope=None):
        """
        Each iter return (frame in 3D torch, hand_landmarks in 2D torch, frame_id in string)
        args:
            dataset: FO Dataset/Video. Could be Frame or Clip
            split: "train", "val" or 'inf'
            loading_by: "sample" or "frame".
                If 'sample', __len__ the number of top level samples in given dataset. Then random select one top level sample at each __getitem__ call (if it's a clip, random select one frame in it).
                If 'frame', __len__ is the sum of number of frames across all clips (the given dataset (FO) has to be Clip view and global_frame_no_to_clip_frame_no_ordered_list has to be given). In this case, each __getitem__ call will locate the clip of the frame by frame global index to get clip level info (e.g., narration), and return the frame in the clip.
        
        """
        super(MyDataset_v002, self).__init__()
        # self.dataset_fo, self.window_size, self.hand_side, self.landmark_name, self.grid_len = dataset_fo, window_size, hand_side, landmark_name, grid_len
        # self.transform = Preprocessing_SingleCam(dataset_fo, wandb.config['scope_dict']['recipe_scope'][-1], split, wandb.config['scope_dict']['sensor_scope'][-1])
        # self.camera = 'pv'
#        # TODO: need a general way to generate input_view from arg `querys`
#        # self.dataset = self.dataset.match(querys)
#        # self.dataset = self.dataset.match_tags(split)
#        self.dataset = self.dataset.match(FF("recipe.label") == self.dataset.info["recipe_no2text_map"][recipe_no])
#        self.dataset = self.dataset.select_group_slices(camera).to_frames()
        # self.frame_ids = self.dataset_fo.values("id")
#         self.frame_ids = []
#         for vid in self.dataset_fo.values('id'):
#             self.frame_ids += self.dataset_fo.info["vid_frame_id_map.val"][vid]        

        # open vocab inference
        self.inference_scope = inference_scope
        if inference_scope != None and split != 'train':
            # narrow down scope if inference
            filepath_list = [filepath for filepath in list(inference_scope.keys())]
            dataset = dataset.match(FF("filepath").is_in(filepath_list))
            self.dataset = dataset
        else:
            self.dataset = dataset

        # self.dataset = dataset.clone(f"{dataset.name}_training_{split}")
        if loading_by == "sample":
            self.sample_ids = self.dataset.values("id")
        elif loading_by == "frame":
            self.sample_ids = fu.make_sample_ids(self.dataset)
            # assert loading_by_split_name != None
            # if f'global_frame_no_to_clip_frame_no_ordered_list_{loading_by_split_name}' in self.dataset.info.keys() and self.dataset.info[f"global_frame_no_to_clip_frame_no_ordered_list_{loading_by_split_name}"] != None:
            #     self.sample_ids = self.dataset.info[f"global_frame_no_to_clip_frame_no_ordered_list_{loading_by_split_name}"]
            # else:
            #     with open(self.dataset.info[f"global_frame_no_to_clip_frame_no_ordered_list_{loading_by_split_name}_path"], 'r') as f:
            #         self.sample_ids = json.load(f)
        self.loading_by = loading_by
        # self.dataset_og = dataset_og
        self.split = split
        # self.dataset_fo = dataset_fo
#         self.vid_frame_dict = {}
#         self.frame_info_dict = {}
#         self.frame_ids, all_vid, all_fno = self.dataset_fo.values(["id", "sample_id", "frame_number"])
#         for i, f_id in enumerate(self.frame_ids):
#             f_no = all_fno[i]
#             v_id = all_vid[i]
#             self.frame_info_dict[f_id] = {"v_id": v_id, "f_no": f_no}
#             if v_id not in self.vid_frame_dict.keys():
#                 self.vid_frame_dict[v_id] = []
#             self.vid_frame_dict[v_id] += [{"f_no": f_no, "f_id": f_id}]
        self.encoder_image_preprocess = None
        self.encoder_text_preprocess = None
        self.decoder_pose_preprocess = None
        self.gradcam_vis_processors = None
        self.gradcam_text_processors = None
        
        self.image_failed_times = 0
        self.use_open_vocab_phrase = False
        # open vocab quantitative results
        if wandb.config['use_open_vocab_phrase']:
            self.use_open_vocab_phrase = True
            self.open_vocab_phrase_dict = self.dataset.info['open_vocab_phrase_dict']

    def __len__(self):
        if wandb.config['debug_mode'] == True:
            return 512
        else:
            return len(self.sample_ids)
        
    def __getitem__(self, index):
        """
        args:
            index: index from 0 to __len__-1 to enumerate
        return:
            image: tensor with shape (3, H, W). Preprocessed image
            text: tensor with shape (context_len, ). Text tokens padded to predefined context length (default 77)
            pose: tensor with shape (forcasting_steps+2, num_indicators * num_joints * num_coords). The poses of visual indocators at current and future timestamps.
            frame_id: string. frame_id of current frame(image) in the Fiftyone Dataset
        """
        if self.loading_by == "sample":
            sample_id = self.sample_ids[index]
            sample = self.dataset[sample_id]
            # check sample type. Video / Frame
            if sample.filepath.split(".")[-1] in ["mp4", "MP4"]:
                # video
                if wandb.config['use_first_frame'] == False:
                    frame_pick = np.random.randint(len(sample.frames)) # Should use len(sample.frmaes); The (support[1] - support[0] + 1) is not the accurate number of frames since it's from anntoation file which may be effected by the way annotation software counting the frames.
                elif wandb.config['use_first_frame'] == True:
                    frame_pick = 0
                frame_no = sample.support[0] + frame_pick # fo frame_no
                frame_sample = sample.frames[frame_no]
            else:
                # image
                frame_sample = sample
        elif self.loading_by == "frame":
            sample = self.dataset[self.sample_ids[index][0]] # clip sample
            self.sample = sample
            frame_sample = sample.frames[self.sample_ids[index][1]]
            self.frame_sample = frame_sample
            
        frame_id = frame_sample.id
        frame_path = frame_sample.filepath
        Keypoints_failed = False
        # text
        if wandb.config['fix_language'] != None:
            # no language guidance
            text = wandb.config['fix_language']
        else:
            # with language guidance
            # text_fo = sample[wandb.config["narration_field_name"]].detections[0]
            # text_fo = frame_sample[wandb.config["narration_field_name"]]
            if self.split != 'inf':
                text_fo = sample[wandb.config["narration_field_name"]]
            else:
                if wandb.config["narration_field_name_inf"] in frame_sample.field_names and frame_sample[wandb.config["narration_field_name_inf"]] != None:
                    text_fo = frame_sample[wandb.config["narration_field_name_inf"]]
                else:
                    text_fo = sample[wandb.config["narration_field_name_inf"]]
                
            # text_fo = fo.Classification(label="test narration")

            # if 'gradcam' not in wandb.config['use_embeddings']:
            #     text = self.text_fo2torch(text_fo)            
            # elif 'gradcam' in wandb.config['use_embeddings']:
            #     text, text_gradcam = self.text_fo2torch(text_fo)
    #        text = self.encoder_text_preprocess(text)
            torch_text_return = self.text_fo2torch(text_fo)
            text = torch_text_return['text']
            text_gradcam = torch_text_return['text_gradcam']
        
        # pose
        if wandb.config["use_path_forecgt"]:
            try:
                pose_fo = P.read_Keypoints(frame_sample[wandb.config["groundtruth_field_name"]+"_path"])
                pose = self.pose_fo2torch(pose_fo)
            except Exception as e:
                print(f"loading Keypoints {frame_sample.filepath=} failed {e=}")
                # pose_fo = P.read_Keypoints("/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P26/P26_01/0000000265/forecasting_groundtruth_v001.json")
                # pose = self.pose_fo2torch(pose_fo)
                # Keypoints_failed = True
        else:
            pose_fo = frame_sample[wandb.config["groundtruth_field_name"]]
            pose = self.pose_fo2torch(pose_fo)
#        hand_confidence = torch.tensor([Keypoint.hand_confidence for Keypoint in pose_fo.keypoints]).reshape(wandb.config["forcasting_steps"]+1,wandb.config["num_indicators"],1).repeat(1,1,wandb.config["num_joints"]*wandb.config["num_coords"]).reshape(wandb.config["forcasting_steps"]+1, wandb.config["num_indicators"]*wandb.config["num_joints"]*wandb.config["num_coords"]).float()
        prep_seq_len = pose.shape[0]
        try:
            if 'hand_confidence' in pose_fo.keypoints[0].field_names:
                hand_confidence = torch.tensor([Keypoint.hand_confidence for Keypoint in pose_fo.keypoints]).reshape(prep_seq_len,wandb.config["num_indicators"],1).repeat(1,1,wandb.config["num_joints"]*wandb.config["num_coords"]).reshape(prep_seq_len, wandb.config["num_indicators"]*wandb.config["num_joints"]*wandb.config["num_coords"])[1:].float() # remove the first hand -- cur hand since this hand_confidence will be the mask for the predictions
            else:
                hand_confidence = torch.ones((prep_seq_len, wandb.config['num_all_coords'])).float() # shape (prep_seq_len, num_all_coords)
        except Exception as e:
            hand_confidence = torch.ones((prep_seq_len, wandb.config['num_all_coords'])).float()
        
        if Keypoints_failed:
            print(f"Keypoints_failed==True. set weight (hand_confidence) to 0 ")
            hand_confidence = torch.zeros_like(hand_confidence)
            
        # image
        image_fo = frame_sample["filepath"]
        try:
            if 'gradcam' not in wandb.config['use_embeddings']:
                image = self.image_fo2torch(image_fo)
            else:
                image, image_gradcam = self.image_fo2torch(image_fo)
        except Exception as e:
            print(f"loading image {image_fo=} failed {e=}")
            image = torch.zeros((3,) + self.encoder_image_preprocess.transforms[1].size)
            hand_confidence = torch.zeros_like(hand_confidence)
            
            image_gradcam = torch.zeros(3, 384, 384) + 0.1
            
        if None in [image, text, pose, hand_confidence, frame_path]:
            print(f"*****Hey there is None when __getitem__ from {sample=} \n {frame_sample=}*****")
            
        if self.split != "inf":
            if 'gradcam' not in wandb.config['use_embeddings']:
                return image, text, pose, hand_confidence, frame_path # fo_id
            elif 'gradcam' in wandb.config['use_embeddings']:
                return image, text, pose, hand_confidence, image_gradcam, text_gradcam, frame_path
        else:
            return image, text, pose, hand_confidence, frame_path, text_fo.label
        
    def image_fo2torch(self, image_fo):
        """
        image_fo: image path
        """
        # read image
        raw_image = Image.open(image_fo) # readed image inbuffer in PIL format
        # preprocess
        image = self.encoder_image_preprocess(raw_image)
        if 'gradcam' not in wandb.config['use_embeddings']:
            return image
        else:
            image_gradcam = self.gradcam_vis_processors(raw_image)
            return image, image_gradcam
    
    def text_fo2torch(self, text_fo):
        """
        text_fo: an instance of fiftyone.core.labels.Classification
        """
        # define return
        text_gradcam = None

        # get string
        if 'label' in dir(text_fo):
            text = text_fo.label # a string # TODO
        else:
            text = text_fo

        if self.use_open_vocab_phrase:
            text = random.choice(self.open_vocab_phrase_dict[text])
        elif self.inference_scope != None:
            sample_filepath = self.sample.filepath
            for sample_filepath_, inference_text in self.inference_scope.items():
                if sample_filepath == sample_filepath_:
                    text = inference_text
                    break

        # get token
        if wandb.config['network_class_name'] == "MyNetwork":
            text = self.encoder_text_preprocess(text)[0] # torch with shape (context_length (default 77))
        elif wandb.config['network_class_name'] == "MyNetworkV002":
            text = self.encoder_text_preprocess(text) # String
            
        if 'gradcam' in wandb.config['use_embeddings']:
            text_gradcam = self.gradcam_text_processors(text)

        return {'text': text, 'text_gradcam': text_gradcam}

    def pose_fo2torch(self, pose_fo):
        """
        pose_fo: an instance of fiftyone.core.labels.Keypoints
        """
        pose = [ ]
        for indicator_i, indicator_fo in enumerate(pose_fo["keypoints"]):
            # indicator_fo is a fo.Keypoint instance                
            pose.append(indicator_fo["points"][:wandb.config["num_joints"]])
        try:
            pose = torch.tensor(pose).view(-1, wandb.config["num_all_coords"])[:wandb.config["forcasting_steps"]+1] # tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
        except Exception as e:
            pose = [pose, pose]
            pose = torch.tensor(pose).view(-1, wandb.config["num_all_coords"])[:wandb.config["forcasting_steps"]+1]
        pose = self.decoder_pose_preprocess(pose)
        
        return pose
    
    
class MyDataset_v001(Dataset):
    def __init__(self, split, dataset_fo, dataset_og, window_size, hand_side, landmark_name, grid_len):
        """
        Each iter return (frame in 3D torch, hand_landmarks in 2D torch, frame_id in string)
        args:
            dataset_fo: frame dataset of specific split 
        """
        super(MyDataset_v001, self).__init__()
        # self.dataset_fo, self.window_size, self.hand_side, self.landmark_name, self.grid_len = dataset_fo, window_size, hand_side, landmark_name, grid_len
        # self.transform = Preprocessing_SingleCam(dataset_fo, wandb.config['scope_dict']['recipe_scope'][-1], split, wandb.config['scope_dict']['sensor_scope'][-1])
        # self.camera = 'pv'
#        # TODO: need a general way to generate input_view from arg `querys`
#        # self.dataset = self.dataset.match(querys)
#        # self.dataset = self.dataset.match_tags(split)
#        self.dataset = self.dataset.match(FF("recipe.label") == self.dataset.info["recipe_no2text_map"][recipe_no])
#        self.dataset = self.dataset.select_group_slices(camera).to_frames()
        # self.frame_ids = self.dataset_fo.values("id")
#         self.frame_ids = []
#         for vid in self.dataset_fo.values('id'):
#             self.frame_ids += self.dataset_fo.info["vid_frame_id_map.val"][vid]        
        self.dataset_og = dataset_og
        self.split = split
        self.dataset_fo = dataset_fo
        self.vid_frame_dict = {}
        self.frame_info_dict = {}
        self.frame_ids, all_vid, all_fno = self.dataset_fo.values(["id", "sample_id", "frame_number"])
        for i, f_id in enumerate(self.frame_ids):
            f_no = all_fno[i]
            v_id = all_vid[i]
            self.frame_info_dict[f_id] = {"v_id": v_id, "f_no": f_no}
            if v_id not in self.vid_frame_dict.keys():
                self.vid_frame_dict[v_id] = []
            self.vid_frame_dict[v_id] += [{"f_no": f_no, "f_id": f_id}]
        self.encoder_image_preprocess = None
        self.encoder_text_preprocess = None
        self.decoder_pose_preprocess = None
        
        
    def __len__(self):
        return len(self.frame_ids)
        
    def __getitem__(self, index):
        """
        args:
            index: index from 0 to __len__-1 to enumerate
        return:
            image: tensor with shape (3, H, W). Preprocessed image
            text: tensor with shape (context_len, ). Text tokens padded to predefined context length (default 77)
            pose: tensor with shape (forcasting_steps+2, num_indicators * num_joints * num_coords). The poses of visual indocators at current and future timestamps.
            frame_id: string. frame_id of current frame(image) in the Fiftyone Dataset
        """
        frame_id = self.frame_ids[index]
        # print(f"{frame_id=}")
        frame_sample = self.dataset_fo[frame_id]
        # image
        image_fo = frame_sample["filepath"]
        # image_fo = "/z/dat/CookBook/MILLYCookbook_media_v000/A_pin/mevo/0/video-0000/pv_frames/frame_0000000000.jpg"
        image = self.image_fo2torch(image_fo)
#        image = self._read_image(frame_sample) # TODO: PLI
#        # preprocess
#        image = self.encoder_image_preprocess(image) 
        
        # text
        text_fo = frame_sample[wandb.config["narration_field_name"]]
        # text_fo = fo.Classification(label="test narration")
        text = self.text_fo2torch(text_fo)
#        text = self.encoder_text_preprocess(text) 
        
        # pose
        pose_fo = frame_sample[wandb.config["groundtruth_field_name"]]
#         pose_fo = fo.Keypoints(
#             keypoints=[
#                 fo.Keypoint(
#                     label="rectangle",
#                     kind="square",  # custom object attribute
#                     points=[(0.5,)*wandb.config["num_coords"]]*wandb.config["num_joints"],
#                     confidence=[0.6, 0.7, 0.8, 0.9],
#                     occluded=[False, False, True, False],  # custom per-point attributes
#                 )
#             ]*((wandb.config["forcasting_steps"]+1)*wandb.config["num_indicators"])
#         )
        pose = self.pose_fo2torch(pose_fo)
#        # hand_landmarks = self._get_landmark_input(frame_sample, self.window_size, self.hand_side, self.landmark_name, self.grid_len) 
#        try:
#            hand_seq = torch.tensor(frame_sample["hand_seq_list"]) # a list of int w/ len=window_size. it's input of nn.Embedding (Decode)
#        except Exception as e:
#            print(f"{e=}")
#            print(frame_sample)
#  print(f"__getitem__ done")

        # TODO: shoule make sure dataset is clean when preparing, not here.
        if torch.any(torch.isnan(pose)):
            print(f"got nan pose at position {torch.nonzero(torch.isnan(pose))}")
            print(f"{frame_id=}")
            pose = torch.where(torch.isnan(pose), torch.tensor(0.5).to(pose.device), pose)
            
        return image, text, pose, frame_id
    def image_fo2torch(self, image_fo):
        """
        image_fo: image path
        """
        # read image
        image = Image.open(image_fo) # readed image inbuffer in PIL format
        # preprocess
        image = self.encoder_image_preprocess(image)
        return image
    
    def text_fo2torch(self, text_fo):
        """
        text_fo: an instance of fiftyone.core.labels.Classification
        """
        # get string
        text = text_fo.label # a string # TODO
        # get token
        text = self.encoder_text_preprocess(text)[0] # torch with shape (context_length (default 77))
        return text
        
    def pose_fo2torch(self, pose_fo):
        """
        pose_fo: an instance of fiftyone.core.labels.Keypoints
        """
        pose = [ ]
        for indicator_i, indicator_fo in enumerate(pose_fo["keypoints"]):
            # indicator_fo is a fo.Keypoint instance                
            pose.append(indicator_fo["points"][:wandb.config["num_joints"]])
        pose = torch.tensor(pose).view(-1, wandb.config["num_all_coords"])[:wandb.config["forcasting_steps"]+1] # tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
        pose = self.decoder_pose_preprocess(pose)
        
        return pose
    
    
    def load_prediction(self, pred_dict, epoch):
        """
        for each frame_id, convert the seq of position no to FO position and load the seq as FO.Keypoint
        args:
            pred_dict: frame_id->pred seq, a tensor with shape (forcasting_steps, num_indocators * num_joints * num_coords)
        """
#        for frame_id, hand_seq_grid in pred_dict.items():
#            hand_seq_xy = self._translate_pos_grid2float(hand_seq_grid, self.grid_len)
#            self.dataset_fo[frame_id]["hand_seq_Keypoints_pred"] = fo.Keypoints(keypoints=[fo.Keypoint(frame_shift=frame_shift, label=self.hand_side, points=[(x, y)]) for frame_shift, (x,y) in enumerate(hand_seq_xy)])
#            self.dataset_fo[frame_id]["hand_seq_list_pred"] = hand_seq_grid
         
        for v_id, list_of_frame_dict_id_no in self.vid_frame_dict.items():
            print(f"loading prediction for {v_id}")
            v_sample = self.dataset_og[v_id]
            for frame_dict_id_no in list_of_frame_dict_id_no:
                f_sample = v_sample.frames[frame_dict_id_no["f_no"]]
                poses_all_stamps = pred_dict[frame_dict_id_no["f_id"]] # tensor (#forcasting_steps, #indicators*#joints*#coordinates)
                poses_all_stamps = poses_all_stamps.view((wandb.config["forcasting_steps"]+1) * wandb.config["num_indicators"], wandb.config["num_joints"], wandb.config["num_coords"]) # tensor ((#forcasting_steps+1)*#indicators, #joints, #coordinates)
                keypoints = []
                for indicator_i, indicator in enumerate(poses_all_stamps):
                    indicator_fo = fo.Keypoint(points=indicator.tolist(), label="Left" if indicator_i%2==0 else "Right", index=indicator_i//wandb.config["num_indicators"], loading_epoch=epoch)
                    keypoints.append(indicator_fo)
                f_sample["pose_pred"] = fo.Keypoints(keypoints=keypoints)
            v_sample.save()
                
            
            
#        for frame_id, all_poses in pred_dict.items(): # all_pose is tensor with shape (#forcasting_steps, #all_coords)
#            all_poses = all_poses.view((wandb.config["forcasting_steps"]) * wandb.config["num_indicators"], wandb.config["num_joints"], wandb.config["num_coords"]) # tensor with shape (#forcasting_steps*#indicators, #joints, #coords)
#            keypoints = []
#            for indicator_i, indicator in enumerate(all_poses, wandb.config["num_indicators"]): # skip current timestamp since this are predictions
#                # indicator_fo = fo.Keypoint(points=np.random.rand(42).reshape(21, 2).tolist(), label="Left" if indicator_i%2==0 else "Right", index=indicator_i//wandb.config["num_indicators"])
#                indicator_fo = fo.Keypoint(points=indicator.tolist(), label="Left" if indicator_i%2==0 else "Right", index=indicator_i//wandb.config["num_indicators"])
#                keypoints.append(indicator_fo)
#            vid_sample = self.dataset_og[self.dataset_fo[frame_id].sample_id]            
#            frame_sample = vid_sample.frames[self.dataset_fo[frame_id].frame_number]
#            frame_sample["pose_pred"] = fo.Keypoints(keypoints=keypoints) # TODO: dive in to see why can't write it in
#        [vid_sample.save() for vid_sample in self.dataset_og]

    
    def _get_landmark_input(self, frame_sample, window_size, hand_side, landmark_name, grid_len):
        """
        1. get hand_pos_sequence [(x1, y1), (x2, y2), ...] from specified hand and landmark
        2. map on grid
        """
        hand_pos_sequence = []
        landmark_no = mc.landmark_no2name_map[landmark_name]
        hand_side_map = {"right": 1, "left": 0} # pos based, not label based
        hand_side_no = hand_side_map[hand_side]
        # cur hand
        keypoints_cur = frame_sample.hand_landmarks.keypoints
        right_hand_no_cur, right_hand_mean_pos_cur = sorted(dict([(hand_no, keypoints_cur[hand_no].points[landmark_no]) for hand_no in range(len(keypoints_cur))]).items(), key=lambda handno2meanpos: handno2meanpos[1][0])[hand_side_no] # hand_no, [x, y]
        hand_pos_sequence.append(right_hand_mean_pos_cur)
        
        # future hands
        keypoints_fut = frame_sample.registered_hand_landmarks.keypoints
        seq2hand2pos_map_fut = {str(key): [] for key in range(1, window_size+1)} # seq_no -> hand_no -> [x, y]
        # for each future frame, get all hand no and pos
        for hand_no in range(len(keypoints_fut)):
            hand_fo = keypoints_fut[hand_no]
            mean_pos = hand_fo.points[landmark_no]
            frame_shift = hand_fo.frame_shift
            seq2hand2pos_map_fut[frame_shift].append((hand_no, mean_pos))
        # for each future frame, get desired hand pos
        for seq_no, hand_no_pos in seq2hand2pos_map.items():    
            if hand_no_pos == []:
                right_hand_no, right_hand_pos = -1, [-1, -1] 
            else:
                right_hand_no, right_hand_pos = sorted(dict(hand_no_pos).items(), key=lambda no_pos: no_pos[1][0])[hand_side_no]
            seq2hand2pos_map_fut[seq_no] = (right_hand_no, right_hand_pos)
            hand_pos_sequence.append(right_hand_pos)
            
        hand_pos_sequence_grid = self._translate_pos_float2grid(hand_pos_sequence, grid_len)
        
        return hand_pos_sequence_grid
    
    def _translate_pos_float2grid(self, hand_pos_sequence, grid_len):
        """
        given a hand pos sequence [(x1, y1), (x2, y2)], return the 1D (flatten) pos on grid [g1, g2, g3...].
        args:
            hand_pos_sequence: x,y are from 0 to 1
        return:
            hand_pos_sequence_grid: [g1, g2, ....] g is int
        """
        hand_pos_sequence_grid = [ ]
        grid_interval = 1/grid_len
        for x, y in hand_pos_sequence:
            num_pre_rows_boxs = (x // grid_interval) * grid_len # the number of box in previous rows
            num_col_box = (y // grid_interval)
            
            g = num_pre_rows_boxs + num_col_box
            hand_pos_sequence_grid.append(g)
            
        return hand_pos_sequence_grid
    
    def _translate_pos_grid2float(self, hand_pos_sequence_grid, grid_len):
        """
        Given a list of int w/o None
        """
        hand_pos_sequence_grid_fo = [ ] 
        grid_interval = 1/grid_len
        for grid_pos in hand_pos_sequence_grid:
            y = (grid_pos // grid_len) * grid_interval + grid_interval/2
            x = (grid_pos % grid_len) * grid_interval + grid_interval/2
            hand_pos_sequence_grid_fo.append((x,y))
        return hand_pos_sequence_grid_fo
    
    def _read_image(self, frame_sample):
        """
        based on different sensor, read the frame in different way.
        pv, rm_vlc_* use torchvision
        ab, depth use numpy
        microphone raise error (even though I will set frame for it)

        return image in torch (unnormalized)
        """
        sensor = self.camera
        if sensor in ["pv", "rm_vlc_lf", "rm_vlc_rf"]: 
            image = torchvision.io.read_image(frame_sample.filepath)
        elif sensor in ["ir", "depth"]:
            image = np.load(frame_sample.filepath)
            image = torch.from_numpy(image).permute(2,0,1)
        else:
            raise Exception(f"{sensor=} doesn't have legal frames to be read")
        
        return image
        
    def get_weighted_sampler():
        weights = torch.DoubleTensor((1 / self.annotations["step_no"].value_counts()[self.annotations["step_no"]]).tolist())
        num_samples = len(self)
        return torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=self.generator)
    
    def _get_transform(self):
        """
        This function return preprocessing with fixed mean and var values from ImageNet.
        Recommend to use Preprocessing_SingleCam class instead
        """
        meanv = [0.4914, 0.4822, 0.4465]
        stdv = [0.2023, 0.1994, 0.2010]
        
        if self.split == 'train' or self.split == "trainval":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(meanv, stdv),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
            ])
        if self.split == 'val' or self.split == "test":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(meanv, stdv),
                T.Resize(256),
                T.CenterCrop(224),
            ])    
        return transform
     
        
    def pred_dict2vid_sample(pred_dict):
        vid_samples = {}
        for frame_id, all_poses in pred_dict.items():
            vid_samples[self.dataset_fo[frame_id].sample_id] = self.dataset_fo[frame_id].frame_number
        return vid
        
    
class MyDataset_vtest(Dataset):
    def __init__(self, split, dataset_fo, window_size, hand_side, landmark_name, grid_len):
        """
        Each iter return (frame in 3D torch, hand_landmarks in 2D torch, frame_id in string)
        args:
            dataset_fo: frame dataset of specific split 
        """
        super(MyDataset_vtest, self).__init__()
        import clip
        _, self.encoder_image_preprocess = clip.load("ViT-B/32", device=wandb.config["device"])
        self.encoder_text_preprocess = clip.tokenize
    def __len__(self):
        return 200
    
    def __getitem__(self, index):        
#        return torch.randn(3, 224, 224), torch.randint(9999, (77,)), torch.rand(wandb.config["window_size"], 2*21*2), "asdf"
        frame_id = "asdf"
        image_fo = "/z/dat/CookBook/MILLYCookbook_media_v000/A_pin/mevo/0/video-0000/pv_frames/frame_0000000000.jpg"
        image = self.image_fo2torch(image_fo)
#        image = self._read_image(frame_sample) # TODO: PLI
#        # preprocess
#        image = self.encoder_image_preprocess(image) 
        
        # text
        # text_fo = frame_sample["narration"] # TODO:
        text_fo = fo.Classification(label="test narration")
        text = self.text_fo2torch(text_fo)
#        text = self.encoder_text_preprocess(text) 
        
        # pose
        # pose_fo = frame_sample["pose"] # 
        pose_fo = fo.Keypoints(
            keypoints=[
                fo.Keypoint( 
                    label="rectangle",
                    kind="square",  # custom object attribute
                    points=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
                    confidence=[0.6, 0.7, 0.8, 0.9],
                    occluded=[False, False, True, False],  # custom per-point attributes
                )
            ]*wandb.config["num_all_coords"]
        )
        pose = self.pose_fo2torch(pose_fo)
#        # hand_landmarks = self._get_landmark_input(frame_sample, self.window_size, self.hand_side, self.landmark_name, self.grid_len) 
#        try:
#            hand_seq = torch.tensor(frame_sample["hand_seq_list"]) # a list of int w/ len=window_size. it's input of nn.Embedding (Decode)
#        except Exception as e:
#            print(f"{e=}")
#            print(frame_sample)
        print(f"__getitem__ done")
        return image, text, pose, frame_id
    def image_fo2torch(self, image_fo):
        """
        image_fo: image path
        """
        # read image
        image = Image.open(image_fo) # readed image inbuffer in PIL format
        # preprocess
        image = self.encoder_image_preprocess(image)
        return image
    
    def text_fo2torch(self, text_fo):
        """
        text_fo: an instance of fiftyone.core.labels.Classification
        """
        # get string
        text = text_fo.label # a string # TODO
        # get token
        text = self.encoder_text_preprocess(text)[0] # torch with shape (context_length (default 77))
        return text
        
    def pose_fo2torch(self, pose_fo):
        """
        pose_fo: an instance of fiftyone.core.labels.Keypoints
        """
        pose = [ ]
        for indicator_fo in pose_fo["keypoints"]:
            # indicator_fo is a fo.Keypoint instance
            pose.append(indicator_fo["points"])
        pose = torch.tensor(pose).view(-1, wandb.config["num_all_coords"])[:wandb.config["forcasting_steps"]+1] # tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
        return pose
 
   
    def load_outputs_fo(self, id2seq_dict):
        pass
            
class Preprocessing_SingleCam(nn.Module):
    """
    Return a function that will choose correct normalization parameters based on sensor info stored in sample
    """
    def __init__(self, dataset, recipe_no, split, camera):
        super(Preprocessing_SingleCam, self).__init__()
        self.dataset = dataset
        if split == 'train' or split == "trainval":
            self.augmentations = T.Compose([
                # T.RandomResizedCrop(224),
                T.Resize(256),
                T.CenterCrop(wandb.config["input_img_size"][-1]),
                # T.RandomHorizontalFlip(),
            ])
        if split == 'val' or split == "test":
            self.augmentations = T.Compose([
                T.Resize(256),
                T.CenterCrop(wandb.config["input_img_size"][-1]),
            ])
            
        
        self.maxv, self.minv, self.meanv, self.stdv = mc.dinfo["distribution"][recipe_no][camera].values()
        self.maxv, self.minv, self.meanv, self.stdv = torch.tensor(self.maxv).reshape(-1, 1, 1), torch.tensor(self.minv).reshape(-1, 1, 1), torch.tensor(self.meanv).reshape(-1, 1, 1), torch.tensor(self.stdv).reshape(-1, 1, 1)
        self.scale = lambda image: (image-self.minv)/self.maxv
        self.normalize = T.Normalize(self.meanv, self.stdv)
    def forward(self, image):
        """
        pv, rm_vlc_* : 
        """
        # data augmentation
        image = self.augmentations(image)
        
        # scale to 0 to 1
        image = self.scale(image)
        
        # normalize to 0mean1var
        image = self.normalize(image)
        
        return image
