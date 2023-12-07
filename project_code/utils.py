# utils for experiments
import os
import time
import glob
import datetime
import wandb
import torch
import numpy as np
import random
import torch.nn as nn
import socket
import re

exp_config = {
    # "function": "save_BLIP_ek50", #  # 'train', 'inference_visualize'
    # network
    "network_class_name": "MyNetworkV002", # "MyNetworkV002",
    "duplicate_mode": False,
    # enco
    # "img_size": (1080, 1920), # (1080, 1920) # (H, W)
    "input_img_size": (224, 224),
    "grid_len": 7, # 16,
    "patch_size": None,
    # "enco_num_hiddens": 512,
    "enco_mlp_num_hiddens": 256, # 2048,
    "enco_num_heads": 2, # 8,
    "enco_num_blks": 2,
    "enco_emb_dropout": 0.1,
    "enco_blk_dropout": 0.1,
    "use_embeddings": ['multimodal'],  # ['multimodal', 'gradcam'],
    # deco
    "window_size": 30, # num of future window_size
    "vocab_size": None, # number of grid for visual encoder
    "num_indicators": 1, # 2, 1
    "num_joints": 21, # 13, 21
    "num_coords": 2,
    "num_all_coords": None,
    "forcasting_steps": 30,
    'indicator_list': ['Right'], # ['Left', 'Right'],
    # "o_key_size": 32,
    # "deco_query_size": 32,
    # "deco_value_size": 32,
    # "deco_num_hiddens": -1,
    "num_hiddens": 768, # 256 for MyNetwork, 768 for MyNetworkV002 (BLIP)
    "deco_norm_shape": None,
    # "deco_ffn_num_input": 32,
    "deco_ffn_num_hiddens": 256, # 1024,
    "deco_num_heads": 2, # 4,
    "deco_num_layers": 2,
    "deco_dropout": 0.1,
    # train
    "lr": 0.001,
    "num_epochs": 25,
    "device": "cuda" if torch.cuda.is_available() else "cpu", # "cuda" if torch.cuda.is_available() else "cpu", "cpu"
    "batch_size": 128,
    "use_first_frame": False,
    "loss": "mse", # 'rmse', 'weighted_rmse', 'mse', 'weighted_rmse', "mse_linear_down", "mse_sine", "mse_revsine", "mse_learnable"
    "optimizer": "adam", # 'adam', 'sgd'
    "scheduler": "", # "cos", 'step'
    # inference
    # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp8/outputs/experiments/exp_0007_202306031803/outputs/best_model_val.pt", # exp8.1 total epoch=100. EK 10k clip
    # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp6/outputs/experiments/exp_0008_202303290239/outputs/best_model_val.pt", # exp6, EK 1k clip
    "inf_model_path_dict": {
        # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp6/outputs/experiments/exp_0008_202303290239/outputs/best_model_val.pt", # exp6, EK 1k clip
        # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp8/outputs/experiments/exp_0007_202306031803/outputs/best_model_val.pt", # exp8.1 total epoch=100. EK 10k clip
        # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp10/outputs/experiments/exp_0092_202306131957/outputs/best_model_inf.pt" # random
        # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp10/outputs/experiments/exp_0132_202306141427/outputs/best_model_val.pt", # exp10.1,
        # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp10/outputs/experiments/exp_0132_202306141427/outputs/best_model_val.pt" # exp10.2. by frame
        # "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp12/outputs/experiments/exp_0037_202306301606/outputs/best_model_val.pt" # exp12.2. gradcam
        # "fgtv005_exp12.6": "/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp12/outputs/experiments/exp_0048_202307012137/outputs/best_model_val.pt" # zoom
        # "fgtv005_exp12.7": "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp12/outputs/experiments/exp_0055_202307012208/outputs/best_model_val.pt"
        "fgtv005_exp12.10": "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp12/outputs/experiments/exp_0059_202307031823/outputs/best_model_val.pt"
                            },
    "vis_exp_path_dict": {
        # "ek_exp12.10": "/z/exp/lgpf/cog/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp12/outputs/experiments/exp_0059_202307031823",
        # "fphabv001_exp13.10": "/workspace/ptg_research/exp13/outputs/experiments/exp_0010_202308012328",
        "fphabv001_exp13.10_newformat": "/z/exp/lgpf/workspace/ptg_research/exp15/outputs/experiments/exp_0099_202308310404",
        # "fphabv001_exp13.10_fo5": "F-PHAB_FO_v001_vis_paper",
        # "fphabv001_exp13.10_inference": "/z/exp/lgpf/workspace/ptg_research/exp15/outputs/experiments/exp_0106_202308310752",
        # "fphabv001_exp13.10_openvocab": "/z/exp/lgpf/workspace/ptg_research/exp15/outputs/experiments/exp_0077_202308282055",
        # "fgtv001_exp14.4_sopranos": "/workspace/exp14/outputs/experiments/exp_0011_202308011844",
        # "pagtv001_exp15.1": "/z/exp/lgpf/workspace/ptg_research/exp15/outputs/experiments/exp_0010_202308092211"
        # "pagtv001_exp15.1_newformat": "/z/exp/lgpf/workspace/ptg_research/exp15/outputs/experiments/exp_0088_202308300154"
                          },
    "pck_alpha": 0.15,
    # experiment
    "debug_mode": False,
    "visualize_prediction_onthefly": False,
    # "network_structure": 2,
    "hostname": socket.gethostname(),
    # fo
    "visualize": False,
    "force_reload": False,
    # dataset
    "hand_side": 'right', # 'Human_01', 'right' # spatially, not at label-wise. could be wrong. if one hand, then pick it anyway. otherwise, choose the righter one
    "landmark_name": 'MEAN_ALL',
    # "scope_dict": {
    #     "recipe_scope": [0],
    #     "sensor_scope": ["pv"],
    #     "step_scope": {0: [6]}, # recipe_no -> list of step no 
    # },
    "dataset_public_name": "F-PHAB", # "MILLYCookbook", "EpicKitchens50", "F-PHAB", "PennAction"
    "dataset_prefix": "F-PHAB", # "MILLYCookbook", "EpicKitchens50", "F-PHAB", "PennAction"
    "groundtruth_field_name": "forecasting_groundtruth",
    "use_path_forecgt": True,
    "narration_field_name": "narration", # "step",
    "narration_field_name_inf": "narration_inf", # "narration_inf", 
    "DATASET_FO_VERSION": "001", # "016", "001"
    "DATASET_TORCH_VERSION": "002",
    "dataset_local_dir": "/z/dat/F-PHAB/F-PHAB_media_v000",  # F-PHAB_media_v000, # "/z/dat/CookBook/MILLYCookbook_media_v000", "/z/dat/EpicKitchens50/EpicKitchens50_media_v000", "/Users/yayuanli/dat/EpicKitchens50/EpicKitchens50_media_v000", "/z/dat/PennAction/Penn_Action_media_v000", "/z/dat/F-PHAB/F-PHAB_media_v000"
    "num_workers":  os.cpu_count(), # ,
    "train_shuffle": True, # otherwise, all False
    "dataset_loading_by": 'frame', # 'sample', 'frame'
    'check_videos_flag': True,
    'skeleton_info': 
                    # {'labels': ["nose", "right shoulder", "left shoulder", "right elbow", "left elbow", "right wrist", "left wrist", "right hip", "left hip", "right knee", "left knee", "right ankle", "left ankle"], 'edges': [[5, 3, 1, 2, 4, 6], [11, 9, 7, 8, 10, 12]]},
                    {"labels": ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], "edges": [[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]]},
    'fix_language': None, # "do something", # None or String
    'use_open_vocab_phrase': False, # If true, use predefined open vocabulary variant phrase. Otherwise, use the original description (action annotation)
    # pose
    "pose_version": "v001",
    "complete_pose_version": "v001",
    "forecasting_groundtruth_version": "v001",
    # saving feature
    "feature_name": "blip_feature_extractor.pt",
    # savings
    "vol_dir": "/z/exp/lgpf",
}
# exp_config["patch_size"] = (exp_config["img_size"][0]//exp_config["grid_len"], exp_config["img_size"][1]//exp_config["grid_len"])
exp_config["vocab_size"] = exp_config["grid_len"]**2
# wandb.config["deco_num_hiddens"] = wandb.config["enco_num_hiddens"]
exp_config["deco_norm_shape"] = [exp_config["num_hiddens"]]
exp_config["num_all_coords"] = exp_config["num_indicators"] * exp_config["num_joints"] * exp_config["num_coords"]
exp_config["_seq_len"] = exp_config["forcasting_steps"]
exp_config["prefetch_factor"] = 4 if not exp_config["debug_mode"] else None
exp_config['total_batch_size'] = 128 if not exp_config["debug_mode"] else exp_config['batch_size'] * 2

def experiment_init():
    """
    assume the folde structure is
    - project foler
      - code
      - outputs
          - expoeriments
            - exp_0001_202301061327
    And assume this is in code/ fodler
    """
    g = set_determinism()

    vol_dir = exp_config['vol_dir'] # vol_dir = "/z/exp/lgpf"
    code_dir_path = os.getcwd()

    # exp_base_folder
    exp_base_folder = os.path.join(vol_dir, code_dir_path.replace("code", "outputs/experiments")[1:])
    os.makedirs(os.path.join(exp_base_folder, "exp_0"), exist_ok=True)

    pre_exp_path_list = glob.glob(os.path.join(exp_base_folder, "exp_*"))
    pre_exp_path_list = sorted(pre_exp_path_list, key=lambda p: int(p.split("/")[-1].split("_")[1]))
    latest_exp_no = int(pre_exp_path_list[-1].split("/")[-1].split("_")[1])
    exp_no = latest_exp_no + 1
    exp_folder_name_patt = lambda nt: f"exp_{nt[0]:04d}_{nt[1]}"
  
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    exp_name = exp_folder_name_patt((exp_no, timestamp))
    exp_path = os.path.join(exp_base_folder, exp_name)
    os.mkdir(exp_path)
    # experiment info - in note
    
    command_rsync_code = f"rsync -a --dirs {code_dir_path} {exp_path}"
    os.system(f"{command_rsync_code}")
    
    # output path
    exp_output_folder_path = os.path.join(exp_path, "outputs")
    os.mkdir(exp_output_folder_path)
    
    # update config
    exp_config.update({
            "exp_path": exp_path,
            "exp_output_folder_path": exp_output_folder_path,        
        })
    
    # wandb
    project_name = exp_base_folder.replace('/', '_')
    try:
        run = wandb.init(project=project_name, name=f"{exp_name}", save_code=True, dir=exp_path, mode="online", config=exp_config)
    except Exception as e:
        print(f"wandb init failed. do offline:\n{e}")
        run = wandb.init(project=project_name, name=f"{exp_name}", save_code=True, dir=exp_path, mode="offline", config=exp_config)
        
    # print(f"{wandb.config=}")
    # pretty_print_dict(wandb.config)
    pretty_print_dict(wandb.config)
    # return exp_path, g, run
    return run

def pretty_print_dict(input_dict):
    import pprint    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(input_dict)

def set_determinism():
    # determinism 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)       
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def grad_clipping(net, theta):
    """Clip the gradient.
    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Timer:
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        # self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    def reset(self):
        pre_sum = self.times
        self.times = []
        return pre_sum # a new storage

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def myset(self, x, v):
        if "values" in self.__setitem__(x).keys():
            self.__setitem__(x)["values"][0] = v
        else:
            self.__setitem__(x)["value"] = v

    def myget(self, x):
        if "values" in self.get(x).keys():
            return self.get(x)["values"][0]
        elif 'value' in self.get(x).keys():
            return self.get(x)["value"]
        elif 'min' in self.get(x).keys():
            return self.get(x)['min']
    __getattr__ = myget
    __setattr__ = myset
    __delattr__ = (
        lambda self, x: self.__delitem__(x)["values"][0]
        if "values" in self.__delitem__(x).keys()
        else self.__delitem__(x)["value"]
    )

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class MyAccumulator:
    def __init__(self, term_names):
        self.data = {term: [] for term in  term_names}
        
    def add(self, terms_dict):
        [self.data[key].append(val) for key, val in terms_dict.items()]
        
    def sum(self, term_name):
        return sum(self.data[term_name])
    def __str__(self):
        return f"{self=}, {len(self.data.keys())=}"
    def mean(self, term_name):
        if len(self.data[term_name]) == 0:
            return 0
        else:
            return sum(self.data[term_name])/len(self.data[term_name])
class SeqDict:
    """
    The value of this class is mainly in add() function. It zip the given keys (input id or frame_path or (frame_path, text)) and values (forecasting prediction)
    """
    def __init__(self):
        self.data = {}
    def add(self, ids, pred_seq):
        """
        args:
            ids: a list of keys (strings (usually a batch)). e.g., LV input pair [(frame_fileapath, prompt), ...]
            pred_seq: a list of values. e.g., cur pose and forecasted poses [(torch(1, 84), torch(30, 84)), ...] 
        """
        for i in range(len(ids)):
            self.data[ids[i]] = pred_seq[i] # a tensor (seq_len, #all_coords)
            

def dump_prediction(pred_dict):
    new_dict = {}
    for frame_id, pred in pred_dict:
        new_dict[frame_id] = pred
        
def find_folders(base_dir, re_pattern):
    """find 1st-level folders that matches re_pattern"""
    matching_folders = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if re.match(re_pattern, dir_name):
                matched_dir_path = os.path.join(root, dir_name)
                matching_folders.append(matched_dir_path)
        break
    return matching_folders                


def pretty_print_dict(dict):
    print(f'=============dict printing================')
    for k, v in dict.items():
        print(f"{k}: {v}")
    print(f'=============dict printing end================')

def get_easy_samples(forecasting_tensor, threshold):
    """
    args:
        forecasting_tensor: (batch_size, seq_len, #all_coords)
    return:
        duplication_count: (batch_size, seq_len). Each count is the number of the same pose as the pose at current seq in the future seqs. For example, the forecasting annotation in the last frame [batch_i] in a clip has all duplicated future poses for all timestamp -- [30, 29, 28, ..., 1]
    """
    duplication_count = torch.zeros(list(forecasting_tensor.shape[:2]))
    # Loop over the first dimension
    for i in range(forecasting_tensor.shape[0]):
        # Loop over the second dimension
        for j in range(forecasting_tensor.shape[1]):
            if (torch.norm(forecasting_tensor[i,:-1,:]-forecasting_tensor[i,1::,:]) / forecasting_tensor.shape[2] ) <= threshold:
                duplication_count[i, j] += 1
            # for k in range(j+1, forecasting_tensor.shape[1]):
            #     # Check if two vectors along the second dimension are equal
            #     if (torch.norm(forecasting_tensor[i, j, :] - forecasting_tensor[i, k, :]) / forecasting_tensor.shape[2]) <= threshold:
            #         duplication_count[i, j] += 1
    return duplication_count

