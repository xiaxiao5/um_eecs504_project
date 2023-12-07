# %load_ext autoreload
# %autoreload 2
# import sys
# sys.path.insert(0, "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp10/code")
import nvidia_utils
nvidia_utils.set_cuda_visible_device()
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.backends.cudnn as cudnn
cudnn.benchmark = False
import sys
import os
# NUM_GPUs = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = # ",".join(map(str, range(NUM_GPUs)))
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.path.basename(os.getcwd())
import dataset as D
import utils as U
import network as N
import train_val as TV
import time
import wandb
import fiftyone as fo
import fiftyonize_utils as fu
from fiftyone import ViewField as FF
import socket
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import pose as P
import evaluate as E

def train_val(validation_only=False, validation_split='val'):
    # net
    print(f"preparing network")
    net = N.get_network().to(wandb.config["device"])
    # net = nn.DataParallel(net).to(wandb.config["device"])

    # dataset 
    print(f"preparing dataset")
    data_loaders = D.get_iterators()
    D.set_preprocess(data_loaders, net)
    
    # train
    if validation_only == False:
        print(f"train and val")
        TV.train(net, data_loaders)
    elif validation_only == True:
        print(f"val {validation_split=}")

        # load model
        net.eval()
        inf_model_path = os.path.join(list(wandb.config['vis_exp_path_dict'].values())[0], 'outputs', 'best_model_val.pt')
        net.load_state_dict(torch.load(inf_model_path))

        # val
        TV.train(net, {'val': data_loaders[validation_split]})

        metric_exp_path = wandb.config['exp_path']        
        compute_metric(metric_exp_path)


def inf_zero_shot_language(dataset_split_fo, inf_model_path):
    """
    Given a network, FO Frame Dataset which indludes frame path and narration (input pair), this function outputs prediction of the network.
    
    args:
        dataset_split_fo: FO Frmae Dataset. All frames in there will be inferenced with the wandb.config['narration_field_name'] as narration
    wandb.config:
        inf_model_path: the model path to be used to inference. if "", no weights loading
        narration_field_name: the Frame level field name that save the narration to be used for inference for each frame
        batch_size, num_workers, prefetch_factor, train_shuffle
    output:
        the predictions dict ((frame_path, text)->torch prediction) saved in exp folder by TV.inference
    return:
        exp_output_folder_path (str)
    """
    # net
    print(f"preparing network")
    net = N.get_network()
    net.eval()
    if inf_model_path != "":
        net.load_state_dict(torch.load(inf_model_path))
    
    data_loader = D.DataLoader(D.MyDataset_v002(dataset_split_fo, split="inf", loading_by='frame', loading_by_split_name='inf'), wandb.config["batch_size"],
                                num_workers=0, # wandb.config["num_workers"],
                                prefetch_factor=None, # wandb.config["prefetch_factor"],
                                shuffle=False)
    
    D.set_preprocess({'inf': data_loader}, net)
    
    # train
    print(f"inferencing")
    TV.inference(net, data_loader)
    
    return wandb.config['exp_output_folder_path']

def compare_multimodal_feature(inf_modal_path, input_pairs):
    # net
    print(f"preparing network")
    net = N.get_network()
    net.load_state_dict(torch.load(inf_model_path))
    net.eval()
    net.to(wandb.config['device'])
    
    feature_dict_list = []
    for raw_image, raw_text in input_pairs:
        image_input = net.encoder_image_preprocess(raw_image).unsqueeze(0).to(wandb.config['device'])
        text_input = net.encoder_text_preprocess(raw_text).to(wandb.config['device'])
        # print(f"{text_input=}")
        feature_dict = net.get_multimodal_feature(image_input, text_input)
        feature_dict_list.append(feature_dict)
        
    for feature_name in feature_dict_list[0].keys():        
        similarity = torch.sum(torch.diag(feature_dict_list[0][feature_name][0,:,:] @ feature_dict_list[1][feature_name][0,:,:].t()))
        print(f"{feature_name=}: {similarity=}")

def save_BLIP_ek50():
    # net
    print(f"preparing network")
    net = N.get_network()

    import EpicKitchens50 as mydata
    dataset_raw, dataset_fo, session = mydata.make_fiftyone_dataset(DATASET_FO_VERSION=wandb.config["DATASET_FO_VERSION"], dataset_local_dir=wandb.config["dataset_local_dir"], window_size=wandb.config["window_size"], grid_len=wandb.config["grid_len"], hand_side=wandb.config["hand_side"], landmark_name=wandb.config["landmark_name"], scope_dict=wandb.config["scope_dict"], light_mode=wandb.config["debug_mode"], force_reload=wandb.config["force_reload"], visualize=wandb.config["visualize"], dataset_prefix=wandb.config["dataset_prefix"])
    
    dataset_fo = dataset_raw.to_clips("narration")
    
    # get global frame no map
    dataset_dir = dataset_raw.info['dataset_dir']
    dataset_name = dataset_raw.name
    map_split='allframes_inclip'
    dataset_raw.info[f'global_frame_no_to_clip_frame_no_ordered_list_{map_split}'] = None
    dataset_raw.info[f'global_frame_no_to_clip_frame_no_ordered_list_{map_split}_path'] = fu.get_global_frame_no_to_clip_frame_no_ordered_list(dataset_fo, return_path=f"{os.path.join(dataset_dir, 'fiftyone', dataset_name, f'get_global_frame_no_to_clip_frame_no_ordered_list_{map_split}_{socket.gethostname()}.json')}")
    dataset_raw.save()
    
    # make laoder
    split='val'
    if not wandb.config["debug_mode"]:
        dataset = D.MyDataset_v002(dataset_fo, split=split, loading_by='frame', loading_by_split_name=map_split)
        data_loader = D.DataLoader(dataset, wandb.config["batch_size"], num_workers=wandb.config["num_workers"], prefetch_factor=wandb.config["prefetch_factor"], shuffle=False)
    else:
        dataset = D.MyDataset_v002(dataset_fo, split=split, loading_by='frame', loading_by_split_name=map_split)
        data_loader = DataLoader(dataset, wandb.config["batch_size"], num_workers=0, prefetch_factor=None, shuffle=False)
    data_loaders = {'train': None, 'val': data_loader} 
    D.set_preprocess(data_loaders, net)
    
    
    # train
    print(f"running")
    TV.train(net, data_loaders)    
    
def compute_metric(metric_exp_path, split='val'):
    # # metric
    # # # load prediction and label
    prediction_path = os.path.join(metric_exp_path, "outputs", f"best_predictions_{split}.pt")
    label_path = os.path.join(metric_exp_path, "outputs", f"labels_{split}.pt")
    prediction_saving = torch.load(prediction_path)
    prediction_torch = torch.stack(list(prediction_saving.values())).reshape(-1, wandb.config['forcasting_steps'], wandb.config['num_indicators'], wandb.config['num_joints'], wandb.config['num_coords'])
    label_saving = torch.load(label_path)
    label_torch = torch.stack(list(label_saving.values())).reshape(-1, wandb.config['forcasting_steps'], wandb.config['num_indicators'], wandb.config['num_joints'], wandb.config['num_coords'])

    # evaluate
    metric_info = {'rmse': {'axis': [1,2,3,4]}, 
                    'pck': {'axis': [1,2,3], 'alpha': wandb.config['pck_alpha']},
                    'ade': {'axis': [1, 2]},
                    'fde': {'axis': [1]}}
    metric_results = E.evaluate(prediction_torch, label_torch, metric_info=metric_info)
    for metric_name, metric_result in metric_results.items():
        print(f"{metric_name}: {list(metric_result.shape)=} {metric_result.mean().item()=:.4f}, {metric_result.std().item()=:.4f}, {metric_result.max().item()=:.4f}, {metric_result.min().item()=:.4f}")

    return {'metric_results': metric_results, 'prediction_saving': prediction_saving, 'label_saving': label_saving}

def visualize(exp_folder, split, post_fix=""):
    # load prediction and label
    metric_name = 'rmse'
    metric_results, prediction_torch, label_torch = list(compute_metric(metric_exp_path=exp_folder, split=split).values())

    # compute select best clips
    # get indexs of easy samples (forecasting annotation are duplicated like at the last several frames of a clips)
    easy_bar = 0 # 0~wandb.config['forcasting_steps'] # the lower, the more samples are considered easy. 0 means having 1 future duplicate pose in a sample annotation will be considered easy
    easy_index_list = torch.nonzero(torch.sum(U.get_easy_samples(torch.stack(list(label_torch.values())), threshold=0), axis=1) > easy_bar).squeeze().tolist()
    if isinstance(easy_index_list, int):
        easy_index_list = [easy_index_list]
    # sort all samples by metric
    sorted_metric, sorted_index = torch.sort(metric_results[metric_name])
    # explicit check sorted non-easy samples
    # cp_sorted_metric = sorted_metric.tolist()
    # for s_i in range(len(sorted_metric)):
    #     if sorted_index[s_i] in easy_index_list:
    #         cp_sorted_metric.remove(sorted_metric[s_i])
    # print(cp_sorted_metric)
    # get top clips
    top_clips = {}
    top_num = 5
    # iter frames to find top performance clips. break when top_num clips are found
    for s_i in range(len(sorted_metric)):
        if len(top_clips.keys()) >= top_num:
            break
        if sorted_index[s_i] in easy_index_list:
            continue

        video_frame_folder = '/'.join(list(label_torch.keys())[sorted_index[s_i]][0].split('/')[:-2])
        if video_frame_folder not in top_clips.keys():
            top_clips[video_frame_folder] = [sorted_index[s_i]]
        else:
            top_clips[video_frame_folder].append(sorted_index[s_i])
    # print(f"{top_clips=}")

    # visualize selected clips
    dataset = fo.Dataset()
    dataset.default_skeleton = fo.KeypointSkeleton(**wandb.config['skeleton_info'])
    for video_frame_folder, sample_index_list in top_clips.items():
        video_sample = fo.Sample(filepath=video_frame_folder+'.mp4')
        # select all indexs where the frames are in this video folder
        video_frame_index_list = [sample_i for sample_i, input_info_ in enumerate(label_torch.keys()) if video_frame_folder in input_info_[0]]
        # load info for all frames in this video folder
        for sample_i in video_frame_index_list: # sample_index_list:
            frame_path = list(label_torch.keys())[sample_i][0]
            f_number = int(frame_path.split('/')[-2])
            inference_text, pose_start = list(prediction_torch.keys())[sample_i][1:]
            pose_prediction = list(prediction_torch.values())[sample_i]
            pose_label = list(label_torch.values())[sample_i]
            metric_dict = {m_name: fo.Classification(label=f"{m_torch[sample_i]:.4f}") for m_name, m_torch in metric_results.items()}
            pose_prediction_Keypoints, pose_prediction_Keypoints_trace = P.forecasting_torch_to_fo(pose_prediction, 
                                                                                  prompt=inference_text,
                                                                                  include_cur_pose=False,
                                                                                  num_indicators=wandb.config['num_indicators'], 
                                                                                  num_joints=wandb.config['num_joints'], 
                                                                                  num_coords=wandb.config['num_coords'])

            pose_label_Keypoints, pose_label_Keypoints_trace = P.forecasting_torch_to_fo(pose_label, 
                                                                                  prompt=inference_text,
                                                                                  include_cur_pose=False,
                                                                                  num_indicators=wandb.config['num_indicators'], 
                                                                                  num_joints=wandb.config['num_joints'], 
                                                                                  num_coords=wandb.config['num_coords'])
            frame_sample_argument = {'filepath': frame_path,
                                     f'inference_text{post_fix[-10:]}': fo.Classification(label=inference_text),
                                     'pose_start': P.pose_torch_to_fo(pose_start, 
                                                                      num_joints=wandb.config['num_joints'], 
                                                                      num_coords=wandb.config['num_coords']),
                                     f'pose_prediction{inference_text}': pose_prediction_Keypoints,
                                     f'pose_prediction_trace{inference_text}': pose_prediction_Keypoints_trace,
                                     'pose_label': pose_label_Keypoints,
                                     'pose_label_trace': pose_label_Keypoints_trace,
                                     **metric_dict
                                    }
            
            video_sample.frames[f_number] = fo.Frame(**frame_sample_argument)

        dataset.add_sample(video_sample)

    return dataset

def inference(inference_scope):
    validation_split = 'val'
    # net
    print(f"preparing network")
    net = N.get_network().to(wandb.config["device"])
    # load model
    net.eval()
    inf_model_path = os.path.join(list(wandb.config['vis_exp_path_dict'].values())[0], 'outputs', 'best_model_val.pt')
    net.load_state_dict(torch.load(inf_model_path))

    # dataset
    print(f"preparing dataset")
    data_loaders = D.get_iterators(inference_scope)
    D.set_preprocess(data_loaders, net)

    # val
    TV.train(net, {'val': data_loaders[validation_split]}, predicting=True)

    metric_exp_path = wandb.config['exp_path']
    compute_metric(metric_exp_path)

def visualize_multi_narration(exp_folder_dict):
    dataset = fo.Dataset()
    for exp_name, exp_folder in exp_folder_dict.items():
        if os.path.exists(exp_folder):
            dataset_ = visualize(exp_folder, split='val', post_fix=f"_{exp_name}")
        else:
            dataset_ = fo.load_dataset(exp_folder)
        dataset.merge_samples(dataset_)
    return dataset
    

if __name__ == "__main__":
    function="inference"
    start_time = time.time()
    if function == 'train':
        ## train ##
        with U.experiment_init() as run:
            wandb.config['function'] = function
            train_val()
        ## train END ##
    elif function == 'validation':
        with U.experiment_init() as run: 
            wandb.config['function'] = function
            train_val(validation_only=True, validation_split='val')
    elif function == 'metric':
        with U.experiment_init() as run: 
            wandb.config['function'] = function
            for exp_name, exp_folder in wandb.config['vis_exp_path_dict'].items():
                print(f"{exp_name=}")
                compute_metric(exp_folder, 'val')
    elif function == 'visualize':
        with U.experiment_init() as run:
            wandb.config['function'] = function
            for exp_name, exp_folder in wandb.config['vis_exp_path_dict'].items():
                print(f"{exp_name=}") 
                visualize(exp_folder=exp_folder, split='val')
    elif function == 'inference':
        """
        inference specific clips in dataset.
        vis_exp_path_dict: exp path for pretrained model weights
        """
        # with U.experiment_init() as run: 
        #     wandb.config['function'] = function        
        video_filepath = "/z/dat/F-PHAB/F-PHAB_media_v000/Video_files/Subject_2/put_sugar/3/color.mp4"
        inference_text_list = ['put_sugar', 'add sugar', 'sprinkle sugar', 'sweeten with sugar', 'dash sugar', 'scatter sugar', 'charge_cell_phone', 'clean_glasses', 'close_juice_bottle', 'close_liquid_soap', 'close_milk', 'close_peanut_butter', 'drink_mug', 'flip_pages', 'flip_sponge', 'give_card', 'give_coin', 'handshake', 'high_five', 'light_candle', 'open_juice_bottle', 'open_letter', 'open_liquid_soap', 'open_milk', 'open_peanut_butter', 'open_soda_can', 'open_wallet', 'pour_juice_bottle', 'pour_liquid_soap', 'pour_milk', 'pour_wine', 'prick', 'put_salt', 'put_sugar', 'put_tea_bag', 'read_letter', 'receive_coin', 'scoop_spoon', 'scratch_sponge', 'sprinkle', 'squeeze_paper', 'squeeze_sponge', 'stir', 'take_letter_from_enveloppe', 'tear_paper', 'toast_wine', 'unfold_glasses', 'use_calculator', 'use_flash', 'wash_sponge', 'write'][:10]

        # dict of exp name and prediction saving path for visualization
        # exp_path_dict = {"fphabv001_exp13.10_fo5": "F-PHAB_FO_v001_Subject_2_put_sugar_3_compare_two_narr"}
        exp_path_dict = {}


        for inference_text in inference_text_list:
            inference_scope = {video_filepath: inference_text}
            with U.experiment_init() as run: 
                wandb.config['function'] = function                
                inference(inference_scope=inference_scope)
                exp_path_dict.update({inference_text: wandb.config['exp_path'] })
                
        with U.experiment_init() as run: 
            wandb.config['function'] = function                    
            dataset = visualize_multi_narration(exp_folder_dict=exp_path_dict)
            dataset.persistent = True
            dataset.save()
        pass

    elif function == 'visualize_multi_narration':
        with U.experiment_init() as run: 
            wandb.config['function'] = function        
            dataset = visualize_multi_narration(exp_folder_dict=wandb.config['vis_exp_path_dict'])

    elif function == "visualize_annotation":
        with U.experiment_init() as run:
            wandb.config['function'] = function
            dataset_raw_, _, _ = D.get_FO_dataset()
            vis_name = dataset_raw_.name+"_vis"
            if vis_name in fo.list_datasets():
                fo.delete_dataset(vis_name)
            dataset_raw = dataset_raw_.clone(name=vis_name, persistent=True)
            dataset_clips = dataset_raw.load_saved_view("clips")
            session = fo.launch_app()
            
            dataset_clip_vis = dataset_clips
            for clip_sample in dataset_clip_vis.iter_samples(progress=True):
                for frame_i in range(len(clip_sample.frames)):
                    frame_sample = clip_sample.frames[frame_i+1]
                    try:
                        pose_fo = P.read_Keypoints(frame_sample[wandb.config["groundtruth_field_name"]+"_path"])
                    except Exception as e:
                        print(f"[ERROR] {frame_sample['filepath']=} {e=}")
                        pose_fo = fo.Keypoints()
                    frame_sample[wandb.config["groundtruth_field_name"]] = pose_fo
                clip_sample.save()
            session.wait()
            print()

    elif function == "visualize_saved":
        with U.experiment_init() as run:
            wandb.config['function'] = function
            print(f"visualizing...")
            tags_in_name = 'val'
            assert tags_in_name == 'val', "only support val for now since savings for training is not in order while narrations is"
            vis_post_fix, vis_exp_path = list(wandb.config['vis_exp_path_dict'].items())[0]
            
            dataset_fo = D.get_FO_dataset()[0].match_tags(tags_in_name)            
            narrations_ = dataset_fo.values(['frames.filepath', f"frames.{wandb.config['narration_field_name']}"], unwind=True)
            filepath_list = narrations_[0]
            narrations = dict(zip(*narrations_)) # {frame_filepath: narration}

            prediction_path = os.path.join(vis_exp_path, "outputs", "best_predictions_val.pt")
            label_path = os.path.join(vis_exp_path, "outputs", "labels_val.pt")
            prediction_torch = torch.stack(list(torch.load(prediction_path).values())).reshape(-1, wandb.config['forcasting_steps'], wandb.config['num_indicators'], wandb.config['num_joints'], wandb.config['num_coords'])
            label_torch = torch.stack(list(torch.load(label_path).values())).reshape(-1, wandb.config['forcasting_steps'], wandb.config['num_indicators'], wandb.config['num_joints'], wandb.config['num_coords'])
            metric_results = E.evaluate(prediction_torch, label_torch, metric_info={'rmse': {'axis': [1,2,3,4]}, 'pck': {'axis': [1,2,3]}})
            for metric_name, metric_result in metric_results.items():
                print(f"{metric_name}: {metric_result.mean()}")

            dataset_raw = fu.visualize_predictions(prediction_path=prediction_path,
                                                   output_dir=os.path.join(wandb.config['dataset_local_dir'], "lgpf_visualization", "predictions"),
                                                   fps=30,
                                                   to_disk=True,
                                                   label_path=label_path,
                                                   vis_post_fix=vis_post_fix,
                                                   tags_in_name=tags_in_name,
                                                   num_indicators=wandb.config['num_indicators'], 
                                                   num_joints=wandb.config['num_joints'], 
                                                   num_coords=wandb.config['num_coords'],
                                                   narrations=narrations,
                                                   metric_results=metric_results,
                                                   max_length=1000)
            print()

    elif function == 'inference_visualize' or function == 'inference_visualize_byrmse':
        with U.experiment_init() as run:
            print(f"initializing experiment")
            wandb.config['function'] = function
            all_visualized_videos = []
            
            print(f"preparing dataset for all clips to be visualized...")
            dataset_raw, dataset_fo, session = D.get_FO_dataset()
            video_list_train = dataset_raw.match_tags("train").values('filepath')
            video_list_val = dataset_raw.match_tags("val").values('filepath')
            dataset_all_narration = dataset_raw.to_clips("narration").clone()
            if function == 'inference_visualize':
                vis_num = 1
                narration_inf_text = 'baseball_pitch'
                # vis_clip_ids = dataset_all_narration.match_tags("val").match(FF('narration.label')!=narration_inf_text).limit(vis_num).values('id')
                vis_clip_ids = dataset_all_narration.match_tags("train").limit(vis_num).values('id')
                # all_dataset_clips = dataset_all_narration.match_tags("train").skip(10).limit(vis_num) + dataset_all_narration.match_tags("val").limit(vis_num) + dataset_all_narration.match_tags("train").take(vis_num) + dataset_all_narration.match_tags("val").take(vis_num)
                # vis_clip_ids = all_dataset_clips.values('id')
                # 
                # vis_clip_ids_train = dataset_all_narration.match_tags("train").take(vis_num).values('id')
                # vis_clip_ids_val = dataset_all_narration.match_tags("val").take(vis_num).values('id')
                # vis_clip_ids = vis_clip_ids_train + vis_clip_ids_val
                # 
                # vis_clip_ids = dataset_all_narration.match((FF("filepath")=='/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P06/P06_05.MP4') & (FF('support')==[1936, 2193])).values('id')
            elif function == 'inference_visualize_byrmse':
                vis_clip_ids = dataset_all_narration.match_tags("val").values('id')
            
            all_dataset_clips = dataset_all_narration.select(vis_clip_ids)
            if wandb.config['narration_field_name_inf'] != wandb.config['narration_field_name']:
                all_dataset_clips.set_values('narration_inf', [fo.Classification(label=narration_inf_text)]*len(all_dataset_clips))
            for clip_info in list(zip(*all_dataset_clips.values(["filepath", "support", f"{wandb.config['narration_field_name_inf']}.label"]))):
                print(f"{'train' if clip_info[0] in video_list_train else 'val'}:\t{clip_info}")


            if function == 'inference_visualize_byrmse':
                dataset_clips = all_dataset_clips
                print(f"inferencing and visualizing {len(dataset_clips)=} {dataset_clips.count('frames')=}...") 
                dataset_clips.info['global_frame_no_to_clip_frame_no_ordered_list_inf'] = fu.get_global_frame_no_to_clip_frame_no_ordered_list(dataset_clips)
                dataset_split_fo = dataset_clips
                inf_model_path_dict = wandb.config["inf_model_path_dict"]
                assert len(inf_model_path_dict) == 1, f"please visualize one model at a time to have correct vis_post_fix. got {len(inf_model_path_list)}"
                
                print(f"inferencing...")
                prediction_path_list = []
                vis_post_fix_list = []
                for vis_post_fix, inf_model_path in inf_model_path_dict.items():
                    print(f"{inf_model_path=}")
                    prediction_folder_path = inf_zero_shot_language(dataset_split_fo=dataset_split_fo, inf_model_path=inf_model_path)
                    prediction_path_list.append(os.path.join(prediction_folder_path, "best_predictions_inf.pt"))
                    vis_post_fix_list.append(vis_post_fix)
                    print(f"DONE {prediction_folder_path=}")
                vis_post_fix = vis_post_fix_list[0]
                print(f"{prediction_path_list=}")
                print(f"{vis_post_fix_list=}")
                
                # choose low rmse frames to visualize
                vis_num = len(dataset_clips)
                prediction_path = prediction_path_list[0]
                prediction_dict = torch.load(prediction_path)
                label_path = prediction_path.replace("best_predictions_inf.pt", "labels_inf.pt")
                label_dict = torch.load(label_path)
                label_tensor = torch.stack([v['pose_label'] for v in label_dict.values()])
                all_keys = list(label_dict.keys())
                
                # filter frames that have duplicated foreacasting label poses
                indexs_1 = torch.nonzero(((label_tensor[:,-1:,:]-label_tensor).sum(dim=2)==0).sum(dim=1) == 1).reshape(-1).tolist()
                key_1 = [all_keys[index_] for index_ in indexs_1]
                prediction_dict_1 = dict([kv for kv in prediction_dict.items() if kv[0] in key_1])
                
                # filter frames that have lower rmse
                new_prediction = dict(sorted(prediction_dict_1.items(), key=lambda kv: kv[1]['rmse'])[:vis_num])
                # choose low rmse frames to visualize
                new_prediction_path = prediction_path.replace('best_predictions_inf.pt', f'best_predictions_inf_rmselow{vis_num}.pt')
                torch.save(new_prediction, new_prediction_path)
                
                new_keys = list(new_prediction.keys())
                new_label_dict = dict([kv for kv in label_dict.items() if kv[0] in new_keys])
                new_label_path = label_path.replace('labels_inf.pt', f'labels_inf_rmselow{vis_num}.pt')
                torch.save(new_label_dict, new_label_path)
                
                print(f"visualizing...")
                all_frame_draw_path, video_draw_path, dataset_raw = fu.visualize_predictions(prediction_path=new_prediction_path,
                                                                                          output_dir=os.path.join(wandb.config['dataset_local_dir'], "lgpf_visualization", "predictions"),
                                                                                          fps=15,
                                                                                          to_disk=True,
                                                                                          label_path=new_label_path,
                                                                                          vis_post_fix=vis_post_fix,
                                                                                          tags_in_name='val')
                all_visualized_videos.append(video_draw_path)
                
            elif function == "inference_visualize":
                print(f"inferencing and visualizing clips one by one...")
                for ci in range(0,len(all_dataset_clips)):
                    dataset_clips = all_dataset_clips.skip(ci).limit(1)
                    print(f"inferencing and visualizing [{dataset_clips.first()[wandb.config['narration_field_name_inf']].label}] {dataset_clips.first().filepath} {dataset_clips.first().support}...") 
                    dataset_clips.info['global_frame_no_to_clip_frame_no_ordered_list_inf'] = fu.get_global_frame_no_to_clip_frame_no_ordered_list(dataset_clips)
                    dataset_split_fo = dataset_clips
                    inf_model_path_dict = wandb.config['vis_exp_path_dict']# inf_model_path_dict = wandb.config["inf_model_path_dict"]
                    # assert len(inf_model_path_dict) == 1, f"please visualize one model at a time to have correct vis_post_fix. got {len(inf_model_path_list)}"
                    
                    print(f"inferencing...")
                    prediction_path_list = []
                    vis_post_fix_list = []
                    for vis_post_fix, inf_model_dir in inf_model_path_dict.items():
                        inf_model_path = os.path.join(inf_model_dir, 'outputs', 'best_model_val.pt') # print(f"{inf_model_path=}")
                        prediction_folder_path = inf_zero_shot_language(dataset_split_fo=dataset_split_fo, inf_model_path=inf_model_path)
                        prediction_path_list.append(os.path.join(prediction_folder_path, "best_predictions_inf.pt"))
                        vis_post_fix_list.append(vis_post_fix)
                        print(f"DONE {prediction_folder_path=}")
                    vis_post_fix = vis_post_fix_list[0]
                    print(f"{prediction_path_list=}")
                    print(f"{vis_post_fix_list=}")
                    
                    
                    print(f"visualizing...")                    
                    fps=15
                    output_dir = os.path.join(wandb.config['dataset_local_dir'], "lgpf_visualization", "predictions")
                    
                    video_draw_list = fu.visualize_forecasting_annotation(dataset_clips=dataset_clips,
                                                                          output_dir=output_dir,
                                                                          narration_field=wandb.config['narration_field_name'],
                                                                          vis_post_fix=vis_post_fix,
                                                                          frame_scope=None,
                                                                          prediction_path_list=prediction_path_list,
                                                                          num_indicators=wandb.config['num_indicators'], 
                                                                          num_joints=wandb.config['num_joints'],
                                                                          num_coords=wandb.config['num_coords'],
                                                                          )
                    print(f"{video_draw_list=}\n")
                    all_visualized_videos.append(video_draw_list)
        
        print(f"{all_visualized_videos=}")
        
    elif function=="save_BLIP_ek50": 
        with U.experiment_init() as run:
            wandb.config['function'] = function
            save_BLIP_ek50()
        
    elif function=="fiftyonize":
        with U.experiment_init() as run:
            wandb.config['function'] = function
            D.get_FO_dataset()
    print(f"DONE in {time.time()-start_time:.2f}s")




