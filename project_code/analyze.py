from moviepy.editor import ImageSequenceClip
import wandb 
import fiftyone as fo
from fiftyone import ViewField as FF
import time
import torch
import json  
import MILLYCookbook as mc
import glob
import network as N
from PIL import Image
import fiftyone.utils.annotations as foua
import os
import shutil
import numpy as np
import pose as P

def visualize(dataset, dataset_clips, config, frame_info):
    """
    assumeption:
        dataset_frame and dataset_clips are fo.core.dataset only contains samples I'm interested
    args:
        frame_info: frame_id -> prompt -> {"prediction", "loss"}
        
    """
    max_forcasting_steps = 31
#     dataset.skeletons["pose_pred"] = dataset.skeletons["pose"]                
#     dataset.skeletons["pose_pred_trace"] = fo.KeypointSkeleton(labels=[f"forcasting_step_{forcasting_step}" for forcasting_step in range(max_forcasting_steps)], edges=[list(range(max_forcasting_steps))])
    with fo.ProgressBar(iters_str="clips") as pb:
        for clip_sample in pb(dataset_clips):
            v_sample = dataset[clip_sample.sample_id]
            for frame_sample_id in clip_sample:
                frame_sample = v_sample.frames[frame_sample_id]
                frame_id = frame_sample["id"]
                prompt_info_dict = frame_info[frame_id]
#                 fo_poses = []
#                 fo_traces = []
                frame_sample["loss"] = {}
                frame_sample["prompt_list"] = []
                for text_i, (text, prompt_info) in enumerate(prompt_info_dict.items()):
                    text_short = get_short_text(text)
                    if "loss" in prompt_info.keys():
                        frame_sample["loss"][text] = prompt_info["loss"]
                        # frame_sample["loss"] = fo.Classification(prompt=text, label=prompt_info["loss"])
                    fo_poses = []
                    fo_traces = []                    
                    forcasting = prompt_info["prediction"]                
                    forcasting = forcasting.view(-1, config["num_indicators"], config["num_joints"], config["num_coords"])
                    mean_points = {"Left": [], "Right": []}
                    for forcasting_step, all_indicators in enumerate(forcasting):
                        
                        for indicator_i, indicator in enumerate(all_indicators):
                            side = "Left" if indicator_i%2==0 else "Right"
                            indicator_fo = fo.Keypoint(points=indicator.tolist(), prompt=text, hand_side=side, label=side, forcasting_step=forcasting_step, index=forcasting_step)
                            fo_poses.append(indicator_fo)
                            mean_points[side].append(indicator.mean(dim=0).tolist()) # mean point of one hand
                            
                    # connect mean points among all forcasting steps for each indicator
                    for side, points in mean_points.items():
                        trace_fo = fo.Keypoint(points=points, hand_side=side, prompt=text, label=side)
                        fo_traces.append(trace_fo)
                        
                    
                    frame_sample[f"pose_pred_{text_short}"] = fo.Keypoints(keypoints=fo_poses)
                    frame_sample[f"pose_pred_trace_{text_short}"] = fo.Keypoints(keypoints=fo_traces)
                    frame_sample["prompt_list"].append(text)
#                     if "pose_pred" not in frame_sample.field_names or frame_sample["pose_pred"] == None:
#                         frame_sample["pose_pred"] = fo.Keypoints(keypoints=fo_poses)
#                     else:
#                         frame_sample["pose_pred"].keypoints += fo_poses
#                     if "pose_pred_trace" not in frame_sample.field_names or frame_sample["pose_pred_trace"] == None:
#                         frame_sample["pose_pred_trace"] = fo.Keypoints(keypoints=fo_traces)
#                     else:
#                         frame_sample["pose_pred_trace"].keypoints += fo_traces
#                     if len(frame_sample["pose_pred_trace"].keypoints[0].points) > max_forcasting_steps:
#                         max_forcasting_steps = len(frame_sample["pose_pred_trace"].keypoints[0].points)
                        
            v_sample.save()
#     dataset.skeletons[f"pose_pred_{text_short}"] = dataset.skeletons["pose"]                
#     dataset.skeletons[f"pose_pred_trace_{text_short}"] = fo.KeypointSkeleton(labels=[f"forcasting_step_{forcasting_step}" for forcasting_step in range(max_forcasting_steps)], edges=[list(range(max_forcasting_steps))])
    
def analyze(exp_name, config, dataset, dataset_clips):
    # dataset = fo.load_dataset(config["DATASET_FO_VERSION"]).clone(f"{config['DATASET_FO_VERSION']}_{exp_name}_analysis", persistent=True)
    dataset.info["exp_name"] = exp_name
    for split in ["train", "val"]:
        print(f"{split=}")
        predictions = torch.load(f"/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp2/outputs/experiments/{exp_name}/outputs/best_predictions_{split}.json")
        labels = torch.load(f"/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp2/outputs/experiments/{exp_name}/outputs/labels_{split}.json")
        with open(f"/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp2/outputs/experiments/{exp_name}/outputs/best_stats_{split}.json") as f:
            stats = json.load(f)
        # load exp/dataset level info
        dataset.info[f"best_epoch_{split}"] = stats["epoch"]
        dataset.info[f"mean_sample_loss_{split}"] = torch.sqrt(torch.nn.functional.mse_loss(torch.stack(list(predictions.values()), dim=0), torch.stack(list(labels.values()), dim=0))).item()
        # TODO: Attention Map
        dataset_split_clip = dataset_clips.match_tags(split)
        # dataset_video = dataset.select_by("frames.id", list(predictions.keys()))
        # dataset_frame = dataset_video.load_saved_view("frame")
        
        # frame level info
        frame_info = {}
        losses = torch.mean(torch.sqrt(torch.nn.functional.mse_loss(torch.stack(list(predictions.values()), dim=0), torch.stack(list(labels.values()), dim=0), reduction='none')), dim=[1,2]).tolist() # avg loss per frame
        all_ids, all_steps = dataset_split_clip.values(["frames.id", "frames.step"], unwind=True)
        for i in range(len(all_ids)):
            fid = all_ids[i]
            fstep = all_steps[i].classifications[0].label
            frame_info[fid] = {fstep:{"prediction": predictions[fid], "loss": losses[i]}} # "label": labels[fid],
        print(f"visualizing")
        visualize(dataset, dataset_split_clip, config, frame_info)
        
def inference(net, config, dataset, dataset_clip, dataset_frame, prompts):
      
    # inference
    frame_id_list = dataset_clip.values("frames.id", unwind=True)
    frame_info = {}
    with fo.ProgressBar(iters_str="frame") as pbf:
        for i, frame_id in pbf(enumerate(frame_id_list)):
            # if i> 2: break
            frame_sample = dataset_frame[frame_id]
            frame_info[frame_id] = {}
#             print(f"{i=}, {frame_id=}")
#             prompt_list = list({keypoint["prompt"]: None for keypoint in frame_sample.pose_pred.keypoints}.keys()) # TODO: make list when loading prediction
            prompt_list = [frame_sample["narration"]["label"]]
#             prompt_list = []
            # image
            frame = Image.open(frame_sample.filepath)
            image = torch.unsqueeze(net.encoder_image_preprocess(frame), dim=0).to(wandb.config["device"])
            
            for prompt in prompts:
                if prompt in prompt_list:
                    continue
                text = net.encoder_text_preprocess(prompt).to(config["device"])
                pose_input = torch.full((1, 1, config["num_indicators"]*config["num_joints"]*config["num_coords"]), -1.).to(config["device"]) # TODO: net config["pose_trigger_value"] when training
                enco_valid_lens = torch.tensor([config["vocab_size"]+1+len(text_i.nonzero()) for text_i in text]).to(config["device"])
                
                with torch.no_grad():
                    prediction, attention_weights = net.inference(image, text, pose_input, enco_valid_lens, config["forcasting_steps"]) # (1, forcasting_steps, #all_coords)
                frame_info[frame_id][prompt] = {"prediction": prediction[0], "attention_weights": attention_weights}
                
    return frame_info
    
def launch_app(view=None):
    app_config = fo.app_config.copy()
    app_config.color_by = "field"
    app_config.multicolor_keypoints = True
    app_config.use_frame_number  = True
    session = fo.launch_app(config=app_config)
    session.view = view
    
    return session
  
def set_skeletons(dataset, prompts, max_forcasting_steps=31):
    for text in prompts:
        text_short = get_short_text(text)
        dataset.skeletons[f"pose_pred_{text_short}"] = dataset.skeletons["pose"]                
        dataset.skeletons[f"pose_pred_trace_{text_short}"] = fo.KeypointSkeleton(labels=[f"forcasting_step_{forcasting_step}" for forcasting_step in range(max_forcasting_steps)], edges=[list(range(max_forcasting_steps))])
        
def get_short_text(text):
    return "".join([text.split(".")[0]] + [word[0] for word in text.split(" ")[1:]])    
    
def save_forecasting_process(dataset_frame, fields, output_dir):
#     import pdb; pdb.set_trace()
    # Customize annotation rendering
    config = foua.DrawConfig(
        {
#             "font_size": 18,
            "per_object_label_colors": False,
            "keypoints_skeleton": {
                                   "labels": ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], 
                                      "edges": [[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]]
                                },
            "per_keypoints_name_colors": False,
        "per_keypoints_label_colors": True,
#         "text_color": "#FF0000",
            "show_all_names": False,
            "show_keypoints_names": False,
                }
    )    
    meta_fields = ["narration"]
    num_indicators = 2
    for frame_sample in dataset_frame.select_fields(fields+meta_fields):
        frame_dir = os.path.join(output_dir, frame_sample.filepath.replace(".jpg", "").replace("/", "_")+f"_{frame_sample.id}")
        # reset
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        if os.path.exists(frame_dir+".mp4"):
            os.remove(frame_dir+".mp4")
        os.makedirs(frame_dir, exist_ok=True)
        max_forecasting_steps = len(frame_sample[fields[0]].keypoints)//num_indicators
        num_draws = 10
        drawing_keypoints_index_list = list(range(0, len(frame_sample[fields[0]].keypoints), len(frame_sample[fields[0]].keypoints)//num_draws))
        foua.draw_labeled_image(fo.Frame(filepath=frame_sample.filepath, media_type=frame_sample.media_type), os.path.join(frame_dir, f"forecasting_step_.jpg"), config=config)        
        for f_step_idx in range(len(drawing_keypoints_index_list)):
            f_step = drawing_keypoints_index_list[f_step_idx]
            step_path = os.path.join(frame_dir, f"forecasting_step_{f_step:010d}.jpg")
            frame_draw = frame_sample.copy()
            for field_i, field in enumerate(fields):
                draw_keypoints = []
                for prev_step in drawing_keypoints_index_list[:f_step_idx+1]:
                    draw_keypoints += frame_sample[field].keypoints[prev_step:prev_step+2]
#                 for Keypoint in draw_keypoints:
#                     Keypoint.label = field
                for ki in range(len(draw_keypoints)):
                    if field != "pose":
                        # prediction
                        draw_keypoints[ki].label = "Left" if ki%2==0 else "Right"
                    draw_keypoints[ki].index = 0 # field_i
                frame_draw[field].keypoints = draw_keypoints
                prompt = f"{frame_draw[field].keypoints[0].prompt}" if field!='pose' else f"{frame_draw['narration'].label}"
                prompt = ".".join(prompt.split(".")[1:])
                frame_draw[f"{field_i}" + ("(groundtruth)" if field=='pose' else "(prediction)")] = fo.Classification(label=prompt)
                frame_draw[f"FORECASTING STEP"] = fo.Classification(label=f"forecasting_{f_step//num_indicators:02d}")
            del(frame_draw["narration"])
            foua.draw_labeled_image(frame_draw, step_path, config=config)           
#         os.system(f"ffmpeg -framerate 30 -i {os.path.join(frame_dir, 'forecasting_step_*.jpg')} -c:v libx264 -profile:v high -pix_fmt yuv420p {frame_dir}.mp4")
        #os.system(f"ffmpeg -framerate 1 -pattern_type glob -i {os.path.join(frame_dir, 'forecasting_step_*.jpg')} -c:v libx264 -pix_fmt yuv420p output.mp4")
        os.system(f"ffmpeg -framerate 1 -pattern_type glob -i '{os.path.join(frame_dir, 'forecasting_step_*.jpg')}' -c:v libx264 -pix_fmt yuv420p {os.path.join(frame_dir, 'output.mp4')}")
    


# #         import pdb; pdb.set_trace()
#         for field in fields:
#             field_dir = os.path.join(frame_dir, field)
#             os.makedirs(field_dir, exist_ok=True)
#             frame_draw = fo.Frame(filepath=frame_sample.filepath, media_type=frame_sample.media_type)
#             frame_draw[field] = fo.Keypoints()
#             for f_step_idx in range(len(drawing_keypoints_index_list)):
#                 f_step = drawing_keypoints_index_list[f_step_idx]
#                 step_path = os.path.join(field_dir, f"{f_step:010d}.jpg")
    
# #                 for field in fields:
#                 draw_keypoints = []
#                 for prev_step in drawing_keypoints_index_list[:f_step_idx+1]:
#                     # only add Keypoint from key frames
#                     draw_keypoints += frame_sample[field].keypoints[prev_step:prev_step+2]
#                 # unify label for drawing
#                 for ki in range(len(draw_keypoints)):
# #                     draw_keypoints[ki].label = ""
#                     draw_keypoints[ki].index = 0
#                 frame_draw[field].keypoints = draw_keypoints
#                 prompt = f"(prediction) {frame_draw[field].keypoints[0].prompt}" if field!='pose' else f"(groundtruth) {frame_sample['narration'].label}"
#                 frame_draw[f"narr_{field}"] = fo.Classification(label=prompt)
# #                 import pdb; pdb.set_trace()
#                 foua.draw_labeled_image(frame_draw, step_path, config=config)
#             os.system(f"ffmpeg -framerate {1/num_draws} -i {os.path.join(field_dir, '%010d.jpg')} -c:v libx264 -profile:v high -pix_fmt yuv420p {field_dir}.mp4")            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        

