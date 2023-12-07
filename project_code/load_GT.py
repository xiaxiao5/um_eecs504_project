# %load_ext autoreload
# %autoreload 2
# import sys
# sys.path.insert(0, "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp9/code")

import fiftyone as fo
from fiftyone import ViewField as FF
import network as N
import utils as U
import pose as P
import torch.nn as nn
import torch
from PIL import Image
import MILLYCookbook as mc
import os 
import fiftyone.utils.annotations as foua


# load LGPF model
## target config
config = U.exp_config
# config["forcasting_steps"] = 30
## target config DONE

net = N.get_network(config)
model_path = "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp8/outputs/experiments/exp_0007_202306031803/outputs/best_model_val.pt" # exp8.1 total epoch=100. EK
# model_path = "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp8/outputs/experiments/exp_0008_202306051742/outputs/best_model_val.pt" # exp8.1 total epoch=3. EK
# model_path = "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp9/outputs/experiments/exp_0020_202306071257/outputs/best_model_val.pt" # exp9.1 total epoch=25. step 6. V1. MC
dataset_raw = fo.load_dataset("EpicKitchens50_FO_v003") # MILLYCookbook_FO_v015: GT_V1

## inject EK##
# vs = fo.load_dataset("EpicKitchens50_FO_v003").match_tags("val").to_clips("narration").skip(10).first()
# frame_sample = vs.frames[vs.support[0]+10]
## inject EK##

net.load_state_dict(torch.load(model_path))
net.eval()
net.cuda()
pose_detector = P.get_pose_detector()
mse_loss = nn.MSELoss()

## prepare clip_sample and narration ##
step = 6
#MILLYCookBook#

draw_dir = "/z/home/yayuanli/web/tmp/forecasting_GT_v1"


# dataset_videos = dataset_raw.load_saved_view("videos")
# dataset_videos = dataset_raw.match(FF("recipe_no").is_in([0])).select_group_slices(["pv"]).match_tags("val")
# all_clips = dataset_videos.to_clips("step").match(FF("step.label")==mc.task_spec[0][step])
all_clips = dataset_raw.load_saved_view("clips").match_tags("train").match(FF("narration.label")=="take knife").skip(1)

# select_videos = ['/1/video-0008/']
# select_frames = ["frame_0000002768.jpg", "frame_0000002757.jpg", "frame_0000002849.jpg", "frame_0000002758.jpg", "frame_0000002657.jpg"]
for clip_no in range(len(all_clips)):
    dataset_clip = all_clips.skip(clip_no).limit(1)
#     if '/'+'/'.join(dataset_clip.first().filepath.split("/")[-3:-1])+'/' not in select_videos:
#         continue
    clip_dataset_name = f"{dataset_raw.name}_clip:{dataset_clip.first().id}_GT"
    clip_dataset = dataset_clip.clone(clip_dataset_name)
    #MILLYCookBook over#
    clip_dataset.default_skeleton = fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])
    clip_dataset.save()
    clip_sample = clip_dataset.first()
    # "take knife" # "pick up knife", "take plate" ...
    
    # clip_sample.filepath = clip_sample.filepath.replace("pv.mp4", "pv_fo.mp4")
    # clip_sample.save()
    clip_dataset.save_view("clips", clip_dataset.to_clips("support"))
    
    # target_narration = clip_sample.step.label
    target_narration = clip_sample.narration.label
    
    # load fram
    for frame_no in clip_sample.frames:
        ##frame_sample
        frame_sample = clip_sample.frames[frame_no]
#         if frame_sample.filepath.split('/')[-1] not in select_frames:
#             continue
        
        ##frame_sample done    
        # GT
        gt_forecasting = P.forecasting_fo_to_torch(P.read_Keypoints(frame_sample.forecasting_groundtruth_path), config)
        # prediction: input frame_sample, output forecasted_poses
        # current pose
        detected_pose = P.get_pose(frame_sample.filepath, frame_sample.frame_info_dir, pose_detector, force_generate=True, return_type="torch", config=config) # shape (1(timestamp), #all_coords)
        check_completion = P.check_first_frame(P.pose_torch_to_fo(detected_pose, config), frame_sample.frame_info_dir, return_type="instance")
        if check_completion != None: # imcomplete
            current_pose = P.pose_fo_to_torch(check_completion, config)
        else:
            current_pose = detected_pose
        # generate expected series of pose. write to FO
        raw_image = Image.open(frame_sample.filepath)
        image = torch.unsqueeze(net.encoder_image_preprocess(raw_image), dim=0).to(config["device"])
        text = net.encoder_text_preprocess(target_narration).to(config["device"])
        # pose_input = torch.full((1, 1, config["num_indicators"]*config["num_joints"]*config["num_coords"]), -1.).to(config["device"])
        pose_input = current_pose.unsqueeze(0).to(config["device"])
        enco_valid_lens = torch.tensor([config["vocab_size"]+1+len(text_i.nonzero()) for text_i in text]).to(config["device"])    
        with torch.no_grad():
            prediction = net.inference(image, text, pose_input, enco_valid_lens, config["forcasting_steps"]) # (1, forcasting_steps, #all_coords)
        # prediction
        forecasted_poses = torch.cat((current_pose, prediction[0]), dim=0) # (30, #all_coords)
        frame_sample = P.load_forecasting_torch_to_fo(frame_sample, target_narration, forecasted_poses, config, field_name="prediction")
        frame_sample = P.load_forecasting_torch_to_fo(frame_sample, target_narration, gt_forecasting, config, field_name="groundtruth")
        frame_sample["frame_no"] = fo.Classification(label=str(frame_no))
        frame_sample["narration"] = fo.Classification(label=target_narration)
        rmse = torch.sqrt(mse_loss(forecasted_poses[:len(gt_forecasting)], gt_forecasting))
        frame_sample["rmse"] = fo.Classification(label=str(rmse))
        print(f"{rmse=}")
        
        # draw frame
        frame_drew_path = P.draw_frame(frame_sample, output_dir=draw_dir)
        print(f"drew {frame_drew_path}")
        
        
    # make a video
    video_drew_path = os.path.dirname(os.path.dirname(frame_drew_path)) +f'_n{target_narration.replace(" ", "").replace(".", "")}.mp4'
    os.system(f"ffmpeg -y -framerate 1 -pattern_type glob -i '{os.path.join(os.path.dirname(os.path.dirname(frame_drew_path)), '*', 'full_scale.jpg')}' -c:v libx264 -pix_fmt yuv420p {video_drew_path}")
    print(f"{video_drew_path=}")
    fo.delete_dataset(clip_dataset.name)
print(f"DONE")
    
