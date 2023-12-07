# %load_ext autoreload
# %autoreload 2
# import sys
# sys.path.insert(0, "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp10/code")

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

## prepare clip_sample and narration ##
#EK#
# # choose test clip from FO (EK_FO_V003): The (clips, narration pair) i) from which good forecasting can be generated; ii) the clip is actually doing the same action but slip mistake. iii)  (or another action -- lapse detection)
# whole_dataset = fo.load_dataset("EpicKitchens50_FO_v003")
# clip_dataset = whole_dataset.load_saved_view("clips").match(FF("narration.label")=="take knife").skip(3).limit(1) # whole_dataset.load_saved_view("clips").skip(9).limit(1)
# clip_dataset = clip_dataset.clone(f"EpicKitchens50_FO_v003_clip:{clip_dataset.first().id}")
#EK over#

step=6    
draw_dir = "/z/home/yayuanli/web/tmp/slip_detection_V1"
#MILLYCookBook#
dataset_raw = fo.load_dataset("MILLYCookbook_FO_v015")
# dataset_videos = dataset_raw.load_saved_view("videos")
dataset_videos = dataset_raw.match(FF("recipe_no").is_in([0])).select_group_slices(["pv"]).match_tags("val")
all_clips = dataset_videos.to_clips("step").match(FF("step.label")==mc.task_spec[0][step])

# load LGPF model
## target config
config = U.exp_config
config["forcasting_steps"] = 90
## target config DONE

net = N.get_network(config)
# model_path = "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp8/outputs/experiments/exp_0007_202306031803/outputs/best_model_val.pt" # exp8.1 total epoch=100
# model_path = "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp8/outputs/experiments/exp_0008_202306051742/outputs/best_model_val.pt" # exp8.1 total epoch=3
model_path = "/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp9/outputs/experiments/exp_0020_202306071257/outputs/best_model_val.pt" # exp9.1 total epoch=25. step 6

net.load_state_dict(torch.load(model_path))
net.eval()
net.cuda()
pose_detector = P.get_pose_detector()
mse_loss = nn.MSELoss()


for clip_no in range(len(all_clips)):
    dataset_clip = all_clips.skip(clip_no).limit(1)
    clip_dataset = dataset_clip.clone(f"{dataset_raw.name}_clip:{dataset_clip.first().id}")
    # clip_dataset = whole_dataset.limit(1).to_clips("step").skip(6).limit(1)
    # clip_dataset = clip_dataset.clone(f"{clip_dataset.name}_clip:{clip_dataset.first().id}")
    #MILLYCookBook over#
    # clip_dataset.default_skeleton = fo.load_dataset("EpicKitchens50_FO_v003").default_skeleton
    clip_dataset.default_skeleton = fo.KeypointSkeleton(labels=["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP", "MEAN_ALL"], edges=[[0,1,2,3,4],[0,5,6,7,8],[9,10,11,12],[13,14,15,16],[0,17,18,19,20],[5,9,13,17]])
    clip_dataset.save()
    clip_sample = clip_dataset.first()
    # "take knife" # "pick up knife", "take plate" ...
    
    # fix filepaths
    # video
    # clip_sample.filepath = clip_sample.filepath[40:].replace("pv_clean.mp4", "pv_fo.mp4")
    clip_sample.filepath = clip_sample.filepath.replace("pv.mp4", "pv_fo.mp4")
    clip_sample.save()
    # # frames
    # frame_path_list = clip_dataset.values("frames.filepath", unwind=True)
    # clip_dataset.set_values("frames.filepath", [[frame_path[40:].replace("pv_clean/", "pv_frames/frame_") for frame_path in frame_path_list]])
    # clip_dataset.set_values("frames.frame_info_dir", [[frame_path[40:-4].replace("pv_clean/", "pv_frames/frame_") for frame_path in frame_path_list]])
    # clip_sample.save()
    clip_dataset.save_view("clips", clip_dataset.to_clips("support"))
    
    # try:
    #     target_narration = clip_sample.narration.label 
    # except:
    #     target_narration = clip_sample.step.label
    # target_narration = "take knife"
    target_narration = clip_sample.step.label
    
    ## prepare clip_sample and narration DONE ##
    
    # actual series of poses
    ##frame_sample
    start_frame_no = clip_sample.support[0] + (clip_sample.support[1] - clip_sample.support[0])//2 * 1
    frame_sample = clip_sample.frames[start_frame_no]
    ##frame_sample done
    detected_pose = P.get_pose(frame_sample.filepath, frame_sample.frame_info_dir, pose_detector, force_generate=False, return_type="torch", config=config) # shape (1(timestamp), #all_coords)
    check_completion = P.check_first_frame(P.pose_torch_to_fo(detected_pose, config), frame_sample.frame_info_dir, return_type="instance")
    if check_completion != None: # imcomplete
        current_pose = P.pose_fo_to_torch(check_completion, config)
    else:
        current_pose = detected_pose
    
    actual_poses = current_pose
    frame_sample["pose"] = P.pose_torch_to_fo(current_pose, config)
    clip_sample.save()
    
    # generate expected series of pose. write to FO
    raw_image = Image.open(frame_sample.filepath)
    image = torch.unsqueeze(net.encoder_image_preprocess(raw_image), dim=0).to(config["device"])
    text = net.encoder_text_preprocess(target_narration).to(config["device"])
    # pose_input = torch.full((1, 1, config["num_indicators"]*config["num_joints"]*config["num_coords"]), -1.).to(config["device"])
    pose_input = current_pose.unsqueeze(0).to(config["device"])
    enco_valid_lens = torch.tensor([config["vocab_size"]+1+len(text_i.nonzero()) for text_i in text]).to(config["device"])    
    with torch.no_grad():
        prediction = net.inference(image, text, pose_input, enco_valid_lens, config["forcasting_steps"]) # (1, forcasting_steps, #all_coords)
    ##expected poses
    # prediction
    expected_poses = torch.cat((current_pose, prediction[0]), dim=0) # (30, #all_coords)
    ##expected poses done
    frame_sample = P.load_forecasting_torch_to_fo(frame_sample, target_narration, expected_poses, config)
    frame_sample["forecasting_timestamp"] = fo.Classification(label='0')
    clip_sample.save()
    # draw frame
    # draw_dir = "/z/home/yayuanli/web/tmp"
    print(f"drew expected poses on a frame: {P.draw_frame(frame_sample, output_dir=draw_dir)}")
    
    # register
    first_frame_sample = frame_sample
    registor = P.SIFT_RANSC()
    
    # iterate frames:
    for frame_no in range(start_frame_no+1, start_frame_no+1+config["forcasting_steps"]):
        # detect current pose 
        frame_sample = clip_sample.frames[frame_no]
        current_pose = P.get_pose(frame_sample.filepath, frame_sample.frame_info_dir, pose_detector, force_generate=False, return_type="torch", config=config) # shape (1(timestamp), #all_coords)
        actual_poses = torch.cat((actual_poses, current_pose), dim=0)
        frame_sample["pose"] = P.pose_torch_to_fo(current_pose, config)    
        
        # register forecasted poses positions to current frame
        # registered_keypoints = []
        # for timestamp in range(len(expected_poses)):
        #     Keypoints_up_list = [P.pose_torch_to_fo(expected_poses[timestamp:timestamp+1], config)]
        #     registered_one_pose_Keypoints = registor.register(filepath_up_list=[first_frame_sample.filepath], filepath_down=frame_sample.filepath,
        #                                                     Keypoints_up_list=Keypoints_up_list, shape=(clip_sample.metadata.frame_height, clip_sample.metadata.frame_width),
        #                                                     load_frame_shift=True)
        #     registered_keypoints += registered_one_pose_Keypoints.keypoint
        # registered_Keypoints = fo.Keypoints(keypoints=registered_keypoints)
            
        registered_Keypoints = registor.register(filepath_up_list=[first_frame_sample.filepath]*len(expected_poses),
                                                          filepath_down=frame_sample.filepath,
                                                          Keypoints_up_list=[P.pose_torch_to_fo(expected_poses[ts:ts+1], config) for ts in range(len(expected_poses))],
                                                          shape=(clip_sample.metadata.frame_height, clip_sample.metadata.frame_width),
                                                          load_frame_shift=False)
        
        registered_Keypoints_torch = P.pose_fo_to_torch(registered_Keypoints, config)
        
        # compare expected and actual pose and calculate mistake probability. write to FO    
        # rmse = torch.sqrt(mse_loss(expected_poses[:len(actual_poses)], actual_poses))
        registered_Keypoints_torch_till_current = registered_Keypoints_torch[:len(actual_poses)]
        rmse = torch.sqrt(mse_loss(registered_Keypoints_torch_till_current, actual_poses))
        print(rmse)
        
        # draw expected/forecasted poses (since current pose)
        registered_Keypoints_torch_since_current = registered_Keypoints_torch[len(actual_poses)-1:]
        # frame_sample = P.load_forecasting_torch_to_fo(frame_sample, target_narration, registered_Keypoints_torch_since_current, config)
        frame_sample = P.load_forecasting_torch_to_fo(frame_sample, target_narration, registered_Keypoints_torch, config)
        frame_sample["forecasting_timestamp"] = fo.Classification(label=str(frame_no-start_frame_no))
        frame_drew_path = P.draw_frame(frame_sample, output_dir=draw_dir)
        print(f"drew {frame_drew_path}")
        
    clip_sample.save()
    
    # make a video
    video_drew_path = os.path.dirname(frame_drew_path)+'.mp4'
    os.system(f"ffmpeg -y -framerate 1 -pattern_type glob -i '{os.path.join(os.path.dirname(frame_drew_path), '*.jpg')}' -c:v libx264 -pix_fmt yuv420p {video_drew_path}")
    print(f"{video_drew_path=}")
    
print("DONE")

    
    
    
    
