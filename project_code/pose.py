from fiftyone import ViewField as FF
import fiftyone.utils.annotations as foua
import torch
import fiftyone as fo
import os
import json
import mediapipe
import json
from PIL import Image
import numpy as np
import cv2
from collections import deque
import fiftyonize_utils as fu
import shutil
import wandb
import utils as U

# indicator_labels = sorted(["Left", "Right"])
indicator_labels = sorted(U.exp_config['indicator_list'])
pose_file_name = f"pose_{U.exp_config['pose_version']}.json" # #  when check completeion: v1: isn't registration; 2: registrate w/ SIFT+RANSAC. pose detection didnt' chaged
complete_pose_file_name = f"complete_pose_{U.exp_config['complete_pose_version']}.json"
get_forecasting_gt_file_name = lambda narr_spt0_spt1: f"forecasting_groundtruth_{narr_spt0_spt1[0].replace(' ','').replace('.','')}_{narr_spt0_spt1[1]}_{narr_spt0_spt1[2]}_{U.exp_config['forecasting_groundtruth_version']}.json" # v1: isn't registration; 2: registrate w/ SIFT+RANSAC
sift_file_name = "sift_feature.json"
total_coords = U.exp_config['num_indicators'] * U.exp_config['num_joints'] * U.exp_config['num_coords']
forecasting_num_Keypoint = (U.exp_config['forcasting_steps'] + 1) * U.exp_config['num_indicators']

class LimitedDict:
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.data = {}
        self.order = deque()
        
    def put(self, key, value):
        if key not in self.data:
            if len(self.order) == self.max_size:
                oldest_key = self.order.popleft()
                del self.data[oldest_key]
        else:
            # If the key already exists, we remove it from the queue
            # to later add it again and update its "age"
            self.order.remove(key)
        self.data[key] = value
        self.order.append(key)
        
    def has_key(self, key):
        if key in self.data:
            return True
        else:
            return False
        
    def get(self, key):
        return self.data[key]
        
def get_pose_detector():
    return mediapipe.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.1, min_tracking_confidence=0.1)

def get_pose(filepath, frame_info_dir, landmark_detector, force_generate=False, return_type="path", config=None):
    """
    Get pose, by Mediapipe, for an image on disk by providing filepath and frame_info_dir etc... and return path or instance of fo.Keypoints(),...
    """
    local_hand_path = get_local_path(frame_info_dir, pose_file_name)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath=} doesn't exist")
        indicators_keypoints = fo.Keypoints()    
    if os.path.exists(local_hand_path) and not force_generate:
        if return_type != "path":
            indicators_keypoints = read_Keypoints(local_hand_path)
    else:
        with Image.open(filepath) as img:
            frame = np.array(img)
        indicators_keypoints = mediapipe_detect_pose(frame, landmark_detector)
        write_Keypoints(indicators_keypoints, local_hand_path)
        
    if return_type == "path":
        return local_hand_path
    elif return_type == "instance":
        return indicators_keypoints
    elif return_type == "torch":
        assert config != None, "can't return pose as torch tensor when config is None"
        return pose_fo_to_torch(indicators_keypoints, config)
    else:
        raise NotImplementedError(f"got {return_type=}")

def mediapipe_detect_pose(frame, landmark_detector):
    """
    Detect pose by mediapipe for given image (in memory)j and phrase it into specific format (Keypoints, torch)。
    frame: numpy.array with shape (H, W, 3(RGB)). read by `with Image.open(filepath) as img: frame = np.array(img)`
    landmark_detector: mediapipe.solutions.hands.Hands(). from get_pose_detector().
    return_type: "Keypoints"
    
    return
    """
    results = landmark_detector.process(frame)
    Keypoints = fo.Keypoints() 
    if results.multi_handedness != None: # hands detected
        # go through hands
        for i in range(len(results.multi_handedness)):
            # one FO.Keypoint for each hand
            Keypoint = fo.Keypoint()
            hand_lev = results.multi_handedness[i].classification[0]
            label = hand_lev.label
            score = hand_lev.score
            Keypoint["label"] = "Right" if label == "Left" else "Left" # switch since egocenteric
            Keypoint["hand_confidence"] = score
            Keypoint["index"] = 0
            
            landmarks = results.multi_hand_landmarks[i].landmark
            # go through landmarks
            all_x = []
            all_y = []
            for j in range(len(landmarks)):
                Keypoint["points"].append((landmarks[j].x, landmarks[j].y))
                all_x.append(landmarks[j].x)
                all_y.append(landmarks[j].y)
            # make a mean landmarks for one hand
            Keypoint["points"].append((np.mean(all_x).item(), np.mean(all_y).item()))
            # add this hand to FO.Keypoints
            Keypoints.keypoints.append(Keypoint)
        Keypoints.keypoints = sorted(Keypoints.keypoints, key=lambda x: x.label)
    return Keypoints

    
def check_first_frame(Keypoints, frame_info_dir, return_type):
    """complete return None, else return completed comple_keypoints"""
    return_of_checking = _check_first_frame(Keypoints)
    if return_of_checking != None:
        complete_Keypoints = return_of_checking
        local_hand_path = get_local_path(frame_info_dir, pose_file_name)
        write_complete_hand_path = get_local_path(frame_info_dir, complete_pose_file_name)
        write_Keypoints(complete_Keypoints, write_complete_hand_path)
        if return_type=="path":
            return write_complete_hand_path
        elif return_type=="instance":
            return complete_Keypoints
    else:
        write_complete_hand_path = get_local_path(frame_info_dir, complete_pose_file_name)
        write_Keypoints(Keypoints, write_complete_hand_path)
        return None

def _check_first_frame(Keypoints):
    """
    check the completion of pose in the first or only frame w/o writting to disk.
    """
    missing_indicators = _check_completion(Keypoints)
    complete_Keypoints = None
    if len(missing_indicators) >= 1: # not complete
        complete_Keypoints = Keypoints.copy()
#        indicator_keypoint_dict = {indicator_label: None for indicator_label in indicator_labels}
#        for indicator_keypoint in Keypoints.keypoints:
#            indicator_keypoint_dict[indicator_keypoint.label] = indicator_keypoint
        for ind_lab in missing_indicators:
            missed_points = get_default_indicator_coor(ind_lab)
            # print(f"{missed_points=}")
            complete_Keypoints.keypoints.append(fo.Keypoint(label=ind_lab, points=missed_points, hand_confidence=0.0, index=0))                
            # TODO: generalize to more indicators
            if len(complete_Keypoints.keypoints) > len(indicator_labels):
                complete_Keypoints.keypoints = complete_Keypoints.keypoints[len(complete_Keypoints.keypoints)-len(indicator_labels):]
        complete_Keypoints.keypoints = sorted(complete_Keypoints.keypoints, key=lambda x: x.label)
    return complete_Keypoints

def get_local_path(frame_info_dir, name=pose_file_name):        
    local_hand_path = os.path.join(frame_info_dir, name)
    return local_hand_path
        
def read_Keypoints(local_hand_path):
    try:
        with open(local_hand_path, "r") as f:
            Keypoints = fo.Keypoints().from_json(json.load(f))
        return Keypoints
    except Exception as e:
        print(f"error reading {local_hand_path=}, {e=}")
        if "pose" in local_hand_path:
            return fo.Keypoints()
        raise e
        
def write_Keypoints(Keypoints, local_path):
    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))
    with open(local_path, "w") as f:
        json.dump(Keypoints.to_json(), f)
        
def check_rest_frames(Keypoints_down, filepath_up_list, filepath_down, frame_info_dir_down, Keypoints_up_list, shape, registeror, return_type, registration_method):
    missing_indicators = _check_completion(Keypoints_down)
    if len(missing_indicators) > 0:
        complete_Keypoints = Keypoints_down.copy()
        if len(complete_Keypoints.keypoints) == len(indicator_labels):
            complete_Keypoints.keypoints.pop()
        registered_Keypoints = registeror.register(filepath_up_list, filepath_down, Keypoints_up_list, shape, registration_method=registration_method)
    #    indicator_keypoint_dict = {indicator_label: None for indicator_label in indicator_labels}
    #    for indicator_keypoint in Keypoints_under.keypoints:
    #        indicator_keypoint_dict[indicator_keypoint.label] = indicator_keypoint
        for ind_lab in missing_indicators:
            try:
                missed_points = [Keypoint.points for Keypoint in registered_Keypoints.keypoints if Keypoint.label==ind_lab][0] # TODO: assume only one pred Keypoint match ind_lab
            except Exception as e:
                print(f"{filepath_down=} {e=}")
            complete_Keypoints.keypoints.append(fo.Keypoint(label=ind_lab, points=missed_points, hand_confidence=0.0))
            if len(complete_Keypoints.keypoints) > len(indicator_labels):
                complete_Keypoints.keypoints = complete_Keypoints.keypoints[len(complete_Keypoints.keypoints)-len(indicator_labels):]
        complete_Keypoints.keypoints = sorted(complete_Keypoints.keypoints, key=lambda x: x.label)
        if len(complete_Keypoints.keypoints) != len(indicator_labels):
            print("missed incompletion. pose.py")
            return None
        local_hand_path = get_local_path(frame_info_dir_down, pose_file_name)
        write_complete_hand_path = get_local_path(frame_info_dir_down, complete_pose_file_name)
        write_Keypoints(complete_Keypoints, write_complete_hand_path)        
        
        if return_type=="path":
            return write_complete_hand_path
        elif return_type=="instance":
            return complete_Keypoints
    else:
        write_complete_hand_path = get_local_path(frame_info_dir_down, complete_pose_file_name)
        write_Keypoints(Keypoints_down, write_complete_hand_path)
        return None
    
def _check_completion(keypoints):
    missing_indicators = indicator_labels.copy()
    for keypoint in keypoints.keypoints:
        if keypoint.label in missing_indicators:
            missing_indicators.remove(keypoint.label)
    return missing_indicators

def get_default_indicator_coor(indicator_label):
    num_joints = 21
    if indicator_label == "Left":
        return [(0.25, 0.99)]*num_joints
    else:
        return [(0.75, 0.99)]*num_joints

class SIFT_RANSC():
    def __init__(self, sift_dict_size=31):
        self.sifter = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.sift_feature_dict = LimitedDict(sift_dict_size)
        
    def register(self,filepath_up_list, filepath_down, Keypoints_up_list, shape, registration_method, load_frame_shift=False):
        registered_Keypoints = fo.Keypoints()
        
        if registration_method=="sift+ransc":
            # get sift feature of down. TODO: standardize folder structure
            if "EpicKitchens50" in filepath_down:
                sift_down = self.get_sift_feature(filepath_down, "/".join(filepath_down.split("/")[:-1]))
            elif "MILLY" in filepath_down:
                sift_down = self.get_sift_feature(filepath_down, filepath_down[:-4])
            else:
                raise Exception(f"In register, cannot phrase dataset from path : {filepath_down=}")
            
        for i, filepath_up in enumerate(filepath_up_list):
            if registration_method=="sift+ransc":
                # get sift feature of up(s). TODO: standardize folder structure
                if "EpicKitchens50" in filepath_down:
                    sift_up = self.get_sift_feature(filepath_up, "/".join(filepath_up.split("/")[:-1]))
                elif "MILLY" in filepath_down:
                    sift_up = self.get_sift_feature(filepath_up, filepath_up[:-4])
                else:
                    raise Exception(f"In register, cannot phrase dataset from path : {filepath_down=}")
                
            # get affine transform matrix between frames
            if registration_method=="sift+ransc":
                try:
                    affine_matrix = self.get_affine_matrix(sift_up, sift_down)
                except Exception as e:
                    print(f"when getting affine matrix, got error and set affine_matrix to identity. {e=}")
                    affine_matrix = np.identity(3) # TODO: np.array([[1,0,0],[0,1,0]])
                
            # register all indicators (points)
            if len(Keypoints_up_list[i].keypoints) != len(indicator_labels):
                with open("./pose_error.out", "w+") as f:
                    print(filepath_up, file=f)
                    print(f'invalid pose num: {len(Keypoints_up_list[i].keypoints)}', file=f)
            for Keypoint_up in Keypoints_up_list[i].keypoints:
                if registration_method=="sift+ransc":
                    registered_points = self.affine_transform(affine_matrix, Keypoint_up.points, shape) # list
                elif registration_method=="copy":
                    registered_points = Keypoint_up.points
                    
                if Keypoint_up.has_attribute("hand_confidence"):
                    hand_confidence = Keypoint_up.hand_confidence
                else:
                    hand_confidence = None
                if load_frame_shift: # load_frame_shift = True means the first up frame is the next frame of down frame
                    registered_Keypoint = fo.Keypoint(label=Keypoint_up.label, points=registered_points, hand_confidence=hand_confidence, index=i+1)
                else:
                    registered_Keypoint = fo.Keypoint(label=Keypoint_up.label, points=registered_points, hand_confidence=hand_confidence, index=i)
                registered_Keypoints.keypoints.append(registered_Keypoint)
        return registered_Keypoints
    
    def get_sift_feature(self, filepath, filedir):
        """
        return (keypoints, descriptors)
        """
        if self.sift_feature_dict.has_key(filepath):
            pass
        else:        
            local_sift_feature_path = get_local_path(filedir, sift_file_name)
            if os.path.exists(local_sift_feature_path):
                try:
                    keypoints, descriptors = self._sift_load(local_sift_feature_path)
                except Exception as e:
                    keypoints, descriptors = self._sift_compute(filepath)
                    self._sift_write(keypoints, descriptors, local_sift_feature_path)                
                    print(f"sift local updated {e=}, {filepath=}")
            else:
                keypoints, descriptors = self._sift_compute(filepath)
                self._sift_write(keypoints, descriptors, local_sift_feature_path)
                
            self.sift_feature_dict.put(key=filepath, value=(keypoints, descriptors))
            
        return self.sift_feature_dict.get(filepath)
        
    def _sift_load(self, local_path):
        with open(local_path, 'r') as f:
            # Read the JSON object from the file
            json_data = f.read()
        # Convert the JSON object back to the keypoints and descriptors
        keypoints_dict, descriptors = json.loads(json_data)
        # Convert the keypoints back to cv2.KeyPoint objects
        keypoints = tuple([cv2.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'], kp['response'], kp['octave'], kp['class_id'])
                     for kp in keypoints_dict])
        descriptors = np.array(descriptors).astype(np.float32)    
        return keypoints, descriptors
    
    def _sift_compute(self, filepath):
        with Image.open(filepath) as img:
            frame = np.array(img)
        keypoints, descriptors = self.sifter.detectAndCompute(frame, None)
        return keypoints, descriptors
    
    def _sift_write(self, keypoints, descriptors, local_path):
        # Convert the keypoints to a list of dictionaries
        keypoints_dict = [{'pt': (kp.pt[0], kp.pt[1]), 'size': kp.size, 'angle': kp.angle, 'response': kp.response,
                           'octave': kp.octave, 'class_id': kp.class_id} for kp in keypoints]
        
        # Convert the keypoints and descriptors to a JSON object
        descriptors = descriptors if not isinstance(descriptors, type(None)) else np.array([[]])
        json_data = json.dumps((keypoints_dict, descriptors.tolist()))
                                                          
        # Open a file in write mode
        with open(local_path, 'w') as f: 
            # Write the JSON object to the file
            f.write(json_data)
        
    def get_affine_matrix(self, sift_up, sift_down):
        """Given two images, return the affine transformation matrix that registers image2 on image1"""
        kp_up, des_up = sift_up
        kp_down, des_down = sift_down
        matches = self.matcher.knnMatch(des_down, des_up, k=2)
        good_matches = []
        for m,n in matches:
            if m.distance < 0.65*n.distance:
                good_matches.append([m])
        # Extract the matching keypoints
        kp_up = np.array([kp_up[m[0].trainIdx].pt for m in good_matches], dtype=np.float32)
        kp_down = np.array([kp_down[m[0].queryIdx].pt for m in good_matches], dtype=np.float32)
        
        # no feature point    
        if len(kp_up)==0 or len(kp_down)==0:
            affine_matrix = np.identity(3)
        else:
            # Estimate the affine transformation using RANSAC
            affine_matrix, inliers = cv2.estimateAffinePartial2D(kp_up, kp_down)
        return affine_matrix
    
    
    def affine_transform_for_forecasting_groundtruth_v001(self, matrix, points, shape):
        """
        ***the input matrix should have been (2,3). The checking force all process to be idenfifyt matrxi***
        Transform points coording to matrix.
        args:
            points: points to be transformed in Cartesian coordinates. a list of tuple (x, y) in float from 0 to 1
            matrix: affine transformation matrix (2,3)
            H,W: height and width of the frames that are used to compute affine transformation matrix. The homogeneous term of matrix depends on it
        return:
            dst_points: transformed points in Cartesian coordinates -- (n,2). normalized to 0-1 based on H,W. A list of tuple (x, y) in float from 0 to 1
        """
        # Add a third column of ones to the points array to represent the homogeneous coordinates
        H, W = shape
        points = np.array(points)
        points[:, 0] *= W
        points[:, 1] *= H
        points = np.hstack((points, np.ones((points.shape[0], 1))))        
        # Perform the matrix multiplication to apply the transformation
        # TODO: why !=
        if type(matrix)==type(None) or matrix.shape != (3,3):
            print(f"force transform matrix to be identity matrix. {matrix=}")
            matrix = np.identity(3)
        matrix = np.vstack((matrix, [0,0,1]))
        transformed_points = points @ matrix.T
        
        # Divide the transformed points by the third coordinate to obtain the Cartesian coordinates
        dst_points = transformed_points[:, :2] / transformed_points[:, 2:]    
        dst_points[:, 0] /= W
        dst_points[:, 1] /= H
            
        return dst_points.tolist()
    
    def affine_transform(self, matrix, points, shape):
        """
        Transform points coording to matrix.
        args:
            points: points to be transformed in Cartesian coordinates. a list of tuple (x, y) in float from 0 to 1
            matrix: affine transformation matrix (2,3)
            H,W: height and width of the frames that are used to compute affine transformation matrix. The homogeneous term of matrix depends on it
        return:
            dst_points: transformed points in Cartesian coordinates -- (n,2). normalized to 0-1 based on H,W. A list of tuple (x, y) in float from 0 to 1
        """
        # Add a third column of ones to the points array to represent the homogeneous coordinates
        H, W = shape
        points = np.array(points)
        points[:, 0] *= W
        points[:, 1] *= H
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Perform the matrix multiplication to apply the transformation
        try:
            # print(f"{matrix.shape=}")
            if type(matrix)==type(None):
                print(f"force transform matrix to be identity matrix. {matrix=}")
                matrix = np.identity(3)
            matrix = np.vstack((matrix, [0,0,1]))
            transformed_points = points @ matrix.T
        except Exception as e:
            print(f"error when apply affine transform. {matrix=}")
        
        # Divide the transformed points by the third coordinate to obtain the Cartesian coordinates
        dst_points = transformed_points[:, :2] / transformed_points[:, 2:]    
        dst_points[:, 0] /= W
        dst_points[:, 1] /= H
            
        return dst_points.tolist()


def make_forecasting_groundtruth(filepath_list, Keypoints_list, shape, registeror, frame_info_dir, narration, support, registration_method, force_generate=False, return_type="path", skip_generate=False):
    local_path = get_local_path(frame_info_dir, name=get_forecasting_gt_file_name((narration, support[0], support[1])))
    # load from local    
    if os.path.exists(local_path) and not force_generate:
        if return_type=="instance":
            forecasting_Keypoints = read_Keypoints(local_path)
    # compute
    elif not skip_generate:
        forecasting_Keypoints = Keypoints_list[0].copy()
        registered_Keypoints = registeror.register(filepath_list[1:], filepath_list[0], Keypoints_list[1:], shape, registration_method=registration_method, load_frame_shift=True)
        forecasting_Keypoints.keypoints += registered_Keypoints.keypoints
        assert len(forecasting_Keypoints.keypoints) == forecasting_num_Keypoint, "invalid number of pose Keypoint"
        write_Keypoints(forecasting_Keypoints, local_path)
    # not exist and don't compute -- skip
    else:
        local_path = None
        forecasting_Keypoints = None
        
    # return 
    if return_type=="path":
        return local_path
    elif return_type=="instance":
        return forecasting_Keypoints
    else:
        raise NotImplementedError(f"{return_type=} not implimented")    
    
def pose_fo_to_torch(pose_fo, num_joints):
    """
    pose_fo: an instance of fiftyone.core.labels.Keypoints
    return: torch tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
    """
    pose = [ ]
    for indicator_i, indicator_fo in enumerate(pose_fo["keypoints"]):
        # indicator_fo is a fo.Keypoint instance                
        pose.append(indicator_fo["points"][:num_joints])
    # pose = torch.tensor(pose).view(-1, config["num_all_coords"])[:config["forcasting_steps"]+1] # tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
    pose = torch.tensor(pose).view(1, -1) # tensor with shape (1, -1)
    # pose = self.decoder_pose_preprocess(pose)
    
    return pose

def load_forecasting_torch_to_fo(frame_sample, prompt, forecasting, config, field_name):
    """
    load forecasting in torch tensor to FO frame_sample
    
    frame_sample: FO frame sample to which forecasting to be loaded. -- image
    prompt: prompt text from which forecasting is generated
    forecasting: torch tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
    config: config dict

    return: FO frame_sample with forecasting loaded
    """
        
    fo_poses = []
    fo_traces = []                    
    # forcasting = prompt_info["prediction"]
    forecasting = forecasting.view(-1, config["num_indicators"], config["num_joints"], config["num_coords"])
    mean_points = {"Left": [], "Right": []}
    for forecasting_step, all_indicators in enumerate(forecasting):
        for indicator_i, indicator in enumerate(all_indicators):
            side = "Left" if indicator_i%2==0 else "Right"
            # indicator_fo = fo.Keypoint(points=indicator.tolist(), prompt=prompt, hand_side=side, label=side, forecasting_step=forecasting_step, index=forecasting_step)
            indicator_fo = fo.Keypoint(points=indicator.tolist(), prompt=prompt, hand_side=side, label=None, forecasting_step=forecasting_step, index=None)
            fo_poses.append(indicator_fo)
            mean_points[side].append(indicator.mean(dim=0).tolist()) # mean point of one hand
            
    # connect mean points among all forecasting steps for each indicator
    for side, points in mean_points.items():
        trace_fo = fo.Keypoint(points=points, hand_side=side, prompt=prompt, label=side)
        fo_traces.append(trace_fo)
        
    # frame_sample[f"forecasting_pred_{prompt.replace('.', '')}"] = fo.Keypoints(keypoints=fo_poses)
    # frame_sample[f"forecasting_pred_trace_{prompt.replace('.', '')}"] = fo.Keypoints(keypoints=fo_traces)
    frame_sample[field_name] = fo.Keypoints(keypoints=fo_poses)
    # frame_sample[f"forecasting_pred_trace_{prompt[:1]}"] = fo.Keypoints(keypoints=fo_traces)
    # frame_sample[f"forecasting_pred_trace_{prompt[:1]}"] = fo.Keypoints()
    
    
    return frame_sample

def forecasting_torch_to_fo(forecasting, prompt, include_cur_pose, num_indicators=2, num_joints=21, num_coords=2):
    """    
    forecasting: torch tensor with shape (#timestamps, num_indicators*num_joints*num_coords)
    prompt: str
    """
        
    fo_poses = []
    fo_traces = []                    
    # forcasting = prompt_info["prediction"]
    forecasting = forecasting.view(-1, num_indicators, num_joints, num_coords)
    mean_points = {"Left": [], "Right": []}
    for forecasting_step, all_indicators in enumerate(forecasting):
        for indicator_i, indicator in enumerate(all_indicators):
            side = "Left" if indicator_i%2==0 else "Right"
            indicator_fo = fo.Keypoint(points=indicator.tolist(), prompt=prompt, hand_side=side, label=side, forecasting_step=forecasting_step if include_cur_pose else forecasting_step+1, index=forecasting_step if include_cur_pose else forecasting_step+1)
            fo_poses.append(indicator_fo)
            mean_points[side].append(indicator.mean(dim=0).tolist()) # mean point of one hand
            
    # connect mean points among all forecasting steps for each indicator -- one indicator has one trace -- one Keypoint which contains forecasting_step(+1) (x,y)s
    for side, points in mean_points.items():
        trace_fo = fo.Keypoint(points=points, hand_side=side, prompt=prompt, label=side)
        fo_traces.append(trace_fo)
    
    return fo.Keypoints(keypoints=fo_poses), fo.Keypoints(keypoints=fo_traces)

def forecasting_fo_to_torch(pose_fo, num_indicators, num_joints):
    """
    args:
        pose_fo: an instance of fiftyone.core.labels.Keypoints
    return:
        torch tensor with shape (-1(forcasting_steps+1), num_indicators*num_joints*num_coords)
    """
    
    pose = [ ]
    for indicator_i, indicator_fo in enumerate(pose_fo["keypoints"]):
        # indicator_fo is a fo.Keypoint instance                
        pose.append(indicator_fo["points"][:num_joints])
        
    num_all_coords = num_indicators * num_joints * len(pose[0][0])
    pose = torch.tensor(pose).view(-1, num_all_coords) # tensor with shape (forcasting_steps+1, num_indicators*num_joints*num_coords)
    
    return pose

def pose_torch_to_fo(pose_torch, num_joints=21, num_coords=2):
    """
    translate pose in torch tensor to FO Keypoints
    
    pose_torch: torch tensor with shape (1 (timestamp), num_indicators*num_joints*num_coords)
    
    return: FO Keypoints
    """
    if len(pose_torch) > 0:
        pose_torch = pose_torch.view(-1, num_joints, num_coords)
        fo_poses = []
        for indicator_i, indicator in enumerate(pose_torch):
            side = "Left" if indicator_i%2==0 else "Right"
            indicator_fo = fo.Keypoint(points=indicator.tolist(), hand_side=side, label=side)
            fo_poses.append(indicator_fo)
        pose_fo = fo.Keypoints(keypoints=fo_poses)
    else:
        pose_fo = fo.Keypoints(keypoints=[])
    
    return pose_fo
        
def draw_frame(frame_sample, draw_frame_path, draw_config):
    try:
        frame_sample["media_type"] = "image"
        foua.draw_labeled_image(frame_sample, draw_frame_path, config=draw_config)
    except Exception as e:
        print(e)
        print(f"draw frame failed {draw_frame_path=}")
        return False
    return True


