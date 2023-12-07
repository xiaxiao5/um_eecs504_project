import pose as P
import cv2
import numpy as np
import os
import json
from PIL import Image
from timeit import timeit
# import fo  # Assuming fo is a module containing the Keypoints class and related classes
import fiftyone as fo
import cProfile
import socket
import time

total_feature_num = 0

class SIFT_RANSC():
    """
    the original method, pose.SIFT_RANSC().register.
    modification is let it always compute whole pipeline (feature->transform->RANSC) instead of using the saved results. So that we can test speed.
        - get_sift_feature
    """
    def __init__(self):
        self.sifter = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
    def register(self,filepath_up_list, filepath_down, Keypoints_up_list, shape, load_frame_shift=False, check_exist_sift=True):
        registered_Keypoints = fo.Keypoints()
        
        # get sift feature of down
        sift_down = self.get_sift_feature(filepath_down, "/".join(filepath_down.split("/")[:-1]), check_exist=check_exist_sift)
        
        for i, filepath_up in enumerate(filepath_up_list):
            # get sift feature of up(s)
            sift_up = self.get_sift_feature(filepath_up, "/".join(filepath_up.split("/")[:-1]), check_exist=check_exist_sift)
            # get affine transform matrix between frames
            try: 
                affine_matrix = self.get_affine_matrix(sift_up, sift_down)
            except Exception as e:
                print(f"when getting affine matrix, got error and set affine_matrix to identity. {e=}")
                affine_matrix = np.identity(3)
            
            # register all indicators (points)
            if len(Keypoints_up_list[i].keypoints) != len(P.indicator_labels):
                print(filepath_up)
                print(f'invalid pose num: {len(Keypoints_up_list[i].keypoints)}')
            for Keypoint_up in Keypoints_up_list[i].keypoints:
                registered_points = self.affine_transform(affine_matrix, Keypoint_up.points, shape) # list
                if load_frame_shift:
                    registered_Keypoint = fo.Keypoint(label=Keypoint_up.label, points=registered_points, hand_confidence=Keypoint_up.hand_confidence, index=i+1)
                else:
                    registered_Keypoint = fo.Keypoint(label=Keypoint_up.label, points=registered_points, hand_confidence=Keypoint_up.hand_confidence)
                registered_Keypoints.keypoints.append(registered_Keypoint)
        return registered_Keypoints
    
    def get_sift_feature(self, filepath, filedir, check_exist=True):
        local_sift_feature_path = P.get_local_path(filedir, P.sift_file_name)
        if check_exist:
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
        else:
            keypoints, descriptors = self._sift_compute(filepath)
            self._sift_write(keypoints, descriptors, local_sift_feature_path)            
        return (keypoints, descriptors)
    
    def _sift_load(self, local_path):
        start_time = time.time()
        with open(local_path, 'r') as f:
            # Read the JSON object from the file
            json_data = f.read()
        # Convert the JSON object back to the keypoints and descriptors
        keypoints_dict, descriptors = json.loads(json_data)
        # Convert the keypoints back to cv2.KeyPoint objects
        keypoints = tuple([cv2.KeyPoint(kp['pt'][0], kp['pt'][1], kp['size'], kp['angle'], kp['response'], kp['octave'], kp['class_id'])
                     for kp in keypoints_dict])
        descriptors = np.array(descriptors).astype(np.float32)
        print(f"{local_path=}, {len(descriptors)=}, loading_time={time.time()-start_time}")
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
        # TODO: why !=
        if type(matrix)==type(None) or matrix.shape != (3,3):
            matrix = np.identity(3)
        matrix = np.vstack((matrix, [0,0,1]))
        transformed_points = points @ matrix.T
        
        # Divide the transformed points by the third coordinate to obtain the Cartesian coordinates
        dst_points = transformed_points[:, :2] / transformed_points[:, 2:]    
        dst_points[:, 0] /= W
        dst_points[:, 1] /= H
            
        return dst_points.tolist()


class RegistrationBenchmark():
    def __init__(self):
        # Define the test data for the benchmark
        if socket.gethostname() == 'Yayuans-MBP.attlocal.net':
            image_template = lambda x: f"/Users/yayuanli/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_01/{x['frame_num']:010d}/{x['filename']}"  # Template for the filepath of the "up" images
        else:
            image_template = lambda x: f"/z/dat/EpicKitchens50/EpicKitchens50_media_v000/videos/train/P01/P01_01/{x['frame_num']:010d}/{x['filename']}"  # Template for the filepath of the "up" images        

        down_frame_num = 5129
        up_frame_num = 30
        self.filepath_down = image_template({'frame_num': down_frame_num, 'filename': 'full_scale.jpg'})    # Filepath for the "down" image
        self.filepath_up_list = [image_template({'frame_num': down_frame_num+i, 'filename': 'full_scale.jpg'}) for i in range(1, 1+up_frame_num)]  # List of filepaths for the "up" images
        Keypoints_up_path_list = [image_template({'frame_num': down_frame_num+i, 'filename': 'pose_v001.json'}) for i in range(1, 1+up_frame_num)] # List of path of pose points saved for the "up" images
        self.Keypoints_up_list = [P.read_Keypoints(path) for path in Keypoints_up_path_list if os.path.exists(path)] # read the pose points for the "up" images
        # shape = cv2.read(filepath_down).shape[:2]  # Shape of the "down" image, h,w
        self.shape = np.array(Image.open(self.filepath_down)).shape[:2]  # Shape of the "down" image, h,w
        
        self.num_iterations = 1  # Number of times to execute the test_register function for better averaging
        
    def get_time(self, method_no):
        if method_no == 1: # SIFT_RANSC, check local sift and load if exist
            def timeit_func():
                SIFT_RANSC().register(self.filepath_up_list, self.filepath_down, self.Keypoints_up_list, self.shape, check_exist_sift=True)
        elif method_no == 2: # SIFT_RANSC, always compute sift
            def timeit_func():
                SIFT_RANSC().register(self.filepath_up_list, self.filepath_down, self.Keypoints_up_list, self.shape, check_exist_sift=False)
        elif method_no == 3: # only _sift_compute
            def timeit_func():
                SIFT_RANSC()._sift_compute(self.filepath_down)
        elif method_no == 4: # write _sift_compute out 
            def timeit_func():
                SIFT_RANSC().sifter.detectAndCompute(np.array(Image.open(self.filepath_down)), None)
        elif method_no == 5: # only _sift_load
            def timeit_func():
                return SIFT_RANSC()._sift_load(self.filepath_down)
        
        avg_execution_time = timeit(timeit_func, number=self.num_iterations)/self.num_iterations
        return avg_execution_time
        
        # print(len(timeit_func()[1]))

        

if __name__ == '__main__':
    benchmark = RegistrationBenchmark()
    for method_no in [1,5]:
        print(f"testing method {method_no}. Runing it for ")
        # avg_execution_time = benchmark.get_time(method_no) 
        # print(f"Average execution time for register method {method_no}: {avg_execution_time:.4f} seconds")
        print(cProfile.run('benchmark.get_time(method_no)', sort=2))



