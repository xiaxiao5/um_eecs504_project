# hand related functions
import mediapipe
import cv2

landmark_no2name_map = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}
class Hand(object):
    def __init__(self):
        pass
    
detector = mediapipe.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
frames = [cv2.flip(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), 1) for path in ["/z/dat/CookBook/CookBookPhase2_v5/A_pin/mevo/0/video-0000/pv_frames/frame_0000000100.jpg", "/z/dat/CookBook/CookBookPhase2_v5/A_pin/mevo/0/video-0000/pv_frames/frame_0000000200.jpg"]]

for frame in frames:
    results = detector.process(frame)
    if results.multi_handedness == None:
        continue
    hand = {"Left": {"confidence": None, "landmarks": {}}, "Right": {"confidence": None, "landmarks": {}}}
    for i in range(len(results.multi_handedness)):
        # a hand
        hand_lev = results.multi_handedness[i].classification[0]
        label = hand_lev.label
        score = hand_lev.score
        
        hand[label]["confidence"] = score
        landmarks = results.multi_hand_landmarks[i].landmark
        for j in range(len(landmarks)):
            hand[label]["landmarks"][j] = {"name": landmark_no2name_map[j], "pos": (landmarks[j].x, landmarks[j].y)}

print()

def gen_landmarks_from_fo(dataset):
    """
    args:
        dataset: FO dataset
        
    return:
        dataset: the FO dataset with the landmarks loaded if load to FO = True. Otherwise, the unmodified input dataset
    """
    
    for frame in frames:
        results = detector.process(frame)
        if results.multi_handedness == None:
            continue
        hand = {"Left": {"confidence": None, "landmarks": {}}, "Right": {"confidence": None, "landmarks": {}}}
        for i in range(len(results.multi_handedness)):
            # a hand
            hand_lev = results.multi_handedness[i].classification[0]
            label = hand_lev.label
            score = hand_lev.score
            
            hand[label]["confidence"] = score
            landmarks = results.multi_hand_landmarks[i].landmark
            for j in range(len(landmarks)):
                hand[label]["landmarks"][j] = {"name": landmark_no2name_map[j], "pos": (landmarks[j].x, landmarks[j].y)}
