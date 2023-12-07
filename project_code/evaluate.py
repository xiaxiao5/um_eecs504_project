import torch
import torch.nn as nn
import numpy as np

def evaluate(predictions, labels, metric_info):
    """
    - predictions (torch.Tensor): The predicted keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    - labels (torch.Tensor): The ground truth keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).        
    """    
    metric_results = {}
    metric_functions = get_metric_function(list(metric_info.keys()))
    for metric_name, metric_args in metric_info.items():
        metric_func = metric_functions[metric_name]
        metric_results[metric_name] = metric_func(predictions, labels, **metric_args)
    
    return metric_results
              
def get_metric_function(metric_names):
    metric_functions = {}
    for metric_name in metric_names:
        if metric_name == 'rmse':
            metric_functions[metric_name] = compute_rmse
        elif metric_name == 'pck':
            metric_functions[metric_name] = compute_pck
        elif metric_name == 'ade&fde':
            metric_functions[metric_name] = evaluate_traj_stochastic
        elif metric_name == 'ade':
            metric_functions[metric_name] = compute_ade
        elif metric_name == 'fde':
            metric_functions[metric_name] = compute_fde
    return metric_functions

def compute_rmse(pred, label, axis=None):
    """
    - predictions (torch.Tensor): The predicted keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    - labels (torch.Tensor): The ground truth keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).        
    """
    rmse = torch.sqrt(torch.nn.functional.mse_loss(pred, label, reduction='none'))
    rmse_value = rmse.mean(axis=axis)
    return rmse_value

def compute_pck(predictions, labels, alpha=0.05, axis=None):
    """
    Compute the Percentage of Correct Keypoints (PCK) metric. 
    `compute_pck(predictions, labels, alpha=0.05, axis=[-1])`

    Parameters:
    - predictions (torch.Tensor): The predicted keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    - labels (torch.Tensor): The ground truth keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    - alpha (float): The normalized distance threshold.
    - axis (list or None): The axes along which the PCK value is computed. If None, the PCK value is computed over all axes. Note the last axis is the #joints. The whole shape is (batch_size, #time_stamps, #visual_indicators, #joints).

    Returns:
    - float: PCK value.
    """

    # Compute bounding box side length (diagonal)
    num_coords = labels.shape[-1]
    bounding_box = torch.cat([labels.min(dim=-2)[0], labels.max(dim=-2)[0]], dim=-1) # (B, T, #indicators, 2*#coords (e.g., minx, miny, maxx, maxy))
    side_length = torch.linalg.norm(bounding_box[:, :, :, num_coords//2:] - bounding_box[:, :, :, :num_coords//2], dim=-1) # (B, T, #indicators), the diagnal of each bbox (visual indicator). norm by default is Frobenius norm
    # print(f"{side_length.shape=}")
    
    # Compute distances between predictions and ground truth
    distances = torch.linalg.norm(predictions - labels, dim=-1) # (B, T, #indicators, #joints) Euclidean distance between all predicted joints and all annotated joints)
    # print(f"{distances.shape=}")
    
    # Check which keypoints are correct based on the threshold
    correctness_keypoints = (distances <= alpha * side_length.unsqueeze(-1)) # shape== shape of distances. the correctness of each keypoint, 0 is not correct, 1 is correct
    # print(f"{correctness_keypoints.shape=}")
    
    # Compute the PCK value
    pck = torch.mean(correctness_keypoints.float(), axis=axis)
    # pck_value = pck*100
    # print(f"PCK value: {pck_value.mean():.2f}%, {pck_value.shape=}")

    return pck


def compute_ade(pred_traj, gt_traj, axis=None):
    """
    ADE is calculated as the l2 distance between the predicted future and the ground-truth averaged over the entire trajectory and two hands
    source: https://github.com/stevenlsw/hoi-forecast/blob/66dab7a62ef283a40283fc48f06a66871d20f73b/evaluation/traj_eval.py

    preds: (torch.Tensor): The predicted keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    gts: (torch.Tensor): The GT keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    axis: (list or None): None, 0, 1, 2. The axes along which the ADE value is computed. If None, the ADE value is computed over all axes. Note the last axis is the #joints. The whole 
    return: (batch_size, num_indicators). e.g., (3513, 2)
    """
    # format
    pred_traj = pred_traj.mean(axis=-2).permute(0, 2, 1, 3) # (batch_size, num_indicators, T, coord(x, y)). compute traj and permute axis
    gt_traj = gt_traj.mean(axis=-2).permute(0, 2, 1, 3) # (batch_size, num_indicators, T, coord(x, y)). compute traj and permute axis

    error = gt_traj - pred_traj # (batch_size, num_indicators, T(?), coord(x, y)). e.g., (3513, 2, 4, 2)
    error = error ** 2
    ade = torch.sqrt(error.sum(dim=3)).mean(axis=axis)

    return ade

def compute_fde(pred_traj, gt_traj, axis=None):
    """
    FDE measures the l2 distance between the predicted future and ground truth at the last time step and averaged over two hands.
    source: https://github.com/stevenlsw/hoi-forecast/blob/66dab7a62ef283a40283fc48f06a66871d20f73b/evaluation/traj_eval.py

    preds: (torch.Tensor): The predicted keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    gts: (torch.Tensor): The GT keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    axis: None, 0, 1
    return: (batch_size, num_indicators). e.g., (3513, 2)
    """    
    # format
    pred_traj = pred_traj.mean(axis=-2).permute(0, 2, 1, 3) # (batch_size, num_indicators, T, coord(x, y)). compute traj and permute axis
    gt_traj = gt_traj.mean(axis=-2).permute(0, 2, 1, 3) # (batch_size, num_indicators, T, coord(x, y)). compute traj and permute axis


    pred_last = pred_traj[:, :, -1, :] # (batch_size, num_indicators, coord(x, y)). e.g., (3513, 2, 2)
    gt_last = gt_traj[:, :, -1, :] # (batch_size, num_indicators, coord(x, y)). e.g., (3513, 2, 2)

    error = gt_last - pred_last # (batch_size, num_indicators, coord(x, y)). e.g., (3513, 2, 2)
    error = error ** 2
    fde = torch.sqrt(error.sum(dim=2)).mean(axis=axis) # (batch_size, num_indicators). e.g., (3513, 2)

    return fde

def evaluate_traj_stochastic(preds, gts, valids=None):
    """
    source: https://github.com/stevenlsw/hoi-forecast/blob/66dab7a62ef283a40283fc48f06a66871d20f73b/evaluation/traj_eval.py
    preds: (torch.Tensor): The predicted keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    gts: (torch.Tensor): The GT keypoints. Shape (batch_size, #time_stamps, #visual_indicators, #joints, #coords).
    """
    # format
    preds = preds.mean(axis=-2).permute(0, 2, 1, 3) # (batch_size, num_indicators, T, coord(x, y)). compute traj and permute axis
    gts = gts.mean(axis=-2).permute(0, 2, 1, 3) # (batch_size, num_indicators, T, coord(x, y)). compute traj and permute axis

    batch_size, num_indicators = preds.shape[0], preds.shape[1]
    if valids == None:
        valids = torch.ones(batch_size, num_indicators).to(preds.device) # (batch_size, num_indicators)

    # ade_list, fde_list = [], []
    fde = compute_fde(preds, gts)  * valids # (batch_size, num_indicators)
    # fde_list.append(fde)
    ade = compute_ade(preds, gts)  * valids # (batch_size, num_indicators)
    # ade_list.append(ade)
    ade_mean = ade.mean(axis=axis)
    fde_mean = ade.mean(axis=axis)
    # ade_mean = ade.sum() / valids.sum()
    # fde_mean = fde.sum() / valids.sum()

    # ade_mean_info = 'ADE: %.3f (%d/%d)' % (ade_mean, valids.sum(), batch_size * num_indicators)
    # fde_mean_info = "FDE: %.3f (%d/%d)" % (fde_mean, valids.sum(), batch_size * num_indicators)

    # print(ade_mean_info)
    # print(fde_mean_info)

    return ade_mean, fde_mean

if __name__ == "__main__":
    preds = torch.rand(128, 2, 30, 2)
    gts = torch.rand(128, 2, 30, 2)
    valids = torch.ones(128, 2) # (B, num_indicators)
    rt = evaluate_traj_stochastic(preds, gts, valids)
    print(rt)

    