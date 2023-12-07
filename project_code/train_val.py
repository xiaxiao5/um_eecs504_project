import utils as U
import metric as M
import wandb
import torch
import torch.nn as nn
import json
import os
import time
import loss as L

def train(net, loaders, rank=None, predicting=False):
    """
    train the net;
    metrics["train"]
    """
    validation_only = 'train' not in loaders.keys()
    if wandb.config['optimizer']=="adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=wandb.config["lr"])
    elif wandb.config['optimizer']=="sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=wandb.config["lr"], momentum=0.9)
    
    if wandb.config['scheduler'] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_0=0, T_max=wandb.config["num_epochs"])
    elif wandb.config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=wandb.config["num_epochs"]//3)
    else:
        scheduler = None

    loss = L.get_loss_func()
        
    if wandb.config["loss"] in ["mse_learnable"]:
        optimizer.add_param_group({'params': loss.weight_predictor.parameters(), 'lr': wandb.config["lr"]})
        
    net.to(wandb.config["device"])
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
    #                         xlim=[10, wandb.config["num_epochs"]])
    
    for epoch in range(wandb.config["num_epochs"]):
        # define ouptus for one epoch
        timers = {"train": U.Timer(), "val": U.Timer()}
        # metric = {"train": U.Accumulator(3), "val": U.Accumulator(3)}  # Sum of training loss, no. of tokens
        metric_terms = ["loss", "num_timestamps", "bleu", "rmse"]
        metric = {"train": U.MyAccumulator(metric_terms), "val": U.MyAccumulator(metric_terms)}
        predictions = {"train": None, "val": None}
        labels = {"train": None, "val": None}
        
        # train
        timers["train"].start()
        net.train()
        if validation_only == False:
            metric["train"], predictions["train"], labels["train"], net = loop_batches(net=net, data_iter=loaders["train"], metric=metric["train"], training=True, optimizer=optimizer, loss=loss, rank=rank, scheduler=scheduler)
        timers["train"].stop()
        
        # val
        timers["val"].start()
        net.eval()
        if 'gradcam' not in wandb.config['use_embeddings']:
            with torch.no_grad():
                metric["val"], predictions["val"], labels["val"], net  = loop_batches(net=net, data_iter=loaders["val"], training=False, metric=metric["val"], loss=loss, predicting=predicting)
        else:
            metric["val"], predictions["val"], labels["val"], net  = loop_batches(net=net, data_iter=loaders["val"], training=False, metric=metric["val"], loss=loss, rank=rank)
        timers["val"].stop()

        if wandb.config['function'] not in ['save_BLIP_ek50']:
            # checkpoint
            training_checkpoint(epoch,metric, predictions, labels, net, timers, loaders)
            
        if scheduler != None:
            scheduler.step()
        if validation_only == True:
            break
        # debug mode
        if wandb.config["debug_mode"] and epoch > 0:
            break

def inference(net, data_loader):
    timers = {"inf": U.Timer()}
    metric_terms = ["loss", "num_timestamps", "bleu", "rmse"]
    metric = {"inf": U.MyAccumulator(metric_terms)} 
    
    net.eval()
    net.to(wandb.config["device"])
    timers["inf"].start()
    with torch.no_grad():
        metric, prediction, label, net  = loop_batches(net=net, data_iter=data_loader, training=False, metric=metric, inference=True, loss=nn.MSELoss(reduction='none'))
    timers["inf"].stop()
    
    # checkpoint
    training_checkpoint(0, metric, predictions={"inf": prediction}, labels={"inf": label}, net=net, timers=timers, loaders={"inf": data_loader})
        
        
def loop_batches(net, data_iter, training, metric, optimizer=None, loss=None, inference=False, rank=None, scheduler=None, predicting=False):
    if data_iter == None:
        return metric, None, None, net
    
    accumulate_num = wandb.config['total_batch_size'] // data_iter.batch_size
    if training:
        optimizer.zero_grad()
    predictions = U.SeqDict() # id->list of redicted grid_no
    labels = U.SeqDict() # id->list of redicted grid_no
    start_time = time.time()
    for batch_i, batch in enumerate(data_iter):
        if batch_i % (len(data_iter)//10 + 1) == 0:
            print(f"{training=} {batch_i=}, {time.time()-start_time} elapsed")
            
        if not inference:
            fo_ids = batch[-1] # frame_path
            if wandb.config['network_class_name'] == "MyNetwork":
                image, text, pose, hand_confidence = [x.to(wandb.config["device"]) for x in batch[:-1]] # image (N,3,H,W); text (N,L); pose (N,#indicators*#joints*#coords)
            elif wandb.config['network_class_name'] == "MyNetworkV002":
                if rank == None:
                    image = batch[0].to(wandb.config["device"])
                    text = batch[1] # String
                    pose = batch[2].to(wandb.config["device"])
                    hand_confidence = batch[3].to(wandb.config["device"])
                    if 'gradcam' in wandb.config['use_embeddings']:
                        image_gradcam = batch[4].to(wandb.config["device"])
                        text_gradcam = batch[5]
                else:
                    image = batch[0].to(rank)
                    text = batch[1] # String
                    pose = batch[2].to(rank)
                    hand_confidence = batch[3].to(rank)
                    if 'gradcam' in wandb.config['use_embeddings']:
                        image_gradcam = batch[4].to(rank)
                        text_gradcam = batch[5]
                    
        elif inference==True:
            frame_path = batch[-2]
            raw_narration = batch[-1]
            if wandb.config['network_class_name'] == "MyNetwork":
                image, text, pose, hand_confidence = [x.to(wandb.config["device"]) for x in batch[:-2]] # image (N,3,H,W); text (N,L); pose (N,#indicators*#joints*#coords)
            elif wandb.config['network_class_name'] == "MyNetworkV002":
                image = batch[0].to(wandb.config["device"])
                text = batch[1] # String
                pose = batch[2].to(wandb.config["device"])
                hand_confidence = batch[3].to(wandb.config["device"])
            
        num_timestamps = wandb.config["forcasting_steps"] # #timestamps will be generated so no current stamp -- seems this comment for exp2 for exp4 this is not true but for exp6 this is true again
        # enco_valid_lens = torch.tensor([wandb.config["vocab_size"]]*wandb.config["batch_size"]).to(wandb.config["device"]) # num_patches == num_box in grid == dim of outpu space of deco = vocab_size
        if wandb.config['network_class_name'] == "MyNetwork":
            enco_valid_lens = torch.tensor([wandb.config["vocab_size"]+1+len(text_i.nonzero()) for text_i in text]).to(wandb.config["device"])
        # deco_valid_lens = torch.tensor([wandb.config["window_size"]]*wandb.config["batch_size"]).to(wandb.config["device"])
        # pose: (batch_size, forecasting_step+1, #num_coords_84)g
        if not inference:
            pose_input = pose[:,:-1]
            pose_label = pose[:, 1:]
            if wandb.config['network_class_name'] == "MyNetwork":
                pose_pred_score, _states = net(image, text, pose_input, enco_valid_lens) # pose_pred_score (N,L,#all_coords)
            elif wandb.config['network_class_name'] == "MyNetworkV002":
                if wandb.config['function'] not in ['save_BLIP_ek50']:
                    if 'gradcam' not in wandb.config['use_embeddings']:
                        if not predicting:
                            pose_pred_score, _states = net(image, text, pose_input) # pose_pred_score (N,L,#all_coords)
                        else:
                            pose_pred_score, _states = net.predict(image, text, pose_input) # pose_pred_score (N,L,#all_coords)
                    elif 'gradcam' in wandb.config['use_embeddings']:
                        pose_pred_score, _states = net(image, text, pose_input, image_gradcam, text_gradcam) # pose_pred_score (N,L,#all_coords)
                elif wandb.config['function'] in ['save_BLIP_ek50']:
                    # feature_path is a list of string for the basic path of each frame feature to be saved to 
                    feature_path = [os.path.join(os.path.dirname(frame_path), wandb.config['feature_name']) for frame_path in fo_ids]
                    net.save_encoder_feature(image, text, feature_path)
                    pose_pred_score = pose_label
            if wandb.config['loss'] == 'weighted_rmse':
                l = torch.sqrt(loss(hand_confidence*pose_pred_score, pose_label)) # b 71, str(l.cuda().item()) == 'nan'
            elif wandb.config['loss'] == 'weighted_mse':
                l = loss(hand_confidence*pose_pred_score, pose_label)
            elif wandb.config['loss'] == 'rmse':
                l = torch.sqrt(loss(pose_pred_score, pose_label))
            elif wandb.config['loss'] == 'mse':
                l = loss(pose_pred_score, pose_label)
            elif wandb.config['loss'] in ["mse_linear_down", "mse_sine", "mse_revsine", "mse_learnable"]:
                loss_forward_kwargs = {
                    'pred': pose_pred_score,
                    'label': pose_label,
                    'mm_ebd': _states[0],
                    "mm_valid_len": _states[1],
                }
                l = loss(loss_forward_kwargs)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(pose_pred_score, pose_label))

            # log samples' predictions and labels
            frame_info_list = []
            pred_info_list = []
            label_info_list = []
            frame_path = fo_ids
            raw_narration = text
            start_token = pose[:, 0]
            for b_i in range(len(start_token.cpu())):
                frame_path_ = frame_path[b_i]
                raw_narration_ = raw_narration[b_i]
                # frame_info_list.append({'frame_path': frame_path_, "raw_narration": raw_narration_})
                pose_input_ = start_token.cpu()[b_i] # (1, #num_coords_84)
                # hand_confidence_ = hand_confidence.cpu()[b_i]
                frame_info_list.append((frame_path_, 
                                        raw_narration_, 
                                        pose_input_, 
                                        # hand_confidence_
                                        )
                                        )
                
                pose_pred_score_ = pose_pred_score.cpu()[b_i] # (forcasting_steps, #num_coords_84)
                # rmse_ = rmse[b_i]
                # pred_info_list.append({'pose_input': pose_input_, 'pose_pred_score': pose_pred_score_, 'rmse': rmse_})
                # pred_info_list.append({
                    # 'pose_input': pose_input_, 
                                    #    'pose_pred_score': pose_pred_score_, 
                                    #    'rmse': rmse_
                                        # }
                                    #    )
                pred_info_list.append(pose_pred_score_)
                pose_label_ = pose_label.cpu()[b_i] # (forcasting_steps, #num_coords_84)
                label_info_list.append(pose_label_)
                
            # predictions.add(list(dict(zip(frame_path, raw_narration)).items()), list(dict(zip(pose_input.cpu(), pose_pred_score.detach().cpu())).items()))
            predictions.add(frame_info_list, pred_info_list)
            labels.add(frame_info_list, label_info_list)

            # hands_pred_pos = pose_pred_score.argmax(dim=2) # get position int; (bs, seq_len) # this is discrete
            # M.handbleu(hands_pred_pos.type(torch.int32).tolist(), hands.type(torch.int32).tolist())
            # labels.add(fo_ids, pose_label.cpu())
            # predictions.add(fo_ids, pose_pred_score.detach().cpu())
            # if wandb.config["save_attention_weights"]:
            #     attention_weight_seq.append(net.decoder.attention_weights)           
        else:
            pose_input = pose[:,:1]
            pose_label = pose[:, 1:]
            if wandb.config['network_class_name'] == "MyNetwork":
                pose_pred_score, _states = net.inference(image, text, pose_input, enco_valid_lens, wandb.config["forcasting_steps"]) # pose_pred_score (N,L,#all_coords)
            elif wandb.config['network_class_name'] == "MyNetworkV002":
                pose_pred_score, _states = net.inference(image, text, pose_input, wandb.config["forcasting_steps"])
            rmse = torch.sqrt(loss(pose_pred_score, pose_label)).mean(dim=[1,2]).tolist() # list len=bs
            frame_info_list = []
            pred_info_list = []
            label_info_list = []
            for b_i in range(len(pose_input.cpu())):
                frame_path_ = frame_path[b_i]
                raw_narration_ = raw_narration[b_i]
                # frame_info_list.append({'frame_path': frame_path_, "raw_narration": raw_narration_})
                frame_info_list.append((frame_path_, raw_narration_))
                
                pose_input_ = pose_input.cpu()[b_i] # (1, #num_coords_84)
                pose_pred_score_ = pose_pred_score.cpu()[b_i] # (forcasting_steps, #num_coords_84)
                rmse_ = rmse[b_i]
                pred_info_list.append({'pose_input': pose_input_, 'pose_pred_score': pose_pred_score_, 'rmse': rmse_})
                
                pose_label_ = pose_label.cpu()[b_i] # (forcasting_steps, #num_coords_84)
                label_info_list.append({'pose_label': pose_label_})
                
            # predictions.add(list(dict(zip(frame_path, raw_narration)).items()), list(dict(zip(pose_input.cpu(), pose_pred_score.detach().cpu())).items()))
            predictions.add(frame_info_list, pred_info_list)
            labels.add(frame_info_list, label_info_list)
            # a list of (frame_path, raw_narration) and a list of (pose_input (cur pose), pose_pred_score (forecasted pose))
            
            
        if training:
            if wandb.config['duplicate_mode']==False:
                l.backward()  # Make the loss scalar for `backward`
                # U.grad_clipping(net, 1)
                if (batch_i+1) % accumulate_num == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                l = torch.tensor(0.0).cuda()
            with torch.no_grad():
                metric.add({"loss": l.cpu().item(), "num_timestamps": num_timestamps, "rmse": rmse.cpu().item()}, )
        elif not inference:
            metric.add({"loss": l.cpu().item(), "num_timestamps": num_timestamps, "rmse": rmse.cpu().item()}, )
        else:
            metric.update({"num_timestamps": num_timestamps})
        # break
        if wandb.config["debug_mode"] and  batch_i >= accumulate_num:
            break
    if not inference:
        return metric, predictions, labels, net
    else:
        return metric, predictions, labels, net
            
            
def training_checkpoint(epoch, metric, predictions, labels, net, timers, loaders):
    """
    args:
        metric: {"train": [], "val": []}. [] is a list of sum of loss, sum of handbleu, num of tokens.
        predictions: {"train": SeqDict({"frame_id": the predicted seq -- a list of int})}
        net: the torch model
    """
    for split_no, split in enumerate(loaders.keys()):
        # wandb
        loss = metric[split].mean("loss")
        bleu = metric[split].mean("bleu")
        rmse = metric[split].mean("rmse")
        bleu_best = bleu if (f"bleu_best_{split}" not in wandb.run.summary.keys()) or bleu > wandb.run.summary[f"bleu_best_{split}"] else wandb.run.summary[f"bleu_best_{split}"]
        loss_best = loss if (f"loss_best_{split}" not in wandb.run.summary.keys()) or loss < wandb.run.summary[f"loss_best_{split}"] else wandb.run.summary[f"loss_best_{split}"]
        rmse_best = rmse if (f"rmse_best_{split}" not in wandb.run.summary.keys()) or rmse < wandb.run.summary[f"rmse_best_{split}"] else wandb.run.summary[f"rmse_best_{split}"]        
        speed = metric[split].sum("num_timestamps") / timers[split].sum() # timestamps / sec
        log_dict = {
            "epoch": epoch,
            f"loss_{split}": loss,
            f"loss_best_{split}": loss_best,
            f"bleu_{split}": bleu,
            f"bleu_best_{split}": bleu_best,
            f"rmse_{split}": rmse,
            f"rmse_best_{split}": rmse_best,
            f"speed_{split}": speed, # tokens / sec
        }
        wandb.log(log_dict)
        print(log_dict)
            
        if rmse <= wandb.run.summary[f"rmse_best_{split}"]:
            wandb.run.summary[f"loss_best_{split}"] = loss
            wandb.run.summary[f"bleu_best_{split}"] = bleu
            wandb.run.summary[f"rmse_best_{split}"] = rmse
            
            # predictions
            torch.save(predictions[split].data, os.path.join(wandb.config["exp_output_folder_path"], f"best_predictions_{split}.pt"))
            if wandb.config["visualize_prediction_onthefly"]:
                start_time = time.time()
                loaders[split].dataset.load_prediction(predictions[split].data, epoch)
                print(f"{epoch=} {split=} loadind predictions to FO taks {time.time()-start_time:.2f}s")
            # labels
            # json.dump(labels[split].data, f)
            torch.save(labels[split].data, os.path.join(wandb.config["exp_output_folder_path"], f"labels_{split}.pt"))
            # net
            net.cpu()
            torch.save(net.state_dict(), os.path.join(wandb.config["exp_output_folder_path"], f"best_model_{split}.pt"))
            # net.cuda()
            net.to(wandb.config["device"])
                        
            # stats
            with open(os.path.join(wandb.config["exp_output_folder_path"], f"best_stats_{split}.json"), 'w') as f:
                json.dump(log_dict, f)                
                
    print(f"="*10)
    
