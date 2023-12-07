# definition of network structures
import transformer as TF
import wandb
import torch
import torch.nn as nn
# import clip
from lavis.models import load_model_and_preprocess
import os
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam


def get_network(config=None):
    """
    return a network instance this file
    """
    if config != None:
        wandb.config = config
    if wandb.config['network_class_name'] == "MyNetwork":
        encoder, encoder_image_preprocess = clip.load("ViT-B/32", device=wandb.config["device"])
        decoder = TF.PoseDecoder(wandb.config["num_indicators"], wandb.config["num_joints"], wandb.config["num_coords"], wandb.config["num_hiddens"], wandb.config["num_hiddens"], wandb.config["num_hiddens"], wandb.config["num_hiddens"], wandb.config["deco_norm_shape"], wandb.config["num_hiddens"], wandb.config["deco_ffn_num_hiddens"], wandb.config["deco_num_heads"], wandb.config["deco_num_layers"], wandb.config["deco_dropout"])
        net = MyNetwork(encoder, decoder, encoder_image_preprocess=encoder_image_preprocess)
        # net.apply(xavier_init_weights)
    elif wandb.config['network_class_name'] == "MyNetworkV002":
        decoder = TF.PoseDecoder(wandb.config["num_indicators"], wandb.config["num_joints"], wandb.config["num_coords"], wandb.config["num_hiddens"], wandb.config["num_hiddens"], wandb.config["num_hiddens"], wandb.config["num_hiddens"], wandb.config["deco_norm_shape"], wandb.config["num_hiddens"], wandb.config["deco_ffn_num_hiddens"], wandb.config["deco_num_heads"], wandb.config["deco_num_layers"], wandb.config["deco_dropout"])
        net = MyNetworkV002(decoder=decoder, duplicate_mode=wandb.config['duplicate_mode'])
    return net
 
def xavier_init_weights(m):
    if type(m) == nn.Linear and m.weight.requires_grad:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU and m.weight.requires_grad:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def count_parameters(net, condition_func):
    """
    condition_func: given a parameter, return True if it should be counted, else False
    """
    return sum([_.numel() for name, _ in net.named_parameters() if condition_func(_)])
    # training_params = [name for name, _ in net.named_parameters() if _.requires_grad]
    # print(f"{training_params=}")
    # wandb.config.update({"training_params": training_params})
    # print(f"{net.log_model_info()=}")
    # net.type(data_type);
    # net = nn.DataParallel(net)
    # wandb.watch(net)
    # print(f"{net=}")
    
class MyNetworkV002(nn.Module):
    """
    This is the very first version of LGPF network. It uses CLIP to encode input images and texts. Then the modality fusion module is as native as feature dimention unifiers (one FC layer for each modality) + concatenating features together at sequence dimention (dim=1).
    """
    def __init__(self, decoder, duplicate_mode=False):
        super(MyNetworkV002, self).__init__()
        self.duplicate_mode = duplicate_mode
        self.encoder, self.encoder_image_preprocess_train_val, self.encoder_text_preprocess_train_val = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True)
        self.freeze(self.encoder)
        
        if "gradcam" in wandb.config['use_embeddings']:
            self.model_gradcam, self.gradcam_vis_processors, self.gradcam_text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device='cpu', is_eval=True)
            self.freeze(self.model_gradcam)
            # self.gradcam_projector = nn.Sequential(nn.Flatten(), nn.Sigmoid(), nn.Linear(576, 512), nn.Sigmoid(), nn.Linear(512, wandb.config["num_all_coords"]))
            self.encoder_post_process_projector = nn.Sequential(nn.Linear(1344, wandb.config['num_hiddens'])) # 768(mm feature dim)+576(gradcam feature dim)
            self.encoder_post_process_projector.apply(xavier_init_weights)
        
        self.decoder = decoder
        self.decoder.apply(xavier_init_weights)
        # self.decoder_pose_preprocess = lambda pose_groudtruth: torch.cat((torch.full((1, 84), -1), pose_groudtruth), dim=0) # add one row as 'sos'
        self.decoder_pose_preprocess = lambda pose_groudtruth: pose_groudtruth # not add one row as 'sos' == return pose_groudtruth itself
        
        self.log_net_info()
        
        # self.apply(xavier_init_weights)
        
    def forward(self, image, text, pose, image_gradcam=None, text_gradcam=None):
        """
        image: preprocessed image. (bs, 3, img_len, img_len)
        text: tokenized text tokens. (bs, seq_len, dim)
        pose: raw pose (joint coordinates) (bs, seq_len, num_indicator*num_joints*coordinator_dim). Poses don't need preprocessing since the have been normalized to 0~1 in experiments/FO settings.
        """
        if self.duplicate_mode == False:
            dec_state = self.encode_and_get_decoder_state(image, text, image_gradcam, text_gradcam)
                
            return self.decoder(pose, dec_state) # (prediction (N, forecasting steps, #coords), state)
        else:
            y = pose[:, 0:1, :].repeat(1,pose.shape[1],1)
            return y, None

    def predict(self, image, text, pose):
        """
        pose: still whole sequence of poses. [bs, sequence_len, num_all_coords]
        """
        start_token = pose[:, :1, :]
        forecasting_steps = pose.shape[1]
        dec_state = self.encode_and_get_decoder_state(image, text)
        outputs, attention_weights = [start_token,], []
        for ts in range(forecasting_steps):
            pose_pred_score, dec_state = self.decoder(outputs[-1], dec_state) # dec_state keeps being updated to contain information from pre-previous timestamp so each time just give self.decoder the previous output of it.
            outputs.append(pose_pred_score)
            attention_weights.append(self.decoder._attention_weights)
        outputs = torch.concat(outputs[1:], dim=1) # torch (bs, forecasting_step, #num_coords_84) not including trigger pose
        return outputs, dec_state
                    

    def get_device(self):
        return self.decoder.dense.weight.device
    
    def get_pose_from_gradcam(self, image, text):
        """
        image: preprocessed image. (bs, 3, img_len, img_len)
        text: preprocessed text. (bs, seq_len, dim)
        """
        # get gradcam features
        text_tokens = self.model_gradcam.tokenizer(text, return_tensors="pt", padding=True).to(self.get_device())
        gradcam, _ = compute_gradcam(self.model_gradcam, image, text, text_tokens, block_num=7)
        gradcam_features = torch.stack(gradcam, dim=0) # (B, L, 24, 24)
        
        # get pose from gradcam features
        pose_from_gradcam = self.gradcam_projector(gradcam_features)
        
        return pose_from_gradcam
    
#     def freeze_encoder(self):
#         for name, param in self.encoder.named_parameters():
#             param.requires_grad = False
            
    def freeze(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False
    def unfreeze(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = True

    def log_net_info(self):
        class_name = type(self).__name__
        
        # parameters
        total_param = count_parameters(self, condition_func=lambda p: True) # sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters()])
        trainable_param = count_parameters(self, condition_func=lambda p: p.requires_grad) # sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters() if _.requires_grad])
        trainable_portion = trainable_param/total_param
        info_dict = {f"{class_name}_parameters_total": total_param,
                     f"{class_name}_parameters_trainable": trainable_param,
                     f"{class_name}_parameters_trainable_portion": trainable_portion}
        print(f"network {class_name} info {info_dict=}")
        
        try:
            wandb.config.update(info_dict)
            print(f"wandb.config updated with {class_name}'s parameter info")
        except Exception as e:
            pass
        try:
            wandb.watch(self)
            print(f"wandb is watching {class_name}")
        except Exception as e:
            pass
        
    def log_net_info_old(self):
        # #parameters
        total_param = sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters()])
        trainable_param = sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters() if _.requires_grad]) #  and 'gradcam' not in name
        trainable_portion = trainable_param/total_param
        wandb.config.update({"parameters_total": total_param,
                             "parameters_trainable": trainable_param,
                             "parameters_trainable_portion": trainable_portion})
        try:
            wandb.watch(self)
        except Exception as e:
            pass
        
    def inference(self, image, text, pose_trigger, forecasting_steps):
        """
        enco_valid_lens: depracated. Keep it just to be consistent with the interface of MyNetwork. (so don't bother to change code outside..)
        """
        
        dec_state = self.encode_and_get_decoder_state(image, text)
        if 'gradcam' not in wandb.config['use_embeddings']:
            outputs, attention_weights = [pose_trigger], []
            for ts in range(forecasting_steps):
                pose_pred_score, dec_state = self.decoder(outputs[-1], dec_state) # dec_state keeps being updated to contain information from pre-previous timestamp so each time just give self.decoder the previous output of it.
                outputs.append(pose_pred_score)
                # attention_weights.append(self.decoder._attention_weights)
            outputs = torch.concat(outputs[1:], dim=1) # torch (bs, forecasting_step, #num_coords_84) not including trigger pose
            return outputs, dec_state
        else:
            NotImplementedError("gradcam not implemented in inference mode")
            
    def encoding(self, image, text):
        encoder_output = self.encoder.extract_features({"image": image, "text_input": text})
        return encoder_output
    
    def get_text_feature_valid_lens(self, text):
        return self.tokenize_blip(text).attention_mask.sum(dim=1) # a tensor of (N), each element is the length of the corresponding valid token sequence
    
    def tokenize_blip(self, text):
        return self.encoder.tokenizer(text, return_tensors='pt', padding=True).to(self.get_device())
        
    def encode_and_get_decoder_state(self, image, text, image_gradcam=None, text_gradcam=None):
        
        if wandb.config['use_embeddings'] == ['multimodal']:
            encoder_output = self.encoding(image, text)
            text_valid_lens = self.get_text_feature_valid_lens(text) # (N)
            post_processed_encoder_outputs = encoder_output.multimodal_embeds # (N, L, D)
            enc_valid_lens = text_valid_lens  # (N)
            
        elif wandb.config['use_embeddings'] == ['multimodal', 'image']:
            ## concate at token dim doesn't make sense ... they are two spaces of features
            encoder_output = self.encoding(image, text)
            text_valid_lens = self.get_text_feature_valid_lens(text) # (N)
            image_embeds = encoder_output.image_embeds # (N, L, D)
            post_processed_encoder_outputs = torch.concat((image_embeds, encoder_output.multimodal_embeds), dim=1) # (N, L, D), text tokend is padded among batch so need to compute text valid lens below. text/multimodal has to be on the tail since it varies in length.
            enc_valid_lens = image_embeds.shape[1] + text_valid_lens # (N)
            
        elif wandb.config['use_embeddings'] == ['multimodal', 'gradcam']:
            encoder_output = self.encoding(image, text)
            
            text_tokens = self.tokenize_blip(text) # a dict
            
            self.unfreeze(self.model_gradcam)
            gradcam, _ = compute_gradcam(self.model_gradcam, image_gradcam, text_gradcam, text_tokens, block_num=7) # [enc token gradcam (the 1st token), average gradcam across token (not includes start and end), gradcam for individual token from 1 to the eos (including) -- varies between samples]
            self.model_gradcam.zero_grad() # this is critical to prevent gradcam from being optimized
            self.freeze(self.model_gradcam)
            
            gradcam_features = torch.stack(gradcam, dim=0) # (B, L+1 (avg at 1), 24, 24)
            gradcam_features = torch.cat((gradcam_features[:, :1], gradcam_features[:, 2:]), dim=1) # drop avg gradcam. (B, L, 24, 24)
            gradcam_features = torch.flatten(gradcam_features, 2, 3).to(self.get_device()) # (B, L, 24*24)
            
            post_processed_encoder_outputs = self.encoder_post_process_projector(torch.concat((encoder_output.multimodal_embeds, gradcam_features), dim=2)) # (N, L, hidden_dim)
            
            enc_valid_lens = text_tokens.attention_mask.sum(dim=1) # (N)
            
        dec_state = self.decoder.init_state(post_processed_encoder_outputs, enc_valid_lens=enc_valid_lens.to(self.get_device()))
        return dec_state
    
    def get_multimodal_feature(self, image, text):
        """this was copied from MyNetworkv001, for CLIP"""
        raise NotImplementedError
        return_dict ={}
        image_features = self.encoder.encode_image_return_seq(image) # (NLD)
        return_dict['image_features'] = image_features.detach()
        
        image_features = self.image_post_processer(image_features.type(torch.float32))
        return_dict['image_features_post'] = image_features.detach()
        
        text_features = self.encoder.encode_text_return_seq(text) # (NLD)
        return_dict['text_features'] = text_features.detach()
        
        text_features = self.text_post_processer(text_features.type(torch.float32))
        return_dict['text_features_post'] = text_features.detach()
        
        post_processed_encoder_outputs = torch.cat([image_features, text_features], axis=1) # (NLD)
        return_dict['multimodal_features'] = post_processed_encoder_outputs.detach()
        
        return return_dict
        
    def save_encoder_feature(self, image, text, path):
        encoder_output = self.encoding(image, text)
        # manual modify # 
        encoder_output_dict = {"multimodal_embeds": encoder_output.multimodal_embeds, "image_embeds": encoder_output['image_embeds']}
        # manual modify end # 
        
        
        for sample_i in range(len(image)):
            text_as_name = text[sample_i].replace(' ', '').replace('.', '')
            for feature_name, batch_feature in encoder_output_dict.items():
                feature = batch_feature[sample_i]
                feature_path = path[sample_i].replace('.pt', f'_{text_as_name}_{feature_name}.pt')
                try:
                    # if not os.path.exists(feature_path):
                    torch.save(feature, feature_path)
                except Exception as e:
                    print(f"Saving encoder output failed. {feature_path=}, {e=}")
                    # raise e


class MyNetwork(nn.Module):
    """
    This is the very first version of LGPF network. It uses CLIP to encode input images and texts. Then the modality fusion module is as native as feature dimention unifiers (one FC layer for each modality) + concatenating features together at sequence dimention (dim=1).
    """
    def __init__(self, encoder, decoder, encoder_image_preprocess):
        super(MyNetwork, self).__init__()
        self.encoder = encoder
        image_output_dim =  encoder.visual.transformer.resblocks[11].ln_2.weight.shape[-1]
        self.image_post_processer = nn.Linear(image_output_dim, wandb.config["num_hiddens"])
        text_output_dim =  encoder.transformer.resblocks[11].ln_2.weight.shape[-1]
        self.text_post_processer = nn.Linear(text_output_dim, wandb.config["num_hiddens"])
        self.decoder = decoder
        self.encoder_image_preprocess = encoder_image_preprocess
        self.encoder_text_preprocess = clip.tokenize
        # self.decoder_pose_preprocess = lambda pose_groudtruth: torch.cat((torch.full((1, 84), -1), pose_groudtruth), dim=0) # add one row as 'sos'
        self.decoder_pose_preprocess = lambda pose_groudtruth: pose_groudtruth # not add one row as 'sos' == return pose_groudtruth itself
        
        self.freeze_encoder()
        self.log_net_info()
        
        self.apply(xavier_init_weights)
        
    def forward(self, image, text, pose, enc_valid_len):
        """
        image: preprocessed image. (bs, 3, img_len, img_len)
        text: tokenized text tokens. (bs, seq_len, dim)
        pose: untokenized pose (joint coordinates) (bs, seq_len, num_indicator, num_joints, coordinator_dim)
        """
        
        image_features = self.encoder.encode_image_return_seq(image) # (NLD)
        image_features = self.image_post_processer(image_features.type(torch.float32))
        text_features = self.encoder.encode_text_return_seq(text) # (NLD)
        text_features = self.text_post_processer(text_features.type(torch.float32))
        post_processed_encoder_outputs = torch.cat([image_features, text_features], axis=1) # (NLD)
        
        dec_state = self.decoder.init_state(post_processed_encoder_outputs, enc_valid_len)
        
        return self.decoder(pose, dec_state)
    
    def freeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            
    def log_net_info(self):
        # #parameters
        total_param = sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters()])
        trainable_param = sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters() if _.requires_grad])
        trainable_portion = trainable_param/total_param
        wandb.config.update({"parameters_total": total_param,
                             "parameters_trainable": trainable_param,
                             "parameters_trainable_portion": trainable_portion})
        try:
            wandb.watch(self)
        except Exception as e:
            pass
        
    def inference(self, image, text, pose_trigger, enco_valid_lens, forecasting_steps):
        """
        image: preprocessed image. (bs, 3, img_len, img_len)
        text: tokenized text tokens. (bs, seq_len, dim)
        """
        image_features = self.encoder.encode_image_return_seq(image) # (NLD)
        image_features = self.image_post_processer(image_features.type(torch.float32))
        text_features = self.encoder.encode_text_return_seq(text) # (NLD)
        text_features = self.text_post_processer(text_features.type(torch.float32))
        post_processed_encoder_outputs = torch.cat([image_features, text_features], axis=1) # (NLD)
        dec_state = self.decoder.init_state(post_processed_encoder_outputs, enco_valid_lens)
        outputs, attention_weights = [pose_trigger], []
        
        for ts in range(forecasting_steps):
            pose_pred_score, dec_state = self.decoder(outputs[-1], dec_state) # dec_state keeps being updated to contain information from pre-previous timestamp so each time just give self.decoder the previous output of it.
            outputs.append(pose_pred_score)
            # attention_weights.append(self.decoder._attention_weights)
        outputs = torch.concat(outputs[1:], dim=1) # torch (bs, forecasting_step, #num_coords_84) not including trigger pose
        return outputs, dec_state
        

    def get_multimodal_feature(self, image, text):
        return_dict ={}
        image_features = self.encoder.encode_image_return_seq(image) # (NLD)
        return_dict['image_features'] = image_features.detach()
        
        image_features = self.image_post_processer(image_features.type(torch.float32))
        return_dict['image_features_post'] = image_features.detach()
        
        text_features = self.encoder.encode_text_return_seq(text) # (NLD)
        return_dict['text_features'] = text_features.detach()
        
        text_features = self.text_post_processer(text_features.type(torch.float32))
        return_dict['text_features_post'] = text_features.detach()
        
        post_processed_encoder_outputs = torch.cat([image_features, text_features], axis=1) # (NLD)
        return_dict['multimodal_features'] = post_processed_encoder_outputs.detach()
        
        return return_dict
        
        
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_len):
        """
        args:
            dec_X: (bs, seq_len(window_size), num_hiddens). Includes the hand pos in cur frame (starting pos)
        """
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_len)        
        return self.decoder(dec_X, dec_state)

class TimestampsWeightTransformer(nn.Module):
    def __init__(self, hidden_size, num_all_coords):
        super(TimestampsWeightTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=1, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, batch_first=True)
        self.tsw_pose_tokenizer = nn.Linear(num_all_coords, hidden_size)
        self.header = nn.Sequential(*[nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)])
        
        self.log_net_info()
        
    def forward(self, mm_ebd, mm_valid_len, pose):
        """
        mm_ebd: (batch_size, mm_token_length, mm_feature_dim)
        mm_valid_len: (B, mm_token_length)
        pose: (batch_size, forecasting_steps, num_all_coords)
        """
        pose_ebd = self.tsw_pose_tokenizer(pose)
        src_key_padding_mask = TF.sequence_mask(torch.zeros(mm_ebd.shape[0], mm_ebd.shape[1]).to(mm_valid_len.device),
                                                mm_valid_len,
                                                value=1)
        memory_key_padding_mask = src_key_padding_mask
        y = self.transformer(mm_ebd, pose_ebd, src_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        y = self.header(y)

        return y
    
    def log_net_info(self):
        class_name = type(self).__name__
        
        # parameters
        total_param = count_parameters(self, condition_func=lambda p: True) # sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters()])
        trainable_param = count_parameters(self, condition_func=lambda p: p.requires_grad) # sum([torch.prod(torch.tensor(_.shape)).item() for name, _ in self.named_parameters() if _.requires_grad])
        trainable_portion = trainable_param/total_param
        info_dict = {f"{class_name}_parameters_total": total_param,
                     f"{class_name}_parameters_trainable": trainable_param,
                     f"{class_name}_parameters_trainable_portion": trainable_portion}
        print(f"network {class_name} info {info_dict=}")
        
        try:
            wandb.config.update(info_dict)
            print(f"wandb.config updated with {class_name}'s parameter info")
        except Exception as e:
            pass
        try:
            wandb.watch(self)
            print(f"wandb is watching {class_name}")
        except Exception as e:
            pass
            
