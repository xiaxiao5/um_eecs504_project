import torch
import torch.nn as nn
import network as N

class ForecastingWeightedMSELoss(nn.Module):
    def __init__(self, kwargs):
        super(ForecastingWeightedMSELoss, self).__init__()
        """

        self.weights is a (1(B), T, 1) tensor to be multiplied on unreduced MSE loss (B,T,num_all_coords)
        """
        self.mse_loss = nn.MSELoss(reduction='none')
        if kwargs['loss_type'] == "mse_linear_down":
            # Create a weight tensor
            self.weights = torch.linspace(1.0, 0.1, steps=kwargs['forecasting_steps'])
            # Expand dims to match the shape of pred and label
            self.weights = self.weights.unsqueeze(0).unsqueeze(2)
            
        elif kwargs['loss_type'] == "mse_sine": 
            # Create a weight tensor
            self.weights = torch.sin(torch.linspace(0, torch.pi, steps=kwargs['forecasting_steps']))
            # Expand dims to match the shape of pred and label
            self.weights = self.weights.unsqueeze(0).unsqueeze(2)
            
        elif kwargs['loss_type'] == "mse_revsine":
            # Create a weight tensor
            sine_wave = torch.sin(torch.linspace(0, torch.pi, steps=kwargs['forecasting_steps']))
            self.weights = -(sine_wave - torch.max(sine_wave)/2) + torch.max(sine_wave)/2
            # Expand dims to match the shape of pred and label
            self.weights = self.weights.unsqueeze(0).unsqueeze(2)
            
        elif kwargs['loss_type'] == "mse_learnable": 
            # changing based on
            self.weight_predictor = N.TimestampsWeightTransformer(kwargs['hidden_size'], kwargs['num_all_coords'])

        else:
            raise NotImplementedError(f"{kwargs['loss_type']=} not implemented")
        try:
            # print(f"{self.weights=}")
            print()
        except:
            pass
        
        self.kwargs = kwargs
        
    def forward(self, kwargs):
        """
        pred, label: (required) (B, T, num_all_coords)
        mm_valid_ebd: (mse_learnable) (B, valid_mm_len, hidden_size)
        """
        pred = kwargs['pred']
        label = kwargs['label']
        # prep
        assert pred.shape == label.shape, "Predicted and label tensors must have the same shape"
        assert pred.device == label.device, "Predicted and label tensors must at the same device"

        # learnable
        if self.kwargs['loss_type'] == "mse_learnable":
            self.weights = self.weight_predictor(kwargs["mm_ebd"], kwargs["mm_valid_len"], label)
        
        # Compute MSE loss
        self.weights = self.weights.to(pred.device)
        loss = self.mse_loss(pred, label)
        
        # Apply the weights
        weighted_loss = loss * self.weights
        
        # Average over all elements
        final_loss = weighted_loss.mean()

        return final_loss

def get_loss_func():
    import wandb
    if wandb.config['loss'] in ["mse", "weighted_mse", "rmse", "weighted_rmse"]:
        loss_func = torch.nn.MSELoss()
    elif wandb.config['loss'] in ["mse_linear_down", "mse_sine", "mse_revsine", "mse_learnable"]:
        kwargs = {
            'loss_type': wandb.config['loss'],
            'forecasting_steps': wandb.config['forcasting_steps'],
            'hidden_size': wandb.config['num_hiddens'],
            'num_all_coords': wandb.config['num_all_coords'],
        }
        loss_func = ForecastingWeightedMSELoss(kwargs)
        if wandb.config['loss'] in ['mse_learnable']:
            loss_func.weight_predictor.to(wandb.config['device'])
    else:
        raise NotImplementedError
    
    return loss_func


if __name__ == "__main__":
    loss_init_kwargs = {
        'loss_type': 'mse_linear_down',
        'hidden_size': 256,
        'num_all_coords': 84,
        'forecasting_steps': 30
    }
    loss_func = ForecastingWeightedMSELoss(loss_init_kwargs)
    loss_forward_kwargs = {
        'pred': torch.rand(32, 30, 84),
        'label': torch.rand(32, 30, 84),
        'mm_ebd': torch.rand(32, 5, 256),
        'mm_ebd': torch.rand(32, 5, 256),
    }
    l = loss_func(loss_forward_kwargs)
    print(l)
    
