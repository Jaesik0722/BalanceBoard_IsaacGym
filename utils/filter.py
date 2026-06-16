import torch

class FILTER_TORCH:
    def __init__(self, num_envs, device):
        
        self.num_envs = num_envs
        self.device = device
        
        self.all_indexes = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        
        self.lowpass_filter_last_output= torch.zeros(self.num_envs, device=self.device)
        
        self.reset_lowpass_filter(self.all_indexes)
        
    def reset_lowpass_filter(self, reset_ids : torch.Tensor):
        self.lowpass_filter_last_output[reset_ids] = 0.0
        
    def update_lowpass_filter(self, input : torch.Tensor, alpha : torch.Tensor, env_ids: torch.Tensor):
        
        output = ((1 - alpha[env_ids]) * self.lowpass_filter_last_output[env_ids]) + (alpha[env_ids] * input)
        self.lowpass_filter_last_output[env_ids] = output
        
        return output
