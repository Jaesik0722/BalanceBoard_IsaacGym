import torch

class PID_TORCH:
    def __init__(self, num_envs, pid_gains : torch.Tensor, 
                 dt, set_point, device):

        self.num_envs = num_envs
        self.device = device
        
        self.all_indexes = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        
        self.kp = pid_gains[:,0]
        self.ki = pid_gains[:,1]
        self.kd = pid_gains[:,2]
        
        self.dt = dt
        
        self.last_error= torch.zeros(self.num_envs, device=self.device)
        
        self.set_point = set_point

        self.reset_pid(self.all_indexes)

    def reset_pid(self, reset_ids : torch.Tensor):
        self.set_point = 0.0
        
        self.last_error[reset_ids] = 0.0
    
    def update_pd_control(self, input : torch.Tensor, env_ids : torch.Tensor):
        error = self.set_point - input
        delta_error = error - self.last_error[env_ids]
        
        p_term = error
        
        d_term = delta_error / self.dt
        
        self.last_error[env_ids] = error

        output = (self.kp[env_ids] * p_term) + (self.kd[env_ids] * d_term)
        
        return output

    def tune_pid_gains(self, p_gain : torch.Tensor, i_gain : torch.Tensor, d_gain : torch.Tensor, env_ids : torch.Tensor):
        self.kp[env_ids] = p_gain[env_ids]
        self.ki[env_ids] = i_gain[env_ids]
        self.kd[env_ids] = d_gain[env_ids]
        
    def tune_pd_gains(self, p_gain : torch.Tensor, d_gain : torch.Tensor, env_ids : torch.Tensor):
        self.kp[env_ids] = p_gain
        self.kd[env_ids] =  d_gain
