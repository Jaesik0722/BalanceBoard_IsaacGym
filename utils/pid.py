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

class PID:
    def __init__(self, kp=0.2, ki=0.0, kd=0.0, dt=0.1, set_point=None):

        self.kp = kp
        self.Ki = ki
        self.Kd = kd
        
        self.dt = dt
        
        self.set_point = set_point

        self.reset_pid()

    def reset_pid(self):
        self.set_point = 0.0

        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.last_error = 0.0
        self.last_input = 0.0

    def update_pid_control(self, input):
        error = self.set_point - input
        delta_error = error - self.last_error

        self.p_term = error
        self.i_term += error * self.dt

        self.d_term = delta_error / self.dt

        self.last_error = error

        output = (self.kp * self.p_term) + (self.ki * self.i_term) + (self.kd * self.d_term)

        return output

    def update_pd_control(self, input):
        error = self.set_point - input
        delta_error = error - self.last_error

        self.p_term = error

        self.d_term = delta_error / self.dt

        self.last_error = error

        output = (self.kp * self.p_term) + (self.kd * self.d_term)
        
        return output

    def tune_pid_gains(self, p_gain, i_gain, d_gain):
        self.kp, self.ki, self.kd = p_gain, i_gain, d_gain
        
    def tune_pd_gains(self, p_gain, d_gain):
        self.kp, self.kd = p_gain, d_gain
