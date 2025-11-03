from pytorch_mppi.src.pytorch_mppi.mppi import MPPI
from torch.distributions.multivariate_normal import MultivariateNormal

class MPPI_R(MPPI):
    def change_direction(self):
        current_max_velocity = self.u_max[1]
        current_min_velocity = self.u_min[1]

        if current_max_velocity > 0:
            new_max_velocity = 0  
            new_min_velocity = -current_max_velocity 
            mu = -current_max_velocity/2
        else:
            new_max_velocity = -current_min_velocity  
            new_min_velocity = 0  
            mu = -current_min_velocity/2

        self.u_max[1] = new_max_velocity
        self.u_min[1] = new_min_velocity

        self.noise_mu[1] = mu
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        self.reset()

    def set_forward(self):
        current_min_velocity = self.u_min[1]
        if current_min_velocity < 0:
            self.change_direction()
    
    def set_reverse(self):
        current_max_velocity = self.u_max[1]
        if current_max_velocity > 0:
            self.change_direction()
