import numpy as np

class PathTracking():
    def set_trajectory(self, trajectory):
        self.reset()
        self.cost.set_trajectory(trajectory)
        self.replan = False

    def smooth_action(self, action, coordination_factor=1.0):
        """
        Enhanced smoothing with coordination factor
        coordination_factor: 1.0 = normal, <1.0 = slower/more cautious
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Apply coordination factor to speed
        action[1] *= coordination_factor
        
        # Enhanced exponential moving average with acceleration limits
        max_accel_change = 0.05  # Limit sudden acceleration changes
        max_steer_change = 0.1   # Limit sudden steering changes
        
        # Limit steering angle changes
        steer_diff = action[0] - self.smoothed_action[0]
        if abs(steer_diff) > max_steer_change:
            action[0] = self.smoothed_action[0] + np.sign(steer_diff) * max_steer_change
        
        # Limit acceleration changes
        accel_diff = action[1] - self.smoothed_action[1]
        if abs(accel_diff) > max_accel_change:
            action[1] = self.smoothed_action[1] + np.sign(accel_diff) * max_accel_change
        
        # Apply exponential moving average
        self.smoothed_action = self.alpha * self.smoothed_action + (1 - self.alpha) * action
        
        # Store in history for future reference
        self.action_history.append(self.smoothed_action.copy())
        if len(self.action_history) > 10:  # Keep last 10 actions
            self.action_history.pop(0)
        
        return self.smoothed_action.copy()
    
    def get_reference_index(self, obs):
        self.index = self.cost.get_reference_index(obs)
        if self.cost.change_dir == True:
            pass
        self.ctrl.sample_null_action = self.cost.sample_null()
        self.reached_goal = self.cost.sample_null()
        self.replan = self.cost.replan
        return self.index
    
    def get_tracking_error(self, obs):
        """Return the current tracking error for coordination purposes"""
        if hasattr(self.cost, 'trajectory') and self.cost.trajectory is not None:
            ref_point = self.cost.trajectory[self.index, :2]
            current_point = obs[:2]
            return np.hypot(ref_point[0] - current_point[0], ref_point[1] - current_point[1])
        return 0.0
    
    def check_direction(self, car, ref):
        v = np.array([ref[0] - car[0], ref[1] - car[1]])
        d = np.array([np.cos(car[2]), np.sin(car[2])])

        dp = np.dot(v, d)
        if dp >= 0:
            forward = True
        else:
            forward = False
        
        if forward != self.forward:
            if forward == True:
                self.ctrl.set_forward()
            else:
                self.ctrl.set_reverse()
        
        self.forward = forward
