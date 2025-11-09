import pdb
import time
from ..environments.simulation_environment import PushingAmongObstaclesEnv
from ..utils.load_config import multi_agent_config as config
from ..utils.rate_for_simulation import Rate
from ..controllers.state_machine_controller import ControlStateMachine
import numpy as np

REACHED_GOAL = 8

def run_carpool_simulation(test_case, at_pushing_pose=True, path_tracking_config=None):
    sim_env = PushingAmongObstaclesEnv(config.ENV_NAME, test_case=test_case, render_mode='human')
    obs = sim_env.set_init_states()
    object_goal_pose = sim_env.object_goal_pose
    rate = Rate(1 / config.dt)

    start_time = time.time()
    max_time = 1200
    state_machine = ControlStateMachine(sim_env, object_goal_pose, obs, at_pushing_pose, path_tracking_config)

    while state_machine.state != REACHED_GOAL and time.time() - start_time < max_time:
        action = state_machine.execute()
        state_machine.obs, _, _, _, _ = sim_env.step(action)
        state_machine.update_poses()
        rate.sleep()
    sim_env.close()

if __name__ == "__main__":
    run_carpool_simulation(test_case=6, at_pushing_pose=False)
