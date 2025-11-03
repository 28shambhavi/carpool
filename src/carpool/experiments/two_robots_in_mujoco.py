import pdb
import time
from ..environments.simulation_environment import PushingAmongObstaclesEnv
from ..utils.load_config import multi_agent_config as config
from ..utils.rate_for_simulation import Rate
from ..controllers.base_controller import ControlStateMachine

REACHED_GOAL = 8
def run_carpool_simulation(test_case):
    sim_env = PushingAmongObstaclesEnv(config.ENV_NAME,
                                       test_case=test_case,
                                       render_mode='human')
    obs = sim_env.set_init_states()
    object_goal_pose = sim_env.object_goal_pose
    rate = Rate(1/config.dt)

    start_time = time.time()
    max_time = 900
    state_machine = ControlStateMachine(sim_env,
                                        rate,
                                        object_goal_pose,
                                        obs)

    while state_machine.state!= REACHED_GOAL:
        action = state_machine.execute()
        state_machine.obs, _, _, _, _ = sim_env.step(action)
        state_machine.update_poses()
        rate.sleep()
    sim_env.close()

    if state_machine.state != REACHED_GOAL:
        print("Did not reach goal, max time exceeded")

if __name__ == "__main__":
    run_carpool_simulation(test_case=7)