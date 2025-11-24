import pdb
import time
from ..environments.simulation_environment import PushingAmongObstaclesEnv
from ..utils.load_config import multi_agent_config as config
from ..utils.rate_for_simulation import Rate
from ..controllers.state_machine_controller import ControlStateMachine
import numpy as np
import os
from tqdm import tqdm

REACHED_GOAL = 8

def run_carpool_simulation(test_case, r1, r2, at_pushing_pose=True, path_tracking_config=None):
    sim_env = PushingAmongObstaclesEnv(config.ENV_NAME, test_case=test_case, render_mode='rgb_array', sweep=True)
    obs = sim_env.set_init_states()
    object_goal_pose = sim_env.object_goal_pose
    rate = Rate(1 / config.dt)

    start_time = time.time()
    max_time = 100
    state_machine = ControlStateMachine(sim_env, object_goal_pose, obs, r1, r2, at_pushing_pose, path_tracking_config)
    car1_history, car2_history, block_history = [], [], []
    while state_machine.state != REACHED_GOAL and time.time() - start_time < max_time:
        action = state_machine.execute()
        state_machine.obs, _, _, _, _ = sim_env.step(action)
        car1, car2, block = state_machine.update_poses()
        car1_history.append(car1)
        car2_history.append(car2)
        block_history.append(block)
        rate.sleep()
    sim_env.close()
    if time.time() - start_time > max_time:
        print("Simulation timed out.")
    original_path = state_machine.object_plan
    execution_time = time.time() - start_time
    return car1_history, car2_history, block_history, original_path, execution_time, object_goal_pose


if __name__ == "__main__":
    test_cases = [9]
    num_runs = 10

    # Create directory for results
    for test_case in test_cases:
        results_dir = f'optimal_correct_test{test_case}'
        os.makedirs(results_dir, exist_ok=True)

        for i in tqdm(range(8, num_runs)):
            print(f"Running simulation {i + 1}/{num_runs}...")

            car1_hist, car2_hist, block_hist, orig_path, exec_time, goal = run_carpool_simulation(
                test_case=test_case, r1=2.1, r2=0.0, at_pushing_pose=False
            )

            # Save this run immediately
            np.savez_compressed(
                os.path.join(results_dir, f'run_{i:03d}.npz'),
                car1_history=np.array(car1_hist),
                car2_history=np.array(car2_hist),
                block_history=np.array(block_hist),
                original_path=np.array(orig_path),
                execution_time=exec_time,
                object_goal_pose=goal,
                run_number=i
            )
            print(f"  Saved run {i + 1}")

        print(f"All results saved to {results_dir}/")