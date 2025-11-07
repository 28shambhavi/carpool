import argparse
from experiments.two_robots_in_mujoco import run_carpool_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_case', default='1')
    args = parser.parse_args()
    print(f"Running simulation with test case: {args.test_case}")
    run_carpool_simulation(args.config)