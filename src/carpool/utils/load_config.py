import yaml
import argparse

class _Config:
    def __init__(self, num_agents=2):
        print("Loading config")
        if num_agents==1:
            with open("config/single_agent.yaml") as f:
                self.config = yaml.full_load(f)

        else:
            with open("carpool/config/decentralized_mujoco.yaml") as f:
                self.config = yaml.full_load(f)

        parser = argparse.ArgumentParser(description = 'MyApp')
        parser.add_argument('--an_arg')
        self.args = parser.parse_args()

    def __getattr__(self, name):
        try:
            return self.config[name]
        except KeyError:
            return getattr(self.args, name)

# config = _Config()
multi_agent_config = _Config(num_agents=2)
# if __name__ == "__main__":
#     print(config.noise_sigma)