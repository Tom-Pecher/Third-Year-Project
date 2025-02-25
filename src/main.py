
from envs.basic.basic_env import BasicTrafficEnv
from envs.basic_sanity.basic_sanity_env import BasicSanityTrafficEnv
from envs.basic_random.basic_random_env import BasicRandomTrafficEnv
from agents.DQN.DQN_Agent import DQNAgent

if __name__ == "__main__":
    # env = BasicSanityTrafficEnv("envs/basic_sanity/sumo/basic_sanity.sumocfg")
    env = BasicRandomTrafficEnv("envs/basic_random/sumo/basic_random.sumocfg")
    # env = BasicTrafficEnv("envs/basic/sumo/basic.sumocfg")

    from agents.Random.Random_Agent import RandomAgent
    # agent1 = DQNAgent(env, wandb_on=True)
    # agent1.train(1000)

    agent2 = RandomAgent(env)
    # agent2.load("DQN_1000.pth")
    agent2.run(1, sumo_gui=True)

    # from agents.FixedDuration.Fixed_Duration_Agent import FixedDurationAgent
    # agent3 = FixedDurationAgent(env)
    # agent3.run(1, sumo_gui=True)
