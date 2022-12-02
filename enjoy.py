import argparse
import gym
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
import numpy as np
import matplotlib.pyplot as plt
from utils.yaml import write_yaml, read_yaml
import modified_envs
from wrapper.wrapper_MLP import RWPPO
from utils.model import model_building

def make_env(env_id, rank):
    def _init():
        env = gym.make("alr_envs:" + env_id)
        return env
    return _init


def step_based(algo: str, env_id: str, model_id: str, step: str):
    path = "./logs/" + algo + "/" + env_id + "_" + model_id
    num_envs = 1
    stats_file = 'env_normalize.pkl'
    stats_path = os.path.join(path, stats_file)
    #env = DummyVecEnv(env_fns=[make_env(env_id, i) for i in range(num_envs)])
    #env = VecNormalize.load(stats_path, env)
    #env = ObsDictWrapper(env)
    env = gym.make("modified_envs:" + env_id)

    model_path = os.path.join(path, "eval/best_model")

    #model_path = os.path.join(path, "model")

    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'rwppo': RWPPO,
        'td3': TD3
    }

    ALGO = ALGOS[algo]
    model = ALGO.load(model_path, env=env)

    obs = env.reset()
    rewards = 0
    reward = 0
    infos = []
    jump_height = []
    goal_dist = []
    if "dmc" in env_id:
        for i in range(int(step)):
            #time.sleep(0.1)
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, dones, info = env.step(action)
            rewards += reward
            #env.render(mode="rgb_array")
            env.render(mode="human")

        env.close()
    elif "Meta" in env_id:
        print("meta")
        for i in range(int(step)):
            #time.sleep(0.05)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            time.sleep(0.1)
            infos.append(info['obj_to_target'])
            reward += rewards
            print("rewrads", i, reward, action)
            env.render(False)
            #if i == 59 or i==89 or i == 199 or i == 19 or i==1:
            #   time.sleep(5)
        env.close()
        infos = np.array(infos)
        a = 1
    else:
        for i in range(int(step)):
            #time.sleep(0.05)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            reward += rewards
            print("rewrads", i, reward, action)
            #reward += info[0]["reward"]
            #jump_height.append(info[0]["height"])
            #goal_dist.append(info[0]["goal_dist"])
            env.render()
            #if i == 0 or i==50-1 or i==100-1 or i==150-1 or i==199 or i==249:
            #    time.sleep(5)
        env.close()

        import matplotlib.pyplot as plt
        jump_height = np.array(jump_height)
        goal_dist = np.array(goal_dist)
        np.savez("jump_hieght"+algo,jump_height)
        np.savez("goal_dist"+algo,goal_dist)
        # plt.plot(goal[:, 0], label='x axis')
        plt.plot(jump_height, label='TD3')
        # plt.plot(goal[:, 1], label='y axis')
        #plt.plot(goal[:, 1], label='y-axis')
        # plt.plot(goal[:, 2], label='x axis')
        plt.legend()
        plt.xlabel("timesteps")
        plt.ylabel("jump height")
        # plt.show()
        # plt.title('Goal Position')
        import tikzplotlib
        tikzplotlib.save("hopper_height.tex")
        #plt.show()
        print(reward)


def episodic(algo: str, env_id, model_id: str, step: str, seed=None):
    file_name = algo + ".yml"
    if "Meta" in env_id:
        data = read_yaml(file_name)["Meta-v2"]
        data['env_params']['env_name'] = data['env_params']['env_name'] + ":" + env_id
    else:
        data = read_yaml(file_name)[env_id]

    env_name = data["env_params"]["env_name"]

    path = "logs/" + algo + "/" + env_id + "_" + model_id + "/algo_mean.npy"
    #path = "logs/" + algo + "/" + env_id + "_" + model_id + "/best_model.npy"
    algorithm = np.load(path)
    print("algorithm", algorithm)

    if 'Meta' in env_id:
        from alr_envs.utils.make_env_helpers import make_env
        env = make_env(env_name, seed=seed)
    elif "Hopper" in env_id:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name[2:-1])
    env.reset()

    if "DeepMind" in env_id:
        env.render("rgb_array")
        env.step(algorithm)
    elif "Meta" in env_id:
        env.render(mode="meta")
        env.step(algorithm)
    else:
        env.render()
        env.step(algorithm)
    env.render()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    parser.add_argument("--model_id", type=str, help="the model")
    parser.add_argument("--step", type=str, help="how many steps rendering")

    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id
    model_id = args.model_id
    step = args.step

    STEP_BASED = ["ppo", "sac", "td3", "rwppo"]
    EPISODIC = ["dmp", "promp"]


    if algo in STEP_BASED:
        step_based(algo, env_id, model_id, step)
    elif algo in EPISODIC:
        episodic(algo, env_id, model_id, step)
    else:
        print("the algorithm (--algo) is false or not implemented")

