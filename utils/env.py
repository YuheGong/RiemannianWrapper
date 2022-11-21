import gym
import os
from gym import wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.obs_dict_wrapper import  ObsDictWrapper

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

def env_maker(data: dict, num_envs: int, training=True, norm_reward=True, seed=None):
    if data["env_params"]['wrapper'] == "VecNormalize":
        env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(num_envs)])
        env = VecNormalize(env, training = training, norm_obs=True, norm_reward=norm_reward)
    #elif "Fetch" in data["env_params"]['env_name']:
        #env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(num_envs)])
        #env = ObsDictWrapper(env)
    elif "Meta" in data["env_params"]['env_name']:
        env = gym.make(data["env_params"]['env_name'], seed=seed)
    elif "Hopper" in data["env_params"]['env_name']:
        env = gym.make(data["env_params"]['env_name'])
    else:
        env = gym.make(data["env_params"]['env_name'])
    return env

def env_save(data: dict, model, env, eval_env):
    model_path = os.path.join(data['path'],  "model.zip")
    model.save(model_path)
    if 'VecNormalize' in data['env_params']['wrapper']:
        # save env
        stats_path = os.path.join(data['path'], "env_normalize.pkl")
        env.save(stats_path)
        # save evaluation env
        eval_stats_path = os.path.join(data['path'], "eval_env_normalize.pkl")
        eval_env.save(eval_stats_path)

def env_continue_load(data: dict):
    if data["env_params"]['wrapper'] == "VecNormalize":
        env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(data["env_params"]['num_envs'])])
        stats_path = os.path.join(data['continue_path'], 'env_normalize.pkl')
        print(stats_path)
        env = VecNormalize.load(stats_path, env)
        # read
        eval_env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(1)])
        eval_stats_path = os.path.join(data['continue_path'], "eval_env_normalize.pkl")
        eval_env = VecNormalize.load(eval_stats_path, eval_env)
    else:
        env = gym.make(data["env_params"]['env_name'])
    return env, eval_env
