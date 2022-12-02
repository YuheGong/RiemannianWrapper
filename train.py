import argparse
from utils.env import env_maker, env_save
from utils.logger import logging
from utils.model import model_building, model_learn, cmaes_model_training
from utils.yaml import write_yaml, read_yaml
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
from utils.csv import csv_save


def train(algo: str, env_id: str, seed=None):
    file_name = algo +".yml"

    if "Meta" in args.env_id:
        data = read_yaml(file_name)["Meta-v2"]
        data['env_params']['env_name'] = data['env_params']['env_name'] + ":" + args.env_id
    else:
        data = read_yaml(file_name)[env_id]

    # create log folder
    path = logging(data['env_params']['env_name'], data['algorithm'])
    data['path'] = path
    data['seed'] = seed

    # make the environment
    env = env_maker(data, num_envs=data["env_params"]['num_envs'], seed=seed)
    eval_env = env_maker(data, num_envs=1, training=False, norm_reward=False, seed=seed)

    # make the model and save the model
    model = model_building(data, env, seed)

    # csv file path
    data["path_in"] = data["path"] #+ '/' + data['algorithm'].upper() + '_1'
    data["path_out"] = data["path"] + '/data.csv'

    try:
        eval_env_path = data['path'] + "/eval/"
        model_learn(data, model, eval_env, eval_env_path)
    except KeyboardInterrupt:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env, eval_env)
        csv_save(data)
        print('')
        print('training interrupt, save the model and config file to ' + data["path"])
    else:
        data["algo_params"]['num_timesteps'] = model.num_timesteps
        write_yaml(data)
        env_save(data, model, env, eval_env)
        csv_save(data)
        print('')
        print('training FINISH, save the model and config file to ' + data['path'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="the algorithm")
    parser.add_argument("--env_id", type=str, help="the environment")
    parser.add_argument("--stop_cri", type=str, help="whether you set up stop criterion or not")
    parser.add_argument("--seed", type=int, help="seed for training")

    args = parser.parse_args()

    if not args.algo and not args.env_id:
        parser.error('Please specify an algorithm (--algo) and an environment (--env_id) to train or enjoy')

    algo = args.algo
    env_id = args.env_id
    stop_cri = args.stop_cri
    STEP_BASED = ["ppo", "sac", "ddpg", "td3", "rwppo"]
    #print("algo", algo)
    EPISODIC = ["dmp", "promp"]
    if algo in STEP_BASED:
        train(algo, env_id, seed=args.seed)
    else:
        print("the algorithm " + algo + " is false or not implemented")

