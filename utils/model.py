import numpy as np
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
import torch as th
from stable_baselines3.common.noise import NormalActionNoise
from .callback import callback_building
from wrapper.wrapper_MLP import RWPPO
from wrapper.custom_policy import CustomActorCriticPolicy
from geoopt.optim import RiemannianAdam, RiemannianSGD


def model_building(data, env, seed=None):
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
    ALGO = ALGOS[data['algorithm']]

    if "policy_kwargs" in data["algo_params"]:
        policy_kwargs = policy_kwargs_building(data)
    else:
        policy_kwargs = None

    if "special_policy" in data['algo_params']:
        policy = CustomActorCriticPolicy
    else:
        policy = data['algo_params']['policy']

    if data['algorithm'] == "ppo" or "rwppo":
       model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, #create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'],
                     n_steps=data["algo_params"]['n_steps'])
    elif data['algorithm'] == "sac":
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, #create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     train_freq=data["algo_params"]["train_freq"],
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'],
                     gradient_steps=data["algo_params"]['gradient_steps'])
    elif data['algorithm'] == "ddpg":
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'])
    elif data['algorithm'] == "td3":
        n_actions =env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1* np.ones(n_actions))
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, #create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     learning_rate=data["algo_params"]['learning_rate'],action_noise=action_noise,
                     batch_size=data["algo_params"]['batch_size'])#,
                     #gradient_steps=data["algo_params"]['gradient_steps'],
                     #train_freq=data["algo_params"]["train_freq"])
    else:
        print("the model initialization function for " + data['algorithm'] + " is still not implemented.")

    return model


def model_learn(data, model, test_env, test_env_path):
    # choose the tensorboard callback function according to the environment wrapper

    eval_callback = callback_building(env=test_env, path=test_env_path, data=data)
    model.learn(total_timesteps=int(data['algo_params']['total_timesteps']) , callback=eval_callback,  eval_env=test_env)



def policy_kwargs_building(data):
    net_arch = {}
    if data["algo_params"]["policy_type"] == "on_policy":
        net_arch["pi"] = [int(data["algo_params"]["policy_kwargs"]["pi"]), int(data["algo_params"]["policy_kwargs"]["pi"])]
        net_arch["vf"] = [int(data["algo_params"]["policy_kwargs"]["vf"]), int(data["algo_params"]["policy_kwargs"]["vf"])]
        net_arch = [dict(net_arch)]
    elif data["algo_params"]["policy_type"] == "off_policy":
        net_arch["pi"] = [int(data["algo_params"]["policy_kwargs"]["pi"]), int(data["algo_params"]["policy_kwargs"]["pi"])]
        net_arch["qf"] = [int(data["algo_params"]["policy_kwargs"]["qf"]), int(data["algo_params"]["policy_kwargs"]["qf"])]

    if data["algo_params"]["policy_kwargs"]["activation_fn"] == "tanh":
        activation_fn = th.nn.Tanh
    elif data["algo_params"]["policy_kwargs"]["activation_fn"] == "sigmoid":
        activation_fn = th.nn.Sigmoid
    else:
        activation_fn = None

    if "optimizer" in data['algo_params']:
        optimizer_class = RiemannianAdam#th.optim.SGD
        return dict(activation_fn=activation_fn, net_arch=net_arch, optimizer_class=optimizer_class)
    return dict(activation_fn=activation_fn, net_arch=net_arch)
