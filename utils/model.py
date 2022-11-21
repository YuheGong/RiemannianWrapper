import numpy as np
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
import torch as th
from stable_baselines3.common.noise import NormalActionNoise
from .callback import callback_building
from wrapper.wrapper_MLP import RWPPO
from wrapper.custom_policy import CustomActorCriticPolicy

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


def cmaes_model_training(algorithm, env, success_full, success_mean, path, log_writer, opts, t, env_id, opt_best):
    fitness = []
    print("----------iter {} -----------".format(t))
    solutions = np.vstack(algorithm.ask())

    #solutions = solutions.clip(-1,1)
    #a = th.distributions.MultivariateNormal(th.zeros(20), th.Tensor(algorithm.sm.covariance_matrix))
    #entro = a.entropy()


    for i in range(len(solutions)):
        env.reset()

        _, reward, done, infos = env.step(solutions[i])
        if "DeepMind" in env_id:
            success_full.append(env.env.success)
        print('reward', -reward)
        fitness.append(-reward)

        #env.optimizer.zero_grad()


    '''
    import torch
    print("infos",infos["trajectory"].shape)
    print("actions", infos['step_actions'].shape)
    print("observations", infos['step_observations'].shape)
    loss = np.sum(infos["trajectory"] - infos['step_observations'],axis=1)
    print("shape", loss.shape)
    
    loss = torch.mean(torch.Tensor(loss))
    import tensorflow as 1tf
    #loss = tf.Variable(loss, requires_grad=True)
    loss_func = torch.nn.MSELoss()
    from torch.autograd import Variable
    #x = torch.unsqueeze(
    x = torch.unsqueeze(torch.Tensor(infos["trajectory"]), dim=1)
    y = torch.unsqueeze(torch.Tensor(infos['step_observations']), dim=1)
    x.requires_grad_()
    y.requires_grad_()
    from torch.autograd import Variable
    #x, y = (x, y)
    loss = loss_func(x, y)
    #loss.requres_grad = True
    #loss_func = torch.nn.MSELoss()
    #loss = loss_func(loss)
    loss.backward()
    env.optimizer.step()
    '''


    algorithm.tell(solutions, fitness)
    sigma = algorithm.sigma
    opt=0
    info = []
    success = 0
    eval_num = 10
    for i in range(eval_num):
        env.reset()
        _, op, __, infos = env.step(np.array(algorithm.mean))
        #print("op", op)
        opt += op
        if "Meta" in env_id:
            success += infos["last_info"]["success"]
        elif "Hopper" in env_id:
            max_height = env.max_height
            min_goal_dist = env.min_goal_dist
        # info["success"].append(infos["last_info"]["success"])
        # info["success"].append(infos["last_info"]["success"])
        # info["success"].append(infos["last_info"]["success"])
    opt = opt/eval_num
    success = success/eval_num

    #opt=env.env.rewards_no_ip

    np.save(path + "/algo_mean.npy", np.array(algorithm.mean).clip(-1,1))
    if t == 0:
        opt_best = opt
    if opt > opt_best:
        opt_best = opt
        np.save(path + "/best_model.npy", algorithm.mean)

    print("opt", opt)
    opts.append(opt)
    t += 1

    if "DeepMind" in env_id:
        success_mean.append(env.env.success)
        if success_mean[-1]:
            success_rate = 1
        else:
            success_rate = 0

        b = 0
        for i in range(len(success_full)):
            if success_full[i]:
                b += 1
        success_rate_full = b / len(success_full)
        success_full = []
        log_writer.add_scalar("iteration/success_rate_full", success_rate_full, t)
        log_writer.add_scalar("iteration/success_rate", success_rate, t)
        log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, t)
        log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, t)

    step = t * env.traj_steps * solutions.shape[0]
    log_writer.add_scalar("eval/mean_reward", opt, step)
    log_writer.add_scalar("eval/sigma", sigma, step)
    #log_writer.add_scalar("eval/entropy", entro, step)
    if "Meta" in env_id:
        log_writer.add_scalar("eval/last_success", success, step)
        log_writer.add_scalar("eval/last_object_to_target", infos["last_info"]["obj_to_target"], step)
        #log_writer.add_scalar("eval/min_object_to_target", infos["last_info"]["obj_to_target"], step)
        log_writer.add_scalar("eval/control_cost", np.sum(np.square(infos["step_actions"])), step)
    elif "Hopper" in env_id:
        log_writer.add_scalar("eval/max_height", max_height, step)
        log_writer.add_scalar("eval/min_goal_dist", min_goal_dist, step)


    #log_writer.add_scalar("iteration/dist_vec", env.env.dist_vec, t)
    '''
    for i in range(len(algorithm.mean)):
        log_writer.add_scalar(f"algorithm_params/mean[{i}]", algorithm.mean[i], t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_mean[{i}]", np.mean(algorithm.C[i]), t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_variance[{i}]", np.var(algorithm.C[i]), t)
    '''
    return algorithm, env, success_full, success_mean, path, log_writer, opts, t, opt_best


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
    else:
        activation_fn = None
    return dict(activation_fn=activation_fn, net_arch=net_arch)
